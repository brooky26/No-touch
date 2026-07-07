"""
Deriv Expiry Range / No Touch Auto-Trading Bot
===============================================
Python port of the "QUANTUM ENDS-IN-OUT / NO TOUCH" Deriv Blockly (DBot)
strategy, extended with Monte Carlo duration + barrier selection for BOTH
contract types.

STRATEGY (ported from the original XML)
----------------------------------------
- EMA(7) / EMA(14) computed on 60s candles built from the live tick stream
  (same candle interval as the original bot).
- A regime read on every new candle decides WHICH contract to trade:
    * Trending / low-choppiness regime  -> NO TOUCH
      (mirrors the original bot's entry trigger: EMA(7) crossing below
      EMA(14) - i.e. was 7>14 on the prior candle, now 7<14 - confirmed by
      a black/bearish candle. A clean, confirmed cross means price is
      moving away from a level with room to place a barrier it likely
      won't come back to touch.)
    * Choppy / ranging regime -> EXPIRY RANGE ("ends in/out")
      (small EMA gap + low directional persistence = price is oscillating,
      which is exactly the condition ends-in-range wants.)
- Regime strength is measured with a lightweight Wilder-style ADX and the
  normalised EMA(7)-EMA(14) gap - no external ML stack required.

MONTE CARLO DURATION + BARRIER SELECTION
-----------------------------------------
Neither the duration nor the barrier is hardcoded. For whichever contract
the regime read selects, the bot:
  1. Estimates live tick volatility (rolling std of log returns).
  2. Simulates thousands of GBM-style price paths at each candidate
     duration.
  3. For NO TOUCH: picks the barrier distance (path-dependent - the
     simulation tracks the running min/max of every path) that puts
     P(no touch) at the target confidence level.
  4. For EXPIRY RANGE: picks the barrier width (terminal-value only -
     ends-in-range only cares where the path finishes, not what it
     touched along the way) that puts P(ends inside range) at the target
     confidence level.
  5. Confirms the surviving candidates against Deriv's LIVE proposal
     price (simulated probability != Deriv's priced payout) and buys
     whichever (duration, barrier) combination has the best expected
     value, never just the "safest" one.

Duration floors (per your spec):
  EXPIRY RANGE : >= 2 minutes
  NO TOUCH     : >  2 minutes (starts at 3 minutes)

MONEY MANAGEMENT (ported 1:1 from the original bot's variables)
-----------------------------------------------------------------
  First Stake, Martingale Factor, Martingale Level ("Do Martingale After"),
  Target Profit, Stop Loss - same semantics as the Blockly bot: on a loss,
  stake is multiplied by the martingale factor for up to N steps; on a win,
  stake resets to First Stake. The bot stops once cumulative profit hits
  +Target Profit or -Stop Loss.

CONNECTION LAYER
-----------------
Reused verbatim from deriv_multisymbol_bot.py: REST OTP bootstrap against
the new Deriv Options API, auto-reconnecting WebSocket client with
exponential backoff and subscription replay.

ENV VARS
---------
  DERIV_APP_ID        - your app_id from developers.deriv.com
  DERIV_API_TOKEN     - API token for your Deriv account
  DERIV_ACCOUNT_TYPE  - "demo" (default) or "real"
  DERIV_ACCOUNT_ID    - optional, skips the accounts lookup

  SYMBOLS             - comma-separated list, default "R_10,R_25,R_50,R_75,R_100"
  TARGET_PROFIT       - default 5
  STOP_LOSS           - default 25
  FIRST_STAKE         - default 1
  MARTINGALE_FACTOR   - default 3.1   (matches the original bot's default)
  MARTINGALE_LEVEL    - default 3     (max martingale steps)
  MC_TARGET_PROB      - default 0.70  (Monte Carlo confidence target)
"""

import asyncio
import json
import os
import random
import sys
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import requests
import websockets

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None

API_BASE      = "https://api.derivws.com"
ACCOUNTS_PATH = "/trading/v1/options/accounts"
OTP_PATH      = "/trading/v1/options/accounts/{account_id}/otp"

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "R_10,R_25,R_50,R_75,R_100").split(",") if s.strip()]

CANDLE_INTERVAL_SEC = 60          # same candle interval as the original bot
EMA_FAST = 7
EMA_SLOW = 14
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 20          # >= this -> trending regime -> NO TOUCH
EMA_GAP_TREND_PCT = 0.0015        # |EMA7-EMA14| / price >= this also counts as trending

# Money management (mirrors the original bot's block-defined defaults)
TARGET_PROFIT     = float(os.getenv("TARGET_PROFIT", "5"))
STOP_LOSS         = abs(float(os.getenv("STOP_LOSS", "25")))
FIRST_STAKE       = float(os.getenv("FIRST_STAKE", "1"))
MARTINGALE_FACTOR = float(os.getenv("MARTINGALE_FACTOR", "3.1"))
MARTINGALE_LEVEL  = int(float(os.getenv("MARTINGALE_LEVEL", "3")))
MIN_STAKE         = 0.35

# Monte Carlo
MC_SIMULATIONS   = int(os.getenv("MC_SIMULATIONS", "8000"))
MC_TARGET_PROB   = float(os.getenv("MC_TARGET_PROB", "0.70"))   # confirmed with you: 70%
MC_SIGMA_GRID    = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

# Duration floors per your spec (minutes)
EXPIRYRANGE_MIN_MINUTES = 2
NOTOUCH_MIN_MINUTES     = 3       # "more than 2 minutes"
EXPIRYRANGE_DURATIONS   = [2, 3, 5, 7, 10, 15]
NOTOUCH_DURATIONS       = [3, 5, 7, 10, 15, 20, 30]

TRADE_COOLDOWN_SEC = 30           # avoid re-entering on every single new candle
CANDIDATE_TOP_K     = 6           # how many MC candidates get a live quote check


# ---------------------------------------------------------------------------
# CONNECTION LAYER - reused from deriv_multisymbol_bot.py
# ---------------------------------------------------------------------------
class DerivClient:
    """
    Client for the new Deriv Options API.

    Auth flow: REST GET .../accounts -> resolve account_id -> REST POST
    .../accounts/{id}/otp -> pre-authenticated WS URL. No `authorize`
    message is sent or needed; the OTP URL is already scoped to the account.

    OTP URLs are short-lived and single-use, so a fresh one is fetched on
    every connect AND every reconnect. After the first successful connect,
    this client auto-reconnects in the background with exponential backoff
    and calls `resubscribe_cb` (if set) so the caller can replay its
    balance/tick subscriptions.
    """

    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE = 2.0
    RECONNECT_CAP = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id = app_id
        self.token = token
        self.account_type = account_type
        self.account_id = account_id
        self.ws = None
        self.req_id = 0
        self.pending = {}
        self.subscriptions = defaultdict(list)  # msg_type -> list[asyncio.Queue]
        self.account = None
        self.resubscribe_cb = None
        self._running = False
        self._reader_task = None
        self._ka_task = None

    def _rest_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Deriv-App-ID": self.app_id,
            "Content-Type": "application/json",
        }

    def _resolve_account_id_sync(self):
        url = f"{API_BASE}{ACCOUNTS_PATH}"
        resp = requests.get(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(
            f"No '{self.account_type}' account found via {ACCOUNTS_PATH}. "
            f"Set DERIV_ACCOUNT_ID explicitly, or create one first via "
            f"POST {ACCOUNTS_PATH}. Accounts returned: {data}"
        )

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        url = f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}"
        resp = requests.post(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    async def connect(self):
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task = asyncio.create_task(self._heartbeat())
        bal = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(
            f"Connected ({self.account_type}). "
            f"loginid={self.account.get('loginid')} balance={self.account.get('balance')}"
        )

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[DerivClient] WS connection lost: {e}")

    async def _supervise(self):
        while self._running:
            if self._reader_task is not None:
                await self._reader_task
            if self._ka_task is not None:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Deriv WS disconnected"))
            self.pending.clear()
            self.ws = None
            if not self._running:
                break
            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(
                    self.RECONNECT_BASE * (2 ** (attempt - 1)), self.RECONNECT_CAP
                ) + random.uniform(0, 1)
                print(f"[DerivClient] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[DerivClient] Reconnect attempt {attempt} failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id = data.get("req_id")
        msg_type = data.get("msg_type")
        if msg_type == "ping":
            return
        if req_id is not None and req_id in self.pending:
            fut = self.pending.pop(req_id)
            if not fut.done():
                fut.set_result(data)
            return
        if msg_type in self.subscriptions:
            for q in self.subscriptions[msg_type]:
                q.put_nowait(data)

    async def send(self, request, timeout=20):
        self.req_id += 1
        req_id = self.req_id
        request = {**request, "req_id": req_id}
        fut = asyncio.get_event_loop().create_future()
        self.pending[req_id] = fut
        await self.ws.send(json.dumps(request))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending.pop(req_id, None)
            return {"error": {"message": "request timed out"}}

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q


# ---------------------------------------------------------------------------
# CANDLE / EMA / REGIME DATA
# ---------------------------------------------------------------------------
class SymbolCandles:
    """Builds fixed-interval OHLC candles from the live tick stream and
    keeps rolling EMA(7)/EMA(14) + tick buffers for Monte Carlo."""

    def __init__(self, symbol, interval_sec=CANDLE_INTERVAL_SEC, maxlen=500):
        self.symbol = symbol
        self.interval = interval_sec
        self.candles = deque(maxlen=maxlen)     # dicts: open/high/low/close/epoch
        self.ticks = deque(maxlen=20000)        # (epoch, price) - for MC volatility
        self._cur = None
        self.ema_fast_list = deque(maxlen=maxlen)
        self.ema_slow_list = deque(maxlen=maxlen)

    def add_tick(self, epoch, price):
        self.ticks.append((epoch, price))
        bucket = int(epoch // self.interval) * self.interval
        if self._cur is None or self._cur["epoch"] != bucket:
            if self._cur is not None:
                self._finish_candle(self._cur)
            self._cur = {"epoch": bucket, "open": price, "high": price,
                         "low": price, "close": price}
        else:
            self._cur["high"] = max(self._cur["high"], price)
            self._cur["low"] = min(self._cur["low"], price)
            self._cur["close"] = price

    def _finish_candle(self, candle):
        self.candles.append(dict(candle))
        closes = [c["close"] for c in self.candles]
        self.ema_fast_list.append(_ema_last(closes, EMA_FAST))
        self.ema_slow_list.append(_ema_last(closes, EMA_SLOW))

    def prices(self):
        return np.array([p for _, p in self.ticks], dtype=float)

    def returns(self):
        p = self.prices()
        if len(p) < 2:
            return np.array([])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(p))
        return r[np.isfinite(r)]

    def ready(self):
        return len(self.candles) >= EMA_SLOW + 2 and len(self.ticks) >= 100


def _ema_last(values, period):
    """EMA of `values` (list of closes) using the standard 2/(n+1) alpha,
    seeded with an SMA of the first `period` values. Returns the last EMA
    value only (this is called incrementally as candles close)."""
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = np.mean(values[:period])
    for v in values[period:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def compute_adx(candles, period=ADX_PERIOD):
    """Wilder's ADX from a list of OHLC candle dicts. Returns None if not
    enough candles yet."""
    if len(candles) < period + 1:
        return None
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    closes = np.array([c["close"] for c in candles])

    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])

    def wilder_smooth(arr, p):
        sm = np.zeros(len(arr))
        sm[p - 1] = arr[:p].sum()
        for i in range(p, len(arr)):
            sm[i] = sm[i - 1] - (sm[i - 1] / p) + arr[i]
        return sm

    if len(tr) < period:
        return None
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / np.where(atr == 0, 1e-9, atr)
    minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr == 0, 1e-9, atr)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-9, (plus_di + minus_di))
    valid = dx[period - 1:]
    if len(valid) == 0:
        return None
    return float(np.mean(valid[-period:]))


@dataclass
class RegimeRead:
    trending: bool
    adx: Optional[float]
    ema_fast: float
    ema_slow: float
    ema_fast_prev: float
    ema_slow_prev: float
    cross_down_confirmed: bool     # mirrors the original bot's entry trigger
    black_candle: bool


def read_regime(sc: SymbolCandles) -> Optional[RegimeRead]:
    if not sc.ready():
        return None
    ema_f = sc.ema_fast_list[-1]
    ema_s = sc.ema_slow_list[-1]
    ema_f_prev = sc.ema_fast_list[-2]
    ema_s_prev = sc.ema_slow_list[-2]
    if None in (ema_f, ema_s, ema_f_prev, ema_s_prev):
        return None

    adx = compute_adx(list(sc.candles))
    price = sc.candles[-1]["close"]
    gap_pct = abs(ema_f - ema_s) / price if price else 0.0

    trending = (adx is not None and adx >= ADX_TREND_THRESHOLD) or (gap_pct >= EMA_GAP_TREND_PCT)

    # Original bot's trigger: EMA(7) was above EMA(14) last candle, now below
    # (a confirmed cross-down), plus the latest candle is black (bearish).
    cross_down = (ema_f_prev > ema_s_prev) and (ema_f < ema_s)
    last = sc.candles[-1]
    black_candle = last["close"] < last["open"]

    return RegimeRead(
        trending=trending, adx=adx,
        ema_fast=ema_f, ema_slow=ema_s,
        ema_fast_prev=ema_f_prev, ema_slow_prev=ema_s_prev,
        cross_down_confirmed=cross_down, black_candle=black_candle,
    )


# ---------------------------------------------------------------------------
# MONTE CARLO - DURATION + BARRIER SELECTION
# ---------------------------------------------------------------------------
def _simulate_paths(vol_per_tick, n_ticks, n_sims=MC_SIMULATIONS):
    """Simulates n_sims GBM-style log-return paths of n_ticks steps.
    Returns (cum_log, running_max, running_min) arrays of shape (n_sims, n_ticks)."""
    steps = np.random.normal(0.0, vol_per_tick, size=(n_sims, n_ticks))
    cum_log = np.cumsum(steps, axis=1)
    running_max = np.maximum.accumulate(cum_log, axis=1)
    running_min = np.minimum.accumulate(cum_log, axis=1)
    return cum_log, running_max, running_min


def _tick_vol(returns):
    if len(returns) < 20:
        return None
    window = returns[-200:] if len(returns) >= 200 else returns
    vol = float(np.std(window))
    return max(vol, 1e-7)


def mc_select_notouch(sc: SymbolCandles, trend_is_bullish: bool):
    """Path-dependent MC: barrier placed on the side opposite the trend
    (bullish trend -> barrier below current price; bearish -> above),
    mirroring the original bot's directional bias. Grid-searches
    (duration, sigma) for the combo whose simulated P(no_touch) is
    closest to MC_TARGET_PROB, for each candidate duration."""
    returns = sc.returns()
    vol = _tick_vol(returns)
    if vol is None:
        return []
    mean_dt = _mean_tick_dt(sc)
    price = sc.prices()[-1]

    candidates = []
    for minutes in NOTOUCH_DURATIONS:
        n_ticks = max(5, int(round(minutes * 60.0 / mean_dt)))
        cum_log, running_max, running_min = _simulate_paths(vol, n_ticks)
        best_for_duration = None
        for sigma_k in MC_SIGMA_GRID:
            scaled_vol = vol * np.sqrt(n_ticks)
            log_dist = sigma_k * scaled_vol
            if trend_is_bullish:
                barrier_log = -log_dist
                touched = running_min[:, -1] <= barrier_log
                side = "below"
            else:
                barrier_log = log_dist
                touched = running_max[:, -1] >= barrier_log
                side = "above"
            p_no_touch = float(np.mean(~touched))
            diff = abs(p_no_touch - MC_TARGET_PROB)
            if best_for_duration is None or diff < best_for_duration["diff"]:
                best_for_duration = {
                    "duration": minutes, "duration_unit": "m", "side": side,
                    "barrier_price": round(float(price * np.exp(barrier_log)), 4),
                    "p_no_touch": round(p_no_touch, 4), "diff": diff,
                }
        if best_for_duration is not None:
            candidates.append(best_for_duration)

    candidates.sort(key=lambda c: c["diff"])
    return candidates[:CANDIDATE_TOP_K]


def mc_select_expiryrange(sc: SymbolCandles):
    """Terminal-value-only MC (ends-in-range doesn't care what the path
    touched along the way, only where it finishes): for each candidate
    duration, picks the symmetric barrier width whose simulated
    P(ends inside range) is closest to MC_TARGET_PROB."""
    returns = sc.returns()
    vol = _tick_vol(returns)
    if vol is None:
        return []
    mean_dt = _mean_tick_dt(sc)
    price = sc.prices()[-1]

    candidates = []
    for minutes in EXPIRYRANGE_DURATIONS:
        n_ticks = max(5, int(round(minutes * 60.0 / mean_dt)))
        scaled_vol = vol * np.sqrt(n_ticks)
        # Simulate only the terminal value - much cheaper than full paths.
        terminal = np.random.normal(0.0, scaled_vol, size=MC_SIMULATIONS)
        best_for_duration = None
        for sigma_k in MC_SIGMA_GRID:
            log_dist = sigma_k * scaled_vol
            inside = np.abs(terminal) <= log_dist
            p_in = float(np.mean(inside))
            diff = abs(p_in - MC_TARGET_PROB)
            if best_for_duration is None or diff < best_for_duration["diff"]:
                best_for_duration = {
                    "duration": minutes, "duration_unit": "m",
                    "barrier_high": round(float(price * np.exp(log_dist)), 4),
                    "barrier_low": round(float(price * np.exp(-log_dist)), 4),
                    "p_in_range": round(p_in, 4), "diff": diff,
                }
        if best_for_duration is not None:
            candidates.append(best_for_duration)

    candidates.sort(key=lambda c: c["diff"])
    return candidates[:CANDIDATE_TOP_K]


def _mean_tick_dt(sc: SymbolCandles):
    if len(sc.ticks) < 2:
        return 2.0
    epochs = [e for e, _ in sc.ticks]
    gaps = np.diff(epochs[-200:]) if len(epochs) >= 200 else np.diff(epochs)
    gaps = gaps[gaps > 0]
    return float(np.mean(gaps)) if len(gaps) else 2.0


# ---------------------------------------------------------------------------
# LIVE QUOTING + EXECUTION
# ---------------------------------------------------------------------------
async def quote_notouch(client, symbol, barrier_price, duration, duration_unit, stake):
    req = {
        "proposal": 1, "amount": round(float(stake), 2), "basis": "stake",
        "contract_type": "NOTOUCH", "currency": "USD",
        "duration": int(duration), "duration_unit": duration_unit,
        "symbol": symbol, "barrier": f"{barrier_price:.4f}",
    }
    resp = await client.send(req)
    if "error" in resp:
        return None
    p = resp["proposal"]
    return {"payout": round(float(p["payout"]), 2),
            "ask_price": round(float(p["ask_price"]), 2),
            "proposal_id": p["id"]}


async def quote_expiryrange(client, symbol, barrier_high, barrier_low, duration, duration_unit, stake):
    req = {
        "proposal": 1, "amount": round(float(stake), 2), "basis": "stake",
        "contract_type": "EXPIRYRANGE", "currency": "USD",
        "duration": int(duration), "duration_unit": duration_unit,
        "symbol": symbol,
        "barrier": f"{barrier_high:.4f}",
        "barrier2": f"{barrier_low:.4f}",
    }
    resp = await client.send(req)
    if "error" in resp:
        return None
    p = resp["proposal"]
    return {"payout": round(float(p["payout"]), 2),
            "ask_price": round(float(p["ask_price"]), 2),
            "proposal_id": p["id"]}


async def select_best_notouch(client, symbol, sc, trend_is_bullish, stake):
    grid = mc_select_notouch(sc, trend_is_bullish)
    best = None
    for cand in grid:
        if cand["duration"] < NOTOUCH_MIN_MINUTES:
            continue
        q = await quote_notouch(client, symbol, cand["barrier_price"],
                                 cand["duration"], cand["duration_unit"], stake)
        if q is None:
            continue
        payout_ratio = q["payout"] / q["ask_price"] if q["ask_price"] > 0 else 0.0
        ev = cand["p_no_touch"] * payout_ratio - 1.0
        if best is None or ev > best["ev"]:
            best = {**cand, **q, "payout_ratio": round(payout_ratio, 3), "ev": round(ev, 3)}
    return best


async def select_best_expiryrange(client, symbol, sc, stake):
    grid = mc_select_expiryrange(sc)
    best = None
    for cand in grid:
        if cand["duration"] < EXPIRYRANGE_MIN_MINUTES:
            continue
        q = await quote_expiryrange(client, symbol, cand["barrier_high"], cand["barrier_low"],
                                     cand["duration"], cand["duration_unit"], stake)
        if q is None:
            continue
        payout_ratio = q["payout"] / q["ask_price"] if q["ask_price"] > 0 else 0.0
        ev = cand["p_in_range"] * payout_ratio - 1.0
        if best is None or ev > best["ev"]:
            best = {**cand, **q, "payout_ratio": round(payout_ratio, 3), "ev": round(ev, 3)}
    return best


async def buy_from_proposal(client, proposal_id, stake):
    resp = await client.send({"buy": proposal_id, "price": round(float(stake), 2)})
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", "buy failed"))
    return resp["buy"]["contract_id"]


async def wait_for_contract_result(client, contract_id):
    q = client.subscribe_channel("proposal_open_contract")
    await client.send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
    while True:
        data = await q.get()
        poc = data.get("proposal_open_contract", {})
        if poc.get("contract_id") == contract_id and poc.get("is_sold"):
            profit = float(poc.get("profit", 0))
            return profit > 0, profit


# ---------------------------------------------------------------------------
# MONEY MANAGEMENT (ported from the original bot's variables)
# ---------------------------------------------------------------------------
@dataclass
class MoneyState:
    stake: float = FIRST_STAKE
    total_profit: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    martingale_step: int = 0

    def on_win(self, profit):
        self.total_profit += profit
        self.win_count += 1
        self.stake = FIRST_STAKE
        self.martingale_step = 0

    def on_loss(self, profit):
        self.total_profit += profit  # profit is negative on a loss
        self.loss_count += 1
        if self.martingale_step < MARTINGALE_LEVEL:
            self.martingale_step += 1
            self.stake = round(self.stake * MARTINGALE_FACTOR, 2)
        else:
            self.martingale_step = 0
            self.stake = FIRST_STAKE

    def should_stop(self):
        if self.total_profit >= TARGET_PROFIT:
            return "target_profit"
        if self.total_profit <= -STOP_LOSS:
            return "stop_loss"
        return None


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
async def tick_consumer(client, candles: Dict[str, SymbolCandles]):
    q = client.subscribe_channel("tick")
    while True:
        data = await q.get()
        t = data.get("tick")
        if not t:
            continue
        symbol = t.get("symbol")
        sc = candles.get(symbol)
        if sc is None:
            continue
        sc.add_tick(float(t["epoch"]), float(t["quote"]))


async def subscribe_ticks(client, symbols):
    for s in symbols:
        resp = await client.send({"ticks": s, "subscribe": 1})
        if "error" in resp:
            print(f"[subscribe] {s}: {resp['error']}")


async def main():
    if not DERIV_APP_ID or not DERIV_API_TOKEN:
        print("ERROR: set DERIV_APP_ID and DERIV_API_TOKEN environment variables.")
        sys.exit(1)

    client = DerivClient(DERIV_APP_ID, DERIV_API_TOKEN, DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)

    async def resubscribe(c):
        await subscribe_ticks(c, SYMBOLS)

    client.resubscribe_cb = resubscribe
    await client.connect()
    await subscribe_ticks(client, SYMBOLS)

    candles = {s: SymbolCandles(s) for s in SYMBOLS}
    asyncio.create_task(tick_consumer(client, candles))

    money = MoneyState()
    last_trade_time = {s: 0.0 for s in SYMBOLS}

    print(f"Auto-Trading Started — Target Profit ${TARGET_PROFIT} / Stop Loss ${STOP_LOSS} / "
          f"First Stake ${FIRST_STAKE} / Martingale x{MARTINGALE_FACTOR} up to {MARTINGALE_LEVEL} steps")
    print(f"Symbols: {SYMBOLS}  |  MC target confidence: {MC_TARGET_PROB:.0%}")

    while True:
        await asyncio.sleep(2)

        stop_reason = money.should_stop()
        if stop_reason:
            print(f"[STOP] {stop_reason} reached. Total profit: {money.total_profit:.2f} "
                  f"(W:{money.win_count} L:{money.loss_count})")
            return

        for symbol in SYMBOLS:
            sc = candles[symbol]
            if time.time() - last_trade_time[symbol] < TRADE_COOLDOWN_SEC:
                continue
            regime = read_regime(sc)
            if regime is None:
                continue

            stake = max(MIN_STAKE, money.stake)

            if regime.trending and regime.cross_down_confirmed and regime.black_candle:
                # Mirrors the original bot's exact entry trigger.
                trend_is_bullish = regime.ema_fast_prev > regime.ema_slow_prev  # was uptrend, now confirmed reversal down
                best = await select_best_notouch(client, symbol, sc, trend_is_bullish, stake)
                if best is None or best["ev"] <= 0:
                    continue
                print(f"[NOTOUCH] {symbol} dur={best['duration']}m barrier={best['barrier_price']} "
                      f"p={best['p_no_touch']:.2f} payout_ratio={best['payout_ratio']} ev={best['ev']}")
                contract_id = await buy_from_proposal(client, best["proposal_id"], stake)
                won, profit = await wait_for_contract_result(client, contract_id)

            elif not regime.trending:
                best = await select_best_expiryrange(client, symbol, sc, stake)
                if best is None or best["ev"] <= 0:
                    continue
                print(f"[EXPIRYRANGE] {symbol} dur={best['duration']}m "
                      f"range=[{best['barrier_low']}, {best['barrier_high']}] "
                      f"p={best['p_in_range']:.2f} payout_ratio={best['payout_ratio']} ev={best['ev']}")
                contract_id = await buy_from_proposal(client, best["proposal_id"], stake)
                won, profit = await wait_for_contract_result(client, contract_id)

            else:
                continue

            last_trade_time[symbol] = time.time()
            if won:
                money.on_win(profit)
                print(f"[WIN] {symbol} profit={profit:.2f} total={money.total_profit:.2f}")
            else:
                money.on_loss(profit)
                print(f"[LOSS] {symbol} profit={profit:.2f} total={money.total_profit:.2f} "
                      f"next_stake={money.stake}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
