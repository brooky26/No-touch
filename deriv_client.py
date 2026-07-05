"""
deriv_client.py — async wrapper around the Deriv WebSocket API (v3).

CONNECTION LAYER: ported from the multi-symbol EXPIRYRANGE bot to match Deriv's
updated auth flow. Deriv retired plain `{"authorize": <token>}` over a static
`wss://ws.derivws.com/websockets/v3?app_id=...` URL for this account-scoped
flow: a REST call resolves the account_id for the given account_type, a second
REST call exchanges that for a one-time-use, account-bound WebSocket URL, and
only THAT url is connected to. There is no separate "authorize" WS message
anymore — auth happens implicitly via the OTP url. This client also carries
over the multi-symbol bot's auto-reconnect supervisor, heartbeat, and
futures-based request/response dispatch, since those are transport-layer
concerns independent of what the bot trades.

TRADING LAYER: kept specific to this bot (NOTOUCH proposals/buy/settlement),
but updated for the schema changes that came with the new API:
  - proposal/buy requests use `underlying_symbol`, not `symbol`
  - buy is a single parameterized call (no separate buy-by-proposal-id step)
  - settlement polling reads `profit`/`status` off proposal_open_contract
    rather than a `status == "won"` field, which Deriv no longer sends

Docs: https://developers.deriv.com/docs/websockets
"""
from __future__ import annotations
import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import requests
import websockets

import config

API_BASE      = "https://api.derivws.com"
ACCOUNTS_PATH = "/trading/v1/options/accounts"
OTP_PATH      = "/trading/v1/options/accounts/{account_id}/otp"


class DerivApiError(RuntimeError):
    pass


@dataclass
class Proposal:
    id: str
    symbol: str
    barrier: float
    duration: int
    duration_unit: str
    ask_price: float   # stake required
    payout: float       # total payout if it wins


@dataclass
class ContractResult:
    contract_id: int
    won: bool
    payout: float
    buy_price: float
    profit: float


class DerivClient:
    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE     = 2.0
    RECONNECT_CAP      = 60.0

    def __init__(self, app_id: str = config.DERIV_APP_ID, token: str = config.DERIV_API_TOKEN,
                 account_type: str = config.ACCOUNT_TYPE,
                 account_id: str | None = getattr(config, "DERIV_ACCOUNT_ID", None)):
        self.app_id       = app_id
        self.token        = token
        self.account_type = account_type
        self.account_id   = account_id

        self.ws            = None
        self.req_id         = 0
        self.pending: dict  = {}
        self.subscriptions  = defaultdict(list)
        self.account        = None       # populated from the post-connect "balance" call
        self.resubscribe_cb = None
        self._running       = False
        self._reader_task    = None
        self._ka_task        = None

    # ------------------------------------------------------------------
    # REST: resolve account_id + exchange it for an account-bound WS url
    # ------------------------------------------------------------------
    def _rest_headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Deriv-App-ID": self.app_id,
                "Content-Type": "application/json"}

    def _resolve_account_id_sync(self):
        resp = requests.get(f"{API_BASE}{ACCOUNTS_PATH}",
                             headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                aid = acc.get("account_id") or acc.get("id")
                if aid:
                    return aid
        raise DerivApiError(
            f"No '{self.account_type}' account found for this token. "
            f"Refusing to start (raw response: {data})"
        )

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
        resp = requests.post(
            f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}",
            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url  = payload.get("url")
        if not ws_url:
            raise DerivApiError(f"OTP exchange missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ------------------------------------------------------------------
    # Connection lifecycle: connect / auto-reconnect / heartbeat
    # ------------------------------------------------------------------
    async def connect(self):
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task     = asyncio.create_task(self._heartbeat())
        # No separate "authorize" message: the OTP url is already account-bound.
        # A "balance" call both confirms the connection works and gives us the
        # loginid/currency/balance we need for verify_account_type()/get_balance().
        bal          = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(f"[Client] Connected ({self.account_type}). "
              f"loginid={self.account.get('loginid')} "
              f"balance={self.account.get('balance')} {self.account.get('currency', '')}")

    async def close(self):
        self._running = False
        if self._ka_task:
            self._ka_task.cancel()
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[Client] WS lost: {e}")

    async def _supervise(self):
        while self._running:
            if self._reader_task:
                await self._reader_task
            if self._ka_task:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WS disconnected"))
            self.pending.clear()
            self.ws = None
            if not self._running:
                break
            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(self.RECONNECT_BASE * (2 ** (attempt - 1)),
                            self.RECONNECT_CAP) + random.uniform(0, 1)
                print(f"[Client] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[Client] Reconnect failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id   = data.get("req_id")
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

    async def send(self, request: dict, timeout: float = 20) -> dict:
        """Send a request, return the raw response dict (may contain an "error" key)."""
        self.req_id += 1
        rid = self.req_id
        request = {**request, "req_id": rid}
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _send_checked(self, request: dict, timeout: float = 20) -> dict:
        """Like send(), but raises DerivApiError if the response contains an error."""
        resp = await self.send(request, timeout=timeout)
        if "error" in resp:
            raise DerivApiError(resp["error"].get("message", "Unknown Deriv API error"))
        return resp

    def subscribe_channel(self, msg_type: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q

    # ------------------------------------------------------------------
    # Safety check + account info
    # ------------------------------------------------------------------
    async def verify_account_type(self, expected: str) -> dict:
        """
        Safety check: the REST account-resolution step (_resolve_account_id_sync)
        is the authoritative source here — it walks Deriv's own
        GET /trading/v1/options/accounts list and only picks an account_id whose
        `account_type` field matches self.account_type. There is no reliable
        loginid-prefix convention to double-check against on top of that (an
        earlier version of this guessed "VR..." == demo, which is wrong for at
        least this account naming scheme and caused false-positive refusals on
        a correctly-resolved demo account). So this just confirms the
        account_type this client was constructed/resolved with actually
        matches `expected` (config.ACCOUNT_TYPE) — catching a config typo or a
        client reused for the wrong purpose, without re-deriving account type
        from unverified string patterns.
        """
        if self.account is None:
            raise DerivApiError("verify_account_type() called before connect()")
        if self.account_type != expected:
            raise DerivApiError(
                f"ACCOUNT_TYPE mismatch: client resolved a '{self.account_type}' account "
                f"but expected='{expected}'. Refusing to start."
            )
        return self.account

    async def get_balance(self) -> float:
        resp = await self._send_checked({"balance": 1})
        return float(resp["balance"]["balance"])

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    async def tick_history(self, symbol: str, count: int = config.HISTORY_TICKS_FOR_CALIBRATION) -> list[float]:
        resp = await self._send_checked({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "ticks",
        })
        return [float(p) for p in resp["history"]["prices"]]

    async def latest_price(self, symbol: str) -> float:
        prices = await self.tick_history(symbol, count=1)
        return prices[-1]

    # ------------------------------------------------------------------
    # NOTOUCH trading
    # ------------------------------------------------------------------
    async def get_notouch_proposal(self, symbol: str, barrier_offset: float, duration: int,
                                    duration_unit: str, stake: float) -> Proposal:
        """
        barrier_offset: signed offset from current spot, Deriv accepts relative barriers
        like "+12.34" / "-12.34" for NOTOUCH contracts.
        """
        barrier_str = f"{'+' if barrier_offset >= 0 else ''}{barrier_offset:.5f}"
        resp = await self._send_checked({
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": "NOTOUCH",
            "currency": "USD",
            "underlying_symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
            "barrier": barrier_str,
        })
        p = resp["proposal"]
        return Proposal(
            id=p.get("id", ""), symbol=symbol, barrier=barrier_offset, duration=duration,
            duration_unit=duration_unit, ask_price=float(p["ask_price"]), payout=float(p["payout"]),
        )

    async def buy(self, proposal: Proposal) -> int:
        """
        Single parameterized buy call (the API no longer buys off a proposal id
        alone) — the same "buy": "1" + "parameters": {...} pattern the
        multi-symbol bot uses for EXPIRYRANGE, adapted for NOTOUCH.
        """
        barrier_str = f"{'+' if proposal.barrier >= 0 else ''}{proposal.barrier:.5f}"
        resp = await self._send_checked({
            "buy": "1",
            "price": proposal.ask_price,
            "parameters": {
                "amount": proposal.ask_price,
                "basis": "stake",
                "contract_type": "NOTOUCH",
                "currency": "USD",
                "underlying_symbol": proposal.symbol,
                "duration": proposal.duration,
                "duration_unit": proposal.duration_unit,
                "barrier": barrier_str,
            },
        }, timeout=30)
        contract_id = resp.get("buy", {}).get("contract_id")
        if not contract_id:
            raise DerivApiError(f"Buy for {proposal.symbol} returned no contract_id: {resp}")
        return int(contract_id)

    async def wait_for_settlement(self, contract_id: int, poll_interval: float = 3.0,
                                   timeout: float = 24 * 3600) -> ContractResult:
        """
        Polls proposal_open_contract until the contract settles. Deriv no longer
        sends a `status == "won"/"lost"` field on this stream; settlement is now
        signalled by status == "sold" (or is_expired/is_settleable), and win/loss
        is read off `profit` (profit > 0 => won), matching the multi-symbol bot.
        """
        start = time.time()
        while time.time() - start < timeout:
            resp = await self._send_checked({"proposal_open_contract": 1, "contract_id": contract_id})
            poc = resp.get("proposal_open_contract", {})
            status = poc.get("status")
            if status == "sold" or poc.get("is_expired") or poc.get("is_settleable"):
                payout    = float(poc.get("payout", 0.0))
                buy_price = float(poc.get("buy_price", 0.0))
                profit    = float(poc.get("profit", payout - buy_price))
                won       = profit > 0
                return ContractResult(contract_id=contract_id, won=won, payout=payout,
                                       buy_price=buy_price, profit=profit)
            await asyncio.sleep(poll_interval)
        raise DerivApiError(f"Contract {contract_id} did not settle within timeout")
