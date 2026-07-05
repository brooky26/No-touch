"""
deriv_client.py — thin async wrapper around the Deriv WebSocket API (v3).

Implements exactly what the bot needs:
  - authorize
  - tick history (for calibration: hurst, vol, regime fitting)
  - live tick subscription
  - proposal (quote) for NOTOUCH contracts at a given barrier/duration
  - buy
  - poll a contract until it settles (won/lost + payout)

Docs: https://developers.deriv.com/docs/websockets
"""
from __future__ import annotations
import asyncio
import itertools
import json
import time
from dataclasses import dataclass

import websockets

import config


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
    def __init__(self, app_id: str = config.DERIV_APP_ID, api_token: str = config.DERIV_API_TOKEN):
        self.url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.api_token = api_token
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._req_id = itertools.count(1)
        self._lock = asyncio.Lock()

    async def connect(self):
        self.ws = await websockets.connect(self.url, ping_interval=20, ping_timeout=10)

    async def close(self):
        if self.ws:
            await self.ws.close()

    async def _send(self, payload: dict) -> dict:
        """Send a request and wait for the matching response (matched by req_id)."""
        if self.ws is None:
            raise DerivApiError("Not connected — call connect() first")
        req_id = next(self._req_id)
        payload = {**payload, "req_id": req_id}
        async with self._lock:
            await self.ws.send(json.dumps(payload))
            while True:
                raw = await self.ws.recv()
                msg = json.loads(raw)
                if msg.get("req_id") == req_id:
                    if msg.get("error"):
                        raise DerivApiError(msg["error"].get("message", "Unknown Deriv API error"))
                    return msg

    async def authorize(self) -> dict:
        resp = await self._send({"authorize": self.api_token})
        return resp

    async def verify_account_type(self, expected: str) -> dict:
        """
        Safety check: confirm the authorized account's actual type (demo/virtual vs real)
        matches config.ACCOUNT_TYPE before any trading starts. Raises DerivApiError on mismatch
        so a wrong/copy-pasted token can never silently trade against the wrong account.
        """
        resp = await self._send({"authorize": self.api_token})
        auth = resp["authorize"]
        is_virtual = bool(auth.get("is_virtual"))
        actual = "demo" if is_virtual else "real"
        if actual != expected:
            raise DerivApiError(
                f"ACCOUNT_TYPE mismatch: expected '{expected}' but the API token authorizes a "
                f"'{actual}' account (loginid={auth.get('loginid')}). Refusing to start."
            )
        return auth

    async def get_balance(self) -> float:
        resp = await self._send({"balance": 1})
        return float(resp["balance"]["balance"])

    async def tick_history(self, symbol: str, count: int = config.HISTORY_TICKS_FOR_CALIBRATION) -> list[float]:
        resp = await self._send({
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

    async def get_notouch_proposal(self, symbol: str, barrier_offset: float, duration: int,
                                    duration_unit: str, stake: float) -> Proposal:
        """
        barrier_offset: signed offset from current spot, Deriv accepts relative barriers
        like "+12.34" / "-12.34" for NOTOUCH contracts.
        """
        barrier_str = f"{'+' if barrier_offset >= 0 else ''}{barrier_offset:.5f}"
        resp = await self._send({
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": "NOTOUCH",
            "currency": "USD",
            "symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
            "barrier": barrier_str,
        })
        p = resp["proposal"]
        return Proposal(
            id=p["id"], symbol=symbol, barrier=barrier_offset, duration=duration,
            duration_unit=duration_unit, ask_price=float(p["ask_price"]), payout=float(p["payout"]),
        )

    async def buy(self, proposal: Proposal) -> int:
        resp = await self._send({"buy": proposal.id, "price": proposal.ask_price})
        return int(resp["buy"]["contract_id"])

    async def wait_for_settlement(self, contract_id: int, poll_interval: float = 3.0,
                                   timeout: float = 24 * 3600) -> ContractResult:
        start = time.time()
        while time.time() - start < timeout:
            resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id})
            poc = resp.get("proposal_open_contract", {})
            if poc.get("is_sold") or poc.get("status") not in (None, "open"):
                won = poc.get("status") == "won"
                payout = float(poc.get("payout", 0.0))
                buy_price = float(poc.get("buy_price", 0.0))
                profit = float(poc.get("profit", payout - buy_price))
                return ContractResult(contract_id=contract_id, won=won, payout=payout,
                                       buy_price=buy_price, profit=profit)
            await asyncio.sleep(poll_interval)
        raise DerivApiError(f"Contract {contract_id} did not settle within timeout")
