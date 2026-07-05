"""
persistence.py — Supabase-backed state store, same pattern as your other bots.

Tables (see supabase_schema.sql for DDL):
  touch_bot_trades              — every trade placed + outcome
  touch_bot_bayesian_state      — serialized BayesianTracker cells
  touch_bot_calibration_runs    — log of each calibration cycle's top-N picks
  touch_bot_active_symbols      — current active symbol slots + consecutive loss counters
"""
from __future__ import annotations
import json
import time
from supabase import create_client, Client

import config
from bayesian import BayesianTracker


class SupabaseStore:
    def __init__(self, url: str = config.SUPABASE_URL, key: str = config.SUPABASE_KEY):
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set in environment")
        self.client: Client = create_client(url, key)

    # -- Bayesian state -----------------------------------------------------
    def load_bayesian_tracker(self) -> BayesianTracker:
        resp = self.client.table("touch_bot_bayesian_state").select("*").eq("id", 1).execute()
        if resp.data:
            return BayesianTracker.from_dict(json.loads(resp.data[0]["state_json"]))
        return BayesianTracker()

    def save_bayesian_tracker(self, tracker: BayesianTracker):
        payload = {"id": 1, "state_json": json.dumps(tracker.to_dict()), "updated_at": time.time()}
        self.client.table("touch_bot_bayesian_state").upsert(payload).execute()

    # -- Trades ---------------------------------------------------------------
    def log_trade(self, symbol: str, regime_label: str, distance_sigma: float, duration_minutes: float,
                  stake: float, payout: float, p_no_touch_est: float, won: bool | None, contract_id: int):
        self.client.table("touch_bot_trades").insert({
            "symbol": symbol, "regime_label": regime_label, "distance_sigma": distance_sigma,
            "duration_minutes": duration_minutes, "stake": stake, "payout": payout,
            "p_no_touch_est": p_no_touch_est, "won": won, "contract_id": contract_id,
            "created_at": time.time(),
        }).execute()

    def update_trade_outcome(self, contract_id: int, won: bool, profit: float):
        self.client.table("touch_bot_trades").update({"won": won, "profit": profit}).eq(
            "contract_id", contract_id
        ).execute()

    # -- Calibration log --------------------------------------------------
    def log_calibration_run(self, picks: list[dict]):
        self.client.table("touch_bot_calibration_runs").insert({
            "picks_json": json.dumps(picks), "created_at": time.time(),
        }).execute()

    # -- Active symbol slots + consecutive loss tracking -------------------
    def save_active_symbols(self, slots: dict):
        """slots: {symbol: {"consecutive_losses": int, "candidate": {...}}}"""
        self.client.table("touch_bot_active_symbols").upsert({
            "id": 1, "slots_json": json.dumps(slots), "updated_at": time.time(),
        }).execute()

    def load_active_symbols(self) -> dict:
        resp = self.client.table("touch_bot_active_symbols").select("*").eq("id", 1).execute()
        if resp.data:
            return json.loads(resp.data[0]["slots_json"])
        return {}
