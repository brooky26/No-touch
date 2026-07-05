"""
bot.py — main orchestration loop for the self-learning TOUCH/NOTOUCH bot.

Flow per the design doc:
  1. Calibrate the universe -> pick top N=3 symbols by EV (config.TOP_N_SYMBOLS).
  2. For each active symbol slot, trade its best (barrier, duration) candidate.
  3. After each settlement: update the Bayesian posterior for that cell, persist the trade,
     track consecutive losses per slot.
  4. If a slot hits config.CONSECUTIVE_LOSS_RECAL_TRIGGER consecutive losses -> recalibrate
     the ENTIRE universe and refresh the top-N picks (a losing symbol may be swapped out).
  5. Safety-net: recalibrate the whole universe periodically regardless
     (config.FULL_RECAL_EVERY_SECONDS), since regimes drift even without a loss streak.

Run:  python bot.py
Requires: .env with DERIV_APP_ID, DERIV_API_TOKEN, SUPABASE_URL, SUPABASE_KEY
"""
from __future__ import annotations
import asyncio
import time
import traceback

import config
from deriv_client import DerivClient
from calibrator import SymbolCalibrator, Candidate
from bayesian import BayesianTracker
from persistence import SupabaseStore
import staking


class TouchBot:
    def __init__(self):
        self.client = DerivClient()
        self.store = SupabaseStore()
        self.tracker: BayesianTracker = self.store.load_bayesian_tracker()
        self.calibrator = SymbolCalibrator(self.client, self.tracker)
        # slots: {symbol: {"consecutive_losses": int, "candidate": Candidate}}
        self.slots: dict[str, dict] = {}
        self.last_full_recal = 0.0

    async def start(self):
        await self.client.connect()
        auth = await self.client.verify_account_type(config.ACCOUNT_TYPE)
        print(f"[bot] connected to Deriv API — account={auth.get('loginid')} "
              f"type={config.ACCOUNT_TYPE} currency={auth.get('currency')} "
              f"balance={auth.get('balance')}")
        if config.ACCOUNT_TYPE == "real" and not config.CONFIRM_LIVE_TRADING:
            raise RuntimeError(
                "ACCOUNT_TYPE=real but CONFIRM_LIVE_TRADING is not set to 'yes'. "
                "Set CONFIRM_LIVE_TRADING=yes in your environment once you've actually "
                "reviewed the config and mean to trade real money."
            )
        await self.recalibrate()
        while True:
            try:
                await self.trade_cycle()
            except Exception:
                print("[bot] error in trade cycle:")
                traceback.print_exc()
            if time.time() - self.last_full_recal > config.FULL_RECAL_EVERY_SECONDS:
                await self.recalibrate()
            await asyncio.sleep(config.POLL_INTERVAL_SECONDS)

    async def recalibrate(self):
        print("[bot] running full universe calibration ...")
        picks = await self.calibrator.top_n_symbols(config.TOP_N_SYMBOLS)
        if not picks:
            print("[bot] WARNING: no symbol cleared the EV/win-prob floor this cycle. "
                  "Standing down until next calibration.")
            self.slots = {}
        else:
            self.slots = {
                symbol: {"consecutive_losses": 0, "candidate": candidate}
                for symbol, candidate in picks
            }
            print(f"[bot] top {len(self.slots)} symbols selected: "
                  f"{[(s, round(c.ev_per_stake, 4)) for s, c in picks]}")
        self.store.log_calibration_run([
            {
                "symbol": s,
                "regime": c.regime_label,
                "distance_sigma": c.distance_sigma,
                "duration_minutes": c.duration_minutes,
                "p_no_touch": c.p_no_touch_calibrated,
                "ev_per_stake": c.ev_per_stake,
            }
            for s, c in ((s, d["candidate"]) for s, d in self.slots.items())
        ])
        self.store.save_active_symbols({
            s: {"consecutive_losses": d["consecutive_losses"]} for s, d in self.slots.items()
        })
        self.last_full_recal = time.time()

    async def trade_cycle(self):
        if not self.slots:
            return
        for symbol, slot in list(self.slots.items()):
            candidate: Candidate = slot["candidate"]
            await self.execute_trade(symbol, candidate, slot)

    async def execute_trade(self, symbol: str, candidate: Candidate, slot: dict):
        try:
            proposal = await self.client.get_notouch_proposal(
                symbol=symbol, barrier_offset=candidate.distance_price,
                duration=candidate.duration_value, duration_unit=candidate.duration_unit,
                stake=config.BASE_STAKE,
            )
        except Exception as e:
            print(f"[bot] proposal failed for {symbol}: {e}")
            return

        payout_per_unit_stake = proposal.payout / proposal.ask_price
        if not staking.passes_ev_floor(candidate.p_no_touch_calibrated, payout_per_unit_stake):
            print(f"[bot] {symbol} candidate no longer clears EV floor at quote time, skipping")
            return

        bankroll = await self.client.get_balance()
        stake = staking.size_stake(
            p_win_mean=candidate.p_no_touch_calibrated,
            p_win_low_ci=candidate.p_no_touch_ci_low,
            payout_per_unit_stake=payout_per_unit_stake,
            bankroll=bankroll,
        )
        if stake <= 0:
            print(f"[bot] {symbol} sized to zero stake (uncertainty too high), skipping this cycle")
            return

        # re-quote at the actual sized stake (Deriv payout ratio ~ stake-invariant but re-quote to be safe)
        proposal = await self.client.get_notouch_proposal(
            symbol=symbol, barrier_offset=candidate.distance_price,
            duration=candidate.duration_value, duration_unit=candidate.duration_unit,
            stake=stake,
        )
        contract_id = await self.client.buy(proposal)
        print(f"[bot] BOUGHT {symbol} NOTOUCH stake={stake} barrier_offset={candidate.distance_price:.4f} "
              f"duration={candidate.duration_value}{candidate.duration_unit} contract_id={contract_id}")

        self.store.log_trade(
            symbol=symbol, regime_label=candidate.regime_label, distance_sigma=candidate.distance_sigma,
            duration_minutes=candidate.duration_minutes, stake=stake, payout=proposal.payout,
            p_no_touch_est=candidate.p_no_touch_calibrated, won=None, contract_id=contract_id,
        )

        result = await self.client.wait_for_settlement(contract_id)
        self.store.update_trade_outcome(contract_id, result.won, result.profit)

        self.tracker.record_outcome(
            symbol=symbol, regime_label=candidate.regime_label, distance_sigma=candidate.distance_sigma,
            duration_min=candidate.duration_minutes, no_touch_won=result.won,
        )
        self.store.save_bayesian_tracker(self.tracker)

        if result.won:
            slot["consecutive_losses"] = 0
            print(f"[bot] {symbol} WON, profit={result.profit}")
        else:
            slot["consecutive_losses"] += 1
            print(f"[bot] {symbol} LOST, consecutive_losses={slot['consecutive_losses']}")
            if slot["consecutive_losses"] >= config.CONSECUTIVE_LOSS_RECAL_TRIGGER:
                print(f"[bot] {symbol} hit consecutive loss threshold -> triggering full recalibration")
                await self.recalibrate()
                return  # slots dict has been rebuilt

        self.store.save_active_symbols({
            s: {"consecutive_losses": d["consecutive_losses"]} for s, d in self.slots.items()
        })


async def main():
    bot = TouchBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
