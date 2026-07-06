"""
self_improvement.py — daily walk-forward validation + rollback protection.

WHAT THIS ACTUALLY VALIDATES, AND WHY: of everything in the Intelligence Stack,
only the Bayesian posterior layer (bayesian.py's BayesianTracker) is PERSISTED
state that accumulates online across trades. The HMM regime detector, GARCH,
Kalman filter, and ARFIMA/GPH estimator are all refit fresh from live tick
history every calibration cycle (see calibrator.py) — they can't go stale in
the way persisted state can, so there's nothing there to roll back. A daily
walk-forward retrain of THOSE models would just mean "fit them on the same
kind of live data we already fit them on every few minutes," which isn't a
meaningfully different validation from what's already continuous.

The Bayesian tracker is different: it's exactly the kind of state that could
silently degrade — e.g. a genuine regime shift makes historically-learned
cells miscalibrated, or a bad run corrupts a specific cell — and unlike the
stateless models, there's no automatic correction; it just keeps averaging in
whatever happens. So this engine's job is specifically to catch that and
protect against it:

  1. Once a day (config.SELF_IMPROVEMENT_INTERVAL_SECONDS), pull the last
     config.SELF_IMPROVEMENT_VALIDATION_DAYS of settled trades (each one
     already logged with the p_no_touch_est that was actually used to size
     it, and the realized won/lost outcome).
  2. Compute the Brier score (mean squared error between predicted
     probability and realized binary outcome) of those predictions — this is
     the walk-forward validation: "how well-calibrated were our live,
     already-made predictions against outcomes we didn't have yet at
     prediction time."
  3. Compare that to the Brier score recorded for the last promoted snapshot.
     If it's degraded by more than the tolerance, ROLL BACK the tracker to
     that snapshot's saved state (discarding however many trades' worth of
     online updates happened since, on the theory that whatever caused the
     degradation is more likely to be corrupting the posterior than
     reflecting a real improvement). Otherwise PROMOTE: snapshot the current
     (already-live) state as the new known-good version.

This is an honest scope call: a from-scratch re-simulation of what the full
stochastic model stack *would have* predicted at each historical trade's
exact tick-history-as-of-that-moment isn't possible without persisting full
tick snapshots per cycle (which this bot doesn't do), so this validates what
we can actually validate from data that's genuinely available — the realized
calibration quality of the persisted learned state — rather than faking a
walk-forward backtest against data we don't have.
"""
from __future__ import annotations
import time

import config
from bayesian import BayesianTracker
from persistence import SupabaseStore


def brier_score(predictions: list[float], outcomes: list[bool]) -> float | None:
    if not predictions:
        return None
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


class SelfImprovementEngine:
    def __init__(self, store: SupabaseStore):
        self.store = store

    def _should_run(self) -> bool:
        if not config.SELF_IMPROVEMENT_ENABLED:
            return False
        last_run = self.store.load_self_improvement_last_run()
        if last_run is None:
            return True
        return (time.time() - last_run) >= config.SELF_IMPROVEMENT_INTERVAL_SECONDS

    def maybe_run_daily(self, tracker: BayesianTracker) -> tuple[BayesianTracker, dict | None]:
        """
        Returns (tracker, run_summary). tracker is the possibly-rolled-back tracker
        the caller should use going forward; run_summary is None if no run happened
        this cycle (not due yet, or disabled).
        """
        if not self._should_run():
            return tracker, None

        summary = {"ran_at": time.time(), "action": "skipped", "reason": None,
                   "current_brier": None, "previous_brier": None}
        try:
            trades = self.store.load_recent_predictions(days=config.SELF_IMPROVEMENT_VALIDATION_DAYS)
            if len(trades) < config.SELF_IMPROVEMENT_MIN_TRADES_FOR_VALIDATION:
                summary["reason"] = (
                    f"only {len(trades)} settled trades in the last "
                    f"{config.SELF_IMPROVEMENT_VALIDATION_DAYS}d "
                    f"(need {config.SELF_IMPROVEMENT_MIN_TRADES_FOR_VALIDATION}) -- skipping validation"
                )
                self.store.save_self_improvement_run(summary)
                self.store.save_self_improvement_last_run(time.time())
                return tracker, summary

            predictions = [t["p_no_touch_est"] for t in trades]
            outcomes = [bool(t["won"]) for t in trades]
            current_brier = brier_score(predictions, outcomes)
            summary["current_brier"] = current_brier
            summary["n_trades"] = len(trades)

            last_snapshot = self.store.load_latest_model_version()
            previous_brier = last_snapshot["validation_brier"] if last_snapshot else None
            summary["previous_brier"] = previous_brier

            degraded = (
                previous_brier is not None
                and current_brier is not None
                and (current_brier - previous_brier) > config.SELF_IMPROVEMENT_ROLLBACK_BRIER_TOLERANCE
            )

            if degraded:
                restored = BayesianTracker.from_dict(last_snapshot["tracker_state"])
                summary["action"] = "rollback"
                summary["reason"] = (
                    f"Brier degraded {previous_brier:.4f} -> {current_brier:.4f} "
                    f"(tolerance {config.SELF_IMPROVEMENT_ROLLBACK_BRIER_TOLERANCE}) -- "
                    f"restored snapshot from {last_snapshot.get('created_at')}"
                )
                print(f"[self-improvement] ROLLBACK: {summary['reason']}")
                self.store.save_self_improvement_run(summary)
                self.store.save_self_improvement_last_run(time.time())
                return restored, summary
            else:
                self.store.save_model_version(tracker.to_dict(), current_brier)
                summary["action"] = "promoted"
                summary["reason"] = (
                    f"Brier {current_brier:.4f} " +
                    (f"(previous {previous_brier:.4f})" if previous_brier is not None else "(first snapshot)")
                )
                print(f"[self-improvement] PROMOTED: {summary['reason']}")
                self.store.save_self_improvement_run(summary)
                self.store.save_self_improvement_last_run(time.time())
                return tracker, summary

        except Exception as e:
            summary["action"] = "error"
            summary["reason"] = str(e)
            print(f"[self-improvement] run failed, leaving tracker unchanged: {e}")
            try:
                self.store.save_self_improvement_run(summary)
                self.store.save_self_improvement_last_run(time.time())
            except Exception:
                pass
            return tracker, summary
