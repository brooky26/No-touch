"""
calibrator.py — the self-learning core / Trade Selection layer.

For every symbol in config.SYMBOL_UNIVERSE:
  1. Pull tick history.
  2. Compute EWMA vol, Hurst exponent, classify the current HMM regime, and build the
     full per-symbol Intelligence Stack context (ensemble.build_symbol_context: GARCH vol
     forecast, Kalman-filtered drift, Hawkes jump-clustering, ARFIMA/GPH long-memory
     horizon-scaling exponent) — see ensemble.py's module docstring for the full design.
  3. Sweep the (barrier distance x duration) grid. For each cell, ensemble.blend_candidate
     combines the GBM/OU analytical prior (fed the ensemble-adjusted drift/vol), the
     block-bootstrap estimator (PRIMARY, empirical/autocorrelation-aware), an independent
     Markov lattice cross-check, and the Bayesian posterior for that (symbol, regime,
     bucket) cell -- using bootstrap/Markov AGREEMENT as a confidence signal that pulls
     the effective win-probability floor down when the two estimators disagree.
  4. Pull a live Deriv proposal (real quoted payout) for the best candidates, compute EV.
  5. For the single best per-symbol candidate that clears both floors, run a final
     confirmation pass at full MC precision (config.MC_SIMULATIONS = 75,000 paths) before
     accepting it as tradable -- the screening sweep uses config.MC_SCREENING_PATHS
     (cheap) since 75k paths x every grid cell x every symbol would be far too slow.
  6. Rank all symbols by their best achievable EV, return the top N.

Recalibration is triggered externally (by bot.py) after N consecutive losses on a traded
symbol, or on a safety-net timer. The Bayesian posterior layer itself is additionally
guarded by self_improvement.py's daily validation + rollback (see that module).
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller
import numpy as np

import config
import stochastic
import ensemble
from regime import RegimeDetector
from bayesian import BayesianTracker, barrier_bucket, duration_bucket
from deriv_client import DerivClient
import staking


@dataclass
class Candidate:
    symbol: str
    regime_label: str
    distance_sigma: float
    distance_price: float
    duration_value: int
    duration_unit: str
    duration_minutes: float
    p_no_touch_model: float          # ensemble-blended estimate (analytical + bootstrap + Markov)
    p_no_touch_calibrated: float     # blended with Bayesian posterior
    p_no_touch_ci_low: float
    ensemble_confidence: float = 0.5  # bootstrap/Markov agreement, 1.0 = perfect agreement
    p_bootstrap: float = 0.0
    p_markov: float | None = None
    p_analytical: float = 0.0
    p_no_touch_confirmed: float | None = None  # full-MC confirmation, set on the final pick only
    payout: float | None = None
    ask_price: float | None = None
    ev_per_stake: float | None = None



class SymbolCalibrator:
    def __init__(self, client: DerivClient, tracker: BayesianTracker):
        self.client = client
        self.tracker = tracker
        self.regime_detectors: dict[str, RegimeDetector] = {}

    async def _calibrate_symbol(self, symbol: str) -> list[Candidate]:
        prices = await self.client.tick_history(symbol, config.HISTORY_TICKS_FOR_CALIBRATION)
        if len(prices) < 200:
            return []
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices))
        S0 = prices[-1]

        hurst = stochastic.hurst_exponent(returns)
        try:
            adf_pvalue = adfuller(prices[-500:])[1]
        except Exception:
            adf_pvalue = 1.0
        is_mean_reverting = (hurst < 0.45) or (adf_pvalue < 0.05)

        detector = self.regime_detectors.setdefault(symbol, RegimeDetector())
        if detector.needs_refit(len(returns)):
            detector.fit(returns)
        regime_id = detector.current_regime(returns)
        regime_params = detector.regime_params(returns)
        regime_info = regime_params.get(regime_id, {"label": "unknown", "mu": 0.0, "sigma": 0.0})
        regime_label = regime_info["label"]

        # Full per-symbol Intelligence Stack context (GARCH/Kalman/Hawkes/ARFIMA), built once
        # and reused across the whole barrier/duration grid below.
        ctx = ensemble.build_symbol_context(symbol, prices, returns, hurst, is_mean_reverting)

        # Sweep the full duration-search-space x barrier-distance grid for this symbol.
        # Duration is the outer loop internally (see sweep_symbol_grid) so each duration's
        # path simulation runs once and is shared across every barrier distance.
        swept = ensemble.sweep_symbol_grid(
            symbol=symbol, S0=S0, prices=prices, ctx=ctx, regime_mu=regime_info["mu"],
            regime_label=regime_label, returns=returns, tracker=self.tracker,
        )

        candidates: list[Candidate] = []
        for d_sigma in config.BARRIER_DISTANCE_GRID_SIGMA:
            # Auto-select which duration(s) are actually worth quoting at this barrier --
            # see ensemble.select_durations_for_barrier's docstring for the selection rule.
            selected = ensemble.select_durations_for_barrier(swept[d_sigma])
            for duration_value, duration_unit, duration_minutes, n_steps, blend in selected:
                d_price = ensemble.scaled_barrier_distance(d_sigma, ctx, S0, n_steps)
                candidates.append(Candidate(
                    symbol=symbol, regime_label=regime_label, distance_sigma=d_sigma,
                    distance_price=d_price, duration_value=duration_value, duration_unit=duration_unit,
                    duration_minutes=duration_minutes, p_no_touch_model=blend.p_no_touch_model,
                    p_no_touch_calibrated=blend.p_no_touch_calibrated,
                    p_no_touch_ci_low=blend.p_no_touch_ci_low,
                    ensemble_confidence=blend.ensemble_confidence, p_bootstrap=blend.p_bootstrap,
                    p_markov=blend.p_markov, p_analytical=blend.p_analytical,
                ))

        return candidates

    async def _quote_candidates(self, symbol: str, candidates: list[Candidate], top_k: int = 5):
        """
        STAGE 2 — market gate: fetch a real Deriv proposal for each MC-confident candidate
        (candidates here have already cleared MIN_WIN_PROB_FLOOR in calibrate_universe --
        this function doesn't re-check that) and compute the actual payout return + EV.
        top_k caps how many of the MC-confident set actually hit the API per symbol per
        cycle, since every call here is a real network round-trip, not simulation.
        """
        candidates_sorted = sorted(candidates, key=lambda c: c.p_no_touch_calibrated, reverse=True)
        for c in candidates_sorted[:top_k]:
            try:
                proposal = await self.client.get_notouch_proposal(
                    symbol=symbol, barrier_offset=c.distance_price,
                    duration=c.duration_value, duration_unit=c.duration_unit,
                    stake=config.BASE_STAKE,
                )
                c.payout = proposal.payout
                c.ask_price = proposal.ask_price
                payout_per_unit_stake = proposal.payout / proposal.ask_price
                c.ev_per_stake = staking.expected_value_per_stake(
                    c.p_no_touch_calibrated, payout_per_unit_stake, stake=1.0,
                )
            except Exception as e:
                # This used to fail silently (ev_per_stake stayed None with no trace), which
                # made a systematic proposal-side problem (bad duration/barrier for this
                # contract, stake below Deriv's minimum, etc.) indistinguishable from "the
                # EV floor correctly rejected everything" in the logs. Surface it instead.
                print(f"[calibrator] {symbol}: proposal failed for barrier={c.distance_price:.5f} "
                      f"duration={c.duration_value}{c.duration_unit}: {e}")
                c.ev_per_stake = None
        return candidates_sorted[:top_k]

    async def calibrate_universe(self) -> list[tuple[str, Candidate]]:
        """
        Scan every symbol in the universe using an explicit two-stage gate:

        STAGE 1 (MC/statistical) — the ensemble sweep already ran every (barrier, duration)
        candidate through simulation in _calibrate_symbol; here we filter down to only the
        ones the model is actually confident about (p_no_touch_ci_low clears
        MIN_WIN_PROB_FLOOR). This is pure model output — no market data, no API calls yet.

        STAGE 2 (market/return) — for ONLY those MC-confident candidates, ask Deriv for a
        real quote and keep the ones that both (a) pay at least MIN_RETURN_PCT and (b) clear
        EV_FLOOR. Both matter: MIN_RETURN_PCT alone doesn't imply positive EV (a candidate
        can pay 40%+ and still be a bad bet at a middling win probability), so EV_FLOOR is
        still the final word on whether a candidate that clears MIN_RETURN_PCT is worth it.

        The single best surviving candidate per symbol then goes through a final
        full-precision (config.MC_SIMULATIONS) confirmation pass before being accepted.
        """
        results: list[tuple[str, Candidate]] = []
        for symbol in config.SYMBOL_UNIVERSE:
            try:
                candidates = await self._calibrate_symbol(symbol)
                if not candidates:
                    continue

                # STAGE 1 — MC gate: which barriers does simulation say won't get touched,
                # and over what duration. No API calls yet.
                mc_confident = [c for c in candidates if c.p_no_touch_ci_low >= config.MIN_WIN_PROB_FLOOR]
                if not mc_confident:
                    near = max(candidates, key=lambda c: c.p_no_touch_ci_low)
                    print(f"[calibrator] {symbol}: MC gate — nothing clears win-prob floor "
                          f"(closest: {near.p_no_touch_ci_low:.3f} at barrier={near.distance_sigma}σ "
                          f"duration={near.duration_value}{near.duration_unit}, "
                          f"floor {config.MIN_WIN_PROB_FLOOR})")
                    continue

                # STAGE 2 — market gate: does Deriv actually pay enough for these specific,
                # MC-confident barrier/duration combos.
                quoted = await self._quote_candidates(symbol, mc_confident, top_k=5)
                viable = [
                    c for c in quoted
                    if c.ev_per_stake is not None and c.ask_price
                    and (c.payout / c.ask_price - 1) >= config.MIN_RETURN_PCT
                    and c.ev_per_stake >= config.EV_FLOOR
                ]
                if not viable:
                    # Same silent-failure problem as above, one level up: previously this
                    # just moved on to the next symbol with zero trace of *how close* (or
                    # far) the best candidate actually was. Log the closest miss so a run
                    # that rejects everything is diagnosable instead of a black box.
                    quoted_ok = [c for c in quoted if c.ev_per_stake is not None and c.ask_price]
                    if quoted_ok:
                        near = max(quoted_ok, key=lambda c: c.ev_per_stake)
                        return_pct = near.payout / near.ask_price - 1
                        print(f"[calibrator] {symbol}: market gate — {len(mc_confident)} "
                              f"MC-confident candidate(s) quoted, none cleared both floors. "
                              f"Closest: return={return_pct:.1%} (floor {config.MIN_RETURN_PCT:.0%}), "
                              f"ev={near.ev_per_stake:.4f} (floor {config.EV_FLOOR}), "
                              f"win_prob_ci_low={near.p_no_touch_ci_low:.3f}, "
                              f"barrier={near.distance_sigma}σ duration={near.duration_value}{near.duration_unit} | "
                              f"model_breakdown: analytical={near.p_analytical:.3f} "
                              f"bootstrap={near.p_bootstrap:.3f} "
                              f"markov={near.p_markov if near.p_markov is None else round(near.p_markov, 3)} "
                              f"confidence={near.ensemble_confidence:.3f}")
                    else:
                        print(f"[calibrator] {symbol}: all {len(quoted)} MC-confident candidates "
                              f"failed at the proposal stage (see errors above)")
                    continue
                best = max(viable, key=lambda c: c.ev_per_stake)

                # Final full-MC confirmation pass before accepting this symbol's pick.
                try:
                    prices = await self.client.tick_history(symbol, config.HISTORY_TICKS_FOR_CALIBRATION)
                    prices_arr = np.asarray(prices, dtype=float)
                    returns_arr = np.diff(np.log(prices_arr))
                    S0 = prices_arr[-1]
                    n_steps = ensemble.n_steps_for_duration(symbol, best.duration_value, best.duration_unit)
                    p_confirmed = await ensemble.confirm_with_full_mc(
                        self.client, symbol, S0, returns_arr, best.distance_price, n_steps,
                    )
                    best.p_no_touch_confirmed = p_confirmed
                    if p_confirmed < config.MIN_WIN_PROB_FLOOR:
                        print(f"[calibrator] {symbol}: failed full-MC confirmation "
                              f"({p_confirmed:.3f} < floor {config.MIN_WIN_PROB_FLOOR}) -- discarding pick")
                        continue
                except Exception as e:
                    print(f"[calibrator] {symbol}: full-MC confirmation failed, keeping screening estimate: {e}")

                results.append((symbol, best))
            except Exception as e:
                # one bad symbol shouldn't kill the whole calibration cycle
                print(f"[calibrator] {symbol} failed: {e}")
                continue

        results.sort(key=lambda item: item[1].ev_per_stake, reverse=True)
        return results

    async def top_n_symbols(self, n: int = config.TOP_N_SYMBOLS) -> list[tuple[str, Candidate]]:
        ranked = await self.calibrate_universe()
        return ranked[:n]
