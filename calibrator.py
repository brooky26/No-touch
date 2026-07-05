"""
calibrator.py — the self-learning core.

For every symbol in config.SYMBOL_UNIVERSE:
  1. Pull tick history.
  2. Compute EWMA vol, Hurst exponent, and classify the current HMM regime.
  3. Choose the process family per the design doc:
       H in [0.4, 0.6]                 -> GBM prior (fast sanity check)
       H < 0.4 (or ADF says stationary) -> OU (mean-reverting) prior
       H > 0.6                          -> fBM-corrected GBM prior
     ...then ALWAYS refine with the stationary block-bootstrap (the robust estimator),
     which is what actually drives trading decisions.
  4. Sweep the (barrier distance x duration) grid, get the no-touch probability for each,
     blend with the Bayesian posterior for that (symbol, regime, bucket) cell.
  5. Pull a live Deriv proposal (real quoted payout) for the best candidates, compute EV.
  6. Rank all symbols by their best achievable EV, return the top N.

Recalibration is triggered externally (by bot.py) after N consecutive losses on a traded
symbol, or on a safety-net timer.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller

import config
import stochastic
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
    p_no_touch_model: float          # block-bootstrap / analytical blended estimate
    p_no_touch_calibrated: float     # blended with Bayesian posterior
    p_no_touch_ci_low: float
    payout: float | None = None
    ask_price: float | None = None
    ev_per_stake: float | None = None


DURATION_UNIT_TO_MINUTES = {"t": 0.033, "s": 1 / 60, "m": 1.0, "h": 60.0, "d": 1440.0}


def _duration_to_T_years_and_minutes(value: int, unit: str) -> tuple[float, float]:
    minutes = value * DURATION_UNIT_TO_MINUTES[unit]
    T_years = minutes / (60 * 24 * 365)
    return T_years, minutes


class SymbolCalibrator:
    def __init__(self, client: DerivClient, tracker: BayesianTracker):
        self.client = client
        self.tracker = tracker
        self.regime_detectors: dict[str, RegimeDetector] = {}

    async def _calibrate_symbol(self, symbol: str) -> list[Candidate]:
        prices = await self.client.tick_history(symbol, config.HISTORY_TICKS_FOR_CALIBRATION)
        if len(prices) < 200:
            return []
        import numpy as np
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices))
        S0 = prices[-1]

        sigma_tick = stochastic.ewma_vol(returns)
        # Hurst must be estimated on the stationary return series, not raw price levels —
        # R/S on price levels trivially reads H~1 since price is an integrated (non-stationary) process.
        hurst = stochastic.hurst_exponent(returns)

        # ADF test as a secondary confirmation of mean reversion
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
        regime_info = regime_params.get(regime_id, {"label": "unknown", "mu": 0.0, "sigma": sigma_tick})
        regime_label = regime_info["label"]

        candidates: list[Candidate] = []

        for duration_value, duration_unit in config.DURATION_GRID:
            T_years, duration_minutes = _duration_to_T_years_and_minutes(duration_value, duration_unit)
            n_steps = max(10, min(500, int(duration_minutes)))  # cap sim resolution
            T_ticks_equivalent = n_steps  # bootstrap works in "tick step" space using historical returns

            barrier_distances_price = [d_sigma * sigma_tick * S0 * (n_steps ** 0.5)
                                        for d_sigma in config.BARRIER_DISTANCE_GRID_SIGMA]

            if is_mean_reverting:
                theta, mu_ou, sigma_ou = stochastic.ou_calibrate(prices[-1000:])
                model_probs = {}
                for d_sigma, d_price in zip(config.BARRIER_DISTANCE_GRID_SIGMA, barrier_distances_price):
                    p = stochastic.ou_no_touch_prob(
                        S0, S0 - d_price, S0 + d_price, theta, mu_ou, sigma_ou,
                        T=n_steps, n_steps=n_steps, n_paths=500,
                    )
                    model_probs[d_sigma] = p
            else:
                # GBM analytical prior, then corrected for Hurst if trending
                hurst_corr = stochastic.fbm_hurst_correction_factor(hurst)
                model_probs = {}
                for d_sigma, d_price in zip(config.BARRIER_DISTANCE_GRID_SIGMA, barrier_distances_price):
                    p_touch = stochastic.gbm_single_barrier_touch_prob(
                        S0, S0 + d_price, regime_info["mu"], max(sigma_tick, 1e-9), T=max(n_steps, 1),
                    )
                    p_touch = min(1.0, p_touch * hurst_corr)
                    model_probs[d_sigma] = 1.0 - p_touch

            # Refine every candidate with the robust block-bootstrap estimator
            boot_probs = stochastic.block_bootstrap_touch_probabilities(
                S0, returns, barrier_distances_price, n_steps=n_steps,
                n_paths=config.BOOTSTRAP_N_PATHS, block_size=config.BOOTSTRAP_BLOCK_SIZE,
                two_sided=True,
            )

            for d_sigma, d_price in zip(config.BARRIER_DISTANCE_GRID_SIGMA, barrier_distances_price):
                p_model_prior = model_probs[d_sigma]
                p_boot = boot_probs[d_price]
                # 60/40 blend favoring the empirical bootstrap, the more robust estimator
                p_no_touch_model = 0.4 * p_model_prior + 0.6 * p_boot

                p_calibrated = self.tracker.calibrated_prob(
                    symbol, regime_label, d_sigma, duration_minutes, p_no_touch_model,
                )
                cell = self.tracker.get_cell(symbol, regime_label, d_sigma, duration_minutes)
                ci_low, _ = cell.credible_interval()
                # shrink CI toward the model estimate when the cell has few observations
                n = cell.n_obs
                w = min(1.0, n / (n + 10.0))
                p_ci_low_blended = w * ci_low + (1 - w) * max(0.0, p_no_touch_model - 0.1)

                candidates.append(Candidate(
                    symbol=symbol, regime_label=regime_label, distance_sigma=d_sigma,
                    distance_price=d_price, duration_value=duration_value, duration_unit=duration_unit,
                    duration_minutes=duration_minutes, p_no_touch_model=p_no_touch_model,
                    p_no_touch_calibrated=p_calibrated, p_no_touch_ci_low=p_ci_low_blended,
                ))

        return candidates

    async def _quote_top_candidates(self, symbol: str, candidates: list[Candidate], top_k: int = 3):
        """Fetch live Deriv proposals for the most promising model candidates and compute EV."""
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
            except Exception:
                c.ev_per_stake = None
        return candidates_sorted[:top_k]

    async def calibrate_universe(self) -> list[tuple[str, Candidate]]:
        """
        Scan every symbol in the universe, return a list of (symbol, best_candidate) sorted
        by EV descending, restricted to candidates that clear the EV floor and win-prob floor.
        """
        results: list[tuple[str, Candidate]] = []
        for symbol in config.SYMBOL_UNIVERSE:
            try:
                candidates = await self._calibrate_symbol(symbol)
                if not candidates:
                    continue
                quoted = await self._quote_top_candidates(symbol, candidates, top_k=3)
                viable = [
                    c for c in quoted
                    if c.ev_per_stake is not None
                    and c.ev_per_stake >= config.EV_FLOOR
                    and c.p_no_touch_ci_low >= config.MIN_WIN_PROB_FLOOR
                ]
                if viable:
                    best = max(viable, key=lambda c: c.ev_per_stake)
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
