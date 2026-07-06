"""
ensemble.py — the Intelligence Stack: blends every per-symbol signal into one
calibrated no-touch probability + a confidence signal, per the APEX-style
four-layer design (Market Perception -> per-symbol Intelligence Stack ->
Trade Selection -> Self-Improvement).

Signals combined per candidate (barrier distance x duration):
  - HMM regime (regime.py)                — which vol/trend regime we're in
  - Kalman-filtered drift (kalman.py)      — smoothed mu, replaces noisy regime mean
  - GARCH(1,1) vol forecast (garch.py)     — forward-looking vol, blended with EWMA
  - Hawkes jump-clustering (hawkes.py)     — inflates effective vol when jumps cluster
  - ARFIMA/GPH long memory (arfima.py)     — corrects the sqrt(T) horizon-scaling exponent
  - GBM/OU analytical prior (stochastic.py)— fast closed-form prior, now fed the above
  - Block-bootstrap (stochastic.py)        — PRIMARY empirical estimator (autocorrelation-aware)
  - Markov lattice first-passage (markov.py) — independent cross-check (memoryless assumption)
  - Bayesian Beta-Binomial posterior (bayesian.py) — online calibration against realized outcomes

The bootstrap remains the dominant weight (it's the most robust single estimator,
per the existing design doc), everything else adjusts its INPUTS (drift, vol,
horizon-scaling exponent) or provides a cross-check. The Markov/bootstrap
agreement is used as a confidence signal: when they disagree a lot, the
candidate's effective win-probability floor is pulled down, independent of
the raw blended point estimate — a wide, uncertain candidate should be traded
more conservatively (or not at all) even if its blended point estimate looks
good.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import config
import stochastic
import garch as garch_mod
import kalman as kalman_mod
import hawkes as hawkes_mod
import arfima as arfima_mod
import markov as markov_mod
from bayesian import BayesianTracker


@dataclass
class SymbolContext:
    """Per-symbol signals computed once per calibration cycle, reused across the
    barrier-distance x duration grid sweep for that symbol."""
    symbol: str
    sigma_tick_ewma: float
    garch_vol_per_tick: float
    garch_trust: float
    kalman_drift_per_tick: float
    kalman_confidence: float
    hurst: float
    is_mean_reverting: bool
    hawkes_intensity: float
    vol_inflation: float
    arfima_d_vol: float
    horizon_exponent: float


def build_symbol_context(symbol: str, prices: np.ndarray, returns: np.ndarray,
                          hurst: float, is_mean_reverting: bool) -> SymbolContext:
    sigma_tick_ewma = stochastic.ewma_vol(returns)
    price_diffs = np.diff(prices) if len(prices) >= 2 else np.array([0.0])
    price_now = float(prices[-1])

    garch_result = garch_mod.fit_garch(returns) if config.GARCH_ENABLED else None
    garch_vol, garch_trust = garch_mod.forecast_vol_per_tick(
        garch_result, sigma_tick_ewma, price_now,
    ) if config.GARCH_ENABLED else (sigma_tick_ewma, 0.5)

    kf = kalman_mod.LocalLevelKalman()
    if config.KALMAN_ENABLED and len(prices) >= 5:
        kf.fit(np.log(prices))
    kalman_drift = kf.drift_per_tick if config.KALMAN_ENABLED else 0.0
    kalman_conf = kf.drift_confidence if config.KALMAN_ENABLED else 0.0

    hawkes_intensity = hawkes_mod.jump_clustering_intensity(price_diffs) if config.HAWKES_ENABLED else 0.0
    vol_inflation = hawkes_mod.vol_inflation_factor(hawkes_intensity) if config.HAWKES_ENABLED else 1.0

    abs_returns = np.abs(returns)
    arfima_d = arfima_mod.gph_long_memory_d(abs_returns) if config.ARFIMA_ENABLED else 0.0
    horizon_exp = arfima_mod.horizon_scaling_exponent(arfima_d) if config.ARFIMA_ENABLED else 0.5

    return SymbolContext(
        symbol=symbol, sigma_tick_ewma=sigma_tick_ewma,
        garch_vol_per_tick=garch_vol, garch_trust=garch_trust,
        kalman_drift_per_tick=kalman_drift, kalman_confidence=kalman_conf,
        hurst=hurst, is_mean_reverting=is_mean_reverting,
        hawkes_intensity=hawkes_intensity, vol_inflation=vol_inflation,
        arfima_d_vol=arfima_d, horizon_exponent=horizon_exp,
    )


def effective_vol_per_tick(ctx: SymbolContext) -> float:
    """Blend EWMA and GARCH vol (weighted by GARCH's own trust score), then
    inflate for jump clustering."""
    blended = ctx.garch_trust * ctx.garch_vol_per_tick + (1 - ctx.garch_trust) * ctx.sigma_tick_ewma
    return blended * ctx.vol_inflation


def effective_drift_per_tick(ctx: SymbolContext, regime_mu: float) -> float:
    """Blend the Kalman-filtered drift with the HMM regime's mean, weighted by
    Kalman's own posterior confidence (falls back to the regime mean when the
    Kalman trend estimate is still noisy)."""
    return ctx.kalman_confidence * ctx.kalman_drift_per_tick + (1 - ctx.kalman_confidence) * regime_mu


def scaled_barrier_distance(d_sigma: float, ctx: SymbolContext, S0: float, n_steps: int) -> float:
    """
    Barrier distance in absolute price units, using the ARFIMA/GPH-corrected
    horizon-scaling exponent instead of a fixed sqrt(n_steps) -- see
    arfima.horizon_scaling_exponent for why persistent vol clustering widens
    this beyond sqrt(T).
    """
    vol = effective_vol_per_tick(ctx)
    return d_sigma * vol * S0 * (n_steps ** ctx.horizon_exponent)


@dataclass
class BlendResult:
    p_no_touch_model: float
    p_no_touch_calibrated: float
    p_no_touch_ci_low: float
    ensemble_confidence: float   # bootstrap/Markov agreement, 1.0 = perfect agreement
    p_bootstrap: float
    p_markov: float | None
    p_analytical: float


def blend_candidate(symbol: str, S0: float, prices: np.ndarray, ctx: SymbolContext, regime_mu: float,
                     regime_label: str, d_sigma: float, d_price: float, n_steps: int,
                     returns: np.ndarray, duration_minutes: float,
                     tracker: BayesianTracker, p_bootstrap: float | None = None) -> BlendResult:
    """
    p_bootstrap: pass a precomputed block-bootstrap no-touch probability when the caller
    has already batched the path simulation across multiple barrier distances at this
    duration (see sweep_symbol_grid) -- re-simulating per (duration, barrier) pair
    separately is the difference between ~17 simulation calls and ~119 per symbol per
    cycle. If None, this runs its own single-distance simulation (kept for standalone use).
    """
    vol = effective_vol_per_tick(ctx)
    drift = effective_drift_per_tick(ctx, regime_mu)

    # 1. Analytical prior (GBM or OU, fed the ensemble-adjusted drift/vol)
    if ctx.is_mean_reverting:
        theta, mu_ou, sigma_ou = stochastic.ou_calibrate(prices[-1000:])
        p_analytical = stochastic.ou_no_touch_prob(
            S0, S0 - d_price, S0 + d_price, theta, mu_ou, max(sigma_ou, 1e-9),
            T=n_steps, n_steps=n_steps, n_paths=500,
        )
    else:
        hurst_corr = stochastic.fbm_hurst_correction_factor(ctx.hurst)
        p_touch = stochastic.gbm_single_barrier_touch_prob(S0, S0 + d_price, drift, max(vol, 1e-9),
                                                            T=max(n_steps, 1))
        p_analytical = 1.0 - min(1.0, p_touch * hurst_corr)

    # 2. Block-bootstrap — PRIMARY estimator (empirical, autocorrelation-aware)
    if p_bootstrap is None:
        boot_probs = stochastic.block_bootstrap_touch_probabilities(
            S0, returns, [d_price], n_steps=n_steps,
            n_paths=config.MC_SCREENING_PATHS, block_size=config.BOOTSTRAP_BLOCK_SIZE,
            two_sided=True,
        )
        p_bootstrap = boot_probs[d_price]

    # 3. Markov lattice cross-check (independent, memoryless assumption)
    p_markov = markov_mod.markov_no_touch_prob(S0, returns, d_price, n_steps) \
        if config.MARKOV_ENABLED else None

    if p_markov is not None:
        agreement = 1.0 - abs(p_bootstrap - p_markov)
        p_no_touch_model = 0.35 * p_analytical + 0.45 * p_bootstrap + 0.20 * p_markov
    else:
        agreement = 0.5  # no cross-check available -- neutral, not "confident"
        p_no_touch_model = 0.4 * p_analytical + 0.6 * p_bootstrap

    # 4. Bayesian posterior blend (online calibration against realized outcomes)
    p_calibrated = tracker.calibrated_prob(symbol, regime_label, d_sigma, duration_minutes, p_no_touch_model)
    cell = tracker.get_cell(symbol, regime_label, d_sigma, duration_minutes)
    ci_low, _ = cell.credible_interval()
    n = cell.n_obs
    w = min(1.0, n / (n + 10.0))
    p_ci_low_blended = w * ci_low + (1 - w) * max(0.0, p_no_touch_model - 0.1)

    # Cross-check disagreement pulls the effective floor down further, independent
    # of the point estimate: a candidate where bootstrap and Markov strongly
    # disagree is genuinely less trustworthy even if its blended mean looks fine.
    p_ci_low_final = p_ci_low_blended * (0.5 + 0.5 * agreement)

    return BlendResult(
        p_no_touch_model=p_no_touch_model, p_no_touch_calibrated=p_calibrated,
        p_no_touch_ci_low=p_ci_low_final, ensemble_confidence=agreement,
        p_bootstrap=p_bootstrap, p_markov=p_markov, p_analytical=p_analytical,
    )


def n_steps_for_duration(symbol: str, duration_value: int, duration_unit: str) -> int:
    """
    Convert a DURATION_SEARCH_SPACE entry into a simulation step count. Tick-based
    entries ("t") map 1:1 to steps. Minute-based entries are converted via the
    symbol's approximate tick cadence, so a "3m" candidate and a "90t" candidate on
    a 2s-tick symbol get treated as roughly the same simulation horizon -- Deriv's
    own tick cadence is what actually governs execution; this is purely for giving
    the bootstrap/lattice/analytical models a consistent step count to work with.
    """
    if duration_unit == "t":
        return max(5, int(duration_value))
    tick_interval = config.SYMBOL_TICK_INTERVAL_SECONDS.get(symbol, config.DEFAULT_TICK_INTERVAL_SECONDS)
    duration_seconds = duration_value * 60.0
    return max(5, int(round(duration_seconds / tick_interval)))


def sweep_symbol_grid(symbol: str, S0: float, prices: np.ndarray, ctx: SymbolContext,
                       regime_mu: float, regime_label: str, returns: np.ndarray,
                       tracker: BayesianTracker) -> dict:
    """
    Sweeps config.DURATION_SEARCH_SPACE x config.BARRIER_DISTANCE_GRID_SIGMA for one
    symbol. Duration is the OUTER loop deliberately: the block-bootstrap path
    simulation only depends on n_steps (not on which barrier we're checking), so for
    each duration we simulate the paths ONCE and check every barrier distance against
    that same batch (stochastic.block_bootstrap_touch_probabilities already supports a
    list of distances for exactly this reason). Structuring it barrier-outer instead
    would mean re-simulating fresh paths for every (barrier, duration) pair separately
    -- 7x more simulation calls per symbol per cycle for no benefit.

    Returns {d_sigma: [(duration_value, duration_unit, duration_minutes, n_steps, BlendResult), ...]}
    covering every duration in the search space, unfiltered -- selection happens in
    select_durations_for_barrier.
    """
    by_barrier: dict = {d_sigma: [] for d_sigma in config.BARRIER_DISTANCE_GRID_SIGMA}

    for duration_value, duration_unit in config.DURATION_SEARCH_SPACE:
        n_steps = n_steps_for_duration(symbol, duration_value, duration_unit)
        duration_minutes = (
            duration_value if duration_unit == "m"
            else n_steps * config.SYMBOL_TICK_INTERVAL_SECONDS.get(
                symbol, config.DEFAULT_TICK_INTERVAL_SECONDS) / 60.0
        )

        barrier_distances_price = [
            scaled_barrier_distance(d_sigma, ctx, S0, n_steps) for d_sigma in config.BARRIER_DISTANCE_GRID_SIGMA
        ]
        # one simulation for this duration, shared across every barrier distance
        boot_probs = stochastic.block_bootstrap_touch_probabilities(
            S0, returns, barrier_distances_price, n_steps=n_steps,
            n_paths=config.MC_SCREENING_PATHS, block_size=config.BOOTSTRAP_BLOCK_SIZE,
            two_sided=True,
        )

        for d_sigma, d_price in zip(config.BARRIER_DISTANCE_GRID_SIGMA, barrier_distances_price):
            blend = blend_candidate(
                symbol=symbol, S0=S0, prices=prices, ctx=ctx, regime_mu=regime_mu,
                regime_label=regime_label, d_sigma=d_sigma, d_price=d_price, n_steps=n_steps,
                returns=returns, duration_minutes=duration_minutes, tracker=tracker,
                p_bootstrap=boot_probs[d_price],
            )
            by_barrier[d_sigma].append((duration_value, duration_unit, duration_minutes, n_steps, blend))

    return by_barrier


def select_durations_for_barrier(swept_for_barrier: list[tuple], top_k: int = None) -> list[tuple]:
    """
    THE DURATION AUTO-SELECTION STEP: given every duration's BlendResult at one barrier
    distance (from sweep_symbol_grid), pick which duration(s) are actually worth
    quoting -- rather than trading a duration a human picked in advance.

    Selection rule: among durations whose p_no_touch_ci_low clears
    config.MIN_WIN_PROB_FLOOR, prefer the ones sitting CLOSEST to that floor (from
    above) -- a duration where the model is almost certain to win (p far above the
    floor) is exactly the kind of candidate that gets a poor payout from Deriv, since
    payout scales inversely with win probability. The duration with the best
    payout-per-risk tradeoff is the shortest/closest one that still clears the risk
    bar, not the "safest-looking" one. If NOTHING clears the floor at this barrier,
    fall back to the single duration with the highest p_no_touch_ci_low (still
    useless once it reaches calibrate_universe's viability filter, but keeps the
    grid populated rather than silently dropping the barrier).
    """
    if top_k is None:
        top_k = config.DURATION_AUTOSELECT_TOP_K

    viable = [s for s in swept_for_barrier if s[4].p_no_touch_ci_low >= config.MIN_WIN_PROB_FLOOR]
    if viable:
        viable.sort(key=lambda s: s[4].p_no_touch_ci_low - config.MIN_WIN_PROB_FLOOR)
        return viable[:top_k]
    scored = sorted(swept_for_barrier, key=lambda s: s[4].p_no_touch_ci_low, reverse=True)
    return scored[:1]


async def confirm_with_full_mc(client, symbol: str, S0: float, returns: np.ndarray,
                                d_price: float, n_steps: int) -> float:
    """
    Final confirmation pass for the single best candidate per symbol before it's
    quoted/traded: re-estimate P(no touch) with the full config.MC_SIMULATIONS
    (75,000) bootstrap paths, rather than the cheaper MC_SCREENING_PATHS used
    to sweep the whole grid. Mirrors the multi-symbol bot's screen-then-confirm
    pattern -- 75k paths for every (symbol x duration x barrier) grid cell
    would be prohibitively slow, but is cheap enough for one confirmation call
    per symbol per cycle.
    """
    boot_probs = stochastic.block_bootstrap_touch_probabilities(
        S0, returns, [d_price], n_steps=n_steps,
        n_paths=config.MC_SIMULATIONS, block_size=config.BOOTSTRAP_BLOCK_SIZE,
        two_sided=True,
    )
    return boot_probs[d_price]
