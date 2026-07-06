"""
garch.py — GARCH(1,1) conditional volatility forecasting.

Part of the per-symbol Intelligence Stack. The block-bootstrap estimator in
stochastic.py is the primary touch-probability driver because it preserves
empirical autocorrelation/vol-clustering, but it's backward-looking over the
whole history window. GARCH(1,1) gives a forward-looking one-step conditional
vol forecast that reacts faster to a recent vol regime shift (e.g., a burst of
large ticks) than the block-bootstrap's blended history does. The ensemble
uses this to scale the barrier-distance grid, not to replace the bootstrap.

Same fit pattern (scaled returns, Zero mean, Garch(1,1)) used in the
multi-symbol EXPIRYRANGE bot, kept consistent here.
"""
from __future__ import annotations
import contextlib
import io
import math
import warnings

import numpy as np
from arch import arch_model

GARCH_SCALE = 1000.0     # scale returns up for numerical stability during MLE fit
MIN_TICKS_FOR_FIT = 200


def fit_garch(returns: np.ndarray):
    """Fit GARCH(1,1) on scaled returns. Returns the fitted result, or None on failure/too little data."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * GARCH_SCALE
        am = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return am.fit(disp="off")
    except Exception:
        return None


def forecast_vol_per_tick(garch_result, baseline_vol_per_tick: float, price_now: float,
                           horizon: int = 1) -> tuple[float, float]:
    """
    Returns (vol_per_tick, trust) where vol_per_tick is in the same units as
    stochastic.ewma_vol's return (log-return std per tick), and trust in [0,1]
    reflects how far the GARCH forecast is from the baseline EWMA estimate
    (a forecast that's wildly different from recent realized vol is suspect —
    e.g. right after a regime break the MLE hasn't stabilized yet).

    Falls back to (baseline_vol_per_tick, 0.5) if GARCH is unavailable or the
    forecast is outside a sane band.
    """
    if garch_result is None:
        return baseline_vol_per_tick, 0.5
    try:
        fc = garch_result.forecast(horizon=horizon, reindex=False)
        cond_vol_scaled = math.sqrt(float(fc.variance.values[-1, horizon - 1]))
        vol_per_tick = cond_vol_scaled / GARCH_SCALE
        lo = baseline_vol_per_tick * 0.2
        hi = baseline_vol_per_tick * 5.0
        if lo <= vol_per_tick <= hi and vol_per_tick > 0:
            ratio = vol_per_tick / max(baseline_vol_per_tick, 1e-12)
            trust = float(np.clip(1.0 / (1.0 + max(ratio - 1.0, 0) * 2), 0.1, 1.0))
            return float(vol_per_tick), trust
    except Exception:
        pass
    return baseline_vol_per_tick, 0.5
