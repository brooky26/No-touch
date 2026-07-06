"""
arfima.py — long-memory (fractional differencing) estimation via GPH.

Part of the per-symbol Intelligence Stack. Hurst (in stochastic.py) captures
long memory in the RETURN series' level. This module estimates long memory in
the VOLATILITY series instead — |returns| here — which is the standard
finance use of the fractional-d idea (cf. FIGARCH): volatility shocks decay
slower than sqrt(T) scaling assumes when d_vol > 0, meaning a NOTOUCH barrier
sized off sqrt(T)-scaled vol will be systematically too tight over longer
durations if the symbol has persistent volatility clustering.

A full ARFIMA(p,d,q) MLE fit is heavy and fragile on the tick-count windows
available here. The GPH (Geweke-Porter-Hudak) log-periodogram regression
estimator gets the same d parameter from a simple OLS regression on the low
frequencies of the periodogram, which is standard practice and numerically
stable on a few thousand points — no new statistical dependency needed beyond
numpy.
"""
from __future__ import annotations
import numpy as np


def gph_long_memory_d(series: np.ndarray, power: float = 0.5) -> float:
    """
    Estimate the fractional differencing parameter d via GPH regression on
    the series' periodogram (typically called on |returns|, i.e. volatility
    proxy, not the returns themselves).

    Uses the lowest ~n^power frequencies (power=0.5 is the standard GPH
    bandwidth choice). Returns d clipped to [-0.49, 0.49] (stationarity /
    invertibility bounds for ARFIMA(0,d,0)); returns 0.0 (no long memory
    adjustment) if there isn't enough data.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 100:
        return 0.0

    x = series - series.mean()
    periodogram = (np.abs(np.fft.rfft(x)) ** 2) / (2 * np.pi * n)
    freqs = 2 * np.pi * np.arange(len(periodogram)) / n

    m = max(4, int(n ** power))
    m = min(m, len(periodogram) - 1)
    # skip freq index 0 (DC component)
    lam = freqs[1:m + 1]
    I = periodogram[1:m + 1]
    valid = I > 0
    if valid.sum() < 4:
        return 0.0
    lam, I = lam[valid], I[valid]

    y = np.log(I)
    x_reg = np.log(4 * np.sin(lam / 2) ** 2)
    try:
        slope, _ = np.polyfit(x_reg, y, 1)
    except Exception:
        return 0.0
    d = -slope / 2.0
    return float(np.clip(d, -0.49, 0.49))


def horizon_scaling_exponent(d_vol: float, base_exponent: float = 0.5) -> float:
    """
    Under long memory in volatility, vol-over-horizon scales as T^(base_exponent + d_vol)
    rather than the GBM-standard T^0.5 (base_exponent=0.5). d_vol > 0 (persistent vol
    clustering, shocks decay slower than short-memory) -> larger effective exponent ->
    vol grows faster than sqrt(T) over longer durations, meaning barriers need to be
    proportionally wider for longer-duration candidates than a pure sqrt(T) rule gives.
    d_vol < 0 (anti-persistent/mean-reverting vol) has the opposite effect.
    """
    return float(np.clip(base_exponent + 0.5 * d_vol, 0.1, 0.95))
