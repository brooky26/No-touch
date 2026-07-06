"""
stochastic.py — the math core.

Implements, per the design doc:
  1. EWMA volatility
  2. Hurst exponent (rescaled range, R/S)
  3. GBM analytical first-passage / touch probability (fast prior)
  4. OU (mean-reverting) simulation + first-passage via simulation
  5. Merton jump-diffusion path simulator (fat tails for barrier estimation)
  6. Fractional Brownian Motion simulation (Davies-Harte / Cholesky) for H != 0.5 correction
  7. Stationary block-bootstrap path simulator — the primary touch-probability estimator,
     since it preserves autocorrelation/vol clustering that Gaussian MC destroys.

All probability functions return P(NO TOUCH) i.e. survival probability, since that's the
NOTOUCH contract's payoff condition. P(touch) = 1 - P(no touch).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. EWMA volatility (RiskMetrics-style)
# ---------------------------------------------------------------------------
def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    """Return the EWMA-annualized-free (per-tick) volatility estimate of a return series."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return float(np.std(returns)) if len(returns) else 0.0
    weights = (1 - lam) * lam ** np.arange(len(returns))[::-1]
    weights /= weights.sum()
    mean = np.average(returns, weights=weights)
    var = np.average((returns - mean) ** 2, weights=weights)
    return float(np.sqrt(var))


# ---------------------------------------------------------------------------
# 2. Hurst exponent via rescaled range (R/S) analysis
# ---------------------------------------------------------------------------
def hurst_exponent(series: np.ndarray, min_lag: int = 8, max_lag: int | None = None) -> float:
    """
    Estimate the Hurst exponent of a price/return series using R/S analysis.
    H ~ 0.5  -> random walk (GBM-consistent)
    H  > 0.5 -> trending / persistent
    H  < 0.5 -> mean-reverting / anti-persistent
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < min_lag * 4:
        return 0.5  # not enough data — assume random walk
    if max_lag is None:
        max_lag = n // 4

    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), num=20).astype(int))
    lags = lags[lags >= min_lag]
    rs_vals = []
    valid_lags = []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i * lag:(i + 1) * lag]
            mean_adj = chunk - chunk.mean()
            cum = np.cumsum(mean_adj)
            r = cum.max() - cum.min()
            s = chunk.std(ddof=0)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return 0.5

    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_vals)
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(slope, 0.01, 0.99))


def fbm_hurst_correction_factor(hurst: float) -> float:
    """
    Rough correction multiplier applied to a GBM-implied touch probability to account for
    H != 0.5. Derived from the scaling relationship Var(t) ~ t^(2H) vs GBM's t^1: effective
    diffusive "reach" at horizon T scales as T^(2H-1) relative to GBM. This is a first-order
    heuristic prior, not a substitute for the block-bootstrap estimate.
    """
    return float(2.0 ** (2 * hurst - 1))


# ---------------------------------------------------------------------------
# 3. GBM analytical first-passage / touch probability (single barrier, fast prior)
# ---------------------------------------------------------------------------
def gbm_single_barrier_touch_prob(S0: float, barrier: float, mu: float, sigma: float, T: float) -> float:
    """
    Analytical probability that a GBM path started at S0 touches `barrier` before time T.
    Uses the reflection principle on log-price. Works for barrier above or below S0.
    Returns P(touch).
    """
    if sigma <= 0 or T <= 0:
        return 0.0
    log_ratio = np.log(barrier / S0)
    drift = (mu - 0.5 * sigma ** 2)
    d1 = (log_ratio - drift * T) / (sigma * np.sqrt(T))
    d2 = (log_ratio + drift * T) / (sigma * np.sqrt(T))
    if barrier > S0:
        p_touch = norm.cdf(-d1) + np.exp(2 * drift * log_ratio / sigma ** 2) * norm.cdf(-d2)
    else:
        p_touch = norm.cdf(d1) + np.exp(2 * drift * log_ratio / sigma ** 2) * norm.cdf(d2)
    return float(np.clip(p_touch, 0.0, 1.0))


def gbm_double_barrier_no_touch_prob(S0: float, lower: float, upper: float, mu: float,
                                      sigma: float, T: float, n_terms: int = 30) -> float:
    """
    Analytical survival probability (no-touch) for a GBM path confined between two barriers,
    via the classic image-series solution to the Kolmogorov PDE on a strip (log-price transform).
    Used for double-no-touch-style contracts / as a sanity-check prior.
    """
    if sigma <= 0 or T <= 0 or lower >= S0 or upper <= S0:
        return 0.0
    x = np.log(S0)
    a, b = np.log(lower), np.log(upper)
    width = b - a
    theta = mu - 0.5 * sigma ** 2
    # Series solution for P(no exit from [a,b] by T) for drifted BM
    total = 0.0
    for k in range(1, n_terms + 1):
        kpi = k * np.pi / width
        sin_term = np.sin(kpi * (x - a))
        decay = np.exp(-0.5 * (kpi ** 2 * sigma ** 2 + (theta / sigma) ** 2) * T)
        coeff = (2.0 / (k * np.pi)) * (1 - np.cos(k * np.pi))
        total += coeff * sin_term * decay
    drift_adj = np.exp(theta * (x - a) / sigma ** 2)
    p_survive = drift_adj * total
    return float(np.clip(p_survive, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 4. Ornstein-Uhlenbeck simulation + first-passage (for mean-reverting regimes, H<0.5)
# ---------------------------------------------------------------------------
def ou_calibrate(series: np.ndarray, dt: float = 1.0) -> tuple[float, float, float]:
    """
    Fit dS = theta*(mu - S)dt + sigma*dW via AR(1) regression on discretized OU.
    Returns (theta, mu, sigma).
    """
    series = np.asarray(series, dtype=float)
    S_t = series[:-1]
    S_tp1 = series[1:]
    # AR(1): S_tp1 = a + b*S_t + eps
    b, a = np.polyfit(S_t, S_tp1, 1)
    b = np.clip(b, 1e-6, 0.999999)
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    resid = S_tp1 - (a + b * S_t)
    resid_std = resid.std(ddof=1)
    # convert AR(1) residual variance to OU diffusion coefficient
    sigma = resid_std * np.sqrt(2 * theta / (1 - b ** 2)) if (1 - b ** 2) > 0 else resid_std
    return float(theta), float(mu), float(sigma)


def ou_simulate_paths(S0: float, theta: float, mu: float, sigma: float, T: float,
                       n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray:
    """Euler-Maruyama simulation of OU paths. Returns array shape (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    noise = rng.standard_normal((n_paths, n_steps))
    for i in range(n_steps):
        paths[:, i + 1] = (paths[:, i] + theta * (mu - paths[:, i]) * dt
                            + sigma * np.sqrt(dt) * noise[:, i])
    return paths


def ou_no_touch_prob(S0: float, lower: float, upper: float, theta: float, mu: float,
                      sigma: float, T: float, n_steps: int = 200, n_paths: int = 2000,
                      seed: int | None = None) -> float:
    """Simulation-based no-touch probability under OU dynamics (analytical OU first-passage
    densities are intractable in closed form for two-sided barriers, so we simulate)."""
    paths = ou_simulate_paths(S0, theta, mu, sigma, T, n_steps, n_paths, seed)
    survived = np.all((paths >= lower) & (paths <= upper), axis=1)
    return float(survived.mean())


# ---------------------------------------------------------------------------
# 5. Merton jump-diffusion path simulator (fat tails, for volatility-spike symbols)
# ---------------------------------------------------------------------------
def merton_jump_diffusion_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int,
                                 n_paths: int, jump_intensity: float = 0.1,
                                 jump_mean: float = 0.0, jump_std: float = 0.02,
                                 seed: int | None = None) -> np.ndarray:
    """
    Simulate Merton jump-diffusion price paths:
      dS/S = mu dt + sigma dW + dJ,  J ~ compound Poisson(jump_intensity, N(jump_mean, jump_std))
    Returns array shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    log_paths = np.empty((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(S0)
    drift = (mu - 0.5 * sigma ** 2 - jump_intensity * (np.exp(jump_mean + 0.5 * jump_std ** 2) - 1)) * dt
    diffusion = sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(jump_intensity * dt, size=(n_paths, n_steps))
    jump_sizes = np.zeros((n_paths, n_steps))
    mask = n_jumps > 0
    if mask.any():
        jump_sizes[mask] = rng.normal(jump_mean, jump_std, size=mask.sum()) * n_jumps[mask]
    increments = drift + diffusion + jump_sizes
    log_paths[:, 1:] = log_paths[:, [0]] + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


# ---------------------------------------------------------------------------
# 6. Fractional Brownian Motion simulation (Davies-Harte via FFT) — for H != 0.5 regimes
# ---------------------------------------------------------------------------
def fbm_paths(S0: float, mu: float, sigma: float, hurst: float, T: float, n_steps: int,
              n_paths: int, seed: int | None = None) -> np.ndarray:
    """
    Simulate geometric fBM price paths using the Davies-Harte circulant-embedding method
    for exact fGn (fractional Gaussian noise) generation, then exponentiate with drift.
    Returns array shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    n = n_steps

    def autocov(k, H):
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))

    k = np.arange(0, n)
    gamma = autocov(k, hurst)
    # circulant embedding
    row = np.concatenate([gamma, [0], gamma[:0:-1][:n - 1]]) if n > 1 else gamma
    # build proper first row of size 2n
    m = 2 * n
    c = np.zeros(m)
    c[:n] = gamma
    c[n + 1:] = gamma[1:][::-1]
    eigvals = np.fft.fft(c).real
    eigvals = np.clip(eigvals, 0, None)  # numerical safety

    fgn_paths = np.empty((n_paths, n))
    for p in range(n_paths):
        w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        w[0] = w[0].real
        if m % 2 == 0:
            w[m // 2] = w[m // 2].real
        z = np.fft.fft(np.sqrt(eigvals / m) * w)
        fgn_paths[p] = z[:n].real * np.sqrt(m)

    # scale fGn increments to have the right per-step variance, then integrate -> fBm
    fgn_paths *= (dt ** hurst)
    fbm_increments = fgn_paths
    log_returns = mu * dt + sigma * fbm_increments
    log_paths = np.log(S0) + np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)])
    return np.exp(log_paths)


# ---------------------------------------------------------------------------
# 7. Stationary block-bootstrap simulator — PRIMARY touch-probability estimator
# ---------------------------------------------------------------------------
def stationary_block_bootstrap_returns(returns: np.ndarray, n_steps: int, n_paths: int,
                                        block_size: int, seed: int | None = None) -> np.ndarray:
    """
    Politis-Romano stationary block bootstrap: resample overlapping blocks of historical
    returns (geometric block-length ~ block_size) to preserve autocorrelation and volatility
    clustering. Returns array shape (n_paths, n_steps) of resampled returns.

    Vectorized across paths (the loop is over n_steps only, with every path's index
    advanced/restarted in one vectorized numpy operation per step) rather than a
    per-path-per-step nested Python loop -- statistically identical to the naive
    per-path loop, but the difference between ~900 Python-level iterations and
    ~67 million is the difference between this being usable for a short-duration,
    fast-turnaround bot (5 ticks to a few minutes) and a full 75,000-path confirmation
    pass taking the better part of a minute.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    n_hist = len(returns)
    p_restart = 1.0 / block_size

    idx = rng.integers(0, n_hist, size=n_paths)
    out = np.empty((n_paths, n_steps))
    for t in range(n_steps):
        out[:, t] = returns[idx]
        idx = idx + 1
        restart_mask = (idx >= n_hist) | (rng.random(n_paths) < p_restart)
        n_restart = int(restart_mask.sum())
        if n_restart:
            idx[restart_mask] = rng.integers(0, n_hist, size=n_restart)
    return out


def block_bootstrap_touch_probabilities(S0: float, returns: np.ndarray, barrier_distances: list[float],
                                         n_steps: int, n_paths: int, block_size: int,
                                         two_sided: bool = True, seed: int | None = None) -> dict:
    """
    For a grid of barrier distances (absolute price units), compute the empirical no-touch
    probability using resampled historical return blocks compounded into price paths.

    If two_sided=True, barrier_distances define a symmetric [S0 - d, S0 + d] corridor
    (typical for NOTOUCH). If False, treat each distance as a single upper barrier.

    Returns {distance: p_no_touch}.
    """
    boot_returns = stationary_block_bootstrap_returns(returns, n_steps, n_paths, block_size, seed)
    log_paths = np.log(S0) + np.hstack([np.zeros((n_paths, 1)), np.cumsum(boot_returns, axis=1)])
    price_paths = np.exp(log_paths)

    results = {}
    for d in barrier_distances:
        if two_sided:
            lower, upper = S0 - d, S0 + d
            survived = np.all((price_paths >= lower) & (price_paths <= upper), axis=1)
        else:
            upper = S0 + d
            survived = np.all(price_paths <= upper, axis=1)
        results[d] = float(survived.mean())
    return results
