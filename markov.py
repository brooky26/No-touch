"""
markov.py — discrete-state Markov chain first-passage estimator.

Part of the per-symbol Intelligence Stack, and an intentionally DIFFERENT
mechanism from the block-bootstrap estimator in stochastic.py: the bootstrap
resamples correlated historical blocks (preserves autocorrelation/vol
clustering), which is a good empirical estimator but isn't strictly Markov.
This module instead builds an empirical single-step transition kernel
(discretized historical tick-to-tick price displacement, memoryless by
construction) and propagates it as a lattice random walk with absorbing
boundaries at the barrier distance, computing survival probability directly.

The two estimators agreeing is a much stronger signal than either alone;
disagreement flags a symbol where autocorrelation structure matters a lot
(bootstrap should be trusted more) or where the sample is too short/noisy for
a stable empirical kernel (Markov estimate should be down-weighted) — the
ensemble uses the SPREAD between them as a confidence signal, not just their
average.
"""
from __future__ import annotations
import numpy as np


def markov_no_touch_prob(S0: float, returns: np.ndarray, barrier_distance_price: float,
                          n_steps: int, resolution_per_sigma: int = 4,
                          max_lattice_halfwidth: int = 150) -> float | None:
    """
    Returns P(no touch) under a discretized Markov/lattice random walk with the
    empirical single-step displacement distribution, absorbing at ±barrier_distance_price.
    Returns None if there isn't enough history to build a stable kernel.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 300:
        return None
    sigma_tick = float(np.std(returns))
    if sigma_tick <= 0:
        return None

    bin_width_price = sigma_tick * S0 / resolution_per_sigma
    if bin_width_price <= 0:
        return None

    displacement_price = returns * S0
    steps_binned = np.round(displacement_price / bin_width_price).astype(int)

    kernel_halfwidth = int(np.clip(np.percentile(np.abs(steps_binned), 99), 2, 60))
    kernel_states = np.arange(-kernel_halfwidth, kernel_halfwidth + 1)
    counts = np.array([(steps_binned == k).sum() for k in kernel_states], dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    kernel = counts / total

    D = int(round(barrier_distance_price / bin_width_price))
    D = min(D, max_lattice_halfwidth)
    if D < kernel_halfwidth:
        # barrier narrower than the kernel's own typical jump -- lattice too coarse
        # to resolve this distance meaningfully at this resolution.
        return None

    lattice_halfwidth = D + kernel_halfwidth
    n_states = 2 * lattice_halfwidth + 1
    center = lattice_halfwidth

    dist = np.zeros(n_states)
    dist[center] = 1.0

    n_steps_sim = min(int(n_steps), 500)
    absorb_lo = center - D
    absorb_hi = center + D

    for _ in range(n_steps_sim):
        dist = np.convolve(dist, kernel, mode="same")
        # absorb (zero out) any mass that has exited [-D, D]
        if absorb_lo > 0:
            dist[:absorb_lo] = 0.0
        if absorb_hi < n_states - 1:
            dist[absorb_hi + 1:] = 0.0

    survival = float(dist.sum())
    return float(np.clip(survival, 0.0, 1.0))
