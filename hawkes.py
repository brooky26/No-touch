"""
hawkes.py — Hawkes self-exciting jump-clustering intensity.

Part of the per-symbol Intelligence Stack. Large moves on these synthetic
indices tend to cluster (a jump raises the short-term probability of another
jump) rather than arrive as an i.i.d. Poisson process — that's exactly the
self-exciting behavior a Hawkes process models. Rather than a full Hawkes MLE
fit (expensive, and easy to destabilize on short/noisy tick windows), this
uses the same practical proxy the multi-symbol bot's structural gate uses:
compare the recent rate of "large move" events against the historical base
rate. A fit-free proxy is deliberate here — it's the same tradeoff Bry's
other bots already made, and it's stable enough for gating decisions.

Output is a jump_intensity in [0, 1]: 0 = no clustering above baseline,
1 = strong clustering (recent large-move rate far exceeds the historical
rate). The ensemble uses this to inflate the effective vol used for barrier
sizing when clustering is high, since a NOTOUCH candidate sized off calm-period
vol is exactly what gets blown through by a jump cluster.
"""
from __future__ import annotations
import numpy as np


def jump_clustering_intensity(price_diffs: np.ndarray, recent_window: int = 20) -> float:
    price_diffs = np.asarray(price_diffs, dtype=float)
    if len(price_diffs) < recent_window + 30:
        return 0.0
    sigma = np.std(price_diffs)
    if sigma <= 0:
        return 0.0
    thresh = 0.5 * sigma
    recent_rate = float(np.mean(np.abs(price_diffs[-recent_window:]) > thresh))
    base_rate = float(np.mean(np.abs(price_diffs) > thresh))
    if base_rate <= 1e-9:
        return 0.0
    excess = (recent_rate / base_rate) - 1.0
    return float(np.clip(excess / 3.0, 0.0, 1.0))


def vol_inflation_factor(jump_intensity: float, max_inflation: float = 1.5) -> float:
    """
    Multiplier applied to the ensemble's effective per-tick vol when jump
    clustering is elevated. 1.0 at intensity=0 (no adjustment), scaling up to
    max_inflation at intensity=1 (strong clustering -> widen the effective
    vol used for barrier sizing, since NOTOUCH pays off on staying INSIDE
    the barrier and jump clusters are exactly what breaks that).
    """
    return float(1.0 + jump_intensity * (max_inflation - 1.0))
