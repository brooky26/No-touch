"""
bayesian.py — online Bayesian calibration of touch/no-touch outcomes.

Maintains a Beta-Bernoulli posterior per (symbol, regime_label, barrier_bucket, duration_bucket)
cell. This is the same win-rate gating pattern used in the EXPIRYRANGE bot, applied here to
calibrate *touch probability estimates* against realized outcomes rather than trusting the
bootstrap/analytical estimate blindly forever.

Cell key: tuple(symbol, regime_label, barrier_bucket, duration_bucket)
"""
from __future__ import annotations
import bisect
from dataclasses import dataclass, field

import config


def barrier_bucket(distance_in_sigma: float) -> str:
    edges = config.BARRIER_BUCKET_EDGES
    idx = bisect.bisect_right(edges, distance_in_sigma) - 1
    idx = max(0, min(idx, len(edges) - 2))
    return f"{edges[idx]:.2f}-{edges[idx+1]:.2f}sigma"


def duration_bucket(duration_minutes: float) -> str:
    if duration_minutes <= 30:
        return "<=30m"
    if duration_minutes <= 120:
        return "30m-2h"
    if duration_minutes <= 360:
        return "2h-6h"
    return ">6h"


@dataclass
class BetaCell:
    alpha: float = config.BETA_PRIOR_ALPHA
    beta: float = config.BETA_PRIOR_BETA

    def update(self, won: bool):
        if won:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n_obs(self) -> float:
        return self.alpha + self.beta - config.BETA_PRIOR_ALPHA - config.BETA_PRIOR_BETA

    def credible_interval(self, z: float = 1.64) -> tuple[float, float]:
        """Normal approximation to the Beta CI (~90% for z=1.64)."""
        a, b = self.alpha, self.beta
        n = a + b
        mean = a / n
        var = (a * b) / (n ** 2 * (n + 1))
        sd = var ** 0.5
        return max(0.0, mean - z * sd), min(1.0, mean + z * sd)


@dataclass
class BayesianTracker:
    cells: dict[tuple, BetaCell] = field(default_factory=dict)

    def _key(self, symbol: str, regime_label: str, distance_sigma: float, duration_min: float) -> tuple:
        return (symbol, regime_label, barrier_bucket(distance_sigma), duration_bucket(duration_min))

    def get_cell(self, symbol: str, regime_label: str, distance_sigma: float, duration_min: float) -> BetaCell:
        key = self._key(symbol, regime_label, distance_sigma, duration_min)
        if key not in self.cells:
            self.cells[key] = BetaCell()
        return self.cells[key]

    def calibrated_prob(self, symbol: str, regime_label: str, distance_sigma: float,
                         duration_min: float, model_estimate: float, prior_weight: float = 0.5) -> float:
        """
        Blend the model-derived probability (block-bootstrap/analytical) with the empirical
        Bayesian posterior for this cell. Early on (few observations) the model estimate
        dominates; as n_obs grows the empirical posterior takes over.
        """
        cell = self.get_cell(symbol, regime_label, distance_sigma, duration_min)
        n = cell.n_obs
        # posterior weight grows with observations, saturating around n=30
        w_post = min(1.0, n / 30.0) * (1 - prior_weight) + (prior_weight if n == 0 else 0)
        w_post = min(1.0, n / (n + 10.0))  # simple shrinkage: more obs -> trust empirical more
        return w_post * cell.mean + (1 - w_post) * model_estimate

    def record_outcome(self, symbol: str, regime_label: str, distance_sigma: float,
                        duration_min: float, no_touch_won: bool):
        cell = self.get_cell(symbol, regime_label, distance_sigma, duration_min)
        cell.update(no_touch_won)

    def to_dict(self) -> dict:
        """Serialize for persistence (Supabase)."""
        return {
            "|".join(map(str, k)): {"alpha": c.alpha, "beta": c.beta}
            for k, c in self.cells.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianTracker":
        tracker = cls()
        for k_str, v in data.items():
            key = tuple(k_str.split("|"))
            tracker.cells[key] = BetaCell(alpha=v["alpha"], beta=v["beta"])
        return tracker
