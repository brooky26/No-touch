"""
kalman.py — local-level Kalman filter for drift/trend estimation.

Part of the per-symbol Intelligence Stack. HMM regime detection tells us WHICH
regime we're in; this gives a smoothed, noise-filtered estimate of the current
drift (mu) within that regime, updated every tick rather than re-estimated
from a whole-window mean. That smoothed drift feeds the GBM/fBM analytical
prior in stochastic.py in place of the regime's raw historical mean, which is
noisy on short windows.

Hand-rolled rather than a pykalman dependency: a 1D local-level-with-trend
model has a closed-form-enough recursion that a small dependency isn't worth
the extra install surface, and it keeps this consistent with the rest of the
codebase's numpy-only math modules (stochastic.py, bayesian.py).
"""
from __future__ import annotations
import numpy as np


class LocalLevelKalman:
    """
    State: x_t = [level_t, trend_t]
    Transition:  level_t = level_{t-1} + trend_{t-1} + w_level
                 trend_t = trend_{t-1} + w_trend
    Observation: z_t = level_t + v   (we observe log-price directly)

    q_level/q_trend are process-noise variances (how much the level/trend can
    drift tick to tick); r is observation-noise variance. Defaults are set
    relative to the series' own variance in fit(), not hardcoded absolutes,
    since tick sizes vary wildly across Deriv synthetic indices.
    """

    def __init__(self):
        self.x = np.zeros(2)              # [level, trend]
        self.P = np.eye(2) * 1e-2
        self.Q = np.eye(2) * 1e-6
        self.R = 1e-4
        self.fitted = False

    def fit(self, log_prices: np.ndarray) -> "LocalLevelKalman":
        log_prices = np.asarray(log_prices, dtype=float)
        if len(log_prices) < 5:
            self.fitted = False
            return self

        diffs = np.diff(log_prices)
        obs_var = float(np.var(diffs)) if len(diffs) > 1 else 1e-8
        obs_var = max(obs_var, 1e-12)

        self.x = np.array([log_prices[0], float(np.mean(diffs[:5])) if len(diffs) >= 5 else 0.0])
        self.P = np.eye(2) * obs_var * 10
        # process noise: level moves ~ observation noise scale, trend moves much slower
        self.Q = np.diag([obs_var * 0.5, obs_var * 0.01])
        self.R = obs_var

        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])

        for z in log_prices:
            # predict
            x_pred = F @ self.x
            P_pred = F @ self.P @ F.T + self.Q
            # update
            y = z - (H @ x_pred)[0]
            S = (H @ P_pred @ H.T)[0, 0] + self.R
            K = (P_pred @ H.T) / S
            self.x = x_pred + (K.flatten() * y)
            self.P = P_pred - K @ H @ P_pred

        self.fitted = True
        return self

    @property
    def level(self) -> float:
        return float(self.x[0])

    @property
    def drift_per_tick(self) -> float:
        """Smoothed trend estimate, in log-return units per tick — a Kalman-filtered mu."""
        return float(self.x[1]) if self.fitted else 0.0

    @property
    def drift_confidence(self) -> float:
        """
        Inverse of the trend state's posterior variance, normalized to [0,1] via a
        soft cap. High confidence = tight posterior on the trend estimate (long,
        stable history); low confidence = trend estimate is still noisy (short/volatile
        history), in which case the ensemble should lean on the HMM regime mean instead.
        """
        if not self.fitted:
            return 0.0
        trend_var = float(self.P[1, 1])
        return float(np.clip(1.0 / (1.0 + trend_var * 1e6), 0.0, 1.0))
