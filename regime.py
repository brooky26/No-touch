"""
regime.py — Markov-modulated regime detection.

Same HMM infrastructure pattern used across your other bots. Fits a Gaussian HMM on
[return, rolling_vol] features and exposes:
  - fit(returns) -> trains/refits the model
  - current_regime(returns) -> int regime id for the most recent observation
  - regime_params(returns) -> per-regime (mu, sigma) dict, used to feed the stochastic layer

3 states by default: low-vol/mean-reverting, trending, high-vol/crisis (config.HMM_N_REGIMES).
"""
from __future__ import annotations
import numpy as np
from hmmlearn.hmm import GaussianHMM

import config


class RegimeDetector:
    def __init__(self, n_regimes: int = config.HMM_N_REGIMES, window: int = 20, seed: int = 42):
        self.n_regimes = n_regimes
        self.window = window
        self.seed = seed
        self.model: GaussianHMM | None = None
        self._fitted_len = 0

    @staticmethod
    def _features(returns: np.ndarray, window: int) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        rolling_vol = np.array([
            returns[max(0, i - window):i + 1].std() if i > 0 else 0.0
            for i in range(len(returns))
        ])
        return np.column_stack([returns, rolling_vol])

    def fit(self, returns: np.ndarray, n_restarts: int = 5) -> "RegimeDetector":
        """
        Fit with multiple random restarts and keep the best-scoring model. GaussianHMM with
        few components/features can otherwise collapse a state to a degenerate near-empty
        cluster with default covariance — restarts + a floor on covariance avoid that.
        """
        X = self._features(returns, self.window)
        best_model, best_score = None, -np.inf
        for i in range(n_restarts):
            model = GaussianHMM(n_components=self.n_regimes, covariance_type="diag",
                                 n_iter=300, random_state=self.seed + i, min_covar=1e-6,
                                 tol=1e-4)
            try:
                model.fit(X)
                score = model.score(X)
            except Exception:
                continue
            # reject degenerate fits where a state's variance blew up to the init default
            max_var = model.covars_.max()
            typical_var = np.var(X)
            if max_var > 50 * max(typical_var, 1e-12):
                continue
            if score > best_score:
                best_model, best_score = model, score
        if best_model is None:
            # fall back to a plain single-restart fit rather than leaving self.model unset
            best_model = GaussianHMM(n_components=self.n_regimes, covariance_type="diag",
                                      n_iter=300, random_state=self.seed, min_covar=1e-6)
            best_model.fit(X)

        # Safety clamp: even after restarts, a state that never gets meaningful posterior
        # mass can retain a degenerate/init covariance far outside the data's range. Clip
        # any state's variance to a sane multiple of the empirical feature variance so a
        # degenerate state can never poison downstream sigma estimates fed to the stochastic
        # layer (which would otherwise size candidates off a nonsense sigma).
        # NOTE: covars_ is a computed property in hmmlearn — mutating elements of the array
        # it returns does not persist. Build the corrected array and reassign it wholesale.
        empirical_var = np.var(X, axis=0)
        cap = 25.0 * np.maximum(empirical_var, 1e-12)
        diag_covars = np.array([np.diag(best_model.covars_[k]) for k in range(self.n_regimes)])
        diag_covars = np.minimum(diag_covars, cap[np.newaxis, :])
        best_model.covars_ = diag_covars

        self.model = best_model
        self._fitted_len = len(returns)
        return self

    def needs_refit(self, n_new_obs: int, refit_every: int = config.HMM_REFIT_EVERY_N_CYCLES) -> bool:
        return self.model is None or (n_new_obs - self._fitted_len) >= refit_every

    def current_regime(self, returns: np.ndarray) -> int:
        if self.model is None:
            self.fit(returns)
        X = self._features(returns, self.window)
        states = self.model.predict(X)
        return int(states[-1])

    def regime_params(self, returns: np.ndarray) -> dict[int, dict]:
        """Per-regime (mu, sigma) of the *return* dimension, ranked by volatility."""
        if self.model is None:
            self.fit(returns)
        means = self.model.means_[:, 0]
        # diag covariance: variances_ shape (n_components, n_features)
        variances = self.model.covars_
        if variances.ndim == 3:
            sigmas = np.sqrt(np.array([variances[k][0, 0] for k in range(self.n_regimes)]))
        else:
            sigmas = np.sqrt(variances[:, 0])
        order = np.argsort(sigmas)  # 0 = calmest regime
        label_map = {}
        labels = ["mean_reverting_lowvol", "trending", "high_vol_crisis"]
        for rank, idx in enumerate(order):
            label = labels[rank] if rank < len(labels) else f"regime_{rank}"
            label_map[int(idx)] = {"label": label, "mu": float(means[idx]), "sigma": float(sigmas[idx])}
        return label_map
