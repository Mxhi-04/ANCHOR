"""ANCHOR and the pure-ML baseline, with normalised CV+ conformal intervals.

ANCHOR:   Tg = Fox(w) + GP(w, Fox)      (GP learns only what physics misses)
Pure ML:  Tg = GP(w)                    (same GP, no physics)

Intervals use CV+ (Barber et al., Ann. Statist. 49, 486 (2021)) with
conformity scores normalised by the GP's own predictive std, so the band
widens where the model has seen little data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .physics import fox_tg

SEED = 42


def make_gp(n_features: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(1e-1, (1e-5, 1e1))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=SEED
    )


class _Base:
    """Fits on (w, Tg_K); predicts mean and std in K."""

    uses_physics = False

    def __init__(self, tg_homo: np.ndarray):
        self.tg_homo = np.asarray(tg_homo, dtype=float)

    def _features(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _offset(self, w: np.ndarray) -> np.ndarray:
        return np.zeros(len(w))

    def fit(self, w: np.ndarray, y: np.ndarray) -> "_Base":
        x = self._features(w)
        self.scaler_ = StandardScaler().fit(x)
        self.gp_ = make_gp(x.shape[1]).fit(self.scaler_.transform(x), y - self._offset(w))
        return self

    def predict(self, w: np.ndarray, return_std: bool = False):
        mean, std = self.gp_.predict(self.scaler_.transform(self._features(w)), return_std=True)
        mean = mean + self._offset(w)
        return (mean, std) if return_std else mean

    def length_scales(self) -> np.ndarray:
        return self.gp_.kernel_.k1.k2.length_scale


class Anchor(_Base):
    name = "ANCHOR"
    uses_physics = True

    def _offset(self, w):
        return fox_tg(w, self.tg_homo)

    def _features(self, w):
        return np.column_stack([w, fox_tg(w, self.tg_homo)])


class PureML(_Base):
    name = "Pure ML"

    def _features(self, w):
        return np.asarray(w, dtype=float)


@dataclass
class ConformalPrediction:
    """Everything needed to build an interval at any confidence level."""

    mean: np.ndarray          # (n_test,) model refit on all training rows
    std: np.ndarray           # (n_test,) GP std of that model
    loo_mean: np.ndarray      # (n_test, n_train) fold-model means
    loo_std: np.ndarray       # (n_test, n_train) fold-model stds
    scores: np.ndarray        # (n_train,) normalised out-of-fold residuals

    def interval(self, confidence: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.scores)
        alpha = 1.0 - confidence
        lo_vals = np.sort(self.loo_mean - self.scores * self.loo_std, axis=1)
        hi_vals = np.sort(self.loo_mean + self.scores * self.loo_std, axis=1)
        # round() guards against float error, e.g. (1 - 0.9) * 10 = 0.9999999999999998
        k_lo = int(np.floor(round(alpha * (n + 1), 9)))
        k_hi = int(np.ceil(round((1 - alpha) * (n + 1), 9)))
        lower = lo_vals[:, k_lo - 1] if k_lo >= 1 else np.full(len(lo_vals), -np.inf)
        upper = hi_vals[:, k_hi - 1] if k_hi <= n else np.full(len(hi_vals), np.inf)
        return lower, upper


def cv_plus(model_cls, tg_homo, w_train, y_train, w_test, n_folds: int = 5) -> ConformalPrediction:
    """Normalised CV+ around `model_cls`, fitted only on the training rows."""
    n = len(y_train)
    loo_mean = np.empty((len(w_test), n))
    loo_std = np.empty((len(w_test), n))
    scores = np.empty(n)
    folds = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=SEED)
    for fit_idx, out_idx in folds.split(w_train):
        m = model_cls(tg_homo).fit(w_train[fit_idx], y_train[fit_idx])
        mu_out, sd_out = m.predict(w_train[out_idx], return_std=True)
        scores[out_idx] = np.abs(y_train[out_idx] - mu_out) / np.maximum(sd_out, 1e-6)
        mu_test, sd_test = m.predict(w_test, return_std=True)
        loo_mean[:, out_idx] = mu_test[:, None]
        loo_std[:, out_idx] = np.maximum(sd_test, 1e-6)[:, None]
    full = model_cls(tg_homo).fit(w_train, y_train)
    mean, std = full.predict(w_test, return_std=True)
    return ConformalPrediction(mean, std, loo_mean, loo_std, scores)
