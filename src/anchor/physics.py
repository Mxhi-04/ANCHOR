"""Physics priors for copolymer glass transition temperature (all in kelvin)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def fox_tg(w: np.ndarray, tg_homo: np.ndarray) -> np.ndarray:
    """Fox equation: 1/Tg = sum_i w_i / Tg_i.

    w: (n, m) weight fractions, each row summing to 1. tg_homo: (m,) in K.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    tg_homo = np.asarray(tg_homo, dtype=float)
    if not np.allclose(w.sum(axis=1), 1.0, atol=5e-3):
        raise ValueError("weight fractions must sum to 1")
    return 1.0 / (w @ (1.0 / tg_homo))


def gordon_taylor_tg(w1, tg1: float, tg2: float, k: float) -> np.ndarray:
    """Binary Gordon-Taylor: Tg = (w1 Tg1 + k w2 Tg2) / (w1 + k w2).

    With k = Tg1 / Tg2 it reduces exactly to Fox.
    """
    w1 = np.asarray(w1, dtype=float)
    w2 = 1.0 - w1
    return (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)


def fit_gordon_taylor_k(w1, tg_measured, tg1: float, tg2: float) -> float:
    """Least-squares Gordon-Taylor k for one binary system."""
    w1 = np.asarray(w1, dtype=float)
    y = np.asarray(tg_measured, dtype=float)

    def sse(log_k: float) -> float:
        return float(np.sum((gordon_taylor_tg(w1, tg1, tg2, np.exp(log_k)) - y) ** 2))

    return float(np.exp(minimize_scalar(sse, bounds=(-5, 5), method="bounded").x))
