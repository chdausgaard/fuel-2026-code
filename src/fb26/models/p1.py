"""P1 estimator (memo §6.1): WLS of the swing on recentered burden with design controls.

Frisch–Waugh–Lovell: outcome and treatment are residualized on the controls
(race FE, E_c × PADD dummies, μ^pt, B̃^lag, Oil_c, X_c) once. The coefficient
and its state-clustered CR1 standard error come from the residualized pair and
equal those of the full regression. RI reuses the same residualizer
(fb26.models.ri).
"""
from dataclasses import dataclass

import numpy as np


class Residualizer:
    """Weighted projection off the column space of Z: v -> v - Z (Z'WZ)^+ Z'W v."""

    def __init__(self, z: np.ndarray, w: np.ndarray, tol: float = 1e-10):
        if np.any(w <= 0):
            raise ValueError('weights must be positive')
        self._sw = np.sqrt(w)
        u, s, _ = np.linalg.svd(z * self._sw[:, None], full_matrices=False)
        self.rank = int((s > tol * s.max()).sum())
        self._q = u[:, :self.rank]

    def __call__(self, v: np.ndarray) -> np.ndarray:
        sw = self._sw if v.ndim == 1 else self._sw[:, None]
        sv = v * sw
        return (sv - self._q @ (self._q.T @ sv)) / sw


def cr1_factor(n: int, k: int, g: int) -> float:
    """Small-sample factor of the CR1 sandwich (as in Stata and statsmodels)."""
    return g / (g - 1) * (n - 1) / (n - k)


def cluster_codes(labels) -> np.ndarray:
    """Dense 0..G-1 integer codes for cluster labels."""
    return np.unique(np.asarray(labels), return_inverse=True)[1]


@dataclass
class P1Fit:
    beta: float
    se_cr1: float
    t: float
    n: int
    n_clusters: int
    n_params: int


def fit_residualized(y: np.ndarray, x: np.ndarray, w: np.ndarray, codes: np.ndarray,
                     n_params: int) -> P1Fit:
    """WLS slope and CR1 SE from residualized y and x; n_params counts all regressors."""
    den = np.sum(w * x * x)
    beta = np.sum(w * x * y) / den
    scores = np.bincount(codes, weights=w * x * (y - beta * x))
    n, g = len(y), int(codes.max()) + 1
    se = np.sqrt(cr1_factor(n, n_params, g) * np.sum(scores ** 2)) / den
    return P1Fit(float(beta), float(se), float(beta / se), n, g, n_params)


def fit(y: np.ndarray, x: np.ndarray, z: np.ndarray, w: np.ndarray, clusters) -> P1Fit:
    res = Residualizer(z, w)
    return fit_residualized(res(y), res(x), w, cluster_codes(clusters), res.rank + 1)
