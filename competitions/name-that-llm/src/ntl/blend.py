"""Log-linear blending of member probabilities with weights fitted on out-of-fold predictions."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .common import K, log_softmax, balanced_accuracy


def member_logp(P: np.ndarray, kind: str = "prob", temp: float = 1.0) -> np.ndarray:
    if kind == "prob":
        return np.log(np.clip(P, 1e-9, 1.0))
    return log_softmax(P, temp)   # raw margins / logits


def fit_weights(LPs: list[np.ndarray], y: np.ndarray, l2: float = 1e-3, class_balance: bool = True) -> np.ndarray:
    """Weights w >= 0 minimising class-balanced NLL of softmax(sum_m w_m * LP_m)."""
    y = np.asarray(y)
    cw = np.ones(len(y))
    if class_balance:
        counts = np.bincount(y, minlength=K)
        cw = (len(y) / (K * counts[y]))
    S = np.stack(LPs, 0)   # (M, n, K)

    def f(w):
        z = np.tensordot(w, S, 1)
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        nll = (cw * (lse - z[np.arange(len(y)), y])).sum() / cw.sum()
        return nll + l2 * (w ** 2).sum()

    w0 = np.full(len(LPs), 1.0 / len(LPs))
    res = minimize(f, w0, method="L-BFGS-B", bounds=[(0, None)] * len(LPs))
    return res.x


def combine(LPs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    z = np.tensordot(w, np.stack(LPs, 0), 1)
    return log_softmax(z)
