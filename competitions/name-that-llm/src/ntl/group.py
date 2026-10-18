"""Test-time prompt recovery and label assignment."""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from .common import K


def tfidf_distance(texts) -> np.ndarray:
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return D


def embedding_distance(E: np.ndarray) -> np.ndarray:
    Z = np.asarray(E, dtype=np.float64)
    Z = Z - Z.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    D = np.clip(1 - Z @ Z.T, 0, None)
    np.fill_diagonal(D, 0)
    return D


def cluster(D: np.ndarray, thresh: float, method: str = "average") -> np.ndarray:
    if D.shape[0] < 2:
        return np.ones(D.shape[0], dtype=int)
    return fcluster(linkage(squareform(D, checks=False), method=method), thresh, criterion="distance")


def pair_prf(groups, true_groups) -> tuple[float, float]:
    g, p = np.asarray(groups), np.asarray(true_groups)
    same_c = g[:, None] == g[None, :]
    same_p = p[:, None] == p[None, :]
    off = ~np.eye(len(g), dtype=bool)
    tp = (same_c & same_p & off).sum()
    return tp / max((same_c & off).sum(), 1), tp / max((same_p & off).sum(), 1)


def assign(L: np.ndarray, groups, max_size: int = K) -> np.ndarray:
    """Within each group use each label at most ceil(n/K) times (Hungarian on log-probabilities L)."""
    L = np.asarray(L)
    out = L.argmax(1).copy()
    groups = np.asarray(groups)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if len(i) < 2:
            continue
        reps = int(np.ceil(len(i) / K))
        r, c = linear_sum_assignment(-np.repeat(L[i], reps, axis=1))
        out[i[r]] = c // reps
    return out


def sinkhorn(P: np.ndarray, groups=None, alpha: float = 0.5, iters: int = 50) -> np.ndarray:
    """Rescale columns toward uniform class mass (alpha = strength), globally or within each group."""
    P = np.asarray(P, dtype=np.float64).copy()
    groups = np.zeros(len(P), dtype=int) if groups is None else np.asarray(groups)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if len(i) < 2:
            continue
        Q = P[i]
        target = (1 - alpha) * Q.sum(0) + alpha * np.full(K, len(i) / K)
        for _ in range(iters):
            Q = Q * (target / (Q.sum(0) + 1e-12))[None, :]
            Q = Q / (Q.sum(1, keepdims=True) + 1e-12)
        P[i] = Q
    return P
