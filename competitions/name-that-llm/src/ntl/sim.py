"""Simulate hidden-test layouts from held-out train prompts: k passages per prompt, cluster, assign, score."""
from __future__ import annotations
import numpy as np
from .common import balanced_accuracy
from .group import tfidf_distance, cluster, assign, sinkhorn, pair_prf


def simulate(LP: np.ndarray, y, pid, texts, per_prompt=(18,), draws: int = 5, thresh: float = 0.9, seed: int = 0, alpha: float = 0.0, sizes=None):
    """LP: blended log-probs for rows (already out-of-fold). per_prompt: int, (lo, hi) range or list; sizes overrides."""
    y, pid, texts = np.asarray(y), np.asarray(pid).astype(str), np.asarray(texts, dtype=object)
    rng = np.random.default_rng(seed)
    uniq = np.unique(pid)
    res = []
    for d in range(draws):
        keep = []
        for p in uniq:
            rows = np.where(pid == p)[0]
            k = per_prompt if isinstance(per_prompt, int) else int(rng.integers(per_prompt[0], per_prompt[1] + 1))
            keep.append(rng.choice(rows, min(k, len(rows)), replace=False))
        keep = np.concatenate(keep)
        plain = balanced_accuracy(y[keep], LP[keep].argmax(1))
        g = cluster(tfidf_distance(texts[keep]), thresh)
        pr = pair_prf(g, pid[keep])
        L = LP[keep]
        if alpha > 0:
            L = np.log(np.clip(sinkhorn(np.exp(L), g, alpha), 1e-12, 1))
        hung = balanced_accuracy(y[keep], assign(L, g))
        orac = balanced_accuracy(y[keep], assign(LP[keep], pid[keep]))
        res.append((plain, hung, orac, pr[0], pr[1]))
    r = np.array(res).mean(0)
    return dict(plain=r[0], hungarian=r[1], oracle=r[2], pair_precision=r[3], pair_recall=r[4])
