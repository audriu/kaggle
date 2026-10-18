#!/usr/bin/env python3
"""Part 2 of exp_grouping: realistic clustering of held-out passages (uses tmp/oof_stylegram.npz)."""
import numpy as np, pandas as pd, time
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "data" / "train.csv")
texts = train.text.astype(str).to_numpy(dtype=object); pid = train.prompt_id.astype(str).to_numpy(dtype=object)
z = np.load(ROOT / "tmp" / "oof_stylegram.npz"); D, y = z["D"], z["y"]; K = D.shape[1]
uniq = np.array(sorted(set(pid))); order = uniq[np.random.default_rng(42).permutation(len(uniq))]
fold_of = {g: i % 5 for i, g in enumerate(order)}; fold = np.array([fold_of[g] for g in pid])
def logp(d, temp=0.5):
    zz = d / temp; zz = zz - zz.max(1, keepdims=True); return zz - np.log(np.exp(zz).sum(1, keepdims=True))
L = logp(D)
def assign_groups(L, groups):
    out = L.argmax(1).copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if 1 < len(i) <= K:
            r, c = linear_sum_assignment(-L[i]); out[i[r]] = c
        elif len(i) > K:
            reps = int(np.ceil(len(i) / K)); r, c = linear_sum_assignment(-np.repeat(L[i], reps, axis=1)); out[i[r]] = c // reps
    return out
def clusters(ts, thresh, method="average"):
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(ts)
    Dm = np.clip(1 - (X @ X.T).toarray(), 0, None); np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), method=method), thresh, criterion='distance')
def pair_prf(g, p):
    same_c = g[:, None] == g[None, :]; same_p = p[:, None] == p[None, :]; off = ~np.eye(len(g), dtype=bool)
    tp = (same_c & same_p & off).sum(); return tp / max((same_c & off).sum(), 1), tp / max((same_p & off).sum(), 1)
print("plain BA", round(balanced_accuracy_score(y, D.argmax(1)), 4), "| oracle Hungarian", round(balanced_accuracy_score(y, assign_groups(L, pid)), 4))
for layout in ("full18", "40pct"):
    rng = np.random.default_rng(0)
    for thresh in (0.7, 0.75, 0.8, 0.85, 0.9):
        bas, plain, prs, nclu, big = [], [], [], [], []
        for k in range(5):
            va = np.where(fold == k)[0]
            if layout == "40pct":
                va = np.concatenate([rng.choice(va[pid[va] == p], max(1, int(round(0.4 * (pid[va] == p).sum()))), replace=False) for p in np.unique(pid[va])])
            g = clusters(texts[va], thresh); prs.append(pair_prf(g, pid[va])); sz = np.bincount(g)[1:]
            nclu.append(len(sz)); big.append((sz > K).sum())
            plain.append(balanced_accuracy_score(y[va], D[va].argmax(1))); bas.append(balanced_accuracy_score(y[va], assign_groups(L[va], g)))
        prs = np.array(prs)
        print(f"{layout} thresh {thresh}: pair P {prs[:,0].mean():.3f} R {prs[:,1].mean():.3f} | clusters/fold {np.mean(nclu):.0f} (true {len(np.unique(pid[fold==0]))}), >18: {np.mean(big):.1f} | plain {np.mean(plain):.4f} -> Hungarian {np.mean(bas):.4f}")
