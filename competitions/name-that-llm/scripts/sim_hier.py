#!/usr/bin/env python3
"""Hierarchical blend simulation: global blend gives P(family); per-family blend weights (fit on that family's rows only, within-
family softmax) give P(model | family). Nested over prompt folds. Usage: python scripts/sim_hier.py [--l2 1e-3]"""
import argparse, json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS, K, log_softmax
from ntl.blend import member_logp, fit_weights, combine
ap = argparse.ArgumentParser(); ap.add_argument("--l2", type=float, default=1e-3); args = ap.parse_args()
tr = load_train(); y = np.asarray(tr.y.values); fold = folds(tr.prompt_id.astype(str).to_numpy(dtype=object)); ids = tr.id.astype(str).values
W = json.load(open(ROOT / "blend" / "weights.json")); members = W["members"]
LP = []
for n in members:
    z = np.load(ROOT / "oof" / f"{n}.npz", allow_pickle=True); S = pd.DataFrame(z["scores"], index=z["ids"].astype(str)).reindex(ids).to_numpy(); LP.append(member_logp(S, str(z["kind"]), W["temps"].get(n, 1.0)))
fam_of = lambda s: "gpt" if s.startswith("gpt") else s.split("-")[0]
fams = sorted(set(fam_of(l) for l in LABELS)); FAM = {f: np.array([i for i, l in enumerate(LABELS) if fam_of(l) == f]) for f in fams}
yf = np.array([fams.index(fam_of(LABELS[c])) for c in y])
def fit_within(LPs, yy, cls):   # weights minimising within-family NLL (logits restricted to cls), class-balanced
    S = np.stack([lp[:, cls] for lp in LPs], 0); loc = np.searchsorted(cls, yy); cw = len(yy) / (len(cls) * np.bincount(loc, minlength=len(cls))[loc])
    def f(w):
        z = np.tensordot(w, S, 1); z = z - z.max(1, keepdims=True); lse = np.log(np.exp(z).sum(1)); return (cw * (lse - z[np.arange(len(yy)), loc])).sum() / cw.sum() + args.l2 * (w ** 2).sum()
    return minimize(f, np.full(len(LPs), 1.0 / len(LPs)), method="L-BFGS-B", bounds=[(0, None)] * len(LPs)).x
pred_flat = np.zeros(len(y), int); pred_hier = np.zeros(len(y), int); pred_hier2 = np.zeros(len(y), int)
for k in range(5):
    a, b = fold != k, fold == k
    w = fit_weights([lp[a] for lp in LP], y[a]); Lb = combine([lp[b] for lp in LP], w); pred_flat[b] = Lb.argmax(1)
    Pb = np.exp(Lb); Lh = np.full_like(Lb, -50.0); Lh2 = Lh.copy()
    for f in fams:
        cls = FAM[f]; rows = a & np.isin(y, cls)
        wf = fit_within([lp[rows] for lp in LP], y[rows], cls)
        zin = log_softmax(np.tensordot(wf, np.stack([lp[b][:, cls] for lp in LP], 0), 1))   # log P(model | family), within-family weights
        lf = np.log(Pb[:, cls].sum(1) + 1e-12)   # log P(family) from the global blend
        Lh[:, cls] = lf[:, None] + zin
        Lh2[:, cls] = lf[:, None] + 0.5 * zin + 0.5 * log_softmax(Lb[:, cls])   # half-way: average within-family evidence
    pred_hier[b] = Lh.argmax(1); pred_hier2[b] = Lh2.argmax(1)
fa = lambda p: float((np.array([fams.index(fam_of(LABELS[c])) for c in p]) == yf).mean())
print(f"flat nested blend      BA {balanced_accuracy(y, pred_flat):.4f}  family acc {fa(pred_flat):.4f}")
print(f"hierarchical (within)  BA {balanced_accuracy(y, pred_hier):.4f}  family acc {fa(pred_hier):.4f}")
print(f"hierarchical (half)    BA {balanced_accuracy(y, pred_hier2):.4f}  family acc {fa(pred_hier2):.4f}")
for f in fams:
    m = np.isin(y, FAM[f]); print(f"  {f:8s} within-family acc given true family: flat {(pred_flat[m] == y[m]).mean():.3f} hier {(pred_hier[m] == y[m]).mean():.3f}")
