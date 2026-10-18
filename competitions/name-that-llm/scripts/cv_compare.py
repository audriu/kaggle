#!/usr/bin/env python3
"""Compare blend member sets by repeated prompt-grouped nested CV (weights fit on 4 folds, scored on the 5th).

Usage: python scripts/cv_compare.py --seeds 10 --set base=m1,m2,... --set plus=m1,m2,...,mN
Member OOF files come from oof/<name>.npz (all rows are out-of-fold, so any prompt-grouped split is valid for the
blend-weight stage). Reports per-set mean balanced accuracy and the paired difference vs the first set.
"""
import argparse, sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS
from ntl.blend import member_logp, fit_weights, combine
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--set", action="append", required=True)
a = ap.parse_args()
sets = [(s.split("=")[0], s.split("=")[1].split(",")) for s in a.set]
tr = load_train(); y = tr.y.values; ids = tr.id.astype(str).values
TEMPS = {}; LP = {}
for n in sorted({m for _, ms in sets for m in ms}):
    z = np.load(ROOT / "oof" / f"{n}.npz", allow_pickle=True); assert list(z["classes"]) == LABELS, n
    S = pd.DataFrame(z["scores"], index=z["ids"].astype(str)).reindex(ids).to_numpy(); kind = str(z["kind"])
    best = None
    for t in ([1.0] if kind == "prob" else [0.25, 0.35, 0.5, 0.75, 1.0, 1.5]):
        lp = member_logp(S, kind, t); nll = -lp[np.arange(len(y)), y].mean()
        if best is None or nll < best[1]: best = (t, nll)
    LP[n] = member_logp(S, kind, best[0]); TEMPS[n] = best[0]
res = {name: [] for name, _ in sets}
for seed in range(a.seeds):
    fold = folds(tr.prompt_id.values, seed=seed)
    for name, ms in sets:
        pred = np.zeros(len(y), dtype=int)
        for k in range(5):
            tr_, te_ = fold != k, fold == k
            w = fit_weights([LP[m][tr_] for m in ms], y[tr_]); pred[te_] = combine([LP[m][te_] for m in ms], w).argmax(1)
        res[name].append(balanced_accuracy(y, pred))
    print(f"seed {seed}: " + "  ".join(f"{name} {res[name][-1]:.4f}" for name, _ in sets), flush=True)
base = np.array(res[sets[0][0]])
print()
for name, ms in sets:
    r = np.array(res[name]); d = r - base
    print(f"{name:10s} mean {r.mean():.4f} sd {r.std(ddof=1):.4f} | vs {sets[0][0]}: {d.mean():+.4f} (se {d.std(ddof=1)/np.sqrt(len(d)):.4f}) members {len(ms)}")
