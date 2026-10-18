#!/usr/bin/env python3
"""Fit log-linear blend weights on out-of-fold member predictions (oof/*.npz), report CV and layout simulations,
and write blend/weights.json for the submission kernel.

Member files: oof/<name>.npz with ids, scores (n, 18), kind ('prob' | 'margin' | 'logit'), classes (sorted labels).
Usage: python scripts/fit_blend.py [member ...]   (default: every file in oof/)
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, family_accuracy, LABELS, K
from ntl.blend import member_logp, fit_weights, combine
from ntl.sim import simulate
tr = load_train(); y = tr.y.values; fold = folds(tr.prompt_id.values); ids = tr.id.astype(str).values
names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "oof").glob("*.npz"))
LP, temps = [], {}
for n in names:
    z = np.load(ROOT / "oof" / f"{n}.npz", allow_pickle=True)
    assert list(z["classes"]) == LABELS, n
    S = pd.DataFrame(z["scores"], index=z["ids"].astype(str)).reindex(ids).to_numpy()
    assert np.isfinite(S).all(), f"{n}: missing rows"
    kind = str(z["kind"])
    best = (None, -1)
    for t in ([1.0] if kind == "prob" else [0.25, 0.35, 0.5, 0.75, 1.0, 1.5]):   # temperature chosen by OOF NLL
        lp = member_logp(S, kind, t); nll = -lp[np.arange(len(y)), y].mean()
        if best[0] is None or nll < best[1]: best = (t, nll)
    temps[n] = best[0]; lp = member_logp(S, kind, best[0]); LP.append(lp)
    print(f"{n:14s} kind {kind:6s} temp {best[0]:<5} OOF balanced acc {balanced_accuracy(y, lp.argmax(1)):.4f} family {family_accuracy(y, lp.argmax(1)):.4f}")
# nested check: fit weights on 4 folds, evaluate on the 5th
pred = np.zeros(len(y), dtype=int)
for k in range(5):
    a, b = fold != k, fold == k
    w = fit_weights([lp[a] for lp in LP], y[a]); pred[b] = combine([lp[b] for lp in LP], w).argmax(1)
print(f"\nnested blend (weights fit on other folds): balanced acc {balanced_accuracy(y, pred):.4f}")
w = fit_weights(LP, y); LPb = combine(LP, w)
print("weights:", {n: round(float(x), 3) for n, x in zip(names, w)}, f"| in-sample blend {balanced_accuracy(y, LPb.argmax(1)):.4f}")
for layout, pp in [("18 per prompt", 18), ("7 per prompt (public 40%)", 7), ("11 per prompt (private 60%)", 11), ("1-8 per prompt", (1, 8))]:
    r = simulate(LPb, y, tr.prompt_id.values, tr.text.values, per_prompt=pp, draws=3, thresh=0.9)
    print(f"  sim {layout:28s} plain {r['plain']:.4f} -> clustered Hungarian {r['hungarian']:.4f} (oracle groups {r['oracle']:.4f}; pair P {r['pair_precision']:.2f} R {r['pair_recall']:.2f})")
json.dump({"members": names, "weights": [float(x) for x in w], "temps": temps, "cv_nested": float(balanced_accuracy(y, pred))}, open(ROOT / "blend" / "weights.json", "w"), indent=1)
print("wrote blend/weights.json")
