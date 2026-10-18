#!/usr/bin/env python3
"""Out-of-fold predictions for the CPU members (stylegram SVC, stylometric LightGBM) -> oof/*.npz."""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS, K
from ntl.style import Stylegram, lgb_proba
OUT = ROOT / "oof"; OUT.mkdir(exist_ok=True)
tr = load_train(); y = tr.y.values; texts = tr.text.to_numpy(dtype=object); fold = folds(tr.prompt_id.values)
t0 = time.time()
for name in sys.argv[1:] or ["stylegram", "style_lgb"]:
    P = np.zeros((len(tr), K), dtype=np.float32)
    for k in range(5):
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]
        P[b] = Stylegram().fit_decision(texts[a], y[a], texts[b]) if name == "stylegram" else lgb_proba(texts[a], y[a], texts[b])
        print(f"{name} fold {k} ({time.time()-t0:.0f}s)", flush=True)
    kind = "margin" if name == "stylegram" else "prob"
    np.savez_compressed(OUT / f"{name}.npz", ids=tr.id.values, scores=P, kind=kind, classes=np.array(LABELS), fold=fold)
    print(f"{name}: OOF balanced acc {balanced_accuracy(y, P.argmax(1)):.4f}")
