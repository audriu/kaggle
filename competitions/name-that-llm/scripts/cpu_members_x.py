#!/usr/bin/env python3
"""OOF for the extra-data stylegram member (stylegram_x): competition rows + public extra passages (class-balanced
weights, share w per class, Kimi excluded) -> oof/stylegram_x.npz. Extra files: --extra-dir (default: the session scratchpad copy)."""
import argparse, sys, time, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS, K, L2I
from ntl.style import Stylegram
ap = argparse.ArgumentParser(); ap.add_argument("--extra-dir", default="/tmp/claude-1000/-home-a-code-kaggle/147b173f-712b-4df3-a802-d662361b5f6d/scratchpad/ntl_external")
ap.add_argument("--w", type=float, default=0.5); ap.add_argument("--exclude", default="kimi-k3"); ap.add_argument("--cap", type=int, default=0, help="per-class cap on extra rows (0 = none)")
ap.add_argument("--name", default="stylegram_x"); ap.add_argument("--extra-csv", default="", help="single merged CSV (text, model) instead of --extra-dir files")
args = ap.parse_args()
E = Path(args.extra_dir); FILES = ["prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages.csv", "seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv",
                                   "prvsiyan_ntl-own-prompt-generations-v1/generated_v1.csv", "dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv"]
parts = [pd.read_csv(args.extra_csv)[["text", "model"]].dropna()] if args.extra_csv else [pd.read_csv(E / f).rename(columns={"label": "model"})[["text", "model"]].dropna() for f in FILES]
ex = pd.concat(parts, ignore_index=True); ex = ex[ex.model.isin(LABELS) & ~ex.model.isin(args.exclude.split(","))].reset_index(drop=True)
if args.cap: ex = ex.sample(frac=1, random_state=0).groupby("model").head(args.cap).reset_index(drop=True)
ex_text = ex.text.astype(str).to_numpy(dtype=object); ex_y = ex.model.map(L2I).values
print("extra rows", len(ex), ex.model.value_counts().to_dict(), flush=True)
tr = load_train(); y = tr.y.values; texts = tr.text.to_numpy(dtype=object); fold = folds(tr.prompt_id.values); t0 = time.time()
P = np.zeros((len(tr), K), dtype=np.float32)
for k in range(5):
    a, b = np.where(fold != k)[0], np.where(fold == k)[0]; ya = y[a]
    sw_tr = np.ones(len(a)); sw_ex = np.zeros(len(ex_y))
    for c in np.unique(ex_y):
        if (ya == c).sum() == 0: continue
        sw_tr[ya == c] = 1 - args.w; sw_ex[ex_y == c] = args.w * (ya == c).sum() / (ex_y == c).sum()
    keep = sw_ex > 0
    P[b] = Stylegram().fit_decision(np.concatenate([texts[a], ex_text[keep]]), np.concatenate([ya, ex_y[keep]]), texts[b], sample_weight=np.concatenate([sw_tr, sw_ex[keep]]))
    print(f"{args.name} fold {k} ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed(ROOT / "oof" / f"{args.name}.npz", ids=tr.id.values, scores=P, kind="margin", classes=np.array(LABELS), fold=fold)
print(f"{args.name}: OOF balanced acc {balanced_accuracy(y, P.argmax(1)):.4f}")
