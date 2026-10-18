#!/usr/bin/env python3
"""Does the public extra data leak train-prompt topics into the CV? For every train passage: max TF-IDF cosine to any extra
passage vs to train passages of other prompts; then per-prompt OOF gain of the extra-data members (stylegram_x vs stylegram,
q8lora_x1 vs q8lora_v1) against that similarity. Usage: python scripts/check_extra_overlap.py --extra-dir DIR"""
import argparse, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--extra-dir", required=True); args = ap.parse_args()
E = Path(args.extra_dir); FILES = {"prvsiyan": "prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages.csv", "seyit": "seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv",
                                   "prv_v1": "prvsiyan_ntl-own-prompt-generations-v1/generated_v1.csv", "dophuchung": "dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv"}
ex = pd.concat([pd.read_csv(E / f).rename(columns={"label": "model"})[["text", "model"]].dropna().assign(src=k) for k, f in FILES.items()], ignore_index=True)
tr = load_train(); y = np.asarray(tr.y.values); pid = tr.prompt_id.astype(str).to_numpy(dtype=object); fold = folds(pid); ids = tr.id.astype(str).values
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=300000).fit(pd.concat([tr.text, ex.text]))
Xt = vec.transform(tr.text); Xe = vec.transform(ex.text)
S_te = (Xt @ Xe.T).toarray(); S_tt = (Xt @ Xt.T).toarray()
same = pid[:, None] == pid[None, :]; S_tt[same] = -1
max_ex = S_te.max(1); arg_ex = S_te.argmax(1); max_other = S_tt.max(1)
print(f"train passages: max cosine to an extra passage: mean {max_ex.mean():.3f} median {np.median(max_ex):.3f} | to a train passage of another prompt: mean {max_other.mean():.3f} median {np.median(max_other):.3f}")
for t in [0.3, 0.4, 0.5, 0.6]: print(f"  passages with extra-cosine > {t}: {int((max_ex > t).sum())} / {len(tr)}  (other-prompt-cosine > {t}: {int((max_other > t).sum())})")
dfp = pd.DataFrame({"pid": pid, "max_ex": max_ex, "max_other": max_other, "src": ex.src.values[arg_ex]})
per = dfp.groupby("pid").agg(max_ex=("max_ex", "mean"), max_other=("max_other", "mean"), top_src=("src", lambda s: s.value_counts().index[0]))
print("\nper-prompt mean max-cosine to extra data: quantiles", per.max_ex.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict())
print("nearest extra source for the 20 most-similar prompts:", per.sort_values("max_ex", ascending=False).head(20).top_src.value_counts().to_dict())
# a few examples of the closest pairs
top = np.argsort(-max_ex)[:3]
for i in top: print(f"\n[cos {max_ex[i]:.2f}] TRAIN {pid[i]}: {tr.text.values[i][:160]!r}\n           EXTRA ({ex.src.values[arg_ex[i]]}): {ex.text.values[arg_ex[i]][:160]!r}")
def oof(n):
    z = np.load(ROOT / "oof" / f"{n}.npz", allow_pickle=True); return pd.DataFrame(z["scores"], index=z["ids"].astype(str)).reindex(ids).to_numpy().argmax(1)
for base, xm in [("stylegram", "stylegram_x"), ("q8lora_v1", "q8lora_x1")]:
    if not (ROOT / "oof" / f"{xm}.npz").exists(): continue
    pb, px = oof(base), oof(xm); g = pd.DataFrame({"pid": pid, "gain": (px == y).astype(float) - (pb == y).astype(float)}).groupby("pid").gain.mean()
    per["gain"] = g; r = spearmanr(per.max_ex, per.gain)
    terc = pd.qcut(per.max_ex, 3, labels=["low-sim", "mid-sim", "high-sim"])
    print(f"\n{xm} vs {base}: overall acc gain {per.gain.mean():+.4f} | spearman(gain, extra-similarity) {r.correlation:+.3f} (p {r.pvalue:.3f})")
    print("  per-prompt gain by similarity tercile:", per.groupby(terc, observed=True).gain.mean().round(4).to_dict())
    print("  BA on prompts in the low-sim tercile:", f"{base} {balanced_accuracy(y[np.isin(pid, per.index[terc == 'low-sim'])], pb[np.isin(pid, per.index[terc == 'low-sim'])]):.4f}",
          f"{xm} {balanced_accuracy(y[np.isin(pid, per.index[terc == 'low-sim'])], px[np.isin(pid, per.index[terc == 'low-sim'])]):.4f}")
