#!/usr/bin/env python3
"""Fidelity and usefulness of the public extra datasets with the CPU stylegram SVC (prompt-grouped CV)."""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS, L2I, FAMILY as fam, K
from ntl.style import Stylegram
E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/claude-1000/-home-a-code-kaggle/147b173f-712b-4df3-a802-d662361b5f6d/scratchpad/ntl_external")
train = load_train(); y = train.y.values; pid = train.prompt_id.astype(str).values; texts = train.text.to_numpy(dtype=object)
fold = folds(pid)
T0 = time.time()

def load_sources():
    src = {}
    d = pd.read_csv(E / "seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv"); src["seyit_claude"] = d[["text", "model", "prompt_id"]]
    d = pd.read_csv(E / "prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages_clean.csv"); src["prvsiyan_clean"] = d[["text", "model", "prompt_id"]]
    d = pd.read_csv(E / "prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages.csv"); src["prvsiyan_all"] = d[["text", "model", "prompt_id"]]
    d = pd.read_csv(E / "prvsiyan_ntl-own-prompt-generations-v1/generated_v1.csv").rename(columns={"label": "model"}); src["prvsiyan_v1"] = d[["text", "model", "prompt_id"]]
    d = pd.read_csv(E / "dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv").rename(columns={"label": "model"}); src["dophuchung"] = d[["text", "model", "prompt_id"]]
    for k in src:
        src[k] = src[k].dropna(subset=["text", "model"]); src[k] = src[k][src[k].model.isin(LABELS)].reset_index(drop=True); src[k]["text"] = src[k].text.astype(str)
    return src
src = load_sources()

def fit_predict(ft, fy, pt, sw=None):
    return Stylegram().fit_decision(ft, fy, pt, sample_weight=sw)

def cv(extra_text=None, extra_y=None, w=0.5):
    Dm = np.zeros((len(y), K), dtype=np.float32)
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        if extra_text is None:
            Dm[va] = fit_predict(texts[tr], y[tr], texts[va])
        else:
            ytr = y[tr]; sw_tr = np.ones(len(tr)); sw_ex = np.zeros(len(extra_y))
            for c in np.unique(extra_y):
                sw_tr[ytr == c] = 1 - w; sw_ex[extra_y == c] = w * (ytr == c).sum() / (extra_y == c).sum()
            Dm[va] = fit_predict(np.concatenate([texts[tr], np.asarray(extra_text, dtype=object)]), np.concatenate([ytr, extra_y]), texts[va], np.concatenate([sw_tr, sw_ex]))
    p = Dm.argmax(1)
    perfam = {f: float(np.mean(p[fam[y] == f] == y[fam[y] == f])) for f in np.unique(fam)}
    return balanced_accuracy(y, p), perfam, Dm

if __name__ == "__main__":
    print({k: len(v) for k, v in src.items()})
    for k, df in src.items():
        dec = fit_predict(texts, y, df.text.values); p = dec.argmax(1); t = df.model.map(L2I).values
        ex = (p == t); fa = (fam[p] == fam[t])
        per = pd.DataFrame({"exact": ex, "family": fa, "model": df.model.values}).groupby("model")[["exact", "family"]].mean().round(2)
        print(f"\nFIDELITY {k}: n={len(df)} exact {ex.mean():.3f} family {fa.mean():.3f}  ({time.time()-T0:.0f}s)"); print(per.T.to_string())
    ba0, pf0, _ = cv(); print(f"\nCV baseline (train only): BA {ba0:.4f} | per family acc { {a: round(b, 3) for a, b in pf0.items()} }")
    combos = {"seyit_claude": ["seyit_claude"], "prvsiyan_clean": ["prvsiyan_clean"], "prvsiyan_all": ["prvsiyan_all"], "prvsiyan_v1": ["prvsiyan_v1"], "dophuchung": ["dophuchung"],
              "ALL": ["prvsiyan_all", "seyit_claude", "prvsiyan_v1", "dophuchung"]}
    for name, parts in combos.items():
        df = pd.concat([src[p] for p in parts], ignore_index=True)
        for w in (0.3, 0.5):
            ba, pf, _ = cv(df.text.values, df.model.map(L2I).values, w)
            print(f"CV + {name} w={w}: BA {ba:.4f} ({ba-ba0:+.4f}) | per family { {a: round(b, 3) for a, b in pf.items()} }  ({time.time()-T0:.0f}s)", flush=True)
