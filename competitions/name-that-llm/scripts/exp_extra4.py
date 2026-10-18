#!/usr/bin/env python3
"""Incremental value of the October-18 batch of public extra datasets, on top of the extras already used by cpu_members_x.

For each new source: dedup (normalised text) against train + current extras, fidelity of a train-only stylegram on it,
then prompt-grouped CV of stylegram trained on train + current extras (+ this source) with the cpu_members_x weighting.
Usage: python scripts/exp_extra4.py --v2 <dir> --v3 <dir> [--w 0.5] [--only src1,src2]
"""
import argparse, re, sys, time, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS, L2I, FAMILY as fam, K
from ntl.style import Stylegram
ap = argparse.ArgumentParser(); ap.add_argument("--v2", required=True); ap.add_argument("--v3", required=True)
ap.add_argument("--w", type=float, default=0.5); ap.add_argument("--only", default=""); ap.add_argument("--exclude", default="kimi-k3")
ap.add_argument("--no-baseline", action="store_true"); ap.add_argument("--combo", action="append", default=[], help="name=src1+src2+...")
a = ap.parse_args()
V2, V3 = Path(a.v2), Path(a.v3); T0 = time.time()
train = load_train(); y = train.y.values; texts = train.text.to_numpy(dtype=object); fold = folds(train.prompt_id.values)
norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()[:400]

def clean(df, src):
    df = df.rename(columns={"label": "model"})[["text", "model"]].dropna(); df = df[df.model.isin(LABELS)].copy()
    df["text"] = df.text.astype(str); df["src"] = src; return df.reset_index(drop=True)
cur = pd.concat([clean(pd.read_csv(V2 / f), f.split("/")[0]) for f in [
    "prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages.csv", "seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv",
    "prvsiyan_ntl-own-prompt-generations-v1/generated_v1.csv", "dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv"]], ignore_index=True)
seen = set(map(norm, texts)) | set(map(norm, cur.text))
new = {}
m = clean(pd.read_csv(V3 / "abdelghanyrageh_ntl-extra-merged/ntl_extra_merged.csv"), "merged")
mraw = pd.read_csv(V3 / "abdelghanyrageh_ntl-extra-merged/ntl_extra_merged.csv")
mraw = mraw[mraw.model.isin(LABELS)].reset_index(drop=True)
for s in ["oleg", "speechmap", "tyz", "ours_proxy", "afify", "keshav"]:
    new[f"merged_{s}"] = clean(mraw[mraw.src == s], f"merged_{s}")
new["merged_oleg_clean"] = clean(mraw[(mraw.src == "oleg") & (mraw["clean"] == True)], "merged_oleg_clean")
mega = pd.read_parquet(V3 / "sunny114514_ntl-mega-corpus/ntl_mega_corpus.parquet"); mega = mega[mega.model.isin(LABELS)]
new["mega_bullshit"] = clean(mega[mega.source.str.contains("bullshit")], "mega_bullshit")
new["mega_hf_traces"] = clean(mega[mega.source.str.contains("angrygiraffe|Crownelius|greghavens|Met4physics")], "mega_hf_traces")
new["mega_axelst"] = clean(mega[mega.source.str.contains("axelst")], "mega_axelst")
new["semenoff"] = clean(pd.read_csv(V3 / "michaelsemenoff_ntl-own-extra-v1/ntl_own_extra_v1.csv"), "semenoff")
sem = pd.read_csv(V3 / "michaelsemenoff_ntl-own-extra-v1/ntl_own_extra_v1.csv")
for s in sem.source.unique(): new[f"semenoff_{s}"] = clean(sem[sem.source == s], f"semenoff_{s}")
new["hailey"] = clean(pd.read_csv(V3 / "haileycheng_name-that-llm-same-prompt-passages/passages.csv"), "hailey")
new["kaplinskiy"] = clean(pd.read_csv(V3 / "kaplinskiy_ntl-gpt-proxy-passages/ntl_gpt_proxy_passages.csv"), "kaplinskiy")
new["lucas"] = clean(pd.read_csv(V3 / "lucasboesen_ntl-free-proxy-generations/gen_passages.csv"), "lucas")
new["afify"] = clean(pd.read_csv(V3 / "abdullah1afify_ntl-generations/ntl_generations.csv"), "afify")
new["keshav"] = clean(pd.read_csv(V3 / "keshavatearth_ntl-gpt-new-prompts-v1/accepted.csv"), "keshav")
for k in list(new):
    df = new[k]; n0 = len(df); df = df[~df.text.map(norm).isin(seen)].drop_duplicates("text").reset_index(drop=True); new[k] = df
    print(f"{k:22s} rows {n0:6d} -> after dedup vs train+current {len(df):6d}  classes {df.model.nunique():2d}  median chars {int(df.text.str.len().median()) if len(df) else 0}", flush=True)
print(f"current extras {len(cur)}; w={a.w}  ({time.time()-T0:.0f}s)", flush=True)
excl = set(a.exclude.split(","))

def fidelity(df):
    dec = Stylegram().fit_decision(texts, y, df.text.values); p = dec.argmax(1); t = df.model.map(L2I).values
    per = pd.DataFrame({"exact": p == t, "family": fam[p] == fam[t], "model": df.model.values}).groupby("model")[["exact", "family"]].mean().round(2)
    return float((p == t).mean()), float((fam[p] == fam[t]).mean()), per

def cv(ex):
    ex = ex[~ex.model.isin(excl)]; ex_text = ex.text.to_numpy(dtype=object); ex_y = ex.model.map(L2I).values
    P = np.zeros((len(y), K), dtype=np.float32)
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]; ya = y[tr]
        sw_tr = np.ones(len(tr)); sw_ex = np.zeros(len(ex_y))
        for c in np.unique(ex_y): sw_tr[ya == c] = 1 - a.w; sw_ex[ex_y == c] = a.w * (ya == c).sum() / (ex_y == c).sum()
        P[va] = Stylegram().fit_decision(np.concatenate([texts[tr], ex_text]), np.concatenate([ya, ex_y]), texts[va], sample_weight=np.concatenate([sw_tr, sw_ex]))
    p = P.argmax(1); perc = {LABELS[c]: round(float((p[y == c] == c).mean()), 2) for c in range(K)}
    return balanced_accuracy(y, p), perc

for c in a.combo:
    nm, parts = c.split("="); new[nm] = pd.concat([new[p] for p in parts.split("+")], ignore_index=True).drop_duplicates("text").reset_index(drop=True)
only = [s for s in a.only.split(",") if s] or list(new)
for k in only:
    df = new[k]
    if len(df) < 30: print(f"skip {k} (n={len(df)})"); continue
    e, f, per = fidelity(df); print(f"\nFIDELITY {k}: n={len(df)} exact {e:.3f} family {f:.3f}  ({time.time()-T0:.0f}s)"); print(per.T.to_string(), flush=True)
if not a.no_baseline:
    ba0, pc0 = cv(cur); print(f"\nCV current extras: BA {ba0:.4f} ({time.time()-T0:.0f}s)\n  per class {pc0}", flush=True)
else: ba0, pc0 = 0.4134, {}
for k in only:
    df = new[k]
    if len(df) < 30: continue
    ba, pc = cv(pd.concat([cur, df], ignore_index=True))
    diff = {c: round(pc[c] - pc0.get(c, np.nan), 2) for c in pc if abs(pc[c] - pc0.get(c, pc[c])) >= 0.03}
    print(f"CV current + {k:20s} ({len(df):5d} rows): BA {ba:.4f} ({ba-ba0:+.4f})  changed classes {diff}  ({time.time()-T0:.0f}s)", flush=True)
