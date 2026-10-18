#!/usr/bin/env python3
"""Merge the public extra passages used by the kernels into one CSV (text, model, prompt_id, src): the four sources already
in use (Oct 17) plus the Oct-18 batch that improved stylegram CV in scripts/exp_extra4.py. Dedup by normalised text.
Usage: python scripts/build_extra_v3.py --v2 <dir> --v3 <dir> --out data/extra_v3.csv
"""
import argparse, re, sys, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--v2", required=True); ap.add_argument("--v3", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); V2, V3 = Path(a.v2), Path(a.v3)
norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()[:400]
def take(df, src, pid=None):
    df = df.rename(columns={"label": "model"}); df = df[df.model.isin(LABELS)].dropna(subset=["text"]).copy(); df["text"] = df.text.astype(str)
    df["prompt_id"] = (df[pid].astype(str) if pid and pid in df else pd.Series(range(len(df)), index=df.index).map(lambda i: f"{src}_{i}"))
    df["src"] = src; return df[["text", "model", "prompt_id", "src"]]
parts = [
    take(pd.read_csv(V2 / "prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages.csv"), "prvsiyan", "prompt_id"),
    take(pd.read_csv(V2 / "seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv"), "seyit", "prompt_id"),
    take(pd.read_csv(V2 / "prvsiyan_ntl-own-prompt-generations-v1/generated_v1.csv"), "prvsiyan_v1", "prompt_id"),
    take(pd.read_csv(V2 / "dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv"), "dophuchung", "prompt_id"),
    take(pd.read_csv(V3 / "michaelsemenoff_ntl-own-extra-v1/ntl_own_extra_v1.csv"), "semenoff", "group"),
    take(pd.read_csv(V3 / "abdullah1afify_ntl-generations/ntl_generations.csv"), "afify", "prompt_id"),
    take(pd.read_csv(V3 / "kaplinskiy_ntl-gpt-proxy-passages/ntl_gpt_proxy_passages.csv"), "kaplinskiy", "prompt_id"),
    take(pd.read_csv(V3 / "keshavatearth_ntl-gpt-new-prompts-v1/accepted.csv"), "keshav", "prompt_id"),
    take(pd.read_csv(V3 / "haileycheng_name-that-llm-same-prompt-passages/passages.csv"), "hailey", "prompt_id"),
    take(pd.read_csv(V3 / "lucasboesen_ntl-free-proxy-generations/gen_passages.csv"), "lucas", "prompt_id"),
]
m = pd.read_csv(V3 / "abdelghanyrageh_ntl-extra-merged/ntl_extra_merged.csv")
parts += [take(m[m.src == "ours_proxy"], "rageh_proxy", "group"), take(m[m.src == "tyz"], "tyz", "group")]
df = pd.concat(parts, ignore_index=True)
seen = set(map(norm, load_train().text)); keep = []
for t in df.text:
    n = norm(t); keep.append(n not in seen); seen.add(n)
df = df[keep].reset_index(drop=True)
df.to_csv(a.out, index=False)
print(len(df), "rows ->", a.out); print(df.src.value_counts().to_string()); print(df.model.value_counts().to_string())
