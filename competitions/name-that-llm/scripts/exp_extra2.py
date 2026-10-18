#!/usr/bin/env python3
"""Why do GLM/Kimi extra passages hurt? Style markers train vs extra per class, then CV with a per-class selection."""
import re, sys, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_extra import src, train, LABELS, L2I, cv
def marks(t):
    w = max(len(t.split()), 1)
    return dict(words=w, emdash=t.count("—") / w * 100, semi=t.count(";") / w * 100, colon=t.count(":") / w * 100, curly=(t.count("’") + t.count("“") + t.count("”")) / w * 100,
                straight=(t.count("'") + t.count('"')) / w * 100, comma=t.count(",") / w * 100, sent_len=w / max(len(re.findall(r"[.!?](\s|$)", t)), 1), caps=sum(c.isupper() for c in t) / max(len(t), 1) * 100)
rows = []
for name, df in [("train", train[["text", "model"]])] + [(k, v) for k, v in src.items()]:
    for m, g in df.groupby("model"):
        if m in ("glm-5.3", "glm-5.3-flash", "kimi-k3", "claude-fable-5", "qwen3.8-max", "gpt-5.5"):
            M = pd.DataFrame([marks(t) for t in g.text.astype(str)]).mean(); rows.append(dict(source=name, model=m, n=len(g), **M.round(2)))
print(pd.DataFrame(rows).sort_values(["model", "source"]).to_string(index=False), flush=True)
ba0, pf0, _ = cv(); print(f"\nbaseline BA {ba0:.4f}", flush=True)
sel_classes = [l for l in LABELS if not l.startswith("glm") and l != "kimi-k3"]
parts = [src[k][src[k].model.isin(sel_classes)] for k in ("prvsiyan_all", "seyit_claude", "prvsiyan_v1", "dophuchung")]
df = pd.concat(parts, ignore_index=True); print("selected extra rows:", len(df), df.model.value_counts().to_dict())
for w in (0.3, 0.5):
    t0 = time.time(); ba, pf, _ = cv(df.text.values, df.model.map(L2I).values, w)
    print(f"CV + selected (no glm/kimi) w={w}: BA {ba:.4f} ({ba-ba0:+.4f}) | per family { {a: round(b, 3) for a, b in pf.items()} } ({time.time()-t0:.0f}s)", flush=True)
