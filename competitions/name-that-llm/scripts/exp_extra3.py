#!/usr/bin/env python3
"""Extra-data variants for the style model: all sources minus Kimi; and Claude capped to 500 per class."""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_extra import src, LABELS, L2I, cv
ba0, pf0, _ = cv(); print(f"baseline BA {ba0:.4f}", flush=True)
allx = pd.concat([src[k] for k in ("prvsiyan_all", "seyit_claude", "prvsiyan_v1", "dophuchung")], ignore_index=True)
variants = {"all minus kimi": allx[allx.model != "kimi-k3"],
            "all minus kimi, claude capped 500/class": pd.concat([g.sample(min(len(g), 500), random_state=0) if m.startswith("claude") else g for m, g in allx[allx.model != "kimi-k3"].groupby("model")]),
            "all minus kimi, no seyit": pd.concat([src[k] for k in ("prvsiyan_all", "prvsiyan_v1", "dophuchung")], ignore_index=True).query("model != 'kimi-k3'")}
for name, df in variants.items():
    for w in (0.3, 0.5):
        t0 = time.time(); ba, pf, _ = cv(df.text.values, df.model.map(L2I).values, w)
        print(f"CV + {name} (n={len(df)}) w={w}: BA {ba:.4f} ({ba-ba0:+.4f}) | per family { {str(a): round(float(b), 3) for a, b in pf.items()} } ({time.time()-t0:.0f}s)", flush=True)
