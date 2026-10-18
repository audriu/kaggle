#!/usr/bin/env python3
"""Simulate test-time pseudo-labelling of the frozen heads.

Each prompt fold plays the hidden test set: the blend (installed weights, OOF member predictions) labels the held-out rows,
confident rows are added to the training side with their pseudo-labels, every frozen head (StandardScaler + LR, as in the
training kernel) is refitted, and the held-out rows are re-blended. Reports balanced accuracy pooled over the 5 folds.

Usage: python scripts/sim_pseudo.py --frozen tmp/out_frozen2 [--quantiles 0,0.5,0.75] [--weights 1.0] [--rounds 1] [--sinkhorn 0.5]
"""
import argparse, json, sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ntl.common import load_train, folds, balanced_accuracy, LABELS, K, log_softmax
from ntl.blend import member_logp, combine

ap = argparse.ArgumentParser()
ap.add_argument("--frozen", default="tmp/out_frozen2"); ap.add_argument("--quantiles", default="0,0.5,0.75"); ap.add_argument("--weights", default="1.0")
ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--crossfit", type=int, default=2); ap.add_argument("--sinkhorn", type=float, default=0.5); ap.add_argument("--min-weight", type=float, default=0.01)
args = ap.parse_args()
C_OF = {"q3_8b_L7_22": 0.0025, "q25_7b_L6_20": 0.0025, "debv3l_L24_6": 0.005, "desk_L23ms": 0.005, "electl_L21_24s": 0.005}
T0 = time.time()
tr = load_train(); y = tr.y.values; fold = folds(tr.prompt_id.values); ids = tr.id.astype(str).values
W = json.load(open(ROOT / "blend" / "weights.json")); members, weights, temps = W["members"], np.array(W["weights"]), W["temps"]
LP = {}
for n in members:
    z = np.load(ROOT / "oof" / f"{n}.npz", allow_pickle=True); S = pd.DataFrame(z["scores"], index=z["ids"].astype(str)).reindex(ids).to_numpy()
    LP[n] = member_logp(S, str(z["kind"]), temps.get(n, 1.0))
FE = {}
for n in members:
    f = ROOT / args.frozen / f"feats_{n}.npz"
    if f.exists() and weights[members.index(n)] > args.min_weight and n in C_OF:
        z = np.load(f, allow_pickle=True); assert (z["ids"].astype(str) == ids).all(); FE[n] = z["X"].astype(np.float32)
print("refit members:", list(FE), "| blend weights", {n: round(float(w), 3) for n, w in zip(members, weights)}, flush=True)
blend = lambda lp, rows: combine([lp[n][rows] for n in members], weights)
def sinkhorn(L, alpha, it=50):
    P = np.exp(L)
    if alpha <= 0: return P
    target = (1 - alpha) * P.sum(0) + alpha * np.full(K, len(P) / K)
    for _ in range(it):
        P = P * (target / (P.sum(0) + 1e-12))[None, :]; P = P / (P.sum(1, keepdims=True) + 1e-12)
    return P
base_pred = np.zeros(len(y), int); res = {}
for k in range(5):
    a, b = np.where(fold != k)[0], np.where(fold == k)[0]
    P0 = sinkhorn(blend(LP, b), args.sinkhorn); base_pred[b] = P0.argmax(1)
    for q in [float(x) for x in args.quantiles.split(",")]:
        for wp in [float(x) for x in args.weights.split(",")]:
            lp = {n: v.copy() for n, v in LP.items()}; P = P0
            for r in range(args.rounds):
                conf = P.max(1); pl = P.argmax(1); sel = conf >= np.quantile(conf, q)
                prec = float((pl[sel] == y[b][sel]).mean())
                half = np.random.RandomState(k).randint(0, args.crossfit, len(b))   # cross-fit: pseudo rows of one half never train the head that scores them
                for n, X in FE.items():
                    for h in range(args.crossfit):
                        fit_rows = sel & (half != h); score_rows = half == h
                        Xa = np.concatenate([X[a], X[b][fit_rows]]); ya = np.concatenate([y[a], pl[fit_rows]]); sw = np.concatenate([np.ones(len(a)), np.full(fit_rows.sum(), wp)])
                        sc = StandardScaler().fit(Xa); clf = LogisticRegression(C=C_OF[n], max_iter=3000).fit(sc.transform(Xa), ya, sample_weight=sw)
                        pr = np.full((score_rows.sum(), K), 1e-9); pr[:, clf.classes_] = clf.predict_proba(sc.transform(X[b][score_rows])); lp[n][b[score_rows]] = np.log(np.clip(pr, 1e-9, 1))
                P = sinkhorn(blend(lp, b), args.sinkhorn)
                key = (q, wp, r + 1); res.setdefault(key, np.zeros(len(y), int))[b] = P.argmax(1)
                print(f"fold {k} q {q} w {wp} round {r+1}: pseudo rows {int(sel.sum())} precision {prec:.3f} | held-out BA {balanced_accuracy(y[b], P0.argmax(1)):.4f} -> {balanced_accuracy(y[b], P.argmax(1)):.4f} ({time.time()-T0:.0f}s)", flush=True)
print(f"\nbaseline (blend + Sinkhorn {args.sinkhorn}) BA {balanced_accuracy(y, base_pred):.4f}")
for key, pred in sorted(res.items()):
    print(f"quantile {key[0]} weight {key[1]} round {key[2]}: BA {balanced_accuracy(y, pred):.4f} ({balanced_accuracy(y, pred) - balanced_accuracy(y, base_pred):+.4f})")
