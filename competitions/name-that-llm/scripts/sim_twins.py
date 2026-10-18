"""Pick TWIN_W (cluster-centred twin-head evidence) by simulating hidden-test layouts on held-out train prompts.

Inputs: blend/weights.json + oof/<member>.npz (plain blend), twin sources:
  frozen heads  <frozen>/feats_<key>.npz   (X fp16 aligned to its ids, fold, c_coef_k / c_intercept_k = per-fold centred head)
  LoRA          <lora>/out/job_<k>/{valid_features.npz, readouts.npz}   (held-out z of fold k, centred readout of job k)
Per draw: k passages per prompt -> content TF-IDF clusters (cut 0.9) -> eligible rows (2 <= cluster size <= 18) ->
twin logp = log_softmax((X - cluster mean) @ coef_fold.T + b_fold) -> L = plain + sum TWIN_W * twin (eligible rows) ->
global Sinkhorn 0.5 -> per-cluster Sinkhorn 0.5 -> Hungarian -> balanced accuracy. Coordinate ascent on a grid.
Usage: python scripts/sim_twins.py [--frozen tmp/out_frozen2] [--lora tmp/out_lora] [--draws 3] [--write]
"""
import argparse, glob, json, os, pathlib, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ntl.common import ROOT, K, load_train, folds, balanced_accuracy, log_softmax
from ntl.blend import member_logp
from ntl.group import tfidf_distance, cluster, assign, sinkhorn

ap = argparse.ArgumentParser()
ap.add_argument("--frozen", default="tmp/out_frozen2"); ap.add_argument("--lora", default="tmp/out_lora")
ap.add_argument("--draws", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fit-layouts", default="1-2,1-5,3-8,7"); ap.add_argument("--report-layouts", default="11,18")
ap.add_argument("--grid", default="0,0.1,0.2,0.3,0.45,0.6,0.8,1.0"); ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--thresh", type=float, default=0.9); ap.add_argument("--write", action="store_true", help="write blend/twins.json")
args = ap.parse_args()
T0 = time.time()

def parse_layouts(s):
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t: continue
        out.append((int(t.split("-")[0]), int(t.split("-")[1])) if "-" in t else int(t))
    return out

tr = load_train(); ids = tr.id.astype(str).values; y = tr.y.values; pid = tr.prompt_id.astype(str).values
texts = tr.text.astype(str).to_numpy(dtype=object); fold = folds(tr.prompt_id.values)
# ---- plain blend (OOF)
w = json.load(open(ROOT / "blend" / "weights.json")); LP = np.zeros((len(ids), K))
for m, wt in zip(w["members"], w["weights"]):
    if wt <= 0: continue
    z = np.load(ROOT / "oof" / f"{m}.npz", allow_pickle=True)
    S = pd.DataFrame(z["scores"], index=z["ids"].astype(str)).reindex(ids).to_numpy()
    LP += wt * member_logp(S, str(z["kind"]), w["temps"].get(m, 1.0))
LP = log_softmax(LP); print(f"plain blend OOF balanced acc {balanced_accuracy(y, LP.argmax(1)):.4f}  members {[m for m, wt in zip(w['members'], w['weights']) if wt > 0]}")
# ---- twin sources
twins = {}
for f in sorted(glob.glob(f"{args.frozen}/feats_*.npz")):
    z = np.load(f, allow_pickle=True); key = os.path.basename(f)[len("feats_"):-4]
    pos = pd.Series(np.arange(len(z["ids"])), index=z["ids"].astype(str)).reindex(ids).to_numpy()
    assert not np.isnan(pos.astype(float)).any(), key
    twins[key] = (z["X"][pos.astype(int)], {k: (z[f"c_coef_{k}"].astype(np.float32), z[f"c_intercept_{k}"].astype(np.float32)) for k in range(5)})
jobs = sorted(glob.glob(f"{args.lora}/out/job_[0-4]"))
if len(jobs) == 5:
    r0 = np.load(f"{jobs[0]}/readouts.npz"); X = np.zeros((len(ids), r0["center_coef"].shape[1]), np.float16); heads = {}; seen = np.zeros(len(ids), bool)
    for j in jobs:
        k = int(os.path.basename(j)[len("job_"):]); v = np.load(f"{j}/valid_features.npz", allow_pickle=True); r = np.load(f"{j}/readouts.npz")
        pos = pd.Series(np.arange(len(ids)), index=ids).reindex(v["ids"].astype(str)).to_numpy().astype(int)
        X[pos] = v["features"]; seen[pos] = True; heads[k] = (r["center_coef"].astype(np.float32), r["center_intercept"].astype(np.float32))
    assert seen.all(); twins["q8lora_v1"] = (X, heads)
print("twin sources:", {k: v[0].shape for k, v in twins.items()}, f"({time.time()-T0:.0f}s)")
assert twins, "no twin sources found"

# ---- precompute draws: rows, clusters, eligible mask, plain L, twin logps
rng = np.random.default_rng(args.seed)
fit_layouts, rep_layouts = parse_layouts(args.fit_layouts), parse_layouts(args.report_layouts)
draws = {}
for lay in fit_layouts + rep_layouts:
    for d in range(args.draws):
        keep = []
        for p in np.unique(pid):
            rows = np.where(pid == p)[0]; k = lay if isinstance(lay, int) else int(rng.integers(lay[0], lay[1] + 1))
            keep.append(rng.choice(rows, min(k, len(rows)), replace=False))
        keep = np.concatenate(keep); g = cluster(tfidf_distance(texts[keep]), args.thresh); sz = np.bincount(g)[g]
        elig = np.where((sz >= 2) & (sz <= 18))[0]; tl = {}
        for key, (X, heads) in twins.items():
            Xk = X[keep].astype(np.float32); Xc = Xk.copy()
            for c in np.unique(g[elig]):
                i = np.where(g == c)[0]; Xc[i] -= Xc[i].mean(0)
            out = np.zeros((len(keep), K), np.float32); fk = fold[keep]
            for k, (coef, b) in heads.items():
                i = np.where(fk == k)[0]
                if len(i): out[i] = log_softmax(Xc[i] @ coef.T + b)
            tl[key] = out
        draws[(str(lay), d)] = dict(keep=keep, g=g, elig=elig, L=LP[keep], twins=tl, y=y[keep])
print(f"prepared {len(draws)} draws ({time.time()-T0:.0f}s)")

def decode(L, g):
    P = sinkhorn(np.exp(L), None, 0.5); P = sinkhorn(P, g, 0.5)
    return assign(np.log(np.clip(P, 1e-12, 1)), g)

def score(tw, layouts):
    res = {}
    for lay in layouts:
        accs = []
        for d in range(args.draws):
            D = draws[(str(lay), d)]; L = D["L"].copy()
            for key, t in tw.items():
                if t > 0: L[D["elig"]] += t * D["twins"][key][D["elig"]]
            accs.append(balanced_accuracy(D["y"], decode(log_softmax(L), D["g"])))
        res[str(lay)] = float(np.mean(accs))
    return res

def objective(tw): return float(np.mean(list(score(tw, fit_layouts).values())))

grid = [float(x) for x in args.grid.split(",")]; keys = sorted(twins)
tw = {k: 0.0 for k in keys}; base = objective(tw); print(f"no twins: fit-layout mean {base:.4f}  {score(tw, fit_layouts + rep_layouts)}")
public = {k: v for k, v in {"q3_8b_L7_22": 0.33, "q25_7b_L6_20": 0.21, "debv3l_L24_6": 0.25, "q8lora_v1": 0.15}.items() if k in tw}
if public: print(f"public-recipe weights {public}: fit-layout mean {objective(public):.4f}  {score(public, fit_layouts + rep_layouts)}")
best = base
for rnd in range(args.rounds):
    for key in keys:
        cand = []
        for v in grid:
            t = dict(tw); t[key] = v; cand.append((objective(t), v))
        obj, v = max(cand); 
        if obj >= best: best, tw[key] = obj, v
        print(f"round {rnd} {key:16s} best {tw[key]:.2f} -> {best:.4f}   grid {[round(c[0], 4) for c in cand]} ({time.time()-T0:.0f}s)", flush=True)
final = score(tw, fit_layouts + rep_layouts)
print(f"\nTWIN_W = {json.dumps({k: v for k, v in tw.items()})}\nfit-layout mean {best:.4f} (no twins {base:.4f})\nper layout: {final}")
if args.write:
    json.dump(dict(twin_w=tw, fit_layouts=args.fit_layouts, objective=best, no_twins=base, per_layout=final, grid=grid, draws=args.draws), open(ROOT / "blend" / "twins.json", "w"), indent=1)
    print("wrote blend/twins.json")
