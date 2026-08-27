"""GPU value-net trainer: (state features at hour 0, day) -> final money.

Regresses the shards train/collect.py wrote (sim/features.extract vectors,
labelled with the player's final money) with a small MLP. The eventual consumer is
inference-time search: instead of rolling a candidate plan out 720 steps, roll to
the next day boundary and let V(state) price the rest of the season -- so what
matters is not just overall fit but fit AS A FUNCTION OF DAY: day-0 states are
nearly indistinguishable (everyone starts with $3000 and an empty NW quadrant, so
the best possible day-0 prediction is close to the mean), while by day ~29 money
nearly equals the label and the net must track it tightly. The report therefore
breaks MAE/R^2 into 5-day buckets and compares each against the trivial
predict-train-mean baseline.

Correlated-rows trap: the 60 rows of one episode share a label and most of a
trajectory, so a row-wise split would leak ~29 sibling rows of every "held-out" row
into training and report fantasy numbers. The split here is BY EPISODE ID, derived
deterministically from --split-seed so a resumed run keeps the exact same split.

Residual head: the net regresses delta = final_money - current_money and V(state)
= money + net(x). Measured on the first 240-episode set: at day 29 |money - final|
averages $577 and delta has sd 999 while the raw target has sd 4166 -- regressing
the raw label makes the net spend its whole budget re-learning the identity
"final ~ money" and it still missed day 29 by ~$1100; the delta head gets the
identity for free and only models future earnings. Metrics below are always in
dollars of FINAL money, so runs before/after this change compare directly.

Mechanics: inputs and delta standardised from TRAIN-split stats (floored at 1e-6);
2x256 ReLU MLP, Adam, MSE on the standardised delta; metrics reported in dollars.
Trains on CUDA when available (this dataset is tiny for a 3060 Ti -- whole splits
live on-device, minibatches are index_selects). Checkpoints every epoch to
<outdir>/last.pt and keeps the best-val-MAE model (plus scaler + feature names,
everything inference needs) in <outdir>/best.pt; both writes are tmp+os.replace
atomic and the script resumes from last.pt, so it is re-runnable mid-training.

Usage:
  python train/value_net.py                       # train/data -> train/runs/vnet
  python train/value_net.py --epochs 300 --fresh
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DAY_BUCKETS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29)]


def load_shards(data_dir):
    files = sorted(glob.glob(str(Path(data_dir) / "shard-*.npz")))
    if not files:
        sys.exit(f"no shards under {data_dir}; run train/collect.py first")
    Xs, ys, ds, es, advs, names = [], [], [], [], [], None
    for f in files:
        z = np.load(f)
        n = list(z["feature_names"])
        if names is None:
            names = n
        elif n != names:
            sys.exit(f"{f}: feature_names differ from {files[0]} -- "
                     f"stale shards from an older sim/features.py; delete and recollect")
        Xs.append(z["features"])
        ys.append(z["targets"])
        ds.append(z["day"])
        es.append(z["episode"])
        # adv tag (collect.py --mix v2): 2 crasher, 1 mirror, 0 quiet; -1 = shard
        # predates the tag.
        advs.append(z["adv"] if "adv" in z.files
                    else np.full(len(z["targets"]), -1, dtype=np.int8))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    day = np.concatenate(ds).astype(np.int64)
    ep = np.concatenate(es).astype(np.int64)
    adv = np.concatenate(advs).astype(np.int64)
    return X, y, day, ep, adv, [str(s) for s in names], len(files)


def split_by_episode(ep, val_frac, seed):
    """Boolean val mask over rows; every row of an episode lands on one side."""
    uniq = np.unique(ep)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(val_frac * len(uniq))))
    val_ids = set(uniq[perm[:n_val]].tolist())
    return np.isin(ep, list(val_ids)), len(uniq) - n_val, n_val


class ValueNet(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _mae_r2(y_true, y_pred):
    mae = float(np.abs(y_true - y_pred).mean())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    sse = float(((y_true - y_pred) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return mae, r2


def bucket_metrics(y_true, y_pred, y_base, day):
    """Overall + per-day-bucket MAE/R^2 for the net and the mean baseline."""
    rows = []
    masks = [("overall", np.ones_like(day, dtype=bool))]
    masks += [(f"day {lo:2d}-{hi:2d}", (day >= lo) & (day <= hi)) for lo, hi in DAY_BUCKETS]
    # extra diagnostic: the truncation floor -- at day 29 money nearly equals the
    # label (collect-time check: mean |money - final| ~ $266), so this row shows
    # how close the net gets to that irreducible bound.
    masks.append(("day 29", day == 29))
    for label, m in masks:
        if not m.any():
            continue
        net_mae, net_r2 = _mae_r2(y_true[m], y_pred[m])
        base_mae, base_r2 = _mae_r2(y_true[m], y_base[m])
        rows.append({"bucket": label, "n": int(m.sum()),
                     "net_mae": net_mae, "net_r2": net_r2,
                     "base_mae": base_mae, "base_r2": base_r2})
    return rows


def atomic_save(obj, path):
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Train the value net on collected shards.")
    ap.add_argument("--data", default=str(ROOT / "train" / "data"))
    ap.add_argument("--outdir", default=str(ROOT / "train" / "runs" / "vnet"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="Adam weight decay; without it val MAE bottoms out within "
                         "~10 epochs and the net memorises the train episodes")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--day-onehot", action="store_true",
                    help="append a one-hot(30) day encoding to the inputs; a single "
                         "global net otherwise trades late-day precision against "
                         "global fit (v1: day-29 MAE 872 vs the ~567 copy floor)")
    ap.add_argument("--compare", default=None,
                    help="path to another best.pt to evaluate on THIS val split "
                         "(the honest upgrade metric, e.g. runs/vnet_cem2/best.pt)")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    X, y, day, ep, adv, names, n_shards = load_shards(args.data)
    if args.day_onehot:
        oh = np.zeros((len(X), 30), dtype=np.float32)
        oh[np.arange(len(X)), np.clip(day, 0, 29)] = 1.0
        X = np.concatenate([X, oh], axis=1)
        names = names + [f"day_oh_{d:02d}" for d in range(30)]
    val_mask, n_train_ep, n_val_ep = split_by_episode(ep, args.val_frac, args.split_seed)
    tr, va = ~val_mask, val_mask
    n_adv = int((adv > 0).sum())
    print(f"data: {X.shape[0]} rows x {X.shape[1]} feats from {n_shards} shards; "
          f"split by episode: {n_train_ep} train / {n_val_ep} val episodes "
          f"({tr.sum()} / {va.sum()} rows); adversarial-market rows: {n_adv} "
          f"({n_adv / len(X):.0%})")

    # train-split standardisation (both sides), metrics in dollars
    money = X[:, names.index("money")].astype(np.float32)
    delta = y - money                      # what the net actually regresses
    x_mean = X[tr].mean(axis=0)
    x_std = np.maximum(X[tr].std(axis=0), 1e-6)
    d_mean = float(delta[tr].mean())
    d_std = max(float(delta[tr].std()), 1e-6)
    y_mean = float(y[tr].mean())           # the trivial-baseline prediction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"device: {device}{gpu}")

    def to_dev(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    Xtr = to_dev((X[tr] - x_mean) / x_std)
    ytr = to_dev((delta[tr] - d_mean) / d_std)
    Xva = to_dev((X[va] - x_mean) / x_std)
    yva_np = y[va]
    money_va = money[va]

    net = ValueNet(X.shape[1], args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_mae = 0, float("inf")

    last_path, best_path = outdir / "last.pt", outdir / "best.pt"
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("d_in") == X.shape[1] and ck.get("hidden") == args.hidden:
            net.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["opt_state"])
            start_epoch = ck["epoch"] + 1
            best_mae = ck.get("best_val_mae", best_mae)
            print(f"resumed from epoch {ck['epoch']} (best val MAE {best_mae:.0f})")
        else:
            print("last.pt shape mismatch (new features?); starting fresh")

    def val_pred():
        """V = money + predicted delta, in dollars of final money."""
        net.eval()
        with torch.no_grad():
            return money_va + (net(Xva) * d_std + d_mean).cpu().numpy()

    n = Xtr.shape[0]
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        va_mae = float(np.abs(val_pred() - yva_np).mean())

        state = {"epoch": epoch, "model_state": net.state_dict(),
                 "opt_state": opt.state_dict(), "best_val_mae": min(best_mae, va_mae),
                 "d_in": X.shape[1], "hidden": args.hidden}
        atomic_save(state, last_path)
        if va_mae < best_mae:
            best_mae = va_mae
            atomic_save({
                "model_state": net.state_dict(), "d_in": X.shape[1],
                "hidden": args.hidden, "epoch": epoch, "target": "delta",
                "x_mean": x_mean, "x_std": x_std, "d_mean": d_mean, "d_std": d_std,
                "y_mean": y_mean, "feature_names": names, "val_mae": va_mae,
                "day_onehot": bool(args.day_onehot),
            }, best_path)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train_mse={total / n:.4f}  "
                  f"val_mae={va_mae:7.0f}  best={best_mae:7.0f}")
    train_s = time.time() - t0
    if start_epoch < args.epochs:
        print(f"trained epochs {start_epoch}..{args.epochs - 1} in {train_s:.1f}s "
              f"on {device.type}")

    # final report from the best checkpoint
    ck = torch.load(best_path, map_location=device, weights_only=False)
    net.load_state_dict(ck["model_state"])
    pred = val_pred()
    base = np.full_like(yva_np, y_mean)  # trivial predict-train-mean baseline
    print(f"\nval metrics (best epoch {ck['epoch']}, {len(yva_np)} rows, "
          f"targets mean {yva_np.mean():.0f} sd {yva_np.std():.0f}):")
    print(f"  {'bucket':<10s} {'n':>5s}  {'net MAE':>8s} {'net R2':>7s}  "
          f"{'mean MAE':>8s} {'mean R2':>7s}")
    for r in bucket_metrics(yva_np, pred, base, day[va]):
        print(f"  {r['bucket']:<10s} {r['n']:5d}  {r['net_mae']:8.0f} {r['net_r2']:7.3f}  "
              f"{r['base_mae']:8.0f} {r['base_r2']:7.3f}")

    adv_va = adv[va]
    if (adv_va >= 0).any():
        for label, m in (("adversarial (crasher/mirror)", adv_va > 0),
                         ("quiet market", adv_va == 0)):
            if not m.any():
                continue
            mae, r2 = _mae_r2(yva_np[m], pred[m])
            print(f"  {label:<28s} n={int(m.sum()):6d}  MAE {mae:6.0f}  R2 {r2:6.3f}")

    if args.compare:
        ck1 = torch.load(args.compare, map_location=device, weights_only=False)
        net1 = ValueNet(ck1["d_in"], ck1["hidden"]).to(device)
        net1.load_state_dict(ck1["model_state"])
        net1.eval()
        names1 = [str(s) for s in ck1["feature_names"]]
        d1 = ck1["d_in"]
        assert names[:d1] == names1, "--compare net's features are not a prefix of ours"
        Xva1 = to_dev((X[va][:, :d1] - ck1["x_mean"]) / ck1["x_std"])
        with torch.no_grad():
            pred1 = money_va + (net1(Xva1).cpu().numpy() * ck1["d_std"] + ck1["d_mean"])
        print(f"\n--compare {args.compare} on THIS val split:")
        for r in bucket_metrics(yva_np, pred1, base, day[va]):
            print(f"  {r['bucket']:<10s} {r['n']:5d}  {r['net_mae']:8.0f} {r['net_r2']:7.3f}")

    print(f"\nbest model + scaler + feature names -> {best_path}")


if __name__ == "__main__":
    main()
