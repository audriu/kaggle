"""Episode dataset collector for the value net: (features at hour 0, day) -> final money.

Plays full FastEnv episodes and snapshots BOTH players' feature vectors
(sim/features.extract) at hour 0 of every day -- 30 snapshots per player, 60 rows
per episode -- labelling every row with that player's FINAL money. The value net
(train/value_net.py) later regresses row -> label so inference-time rollouts can be
truncated at a day boundary instead of played out for 720 steps.

Matchup diversity is the point of the collector: a net fitted only on
baseline-vs-baseline play would never see a crashed carrot market or a 12-hand farm.
Each episode's two seats are drawn deterministically from the episode seed:
30% baseline agent/main.py, 20% builtin starter, 50% a random theta sampled
gauss(default_theta, sigmas) and clipped to the policy bounds -- the same
distribution CEM generation 0 explores, so the net trains on exactly the states the
optimiser visits. Deriving everything from the seed makes shards bit-reproducible.

Sharding: episodes are grouped into chunks of CHUNK=8 consecutive seeds; each worker
plays one chunk and writes one compressed npz shard named by its inclusive seed
range (shard-<lo>-<hi>.npz: features float32 [N,D], targets float32 [N], day int16,
episode int32 = the seed, feature_names). Shards whose file already exists are
SKIPPED, so re-running the same command resumes an interrupted collection and
re-running a finished one is a no-op. Keep --episodes a multiple of CHUNK when
growing a dataset in place, so a rerun never produces a partial shard whose seed
range overlaps a full one.

Writes are tmp+os.replace atomic, so a killed worker never leaves a truncated shard
that a later resume would trust.

Usage (the box also runs CEM on ~16 procs -- default stays at 4):
  python train/collect.py --episodes 48 --procs 4 --seed0 1000000 --out train/data
"""

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from sim import features  # noqa: E402
from sim.arena import build, builtin_spec, file_spec  # noqa: E402
from sim.fastenv import FastEnv  # noqa: E402

CHUNK = 8  # episodes per shard; 8 eps * 60 rows = 480 rows/shard

BASELINE = file_spec(ROOT / "agent" / "main.py", "baseline")
STARTER = builtin_spec("starter")

# seat mix: (cumulative probability, kind)
MIX = [(0.30, "baseline"), (0.50, "starter"), (1.00, "theta")]


def matchup(seed):
    """Two seat descriptors, derived deterministically from the episode seed.

    A descriptor is "baseline", "starter", or a tuple(theta). Each seat gets its
    own salted RNG so seat 1's draw does not depend on how many gauss() calls
    seat 0 consumed.
    """
    out = []
    for side in (0, 1):
        rng = random.Random(f"collect-{seed}-{side}")
        r = rng.random()
        kind = next(k for p, k in MIX if r < p)
        if kind == "theta":
            theta = policy.clip_theta([
                rng.gauss(m, s)
                for m, s in zip(policy.default_theta(), policy.sigmas())
            ])
            out.append(tuple(theta))
        else:
            out.append(kind)
    return out


def _agent_fn(desc):
    if desc == "baseline":
        return build(BASELINE)   # arena's per-process cache: loaded once per worker
    if desc == "starter":
        return build(STARTER)
    return policy.build(list(desc))


def play_episode(seed):
    """Run one episode, return (rows, days, players, finals).

    rows[k] is the feature vector for `players[k]` at hour 0 of `days[k]`;
    label it with finals[players[k]] after the episode ends.
    """
    agents = [_agent_fn(d) for d in matchup(seed)]
    env = FastEnv(seed)
    rows, days, players = [], [], []
    for _ in range(env.steps_total - 1):
        obs0 = env.observation(0)
        if obs0.get("hour", 0) == 0:
            day = obs0.get("day", 0)
            for i in (0, 1):
                rows.append(features.extract(env.observation(i)))
                days.append(day)
                players.append(i)
        env.step([fn(env.observation(i)) for i, fn in enumerate(agents)])
        if env.done:
            break
    return rows, days, players, env.rewards()


def shard_path(out_dir, lo, hi):
    return Path(out_dir) / f"shard-{lo:09d}-{hi:09d}.npz"


def collect_shard(job):
    """Worker: play seeds [lo, hi] and write one npz shard. Returns stats."""
    out_dir, lo, hi = job
    path = shard_path(out_dir, lo, hi)
    if path.exists():
        return {"path": path.name, "rows": 0, "eps": 0, "skipped": True}

    t0 = time.time()
    feats, targets, days, eps = [], [], [], []
    for seed in range(lo, hi + 1):
        try:
            rows, ds, players, finals = play_episode(seed)
        except Exception as e:
            raise RuntimeError(f"episode seed={seed} failed: {e}") from e
        feats.extend(rows)
        targets.extend(finals[p] for p in players)
        days.extend(ds)
        eps.extend([seed] * len(rows))

    # Write via an open handle (savez_compressed would append ".npz" to a bare
    # tmp name), then atomically rename into place.
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            features=np.asarray(feats, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            day=np.asarray(days, dtype=np.int16),
            episode=np.asarray(eps, dtype=np.int32),
            feature_names=np.asarray(features.FEATURE_NAMES),
        )
    os.replace(tmp, path)
    return {"path": path.name, "rows": len(feats), "eps": hi - lo + 1,
            "skipped": False, "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description="Collect value-net training shards.")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--procs", type=int, default=4,
                    help="worker processes (default 4: the box usually also runs CEM)")
    ap.add_argument("--seed0", type=int, default=1_000_000,
                    help="first episode seed; away from arena/CEM seed blocks (100k+)")
    ap.add_argument("--out", default=str(ROOT / "train" / "data"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes % CHUNK:
        print(f"note: --episodes {args.episodes} is not a multiple of {CHUNK}; "
              f"the final shard will be partial -- keep it consistent across reruns")

    jobs = []
    lo = args.seed0
    while lo < args.seed0 + args.episodes:
        hi = min(lo + CHUNK, args.seed0 + args.episodes) - 1
        jobs.append((str(out_dir), lo, hi))
        lo = hi + 1

    t0 = time.time()
    with Pool(args.procs) as pool:
        stats = pool.map(collect_shard, jobs, chunksize=1)
    elapsed = time.time() - t0

    new = [s for s in stats if not s["skipped"]]
    rows = sum(s["rows"] for s in new)
    eps = sum(s["eps"] for s in new)
    for s in stats:
        tag = "skip" if s["skipped"] else f"{s['rows']:5d} rows {s['elapsed']:5.1f}s"
        print(f"  {s['path']}  {tag}")
    print(f"done: {eps} episodes -> {rows} rows in {elapsed:.1f}s "
          f"({eps / elapsed:.2f} eps/s, {rows / elapsed:.0f} rows/s) "
          f"[{len(stats) - len(new)} shards skipped]  D={features.N_FEATURES}")


if __name__ == "__main__":
    main()
