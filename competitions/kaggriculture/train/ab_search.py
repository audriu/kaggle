"""Paired A/B harness for the rollout-search layer (search v2 gate).

Runs search-on vs search-off with the SAME baked theta on the same seeds, both
seats, against selected opponents, and reports the PAIRED per-(seed,seat,
opponent) money difference with its SE -- the plan-§9 ship gate is
  paired > +1 SE on crasher AND mirror, > -1 SE on starter AND baseline,
confirmed on seeds never used for config selection.

Seed-block bookkeeping (never select and confirm on the same block):
  820000+  config sweep (selection)      900000+  final confirmation
  Everything below 820000 is burned by earlier work (arena gates 400k/500k,
  v1 A/Bs 700k/810k, collectors 1M/2M/3M, tests <1000).

The search agent must be constructible inside a Pool worker from a picklable
Spec, hence the module-level builders keyed by config name (sim/arena.Spec
resolves "module:function" strings).

Usage:
  python train/ab_search.py --configs mg150_starter,mg2000_self \
      --opponents crasher,mirror --seeds 24 --seed0 820000 --procs 4
  python train/ab_search.py --configs <winner> --opponents crasher,mirror,starter,baseline \
      --seeds 40 --seed0 900000 --procs 4      # confirmation
"""

import argparse
import math
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from agent.search import build_search_agent  # noqa: E402
from agent.vnet_infer import ValueFn  # noqa: E402
from sim.arena import Result, Spec, builtin_spec, file_spec, params_spec, build  # noqa: E402
from sim.fastenv import FastEnv  # noqa: E402
from train.checkpoint import CheckpointManager  # noqa: E402
from train.exploiters import get_exploiters  # noqa: E402

VNET_RUN = ROOT / "train" / "runs" / "vnet_v2"
THETA_RUN = ROOT / "train" / "runs" / "cem2"

# name -> search cfg overrides; the sweep grid lives here so specs pickle.
CONFIGS = {
    "mg150_starter": {"min_gain": 150.0, "opp_model": "starter"},
    "mg800_starter": {"min_gain": 800.0, "opp_model": "starter"},
    "mg2000_starter": {"min_gain": 2000.0, "opp_model": "starter"},
    "mg150_self": {"min_gain": 150.0, "opp_model": "self"},
    "mg800_self": {"min_gain": 800.0, "opp_model": "self"},
    "mg2000_self": {"min_gain": 2000.0, "opp_model": "self"},
}

_CACHE = {}


def _theta():
    th = _CACHE.get("theta")
    if th is None:
        _, s = CheckpointManager(str(THETA_RUN)).load_best()
        th = _CACHE["theta"] = list(s["best_theta"])
    return th


def _vnet():
    vf = _CACHE.get("vnet")
    if vf is None:
        best = VNET_RUN / "best.pt"
        vf = _CACHE["vnet"] = ValueFn.from_checkpoint(str(best))
    return vf


def build_search(cfg_name):
    """arena Spec builder: params_spec('train.ab_search:build_search', (name,))."""
    (name,) = cfg_name if isinstance(cfg_name, (list, tuple)) else (cfg_name,)
    return build_search_agent(_theta(), _vnet(), CONFIGS[name])


def build_plain(_ignored):
    return policy.build(_theta())


def _opponent_spec(name):
    if name == "crasher":
        th = dict(get_exploiters())["crasher"]
        return params_spec("agent.policy:build", th, "crasher")
    if name == "mirror":
        return params_spec("train.ab_search:build_plain", (0,), "mirror")
    if name == "baseline":
        return file_spec(ROOT / "agent" / "main.py", "baseline")
    return builtin_spec(name)


def _play(job):
    seed, seat, cand, opp = job
    a, b = (cand, opp) if seat == 0 else (opp, cand)
    r = FastEnv(seed).run([build(a), build(b)])
    return Result(seed, opp.name, seat, r[seat], r[1 - seat])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="mg800_starter")
    ap.add_argument("--opponents", default="crasher,mirror")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=820000)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    opps = [_opponent_spec(o) for o in args.opponents.split(",")]
    off_spec = params_spec("train.ab_search:build_plain", (0,), "search-off")

    with Pool(args.procs) as pool:
        jobs_off = [(s, seat, off_spec, o) for o in opps for s in seeds for seat in (0, 1)]
        off = pool.map(_play, jobs_off, chunksize=1)
        off_by_key = {(r.seed, r.seat, r.opponent): r.score for r in off}
        print(f"search-off reference: mean {st.mean([r.score for r in off]):.0f} "
              f"over {len(off)} eps")

        for cfg_name in args.configs.split(","):
            on_spec = params_spec("train.ab_search:build_search", (cfg_name,),
                                  f"on-{cfg_name}")
            jobs_on = [(s, seat, on_spec, o) for o in opps for s in seeds for seat in (0, 1)]
            on = pool.map(_play, jobs_on, chunksize=1)
            print(f"\n== {cfg_name} ==")
            all_d = []
            for o in opps:
                d = [r.score - off_by_key[(r.seed, r.seat, r.opponent)]
                     for r in on if r.opponent == o.name]
                all_d += d
                se = st.pstdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
                print(f"  vs {o.name:<10s} paired {st.mean(d):+8.0f} +- {se:6.0f} SE  "
                      f"(n={len(d)}, on-mean {st.mean([r.score for r in on if r.opponent == o.name]):.0f})")
            se = st.pstdev(all_d) / math.sqrt(len(all_d))
            print(f"  pooled       paired {st.mean(all_d):+8.0f} +- {se:6.0f} SE (n={len(all_d)})")


if __name__ == "__main__":
    main()
