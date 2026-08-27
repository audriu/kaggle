"""Cross-Entropy Method trainer for the parameterised policy.

Why CEM and not deep RL: reward arrives once per 720-step episode with measured noise
sd/mean ~ 0.37-0.56 and near-useless seed pairing, the action space is 1e24+ but the
*strategy* space is ~45 numbers, and the box is 16 CPU cores. Rank-based CEM over
theta is robust to exactly this regime and is embarrassingly parallel.

Each generation:
  1. sample `pop` thetas from N(mu, sigma^2), clipped to per-param bounds
  2. evaluate every candidate (plus mu and best-so-far as controls) against the
     opponent league on a fresh seed block, both seats  -- sim/arena.evaluate_many
  3. elites = top quarter by mean final money; mu/sigma <- elite mean/std (smoothed,
     sigma floored to keep exploring)
  4. league grows with the best theta every `league_every` gens (keeps the population
     from overfitting to a frozen opponent -- the mirror-vs-starter gap is 35%);
     fixed price-crasher exploiters (train/exploiters.py) ride along by default
     (--no-exploiters to drop them) so candidates always feel market flooding
  5. checkpoint EVERYTHING (mu, sigma, best, league, RNG, history) atomically

Resume: `--resume` (the default) picks up from the newest valid checkpoint; a fresh
run needs `--fresh`. Seed blocks derive from the generation index, so a resumed run
evaluates on exactly the seeds it would have used uninterrupted.

Usage:
  python train/cem.py --gens 50                 # start or resume the default run
  python train/cem.py --gens 200 --pop 24 --seeds-per 15
  kill it any time; rerun the same command to continue.
"""

import argparse
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from sim.arena import (DEFAULT_PROCS, builtin_spec, evaluate_many, file_spec,  # noqa: E402
                       params_spec, seed_block)
from train.checkpoint import CheckpointManager, capture_rng, restore_rng  # noqa: E402
from train.exploiters import get_exploiters  # noqa: E402

BUILDER = "agent.policy:build"


def fresh_state(rng_seed, init_theta=None):
    """init_theta: start the search there instead of at the policy defaults, with
    sigma re-widened to the full exploration scale -- the restart move for a run
    whose sigma collapsed to the floor (exploration stalled) around a good theta."""
    random.seed(rng_seed)
    theta = list(init_theta) if init_theta is not None else policy.default_theta()
    return {
        "gen": 0,
        "mu": list(theta),
        "sigma": policy.sigmas(),
        "best_theta": list(theta),
        "best_fitness": None,
        "league": [],           # list of (gen, theta) snapshots, newest last
        "history": [],          # per-gen summary dicts
        "rng": capture_rng(),
    }


def opponents_for(state, use_exploiters=True):
    opps = [builtin_spec("starter"), file_spec(ROOT / "agent" / "main.py", "baseline")]
    if use_exploiters:
        # Fixed price-crasher thetas (train/exploiters.py): pure code, never part of
        # the checkpoint, so toggling them cannot break resume.
        for name, theta in get_exploiters():
            opps.append(params_spec(BUILDER, theta, name))
    for g, theta in state["league"][-2:]:
        opps.append(params_spec(BUILDER, theta, f"league-g{g}"))
    return opps


def run(args):
    cm = CheckpointManager(args.outdir, keep=5)
    step, state = (None, None) if args.fresh else cm.load_latest()
    if state is None:
        init_theta = None
        if args.init_run:
            _, src = CheckpointManager(args.init_run).load_best()
            if src is None:
                sys.exit(f"--init-run: no best checkpoint in {args.init_run}")
            init_theta = src["best_theta"]
            print(f"init from {args.init_run} gen {src['gen']} best_theta "
                  f"(fitness {src['best_fitness']:.0f}), sigma re-widened")
        state = fresh_state(args.rng_seed, init_theta)
        print(f"fresh run -> {args.outdir}")
    else:
        restore_rng(state["rng"])
        print(f"resumed from gen {state['gen']} (checkpoint step {step})")

    names = policy.theta_names()
    n_dim = len(names)
    elite_n = max(2, args.pop // 4)
    sigma_floor = [s * args.sigma_floor for s in policy.sigmas()]
    print("opponents: " + ", ".join(o.name for o in opponents_for(state, args.exploiters)))

    with Pool(args.procs) as pool:
        while state["gen"] < args.gens:
            gen = state["gen"]
            t0 = time.time()

            # 1. sample
            cands_theta = [
                policy.clip_theta([random.gauss(m, s) for m, s in zip(state["mu"], state["sigma"])])
                for _ in range(args.pop)
            ]
            # controls: current mean and best-so-far ride along in the same eval
            cands_theta.append(list(state["mu"]))
            cands_theta.append(list(state["best_theta"]))
            cands = [params_spec(BUILDER, t, f"g{gen}c{i}") for i, t in enumerate(cands_theta)]

            # 2. evaluate on this generation's seed block
            opps = opponents_for(state, args.exploiters)
            seeds = seed_block(gen, args.seeds_per)
            reports = evaluate_many(cands, opps, seeds, pool)
            fitness = [r.mean for r in reports]

            # 3. elites -> new mu/sigma
            order = sorted(range(args.pop), key=lambda i: -fitness[i])
            elites = [cands_theta[i] for i in order[:elite_n]]
            new_mu = [sum(e[d] for e in elites) / elite_n for d in range(n_dim)]
            new_sd = [
                max(
                    sigma_floor[d],
                    (sum((e[d] - new_mu[d]) ** 2 for e in elites) / elite_n) ** 0.5,
                )
                for d in range(n_dim)
            ]
            a = args.alpha
            state["mu"] = [a * n + (1 - a) * o for n, o in zip(new_mu, state["mu"])]
            state["sigma"] = [a * n + (1 - a) * o for n, o in zip(new_sd, state["sigma"])]

            # best-so-far: compare on THIS generation's shared eval (same seeds/opponents),
            # so the incumbent must keep defending its title on fresh worlds.
            gen_best_i = order[0]
            incumbent_fit = fitness[args.pop + 1]
            challenger_fit = fitness[gen_best_i]
            if challenger_fit > incumbent_fit:
                state["best_theta"] = cands_theta[gen_best_i]
                state["best_fitness"] = challenger_fit
            else:
                state["best_fitness"] = incumbent_fit

            # 4. league
            if gen % args.league_every == args.league_every - 1:
                state["league"].append((gen, list(state["best_theta"])))
                state["league"] = state["league"][-args.league_max:]

            state["gen"] = gen + 1
            state["history"].append({
                "gen": gen,
                "mean_pop": sum(fitness[:args.pop]) / args.pop,
                "best_pop": challenger_fit,
                "mu_fit": fitness[args.pop],
                "best_fit": state["best_fitness"],
                "elapsed": time.time() - t0,
            })

            # 5. checkpoint (atomic; RNG captured after all sampling this gen)
            state["rng"] = capture_rng()
            n_eps = len(cands) * len(opps) * len(seeds) * 2
            cm.save(gen, state,
                    metrics={"best_fit": state["best_fitness"], "mu_fit": fitness[args.pop]},
                    is_best=True)
            h = state["history"][-1]
            print(f"gen {gen:3d}  pop_mean={h['mean_pop']:7.0f}  pop_best={h['best_pop']:7.0f}  "
                  f"mu={h['mu_fit']:7.0f}  best={h['best_fit']:7.0f}  "
                  f"({n_eps} eps, {h['elapsed']:.0f}s, league={len(state['league'])})",
                  flush=True)

    print("\nfinal mu vs default:")
    for name, d, m in zip(names, policy.default_theta(), state["mu"]):
        if abs(m - d) > 1e-9:
            print(f"  {name:18s} {d:8.2f} -> {m:8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seeds-per", type=int, default=12)
    ap.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    ap.add_argument("--outdir", default=str(ROOT / "train" / "runs" / "cem1"))
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--sigma-floor", type=float, default=0.15)
    ap.add_argument("--league-every", type=int, default=5)
    ap.add_argument("--league-max", type=int, default=6)
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--exploiters", action=argparse.BooleanOptionalAction, default=True,
                    help="include fixed price-crasher opponents (train/exploiters.py)")
    ap.add_argument("--init-run", default=None,
                    help="run dir whose best_theta seeds a FRESH run (sigma re-widened); "
                         "only read when starting fresh, ignored on resume")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--resume", action="store_true", help="(default behaviour)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
