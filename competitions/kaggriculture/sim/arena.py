"""Parallel match runner and evaluation harness.

Evaluation noise in this game is brutal: over 40 seeds the current agent scores
mean 13035 / sd 4781, and perturbed variants correlate with the baseline at only
r = +0.10..+0.52, so common-random-numbers pairing barely helps. (The cause is real:
the per-day weed RNG stream is consumed by both farms before the town-shop-unlock
draw, so the opponent's tile occupancy shifts your weeds and which shops unlock --
same seed does not mean same world.)

Everything here is therefore built around "many seeds, both seats, report the
standard error". Seats are always swapped so first-player asymmetry cancels.

Agents are described by a small picklable `Spec` so worker processes can rebuild
them; each worker caches by spec key, so a policy is constructed once per process.
"""

import math
import os
import statistics as st
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.fastenv import FastEnv, load_agent  # noqa: E402

DEFAULT_PROCS = min(16, (os.cpu_count() or 4))


@dataclass(frozen=True)
class Spec:
    """A picklable description of an agent. `kind` selects how to build it.

    file    -- path to a Kaggle-style agent .py (exec'd, last callable wins)
    builtin -- one of the env's own agents: "pass", "random", "starter"
    params  -- a parameterised policy: `builder(theta)` -> callable, where
               `builder` is resolved from "module:function" so it stays picklable
    """

    kind: str
    name: str
    payload: Any = None
    theta: tuple = field(default=(), compare=True)

    def key(self):
        return (self.kind, self.name, self.payload, self.theta)


def file_spec(path, name=None):
    p = str(Path(path).resolve())
    return Spec("file", name or Path(p).stem, p)


def builtin_spec(name):
    return Spec("builtin", name, name)


def params_spec(builder, theta, name="cand"):
    """builder: "package.module:function" string, resolved inside the worker."""
    return Spec("params", name, builder, tuple(theta))


_CACHE: dict = {}


def build(spec: Spec) -> Callable:
    k = spec.key()
    fn = _CACHE.get(k)
    if fn is not None:
        return fn
    if spec.kind in ("file", "builtin"):
        fn = load_agent(spec.payload)
    elif spec.kind == "params":
        mod_name, _, attr = spec.payload.partition(":")
        import importlib

        fn = getattr(importlib.import_module(mod_name), attr)(list(spec.theta))
    else:
        raise ValueError(f"unknown spec kind {spec.kind!r}")
    _CACHE[k] = fn
    return fn


@dataclass
class Result:
    seed: int
    opponent: str
    seat: int
    score: float
    opp_score: float

    @property
    def win(self):
        return self.score > self.opp_score


def _play(job):
    seed, seat, cand, opp, steps = job
    a, b = (cand, opp) if seat == 0 else (opp, cand)
    rewards = FastEnv(seed, episode_steps=steps).run([build(a), build(b)])
    return Result(seed, opp.name, seat, rewards[seat], rewards[1 - seat])


@dataclass
class Report:
    results: list
    mean: float
    sd: float
    se: float
    winrate: float
    n: int
    per_opponent: dict

    def __str__(self):
        rows = "\n".join(
            f"    vs {o:<16s} mean={v['mean']:8.0f}  win={v['winrate']:5.1%}  n={v['n']}"
            for o, v in sorted(self.per_opponent.items())
        )
        return (f"  mean={self.mean:8.0f}  sd={self.sd:7.0f}  se={self.se:6.0f}  "
                f"win={self.winrate:5.1%}  n={self.n}\n{rows}")


def summarize(results) -> Report:
    scores = [r.score for r in results]
    n = len(scores)
    sd = st.pstdev(scores) if n > 1 else 0.0
    per = {}
    for r in results:
        per.setdefault(r.opponent, []).append(r)
    per_opponent = {
        o: {
            "mean": st.mean([x.score for x in rs]),
            "winrate": sum(x.win for x in rs) / len(rs),
            "n": len(rs),
        }
        for o, rs in per.items()
    }
    return Report(
        results=results,
        mean=st.mean(scores) if n else 0.0,
        sd=sd,
        se=sd / math.sqrt(n) if n else 0.0,
        winrate=sum(r.win for r in results) / n if n else 0.0,
        n=n,
        per_opponent=per_opponent,
    )


def evaluate(candidate, opponents, seeds, pool=None, steps=720, both_seats=True):
    """Play `candidate` against every opponent on every seed, in both seats."""
    seats = (0, 1) if both_seats else (0,)
    jobs = [(s, seat, candidate, opp, steps)
            for opp in opponents for s in seeds for seat in seats]
    if pool is None:
        results = [_play(j) for j in jobs]
    else:
        results = pool.map(_play, jobs, chunksize=1)
    return summarize(results)


def evaluate_many(candidates, opponents, seeds, pool, steps=720, both_seats=True):
    """Evaluate several candidates in ONE flat fan-out so the pool stays saturated.

    Evaluating candidates one at a time leaves cores idle at each barrier; a single
    flat job list keeps every worker busy until the whole generation is done.
    """
    seats = (0, 1) if both_seats else (0,)
    jobs, owner = [], []
    for ci, cand in enumerate(candidates):
        for opp in opponents:
            for s in seeds:
                for seat in seats:
                    jobs.append((s, seat, cand, opp, steps))
                    owner.append(ci)
    results = pool.map(_play, jobs, chunksize=1)
    buckets = [[] for _ in candidates]
    for ci, r in zip(owner, results):
        buckets[ci].append(r)
    return [summarize(b) for b in buckets]


def seed_block(generation, n_seeds, stride=100_000):
    """Deterministic, reproducible, non-overlapping seed set for a generation.

    Rotating seeds each generation stops the population from overfitting to a fixed
    set of worlds; deriving them from the generation index keeps a resumed run on
    exactly the same schedule it would have had.
    """
    base = stride + generation * n_seeds
    return list(range(base, base + n_seeds))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate an agent against a set of opponents.")
    ap.add_argument("agent", nargs="?", default=str(Path(__file__).parent.parent / "agent" / "main.py"))
    ap.add_argument("--opponents", default="starter,pass,self")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()

    cand = file_spec(args.agent)
    opps = []
    for o in args.opponents.split(","):
        opps.append(cand if o == "self" else
                    (builtin_spec(o) if not o.endswith(".py") else file_spec(o)))
    seeds = seed_block(0, args.seeds)

    import time
    t0 = time.time()
    with Pool(args.procs) as p:
        rep = evaluate(cand, opps, seeds, pool=p, steps=args.steps)
    el = time.time() - t0
    print(f"{Path(args.agent).stem}  ({rep.n} episodes in {el:.1f}s, {rep.n / el:.1f} eps/s)")
    print(rep)


if __name__ == "__main__":
    main()
