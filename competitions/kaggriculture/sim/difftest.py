"""Differential test: FastEnv must reproduce the reference environment exactly.

Runs the same agents on the same seed through `kaggle_environments.Environment.run()`
and through `FastEnv`, then compares the full per-step trajectory (money, market
inventory and prices, town unlocks, and the rendered tile grid) plus final rewards.
Any divergence is a bug in FastEnv, never in the rules -- both use the same interpreter.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_environments import make  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

AGENT = str(Path(__file__).resolve().parent.parent / "agent" / "main.py")


def _priv(step_state):
    """Snapshot both players' private state NOW -- these dicts are mutated in place."""
    return json.dumps(
        [step_state[i]["observation"]["private"] for i in range(2)],
        sort_keys=True, default=str,
    )


def _fingerprint(obs):
    """Everything an agent could observe, flattened for comparison."""
    farms = obs["farms"]
    return {
        "day": obs["day"],
        "hour": obs["hour"],
        "money": [f["money"] for f in farms],
        "farmer": [list(f["farmer"]) for f in farms],
        "hands": [[list(h) for h in f["hands"]] for f in farms],
        "hires": [f.get("hires_today") for f in farms],
        "quads": [sorted(f.get("unlocked_quadrants") or []) for f in farms],
        "tiles": [json.dumps(f["tiles"], sort_keys=True, default=str) for f in farms],
        "inv": dict(obs["market"]["inventory"]),
        "prices": dict(obs["market"]["prices"]),
        "town": sorted(obs["town"]["unlocked_shops"]),
    }


def reference_trajectory(agents, seed, steps):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    t0 = time.time()
    env.run(agents)
    elapsed = time.time() - t0
    traj = [_fingerprint(s[0]["observation"]) for s in env.steps]
    privates = [_priv(s) for s in env.steps]
    return traj, privates, [s.reward for s in env.steps[-1]], elapsed


def fast_trajectory(agents, seed, steps):
    fns = [load_agent(a) for a in agents]
    fe = FastEnv(seed, episode_steps=steps)
    traj, privates = [], []
    t0 = time.time()
    traj.append(_fingerprint(fe.observation(0)))
    privates.append(_priv(fe.state))
    for _ in range(steps - 1):
        acts = [fn(fe.observation(i)) for i, fn in enumerate(fns)]
        fe.step(acts)
        traj.append(_fingerprint(fe.observation(0)))
        privates.append(_priv(fe.state))
        if fe.done:
            break
    return traj, privates, fe.rewards(), time.time() - t0


def compare(agents, seed, steps):
    rt, rp, rr, re_ = reference_trajectory(agents, seed, steps)
    ft, fp, fr, fe_ = fast_trajectory(agents, seed, steps)

    if len(rt) != len(ft):
        return False, f"step count {len(rt)} vs {len(ft)}", re_, fe_
    for i, (a, b) in enumerate(zip(rt, ft)):
        if a != b:
            diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
            return False, f"public state diverged at step {i}: {json.dumps(diff, default=str)[:600]}", re_, fe_
    for i, (a, b) in enumerate(zip(rp, fp)):
        if a != b:
            return False, f"private state diverged at step {i}:\n  ref  {a[:300]}\n  fast {b[:300]}", re_, fe_
    if [float(x) for x in rr] != [float(x) for x in fr]:
        return False, f"rewards {rr} vs {fr}", re_, fe_
    return True, f"identical over {len(rt)} steps, rewards {rr}", re_, fe_


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 720
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3, 7, 42]
    matchups = [[AGENT, AGENT], [AGENT, "starter"], ["starter", AGENT], ["pass", AGENT]]

    ok_all, ref_t, fast_t = True, 0.0, 0.0
    for agents in matchups:
        for seed in seeds:
            names = "/".join(Path(a).stem if a.endswith(".py") else a for a in agents)
            ok, msg, rt, ft = compare(agents, seed, steps)
            ref_t += rt
            fast_t += ft
            ok_all &= ok
            print(f"[{'OK ' if ok else 'FAIL'}] {names:22s} seed={seed:<4} {msg}")
            if not ok:
                return 1
    n = len(matchups) * len(seeds)
    print(f"\n{n} episodes x {steps} steps")
    print(f"  reference : {ref_t:7.2f}s total  ({ref_t / n:.3f} s/episode)")
    print(f"  fastenv   : {fast_t:7.2f}s total  ({fast_t / n:.3f} s/episode)")
    print(f"  speedup   : {ref_t / fast_t:.1f}x")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
