"""Ground-truth divergence experiment for agent/reconstruct.py.

The week-3 rollout search will, every turn on Kaggle, rebuild a simulator from
the agent's observation (make_sim) and score candidate plans by rolling it a
few days forward. This script measures how much that reconstruction can be
trusted, against the only ground truth there is: a real FastEnv whose hidden
state (opponent private, episode seed) the agent never sees.

Five checks, single process (the box runs a 16-proc CEM; procs <= 4 applies):

  A. EXACT-INFO CONTINUATION (hard assert). Reconstruct with the TRUE opponent
     private and TRUE episode seed and lockstep 72 steps against a deepcopy of
     the real env, comparing a full-state JSON fingerprint (farms, market,
     town, both privates, day, hour) after every step. Bit-identical here
     proves the rebuild logic is exact and ALL default-mode divergence comes
     from the two documented estimated/unknowable inputs. Includes a
     (day 12, hour 13) point with hands hired, asserting the hour sequence
     14..23,0 and that no daily event (hand reset, weed spawn) re-fires
     mid-day.

  B. DIVERGENCE (the headline table). Snapshot player 0's obs at day
     5/10/15/20 hour 0 plus day 12 hour 13, seeds x4; roll the real env and
     the default reconstruction (empty opponent private, wrong rng_seed)
     forward with the SAME agent pair (policy.agent / starter) and report
     mean/max |money_real - money_recon| for player 0 at K = 2/3/5 days.
     Two ablations (true seed only / true opponent private only) split the
     error between the weed-RNG and the opponent model.

  C. RANK FIDELITY (the point of the exercise). 10 theta-variant candidates
     (strategic-param perturbations of the default policy); at each of 6
     contexts rank them by real-rollout vs reconstructed-rollout end money and
     report Spearman rho at K=2 and K=3. The search only needs the
     reconstruction to ORDER candidates like reality would.

  D. ROBUSTNESS: 30 reconstruction points spread over days AND hours, 5-day
     rollouts, zero exceptions, all money finite (hard assert).

  E. LATENCY: make_sim cold (includes the one-time make() config load) and
     warm, plus one end-to-end make_sim + 2-day rollout. Loaded-box numbers,
     pessimistic 2-4x.

Run:  python train/test_reconstruct.py    (from competitions/kaggriculture)
"""

import copy
import json
import math
import sys
import time
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP))

from agent import policy  # noqa: E402
from agent.reconstruct import make_sim  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

STARTER = load_agent("starter")
PAIR = (policy.agent, STARTER)  # seat 0 = self (the reconstructing player)
MARKS = (48, 72, 120)  # K = 2, 3, 5 days
WRONG_SEED = 12345  # deliberately not the episode seed: the agent can't know it


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def collect_snapshots(seed, snap_steps, agents=PAIR):
    """Run a real episode, snapshotting (deepcopy env, deepcopy obs0) whenever
    env.step_i hits a requested step (i.e. BEFORE that step's actions)."""
    want = set(snap_steps)
    env = FastEnv(seed)
    snaps = {}
    for _ in range(max(want) + 1):
        if env.step_i in want:
            snaps[env.step_i] = (copy.deepcopy(env), copy.deepcopy(env.observation(0)))
            if len(snaps) == len(want):
                break
        env.step([fn(env.observation(i)) for i, fn in enumerate(agents)])
    return snaps


def roll(sim, agents, marks=MARKS, me=0):
    """Step `sim` forward, recording player `me`'s money at each mark."""
    out = {}
    last = max(marks)
    for t in range(1, last + 1):
        if not sim.done:
            sim.step([fn(sim.observation(i)) for i, fn in enumerate(agents)])
        if t in marks:
            out[t] = sim.rewards()[me]
    return out


def fingerprint(sim):
    """Full mutable game state as a canonical JSON string."""
    o0 = sim.state[0].observation
    privates = [sim.state[i].observation.private for i in range(len(sim.state))]
    return json.dumps(
        [o0.farms, o0.market, o0.town, privates, o0.day, o0.hour],
        sort_keys=True,
    )


def spearman(a, b):
    """Spearman rho with average ranks for ties (stdlib only)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


NAME_IDX = {n: i for i, n in enumerate(policy.theta_names())}

# 10 candidates that all change behaviour from ANY mid-season state (day-gated
# params like dump_day/melon_day would tie at these snapshot days).
VARIANTS = [
    ("default",   {}),
    ("hire_lo",   {"hire_d1": 5.0, "hire_d2": 5.0}),
    ("hire_hi",   {"hire_d1": 14.0, "hire_d2": 14.0}),
    ("wheat_hi",  {"wheat_base": 6.0, "wheat_plus": 4.0, "wheat_buy_cap": 10.0}),
    ("carrot_lo", {"carrot_buy_cap": 2.0}),
    ("goose_hi",  {"goose_target": 10.0, "goose_cash": 400.0}),
    ("cow_early", {"cow_day": 8.0, "cow_target": 6.0}),
    ("sell_cap5", {"sell_cap": 5.0}),
    ("floors_hi", {"sell_floor_egg": 0.9, "sell_floor_wheat": 0.85,
                   "sell_floor_carrot": 0.85}),
    ("feed_hi",   {"feed_res_mult": 4.0, "feed_res_base": 6.0}),
]


def build_variant(overrides):
    theta = policy.default_theta()
    for k, v in overrides.items():
        theta[NAME_IDX[k]] = v
    return policy.build(theta)


def step_of(day, hour):
    return day * 24 + hour


# ---------------------------------------------------------------------------
# E first (cold-start must be measured before anything touches make_sim)
# ---------------------------------------------------------------------------

def run_latency():
    print("== E. latency (loaded box; pessimistic 2-4x) ==")
    snaps = collect_snapshots(999, [step_of(15, 0)])
    _, obs = snaps[step_of(15, 0)]

    t0 = time.perf_counter()
    make_sim(obs)
    cold_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        make_sim(obs)
    warm_ms = (time.perf_counter() - t0) * 1e3 / n

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        sim = make_sim(obs, rng_seed=WRONG_SEED)
        sim.run_steps(PAIR, 48)
        times.append((time.perf_counter() - t0) * 1e3)
    e2e_ms = sum(times) / len(times)

    print(f"  make_sim cold (incl. one-time make() config load): {cold_ms:8.1f} ms")
    print(f"  make_sim warm (n={n}):                             {warm_ms:8.2f} ms")
    print(f"  make_sim + 2-day rollout end-to-end (n=10 mean):   {e2e_ms:8.1f} ms")
    return {"cold_ms": cold_ms, "warm_ms": warm_ms, "e2e_2day_ms": e2e_ms}


# ---------------------------------------------------------------------------
# A. exact-info continuation
# ---------------------------------------------------------------------------

def run_exact(snaps_by_seed):
    print("\n== A. exact-info continuation (true opponent private + true seed) ==")
    seed = 101
    n_steps = 72
    for step, (env_s, obs_s) in sorted(snaps_by_seed[seed].items()):
        real = copy.deepcopy(env_s)
        recon = make_sim(
            obs_s,
            opponent_private=copy.deepcopy(env_s.state[1].observation.private),
            rng_seed=env_s.info["seed"],
        )
        assert recon.me == 0 and recon.step_i == step, (recon.me, recon.step_i, step)
        mid_day = step % 24 != 0
        if mid_day:
            hands0 = len(obs_s["farms"][0]["hands"])
            assert hands0 > 0, "mid-day probe needs hired hands to be meaningful"
        for t in range(n_steps):
            real.step([fn(real.observation(i)) for i, fn in enumerate(PAIR)])
            recon.step([fn(recon.observation(i)) for i, fn in enumerate(PAIR)])
            fa, fb = fingerprint(real), fingerprint(recon)
            assert fa == fb, f"state diverged at snapshot step {step}, +{t + 1}"
            if mid_day and t == 0:
                o = recon.state[0].observation
                # no daily event re-fired: hour advanced by 1, hands intact
                assert o.hour == (step % 24) + 1 and o.day == step // 24
                assert len(o.farms[0]["hands"]) == hands0
        d, h = step // 24, step % 24
        print(f"  snapshot day {d:2d} hour {h:2d}: {n_steps} locksteps bit-identical")
    print("  PASS: reconstruction is exact given full information")


# ---------------------------------------------------------------------------
# B. divergence of the default (act-time-legal) reconstruction
# ---------------------------------------------------------------------------

def run_divergence(snaps_by_seed, div_points):
    print("\n== B. money divergence, real vs default reconstruction ==")
    modes = {
        "default (empty opp, wrong seed)": lambda e, o: make_sim(o, rng_seed=WRONG_SEED),
        "+ true seed (empty opp)": lambda e, o: make_sim(o, rng_seed=e.info["seed"]),
        "+ true opp private (wrong seed)": lambda e, o: make_sim(
            o, opponent_private=copy.deepcopy(e.state[1].observation.private),
            rng_seed=WRONG_SEED),
    }
    diffs = {m: {k: [] for k in MARKS} for m in modes}
    per_point = {}
    for seed, snaps in sorted(snaps_by_seed.items()):
        for step in div_points:
            env_s, obs_s = snaps[step]
            real = roll(copy.deepcopy(env_s), PAIR)
            for mode, mk in modes.items():
                rec = roll(mk(env_s, obs_s), PAIR)
                for k in MARKS:
                    diffs[mode][k].append(abs(real[k] - rec[k]))
                if mode.startswith("default"):
                    per_point.setdefault(step, []).append(
                        [abs(real[k] - rec[k]) for k in MARKS])
    print("  per snapshot point, default mode, mean |diff| $ over "
          f"{len(snaps_by_seed)} seeds:")
    print("    point         K=2d     K=3d     K=5d")
    for step in div_points:
        rows = per_point[step]
        means = [sum(r[i] for r in rows) / len(rows) for i in range(3)]
        print(f"    d{step // 24:02d} h{step % 24:02d}   "
              + "  ".join(f"{m:7.0f}" for m in means))
    print(f"  pooled over {len(diffs[next(iter(modes))][48])} points "
          "(mean / max |diff| $):")
    for mode in modes:
        row = "  ".join(
            f"K={k // 24}d {sum(diffs[mode][k]) / len(diffs[mode][k]):5.0f}/"
            f"{max(diffs[mode][k]):5.0f}" for k in MARKS)
        print(f"    {mode:34s} {row}")
    return diffs


# ---------------------------------------------------------------------------
# C. rank fidelity over theta-variant candidates
# ---------------------------------------------------------------------------

def run_ranks(snaps_by_seed, rank_seeds, rank_points):
    print(f"\n== C. Spearman rank fidelity over {len(VARIANTS)} theta variants ==")
    agents = [(name, build_variant(ov)) for name, ov in VARIANTS]
    rhos = {48: [], 72: []}
    print("    context            rho@K=2d  rho@K=3d")
    for seed in rank_seeds:
        for step in rank_points:
            env_s, obs_s = snaps_by_seed[seed][step]
            real_m, recon_m = {48: [], 72: []}, {48: [], 72: []}
            for _, fn in agents:
                pair = (fn, STARTER)
                r = roll(copy.deepcopy(env_s), pair, marks=(48, 72))
                c = roll(make_sim(obs_s, rng_seed=WRONG_SEED), pair, marks=(48, 72))
                for k in (48, 72):
                    real_m[k].append(r[k])
                    recon_m[k].append(c[k])
            row = []
            for k in (48, 72):
                rho = spearman(real_m[k], recon_m[k])
                rhos[k].append(rho)
                row.append(rho)
            print(f"    seed {seed} d{step // 24:02d} h{step % 24:02d}     "
                  f"{row[0]:8.3f}  {row[1]:8.3f}")
    means = {k: sum(v) / len(v) for k, v in rhos.items()}
    print(f"    mean (n={len(rhos[48])})          "
          f"{means[48]:8.3f}  {means[72]:8.3f}")
    return rhos


# ---------------------------------------------------------------------------
# D. robustness sweep
# ---------------------------------------------------------------------------

def run_robustness():
    print("\n== D. robustness: 30 reconstruction points, 5-day rollouts ==")
    points, failures = 0, 0
    for seed in (11, 12):
        steps = list(range(30, 30 + 45 * 15, 45))  # hours 6,3,0,21,18,15,12,...
        snaps = collect_snapshots(seed, steps)
        for step in steps:
            _, obs_s = snaps[step]
            try:
                sim = make_sim(obs_s, rng_seed=7)
                rw = sim.run_steps(PAIR, 120)
                assert all(math.isfinite(v) for v in rw), rw
                points += 1
            except Exception as ex:  # noqa: BLE001 - counted, then re-raised via assert
                failures += 1
                print(f"    FAIL seed {seed} step {step}: {ex!r}")
    print(f"  {points} points OK, {failures} exceptions "
          f"(days 1-28, hours 0/3/6/9/12/15/18/21)")
    assert failures == 0 and points >= 30
    print("  PASS")


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    lat = run_latency()

    div_points = [step_of(5, 0), step_of(10, 0), step_of(12, 13),
                  step_of(15, 0), step_of(20, 0)]
    seeds = [101, 202, 303, 404]
    snaps_by_seed = {s: collect_snapshots(s, div_points) for s in seeds}

    run_exact(snaps_by_seed)
    run_divergence(snaps_by_seed, div_points)
    run_ranks(snaps_by_seed, rank_seeds=[101, 202],
              rank_points=[step_of(10, 0), step_of(12, 13), step_of(15, 0)])
    run_robustness()

    print(f"\nall checks passed in {time.time() - t_start:.1f}s "
          f"(make_sim warm {lat['warm_ms']:.2f} ms)")


if __name__ == "__main__":
    main()
