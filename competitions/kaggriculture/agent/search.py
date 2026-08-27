"""Inference-time theta-candidate rollout search wrapped around agent/policy.py.

The week-3 stage (research plan section 5 / section 8): the CEM-trained theta is a
fixed season plan, blind to the actual market/opponent it is facing. This layer
re-plans once per day at act time: on a SEARCH TURN it proposes a handful of
CANDIDATE THETAS (small named overrides on the currently acting theta, same spirit
as train/exploiters.py), scores each by reconstructing a forward simulator from
the live observation (agent/reconstruct.make_sim), rolling K days ahead with the
candidate steering our seat and the engine's starter agent in the opponent seat,
and evaluating the end state with the frozen numpy value net
(agent/vnet_infer.ValueFn, V = predicted final money). The winner becomes the
acting theta until the next search turn. Between search turns the agent is
exactly policy.build(acting_theta) -- microseconds per call.

Why theta-candidates and not action search: branching is 1e24+/turn, but the
strategy space is ~46 numbers and rollouts of theta variants rank near-perfectly
under reconstruction (Spearman rho 1.000 at K=2 days vs real rollouts,
train/test_reconstruct.py C), so searching in theta space buys plan adaptivity at
StarCraft-scale branching for the price of ~15 rollouts.

Budget discipline (Kaggle: actTimeout 1 s/turn, remainingOverageTime 60 s/episode):
search fires at most once per day (default hour 0, days 3..27 => <= 25 turns), each
rollout is individually timed and the loop stops as soon as the next rollout (est.
= max observed so far this turn) would overflow cfg["turn_budget_s"] (default
0.7 s < 1 s, so a nominal search turn consumes ZERO overage). Search is skipped
outright once obs remainingOverageTime falls to cfg["overage_floor"].

Safety: the whole search path is inside try/except -- any exception or a hard
budget overrun (> turn_budget_s * overrun_factor) returns the base action, counts
one failure, and after cfg["max_failures"] failures search is disabled for the
rest of the episode. If remainingOverageTime is missing from the obs the episode
budget cannot be policed, so search hard-disables. An exception never propagates
to Kaggle (train/test_search.py B proves a value_fn that raises leaves a
trajectory bit-identical to the plain policy).

Measured (2026-08-27, box fully loaded by the 16-proc cem3 run => 2-4x
pessimistic; scripts/package_search.py timing episode, cem2 theta, seed 31337
vs starter, reward 46711): all 25 search turns scored 15-16 of the 16 library
candidates (mean 15.4; the shortfall is the dedup skip, not the budget) at
~17 ms/rollout; search-turn wall clock mean 256 ms / max 339 ms vs the 700 ms
budget; over all 719 turns median 0.13 ms, p99 264 ms, max 339 ms, total
overage consumed 0.00 s of 60 s.

Effectiveness A/B (2026-08-27, sim.arena, Pool(4) on the loaded box, 40 fresh
seeds 700000+, both seats, SAME cem2-best theta in both arms, opponents
starter + baseline): NULL RESULT -- search-on 46,779 +- 597 SE vs search-off
47,191 +- 929 SE; paired per-(seed,seat,opponent) diff -412 +- 807 SE pooled
(vs baseline -1,282 +- 1,045; vs starter +458 +- 1,223; n=160). Both arms win
100% of matches, so against these weak opponents the theta is already at its
income ceiling and V-noise-driven switches (~13/episode, val MAE ~$3.1k vs
min_gain $150) only reshuffle outcomes (on-arm sd 7.6k vs off-arm 11.7k).
Reported faithfully per plan discipline; NOT tuned post-hoc on these seeds.
Untested-here follow-ups: harder opponents (crasher/mirror), higher min_gain.

Independent verification A/B (2026-08-27, adversarial re-run on DISJOINT fresh
seeds 810000-810039, same protocol, n=160/arm): paired diff -2,010 +- 713 SE
(vs baseline -2,800 +- 1,057; vs starter -1,220 +- 949; both arms 100% win;
on-arm sd 7,669 vs off-arm 8,878). Same sign, now ~2.8 SE below zero -- on
these seeds the search is significantly NEGATIVE, not merely null. Verdict
strengthened: do NOT submit dist/main_search.py over dist/main_cem2.py with
this theta + value net (V noise, val MAE ~$3.1k, vs min_gain $150 makes the
~13 switches/episode net-harmful against opponents both arms already beat).

Per-episode state lives in the closure's `st` dict and resets whenever
day*24+hour goes backwards (new episode -- arena workers reuse one closure across
episodes). Exposed as agent._search_state for tests/telemetry.

Dual-mode on purpose: imported as agent.search in the repo it pulls its
dependencies from agent/ and sim/; inlined into dist/main_search.py by
scripts/package_search.py the guarded import block is skipped because make_sim
et al. are already defined earlier in the flat file (which is exec'd with no
__file__).
"""

from __future__ import annotations

import copy
import time

from kaggle_environments.envs.kaggriculture import kaggriculture as K

if "make_sim" not in globals():
    # Dev mode: running as agent/search.py inside the repo. In the packaged
    # single file every name below is already defined above this section, so
    # this block is skipped and __file__ is never touched.
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from agent.policy import build, clip_theta, theta_names  # noqa: E402,F401
    from agent.reconstruct import make_sim  # noqa: E402,F401
    from agent.vnet_infer import ValueFn  # noqa: E402,F401
    from sim.features import extract  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Candidate library. Plain data on purpose: a later CEM stage can tune the
# override magnitudes (or learn per-day libraries) without touching the search
# loop. Each entry is (name, ops); ops are (param, op, value) applied to the
# CURRENT acting theta, then policy.clip_theta:
#   set   -> value                    add   -> current + value
#   mul   -> current * value          floor -> max(current, value)
#   day   -> current game day + value ("open this gate today")
# The pseudo-param "hire_bucket" resolves to hire_d{0..3} for the current day,
# matching policy._market's bucket edges (10/17/24).
# Order matters: the budget loop truncates from the END, so the historically
# strongest levers (hiring, cows -- see cem2's learned drift, plan section 8)
# come first. "noop" MUST stay first: it is always scored and is the margin
# baseline every switch is measured against.
# ---------------------------------------------------------------------------

PREMIUM_FLOOR_PARAMS = ("sell_floor_straw", "sell_floor_melon",
                        "sell_floor_milk", "sell_floor_wool")

CANDIDATES = [
    ("noop", ()),
    ("hire_up", (("hire_bucket", "add", 2.0),)),
    ("hire_dn", (("hire_bucket", "add", -2.0),)),
    ("cow_now", (("cow_day", "day", 0.0), ("cow_target", "add", 1.0))),
    ("goose_now", (("goose_day", "day", 0.0), ("goose_target", "add", 1.0))),
    ("melon_now", (("melon_day", "day", 0.0), ("melon_seed_max", "floor", 2.0))),
    ("straw_now", (("straw_day", "day", 0.0), ("straw_seed_max", "floor", 2.0))),
    ("land_loose", (("land_cash_buffer", "add", 600.0),)),
    ("land_tight", (("land_cash_buffer", "add", -300.0),)),
    ("floors_up", tuple((k, "mul", 1.5) for k in PREMIUM_FLOOR_PARAMS)),
    ("floors_dn", tuple((k, "mul", 0.5) for k in PREMIUM_FLOOR_PARAMS)),
    ("dump_now", (("dump_day", "day", 0.0),)),
    ("dump_later", (("dump_day", "add", 2.0),)),
    ("sheep_now", (("sheep_day", "day", 0.0), ("sheep_target", "add", 1.0))),
    ("animals_defer", (("goose_day", "set", 31.0), ("cow_day", "set", 31.0),
                       ("sheep_day", "set", 31.0))),
    ("premium_close", (("melon_day", "set", 31.0), ("straw_day", "set", 31.0))),
]

DEFAULT_CFG = {
    "k_days": 2,            # rollout horizon; rho 1.000 vs real at K=2 (test_reconstruct C)
    "turn_budget_s": 0.7,   # < 1 s actTimeout: a nominal search turn eats no overage
    "min_gain": 150.0,      # $ margin over noop's V required to switch (tie -> noop)
    "overage_floor": 15.0,  # stop searching when remainingOverageTime <= this
    "day_min": 3,           # nothing to re-plan before the farm exists
    "day_max": 27,          # a 2-day rollout past day 27 runs off the season
    "search_hour": 0,       # one search per day, at the daily re-plan boundary
    "max_failures": 3,      # failures (exception/overrun) before disabling search
    "overrun_factor": 2.0,  # elapsed > budget*this counts as a failure
    "rng_salt": 977,        # rng_seed = salt + day: fixed WITHIN a turn (common
                            # random numbers across candidates), varies across days
}

_HIRE_BUCKET_PARAMS = ("hire_d0", "hire_d1", "hire_d2", "hire_d3")


def _hire_param(day):
    return _HIRE_BUCKET_PARAMS[0 if day < 10 else (1 if day < 17 else (2 if day < 24 else 3))]


def resolve_candidate(acting_theta, ops, day, name_idx):
    """Apply one candidate's ops to `acting_theta` -> clipped theta list, or
    None when the result is identical to the acting theta (a free skip: no
    point spending a rollout to compare a plan against itself)."""
    theta = list(acting_theta)
    for pname, op, val in ops:
        if pname == "hire_bucket":
            pname = _hire_param(day)
        i = name_idx[pname]
        if op == "set":
            theta[i] = val
        elif op == "add":
            theta[i] = theta[i] + val
        elif op == "mul":
            theta[i] = theta[i] * val
        elif op == "floor":
            theta[i] = max(theta[i], val)
        elif op == "day":
            theta[i] = float(day) + val
        else:
            raise ValueError(f"unknown candidate op {op!r}")
    theta = clip_theta(theta)
    return None if theta == list(acting_theta) else theta


def _merge_cfg(cfg):
    out = dict(DEFAULT_CFG)
    for k, v in (cfg or {}).items():
        if k not in out:
            raise KeyError(f"unknown search cfg key {k!r}")
        out[k] = v
    return out


_PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def build_search_agent(theta, value_fn, cfg=None):
    """Wrap policy.build(theta) with once-a-day theta-candidate rollout search.

    theta     -- base theta (clipped on entry); also the reset state each episode
    value_fn  -- agent/vnet_infer.ValueFn (its .v is used) or any callable
                 feats_row -> $ (what train/test_search.py's poisoned check uses)
    cfg       -- dict of DEFAULT_CFG overrides (unknown keys raise)

    Returns agent(obs) -> action dict. The closure's per-episode state is
    exposed as agent._search_state (telemetry: one dict per search turn).
    """
    cfg = _merge_cfg(cfg)
    base_theta = clip_theta(list(theta))
    name_idx = {n: i for i, n in enumerate(theta_names())}
    score_fn = value_fn.v if hasattr(value_fn, "v") else value_fn
    starter = K.agents["starter"]
    n_steps = int(cfg["k_days"]) * 24

    st = {}

    def _reset():
        st.update(
            last_step=-1,          # detects the next episode: step goes backwards
            acting_theta=list(base_theta),
            acting_agent=build(base_theta),
            failures=0,
            disabled=False,
            searches=0,
            telemetry=[],          # one dict per search turn, capped at 64
        )

    _reset()

    def _search_turn(obs, day):
        """Score candidates by (reconstruct -> K-day rollout -> V); switch the
        acting theta only on a clear (>= min_gain) win over the noop."""
        t0 = time.perf_counter()
        me = int(obs.get("player", 0) or 0)
        budget = float(cfg["turn_budget_s"])
        # ONE reconstruction per turn; candidates deepcopy it. Fixed rng_seed =
        # common random numbers: every candidate faces the same imagined weeds.
        base_sim = make_sim(obs, rng_seed=int(cfg["rng_salt"]) + day)
        acting = st["acting_theta"]
        scored, times = [], []
        for name, ops in CANDIDATES:
            if name == "noop":
                cand = list(acting)     # noop is unconditional: margin baseline
            else:
                cand = resolve_candidate(acting, ops, day, name_idx)
                if cand is None:
                    continue            # identical to acting theta
                est = max(times) if times else budget
                if (time.perf_counter() - t0) + est > budget:
                    break               # next rollout would blow the budget
            t1 = time.perf_counter()
            sim = copy.deepcopy(base_sim)
            pair = [None, None]
            pair[me] = build(cand)
            pair[1 - me] = starter
            sim.run_steps(pair, n_steps)
            v = float(score_fn(extract(sim.observation(me))))
            times.append(time.perf_counter() - t1)
            scored.append((name, cand, v))
        elapsed = time.perf_counter() - t0

        noop_v = scored[0][2]
        best_name, best_theta_c, best_v = max(scored, key=lambda c: c[2])
        entry = {"day": day, "n_scored": len(scored), "elapsed_s": elapsed,
                 "noop_v": noop_v, "best": best_name, "best_v": best_v,
                 "switched": False, "overrun": False}
        if elapsed > budget * float(cfg["overrun_factor"]):
            # The box is far slower than planned: keep the base action, burn a
            # failure so a chronically slow worker stops searching entirely.
            entry["overrun"] = True
            st["failures"] += 1
        elif best_name != "noop" and best_v >= noop_v + float(cfg["min_gain"]):
            st["acting_theta"] = list(best_theta_c)
            st["acting_agent"] = build(best_theta_c)
            entry["switched"] = True
        st["searches"] += 1
        if len(st["telemetry"]) < 64:
            st["telemetry"].append(entry)
        return entry["switched"] and not entry["overrun"]

    def agent(obs):
        try:
            day = int(obs.get("day", 0) or 0)
            hour = int(obs.get("hour", 0) or 0)
            step = day * 24 + hour
            if step < st["last_step"]:
                _reset()               # new episode on a reused closure
            st["last_step"] = step
            base_action = st["acting_agent"](obs)
        except Exception:
            return dict(_PASS_ACTION)  # even the base policy failed: stay legal

        try:
            if st["disabled"] or st["failures"] >= int(cfg["max_failures"]):
                return base_action
            if hour != int(cfg["search_hour"]) or not (
                int(cfg["day_min"]) <= day <= int(cfg["day_max"])
            ):
                return base_action
            overage = obs.get("remainingOverageTime", None)
            if overage is None:
                st["disabled"] = True  # cannot police the 60 s episode budget
                return base_action
            if float(overage) <= float(cfg["overage_floor"]):
                return base_action
            if _search_turn(obs, day):
                return st["acting_agent"](obs)   # act on the winner immediately
            return base_action
        except Exception:
            st["failures"] += 1
            if st["failures"] >= int(cfg["max_failures"]):
                st["disabled"] = True
            return base_action

    agent._search_state = st
    agent._search_cfg = cfg
    return agent
