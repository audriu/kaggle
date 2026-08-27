"""theta v2 wiring suite: the appended params must be live, and only when asked.

Checks, in order:
  (a) PARAMS is append-only: 47 pre-v2 entries byte-identical to the version at git
      HEAD, plus the 17 v2 entries.
  (b) default_theta() still reproduces agent/main.py episode rewards (4 seeds).
  (c) a short 47-dim theta builds an agent that matches the full default
      action-for-action over 100 FastEnv steps.
  (d) every v2 param, perturbed off neutral on its own (plus base overrides that
      make its subject exist, e.g. melons for hold_melon_until), diverges the
      action trace of a probe episode -- catches dead wiring; for hold_milk_until
      the mechanism itself is asserted: no MILK SELL order before the hold day.
  (e) clip_theta pads a short theta before clipping.

Runnable directly; no Pool, ~15 s single-process.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

N_OLD = 47   # PARAMS length before theta v2
N_NEW = 17   # appended by theta v2

# (param, probe value, base overrides applied to BOTH arms). Base overrides create
# the conditions the param acts on -- every config below was verified to diverge:
# - melon/straw holds: those crops are baseline-off AND carrot/tomato seeds outrank
#   them in _unit's planting ladder, so both gates open and the rivals shut off;
# - milk/wool holds: cows early (no goose prerequisite) / sheep from day 0;
# - floor_p2_mul: town demand keeps premium prices ABOVE every default floor
#   (milk 160->285 vs floor 40), so the multiplier can only bind against a raised
#   base floor (sell_floor_milk 1.5 -> floor 240 sits inside the price path);
# - hire_min: work (water+harvest+feed+empty//div) exceeds the hire targets on any
#   normal farm, so hire_min only binds on a bare farm with nothing to do;
# - phase2_day: behaviour-neutral alone by design, gated through a live p2 delta.
EARLY_COWS = {"cow_day": 6.0, "cow_req_geese": 0.0, "cow_cash": 400.0}
PROBES = [
    ("phase2_day",           25.0, {"wheat_base_p2_add": 8.0}),
    ("hire_min_p2_add",       6.0, {"wheat_base": 0.0, "wheat_plus": 0.0,
                                    "wheat_buy_cap": 0.0, "carrot_buy_cap": 0.0,
                                    "goose_target": 0.0, "hire_empty_div": 8.0,
                                    "land_empty_thresh": 0.0}),
    ("feed_res_mult_p2_add", -2.0, {}),
    ("goose_target_p2_add",  10.0, {}),
    ("cow_target_p2_add",    10.0, {}),
    ("wheat_base_p2_add",     8.0, {}),
    ("carrot_cap_p2_mul",     0.0, {}),
    ("tomato_seed_p2_add",    4.0, {}),
    ("floor_p2_mul",          0.0, {**EARLY_COWS, "cow_day": 2.0,
                                    "sell_floor_milk": 1.5, "wheat_base": 8.0}),
    ("sell_cap_wheat",        1.0, {}),
    ("sell_cap_egg",          1.0, {}),
    ("hold_melon_until",     20.0, {"melon_day": 4.0, "melon_seed_max": 4.0,
                                    "carrot_buy_cap": 0.0}),
    ("hold_straw_until",     20.0, {"straw_day": 4.0, "straw_seed_max": 4.0,
                                    "carrot_buy_cap": 0.0, "tomato_seed_max": 0.0}),
    ("hold_milk_until",      25.0, EARLY_COWS),
    ("hold_wool_until",      20.0, {"sheep_day": 0.0, "sheep_target": 2.0,
                                    "sheep_cash": 500.0}),
    ("land_stop_day",         5.0, {}),
    ("hire_hour_max",         0.0, {}),
]
PROBE_SEEDS = (2, 5, 9)


def theta_with(overrides):
    theta = policy.default_theta()
    names = policy.theta_names()
    for name, v in overrides.items():
        theta[names.index(name)] = v
    return theta


def action_trace(theta, seed, starter):
    """Full per-step action list of build(theta) vs starter on one episode."""
    fn = policy.build(theta)
    env = FastEnv(seed)
    trace = []
    for _ in range(719):
        a = fn(env.observation(0))
        trace.append(a)
        env.step([a, starter(env.observation(1))])
        if env.done:
            break
    return trace


def check_params_append_only():
    src = subprocess.run(
        ["git", "show", "HEAD:competitions/kaggriculture/agent/policy.py"],
        capture_output=True, text=True, check=True, cwd=ROOT).stdout
    ns = {}
    exec(compile(src, "policy.py@HEAD", "exec"), ns)
    old = ns["PARAMS"]
    assert len(policy.PARAMS) == N_OLD + N_NEW, (
        f"PARAMS has {len(policy.PARAMS)} entries, want {N_OLD} + {N_NEW}")
    for i, (new_e, old_e) in enumerate(zip(policy.PARAMS[:N_OLD], old[:N_OLD])):
        assert new_e == old_e, f"pre-v2 PARAMS[{i}] changed: {old_e} -> {new_e}"
    print(f"[OK ] (a) {N_OLD} pre-v2 entries identical to HEAD, {N_NEW} appended")


def check_default_equivalence():
    baseline = load_agent(str(ROOT / "agent" / "main.py"))
    para = policy.build(policy.default_theta())
    starter = load_agent("starter")
    for seed in (1, 3, 42, 777):
        r_base = FastEnv(seed).run([baseline, starter])
        r_para = FastEnv(seed).run([para, starter])
        assert r_base == r_para, f"seed {seed}: baseline {r_base} != default theta {r_para}"
    print("[OK ] (b) default theta == agent/main.py episode rewards on 4 seeds")


def check_short_theta():
    starter = load_agent("starter")
    full = policy.build(policy.default_theta())
    short = policy.build(policy.default_theta()[:N_OLD])
    env = FastEnv(11)
    for step in range(100):
        obs = env.observation(0)
        a_full = full(obs)
        a_short = short(obs)
        assert a_full == a_short, f"step {step}: short {a_short} != full {a_full}"
        env.step([a_full, starter(env.observation(1))])
    print("[OK ] (c) 47-dim theta == full default, action-for-action over 100 steps")


def check_probes():
    starter = load_agent("starter")
    cache = {}   # (theta tuple, seed) -> trace

    def trace(theta, seed):
        key = (tuple(theta), seed)
        if key not in cache:
            cache[key] = action_trace(theta, seed, starter)
        return cache[key]

    fails = 0
    for name, value, base in PROBES:
        t_base = theta_with(base)
        t_probe = theta_with({**base, name: value})
        diverged = any(trace(t_base, s) != trace(t_probe, s) for s in PROBE_SEEDS)
        print(f"[{'OK ' if diverged else 'FAIL'}] (d) {name}={value} diverges a probe episode")
        fails += not diverged
    return fails


def check_hold_milk_mechanism():
    starter = load_agent("starter")

    def milk_sell_days(theta):
        fn = policy.build(theta)
        env = FastEnv(3)
        days = []
        for _ in range(719):
            obs = env.observation(0)
            a = fn(obs)
            days += [obs.get("day", 0) for o in a["market"] if o[:2] == ["SELL", "MILK"]]
            env.step([a, starter(env.observation(1))])
            if env.done:
                break
        return days

    free = milk_sell_days(theta_with(EARLY_COWS))
    assert free and min(free) < 25, f"probe episode has no early milk sales ({free[:5]}...)"
    held = milk_sell_days(theta_with({**EARLY_COWS, "hold_milk_until": 25.0}))
    assert held, "hold_milk_until=25 stopped milk sales entirely (dump_day should fire)"
    assert min(held) >= 25, f"MILK SELL issued on day {min(held)} despite hold until 25"
    print(f"[OK ] (d) mechanism: hold_milk_until=25 delays first MILK SELL "
          f"day {min(free)} -> {min(held)}")


def check_clip_pads():
    short = policy.default_theta()[:N_OLD]
    short[0] = 99.0            # hire_d0, hi = 14
    out = policy.clip_theta(short)
    assert len(out) == len(policy.PARAMS), f"clip_theta kept length {len(out)}"
    assert out[0] == 14.0, f"clip_theta did not clip padded input: {out[0]}"
    assert out[N_OLD:] == [p[1] for p in policy.PARAMS[N_OLD:]], "padded tail != defaults"
    print("[OK ] (e) clip_theta pads a short theta then clips")


def main():
    check_params_append_only()
    check_default_equivalence()
    check_short_theta()
    fails = check_probes()
    check_hold_milk_mechanism()
    check_clip_pads()
    print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
