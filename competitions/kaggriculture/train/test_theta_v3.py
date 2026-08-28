"""Theta v3 sanity: append-only schema, default equivalence, live wiring.

Mirrors train/test_theta_v2.py for the v3 tail (sell_cap_milk/wool/straw,
feed_buy_target/max_price, feed_stop_day, sheep_per_yarn, fert_ongoing,
hire_ramp_day/hire_d0b). Checks:
  (a) first 64 PARAMS entries identical to git HEAD's policy.py; 10 appended
  (b) full-default rewards == agent/main.py rewards (3 seeds, both seats)
  (c) 64-dim theta pads to the same actions as the full default (100 steps)
  (d) every v3 param, perturbed to an active value, diverges a probe episode's
      action trace (no dead wiring), incl. a mechanism check: feed_stop_day=22
      must produce zero FEED actions from day 22 in a ranch episode
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

N_V2 = 64
V3 = [("sell_cap_milk", 2.0), ("sell_cap_wool", 2.0), ("sell_cap_straw", 2.0),
      ("feed_buy_target", 40.0), ("feed_buy_max_price", 100.0),
      ("feed_stop_day", 22.0), ("sheep_per_yarn", 4.0), ("fert_ongoing", 1.0),
      ("hire_ramp_day", 2.0), ("hire_d0b", 14.0)]
# feed_buy_max_price alone is inert (feed_buy_target defaults 0); perturb it
# together with an activating partner and assert the PAIR differs from the
# partner alone (proves the price guard is read).
PAIRED = {"feed_buy_max_price": ("feed_buy_target", 40.0),
          "hire_d0b": ("hire_ramp_day", 2.0)}


def head_params():
    src = subprocess.run(
        ["git", "show", "HEAD:competitions/kaggriculture/agent/policy.py"],
        capture_output=True, text=True, cwd=ROOT).stdout
    ns = {"__name__": "head_policy"}
    head = {}
    for line in src.splitlines():
        pass
    exec(compile(src, "head_policy.py", "exec"), ns)
    return ns["PARAMS"]


def trace(theta, seed, steps=680, opp="starter"):
    fn = policy.build(list(theta))
    opp_fn = load_agent(opp)
    env = FastEnv(seed)
    acts = []
    for _ in range(steps):
        a = fn(env.observation(0))
        acts.append(a)
        env.step([a, opp_fn(env.observation(1))])
        if env.done:
            break
    return acts, env.rewards()


def with_over(base, pairs):
    names = policy.theta_names()
    th = list(base)
    for k, v in pairs:
        th[names.index(k)] = v
    return policy.clip_theta(th)


def main():
    # (a) append-only vs HEAD
    old = head_params()
    assert len(policy.PARAMS) == N_V2 + len(V3), (
        f"PARAMS {len(policy.PARAMS)}, want {N_V2}+{len(V3)}")
    for i in range(min(N_V2, len(old))):
        assert tuple(policy.PARAMS[i]) == tuple(old[i]), f"entry {i} changed: {policy.PARAMS[i]}"
    print(f"[OK ] (a) first {N_V2} entries identical to HEAD; {len(V3)} appended")

    # (b) default == baseline heuristic, episode rewards
    base = load_agent(str(ROOT / "agent" / "main.py"))
    for seed in (21, 22, 23):
        for seat in (0, 1):
            pair_b = [base, load_agent("starter")]
            pair_p = [policy.agent, load_agent("starter")]
            if seat == 1:
                pair_b.reverse()
                pair_p.reverse()
            rb = FastEnv(seed).run(pair_b)
            rp = FastEnv(seed).run(pair_p)
            assert rb == rp, f"seed {seed} seat {seat}: {rb} != {rp}"
    print("[OK ] (b) default theta == agent/main.py rewards, 3 seeds x 2 seats")

    # (c) 64-dim pad
    a64, _ = trace(policy.default_theta()[:N_V2], 31, steps=100)
    afull, _ = trace(policy.default_theta(), 31, steps=100)
    assert a64 == afull, "64-dim padded theta diverged from full default"
    print("[OK ] (c) 64-dim theta pads to identical actions")

    # (d) live wiring: perturb each v3 param on a RANCH base so the touched
    # subsystems (milk/wool sales, feeding, sheep, hiring) are actually active.
    from train.ranch_theta import CORE, RANCH
    from train.exploiters import make_theta
    ranch = make_theta({**CORE, **RANCH["ranch_straw"]})
    ref = {}
    for seed in (41, 42, 43):
        ref[seed] = trace(ranch, seed)[0]
    # Per-item caps: unit-level check on _market directly (an episode probe can
    # miss a cap that never binds on a weak base bot — milk/wool did bind, straw
    # trickles < cap). Shed 50 of the item, floor 0 -> order qty must equal cap.
    for item, cap_name in (("MILK", "sell_cap_milk"), ("WOOL", "sell_cap_wool"),
                           ("STRAWBERRY", "sell_cap_straw")):
        p = dict(zip(policy.theta_names(),
                     with_over(policy.default_theta(), [(cap_name, 3.0)])))
        for k in ("sell_floor_milk", "sell_floor_wool", "sell_floor_straw"):
            p[k] = 0.0
        scan = {k: [] for k in ("water", "harvest", "feed", "care", "fert",
                                "weeds", "empty", "empty_coop", "empty_pasture")}
        scan.update(n_geese=0, n_cows=0, n_sheep=0, n_wheat=0, n_carrot=0)
        obs = {"day": 5, "hour": 3, "market": {"prices": {item: 100},
                                               "inventory": {}}}
        orders = policy._market(p, obs, {"money": 0, "hires_today": 9},
                                {"shed": {item: 50}, "seeds": {}}, scan,
                                {"wheat_target": 0})
        sells = [o for o in orders if o[0] == "SELL" and o[1] == item]
        assert sells and sells[0][2] == 3, f"{cap_name}: expected qty 3, got {sells}"
        print(f"[OK ] (d) {cap_name} caps a 50-unit shed to 3/turn (unit-level)")

    # fert_ongoing: unit-level (a probe episode rarely parks a fertilizer-carrying
    # unit on an aged strawberry). Unit on a STRAWBERRY tile, watered, carrying
    # 1 FERTILIZER: gate closed at default, FERTILIZE when >= 0.5.
    for flag, expect in ((0.0, False), (1.0, True)):
        p = dict(zip(policy.theta_names(),
                     with_over(policy.default_theta(), [("fert_ongoing", flag)])))
        tile = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0,
                "watered_today": True, "yield_units": 0, "fertilized_until_day": -1}
        farm = {"tiles": [[None] * 10 for _ in range(10)], "money": 0}
        farm["tiles"][2][2] = tile
        scan = {k: [] for k in ("water", "harvest", "feed", "care", "fert",
                                "weeds", "empty", "empty_coop", "empty_pasture")}
        scan.update(n_geese=0, n_cows=0, n_sheep=0, n_wheat=0, n_carrot=0)
        act = policy._unit(p, (2, 2), 0, farm,
                           {"shed": {}, "seeds": {}, "inventories": [{"FERTILIZER": 1}]},
                           10, 3, scan, set(), {"wheat_target": 0})
        got = act == ["FERTILIZE"]
        assert got == expect, f"fert_ongoing={flag}: action {act}"
    print("[OK ] (d) fert_ongoing gates FERTILIZE on strawberries (unit-level)")

    for name, val in V3:
        if name.startswith("sell_cap_") or name == "fert_ongoing":
            continue  # covered by the unit-level checks above
        pairs = [(name, val)]
        base_pairs = []
        if name in PAIRED:
            base_pairs = [PAIRED[name]]
            pairs = base_pairs + pairs
        base_acts = {s: (trace(with_over(ranch, base_pairs), s)[0] if base_pairs else ref[s])
                     for s in ref}
        diverged = False
        for seed in ref:
            if trace(with_over(ranch, pairs), seed)[0] != base_acts[seed]:
                diverged = True
                break
        assert diverged, f"{name}={val}: no behavioural effect on any probe seed"
        print(f"[OK ] (d) {name} wired (diverges probe episodes)")

    # mechanism: feed_stop_day=22 -> no FEED from day 22
    acts, _ = trace(with_over(ranch, [("feed_stop_day", 22.0)]), 41, steps=719)
    for step, a in enumerate(acts):
        if step // 24 >= 22:
            units = [a.get("farmer", [])] + list(a.get("hands", []))
            assert not any(u and u[0] == "FEED" for u in units), f"FEED at step {step}"
    print("[OK ] (d) feed_stop_day mechanism: zero FEED actions from day 22")

    print("\nPASS")


if __name__ == "__main__":
    main()
