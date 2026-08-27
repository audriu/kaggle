"""Sanity suite for agent/search.py (the theta-candidate rollout search layer).

Four checks, single process (a 16-proc CEM run owns the box; procs <= 4 applies
-- timings printed here are 2-4x pessimistic):

  A. SCHEMA: the search agent returns schema-valid actions on 100 consecutive
     steps from episode start (dict; farmer = list of str+args; one hands entry
     per hired hand; market = list of <= 10 orders) and the two search turns in
     that window (day 3 h0 = step 72, day 4 h0 = step 96) actually ran and
     scored candidates.

  B. FALLBACK PROVEN: a poisoned value_fn (raises on every call) must leave the
     trajectory BIT-IDENTICAL to the plain policy.build(theta) agent -- same
     action dict every step, same rewards -- because every search turn falls
     back to the base action. Also asserts the failure ledger: exactly
     max_failures failures, then search disabled for the rest of the episode
     (no further search attempts show up in the state).

  C. LATENCY: non-search turns must stay well inside the 1 s actTimeout --
     median < 50 ms asserted (measured on this loaded box: ~0.1-0.5 ms).

  D. NO STATE BLEED: run two episodes back-to-back through ONE closure (arena
     workers cache and reuse closures across episodes). The second episode must
     produce exactly the same action stream and rewards as a FRESH closure
     playing that episode, and the per-episode state (telemetry, searches,
     acting theta) must have been reset -- not accumulated. Uses a generous
     turn budget (5 s) so the wall-clock early-stop cannot make candidate
     counts -- and hence trajectories -- timing-dependent on a loaded box.

Run:  python train/test_search.py    (from competitions/kaggriculture)
"""

import statistics as stats
import sys
import time
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP))

from agent import policy  # noqa: E402
from agent.search import DEFAULT_CFG, build_search_agent  # noqa: E402
from agent.vnet_infer import ValueFn  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

try:
    from agent.vnet_weights import VNET_B64  # noqa: E402
except ImportError:
    sys.exit("agent/vnet_weights.py missing (gitignored, generated) -- run "
             "scripts/export_vnet.py first")

STARTER = load_agent("starter")
THETA = policy.default_theta()
VALUE_FN = ValueFn.from_embedded(VNET_B64)

UNIT_OPS = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "DIG",
            "PLANT", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER", "DROP",
            "PICKUP", "PLACE", "BUILD_COOP", "BUILD_PASTURE"}
MARKET_OPS = {"SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "HIRE", "BUY_LAND"}


def check_action(action, obs):
    """Assert the action dict has the shape the engine's interpreter expects."""
    assert isinstance(action, dict), type(action)
    farmer = action.get("farmer")
    hands = action.get("hands")
    market = action.get("market")
    assert isinstance(farmer, list) and farmer and farmer[0] in UNIT_OPS, farmer
    assert isinstance(hands, list), hands
    n_hands = len(obs["farms"][obs.get("player", 0)].get("hands") or [])
    assert len(hands) == n_hands, (len(hands), n_hands)
    for h in hands:
        assert isinstance(h, list) and h and h[0] in UNIT_OPS, h
    assert isinstance(market, list) and len(market) <= 10, market
    for order in market:
        assert isinstance(order, list) and order and order[0] in MARKET_OPS, order


def run_lockstep(agent_fn, seed, episode_steps=720, n_steps=None, validate=False,
                 timings=None):
    """Run agent_fn (seat 0) vs starter, capturing seat-0 actions per step."""
    env = FastEnv(seed, episode_steps=episode_steps)
    actions = []
    for _ in range(n_steps if n_steps is not None else episode_steps - 1):
        obs = env.observation(0)
        t0 = time.perf_counter()
        a0 = agent_fn(obs)
        if timings is not None:
            timings.append((env.step_i, time.perf_counter() - t0))
        if validate:
            check_action(a0, obs)
        actions.append(a0)
        env.step([a0, STARTER(env.observation(1))])
        if env.done:
            break
    return actions, env.rewards()


# ---------------------------------------------------------------------------

def check_schema_and_latency():
    print("== A. schema-valid actions, 100 consecutive steps ==")
    ag = build_search_agent(THETA, VALUE_FN)
    timings = []
    run_lockstep(ag, seed=777, n_steps=100, validate=True, timings=timings)
    st = ag._search_state
    assert st["searches"] == 2, st["searches"]  # day 3 h0 (72), day 4 h0 (96)
    assert all(e["n_scored"] >= 1 for e in st["telemetry"])
    scored = [e["n_scored"] for e in st["telemetry"]]
    print(f"  100 steps valid; searches ran at steps 72/96, "
          f"candidates scored {scored}")

    print("== C. non-search-turn latency ==")
    search_steps = {72, 96}
    base_ms = [t * 1e3 for s, t in timings if s not in search_steps]
    search_ms = [t * 1e3 for s, t in timings if s in search_steps]
    med = stats.median(base_ms)
    print(f"  non-search turns (n={len(base_ms)}): median {med:.3f} ms  "
          f"max {max(base_ms):.1f} ms; search turns: "
          f"{', '.join(f'{t:.0f} ms' for t in search_ms)}  [loaded box]")
    assert med < 50.0, f"non-search median {med:.1f} ms >= 50 ms"
    print("  PASS")


class PoisonedValueFn:
    """Raises on every evaluation -- the fallback path must absorb it."""

    calls = 0

    def v(self, feats_row):
        PoisonedValueFn.calls += 1
        raise RuntimeError("poisoned value_fn (test)")


def check_poisoned_fallback():
    print("== B. poisoned value_fn == plain policy (fallback proven) ==")
    steps = 240  # 10 days: search turns on days 3..9, failures cap at 3
    poisoned = build_search_agent(THETA, PoisonedValueFn())
    acts_p, rew_p = run_lockstep(poisoned, seed=555, episode_steps=steps,
                                 validate=True)
    acts_b, rew_b = run_lockstep(policy.build(THETA), seed=555,
                                 episode_steps=steps)
    assert acts_p == acts_b, "poisoned-search trajectory diverged from base policy"
    assert rew_p == rew_b, (rew_p, rew_b)
    st = poisoned._search_state
    assert st["failures"] == DEFAULT_CFG["max_failures"], st["failures"]
    assert st["disabled"] is True
    assert st["searches"] == 0 and st["telemetry"] == []  # every attempt died pre-commit
    assert PoisonedValueFn.calls == DEFAULT_CFG["max_failures"]
    print(f"  {len(acts_p)} actions identical, rewards {rew_p} == {rew_b}; "
          f"{st['failures']} failures then disabled (V called "
          f"{PoisonedValueFn.calls}x)")
    print("  PASS")


def check_episode_reset():
    print("== D. winner-theta cache resets between episodes (no bleed) ==")
    steps = 240
    cfg = {"turn_budget_s": 5.0}  # early-stop cannot trigger => deterministic
    reused = build_search_agent(THETA, VALUE_FN, cfg=cfg)

    _, rew1 = run_lockstep(reused, seed=901, episode_steps=steps)
    st = reused._search_state
    ep1_searches = st["searches"]
    ep1_theta = list(st["acting_theta"])
    assert ep1_searches > 0
    dirty = ep1_theta != policy.clip_theta(THETA)
    print(f"  ep1 (seed 901): {ep1_searches} searches, "
          f"acting theta {'DID' if dirty else 'did NOT'} drift from base")

    acts_reused, rew2 = run_lockstep(reused, seed=902, episode_steps=steps)
    fresh = build_search_agent(THETA, VALUE_FN, cfg=cfg)
    acts_fresh, rew2f = run_lockstep(fresh, seed=902, episode_steps=steps)

    st = reused._search_state
    stf = fresh._search_state
    assert st["searches"] == stf["searches"], (st["searches"], stf["searches"])
    assert len(st["telemetry"]) == st["searches"], "telemetry accumulated across episodes"
    assert acts_reused == acts_fresh, "ep2 trajectory depends on ep1 state (bleed!)"
    assert rew2 == rew2f, (rew2, rew2f)
    assert st["acting_theta"] == stf["acting_theta"]
    print(f"  ep2 (seed 902) reused vs fresh closure: {len(acts_reused)} actions "
          f"identical, rewards {rew2} == {rew2f}, "
          f"{st['searches']} searches each -- state fully reset")
    print("  PASS")


def main():
    t0 = time.time()
    check_schema_and_latency()
    check_poisoned_fallback()
    check_episode_reset()
    print(f"\nall checks passed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
