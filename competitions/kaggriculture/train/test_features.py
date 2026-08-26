"""Sanity checks for sim/features.py: shape, finiteness, and NO observation mutation.

The mutation check is the one that matters: FastEnv hands agents the LIVE state
object (that is where its 16x speedup comes from), so a featuriser that writes even
one key -- e.g. a defaulting `setdefault` -- would silently corrupt every episode
train/collect.py plays. We deep-copy the observation before extract() and deep-compare
after; Struct subclasses dict, so == recurses by value through the whole tree.

Run directly: python train/test_features.py
"""

import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import features  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

CHECK_STEPS = {0, 1, 5, 23, 24, 25, 40, 49}  # spans a day boundary (hour 23 -> 0)


def check_obs(env, player, step_i):
    obs = env.observation(player)
    before = copy.deepcopy(obs)
    vec = features.extract(obs)

    assert isinstance(vec, list), f"extract returned {type(vec)}"
    assert len(vec) == len(features.FEATURE_NAMES), (
        f"step {step_i} p{player}: {len(vec)} values vs "
        f"{len(features.FEATURE_NAMES)} names")
    for name, v in zip(features.FEATURE_NAMES, vec):
        assert isinstance(v, float), f"{name} is {type(v).__name__}, not float"
        assert math.isfinite(v), f"{name} is not finite: {v}"
    assert obs == before, f"extract() MUTATED the observation at step {step_i} p{player}"
    return vec


def main():
    names = features.FEATURE_NAMES
    assert len(names) == len(set(names)), "FEATURE_NAMES has duplicates"
    assert len(names) == features.N_FEATURES

    agent = load_agent(str(ROOT / "agent" / "main.py"))
    env = FastEnv(seed=7)
    checked = 0
    last_vec = None
    for step_i in range(50):
        if step_i in CHECK_STEPS:
            for player in (0, 1):
                last_vec = check_obs(env, player, step_i)
                checked += 1
        env.step([agent(env.observation(i)) for i in (0, 1)])
        assert not env.done, f"episode ended prematurely at step {step_i}"

    # Spot-check a few values against what 50 steps of baseline play must look like.
    v = dict(zip(names, last_vec))
    assert v["day"] == 2.0 and v["hour"] == 1.0, (v["day"], v["hour"])
    # locked tiles and unlocked-quadrant count must agree (25 tiles per quadrant)
    assert v["locked"] == 25.0 * (4 - v["quadrants"]), (v["locked"], v["quadrants"])
    assert v["money"] > 0.0
    assert v["mkt_sold_WHEAT"] <= 0.0              # town drains, day-2 sales tiny

    print(f"ok: {features.N_FEATURES} features, {checked} obs checked over 50 steps, "
          f"no mutation, all finite")
    sample = ["day", "money", "crop_WHEAT", "crop_CARROT", "shed_WHEAT",
              "price_CARROT", "mkt_sold_CARROT", "opp_money", "opp_crop_CARROT"]
    print("  " + "  ".join(f"{k}={v[k]:.0f}" for k in sample))


if __name__ == "__main__":
    main()
