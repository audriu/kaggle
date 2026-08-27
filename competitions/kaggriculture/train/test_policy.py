"""agent/policy.py with default theta must reproduce agent/main.py exactly.

Same seeds, same opponents: episode rewards must be equal to the cent. Any drift
means a refactor bug, and training would start from a different (unvalidated) agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import policy  # noqa: E402
from sim.fastenv import FastEnv, load_agent  # noqa: E402

BASE = str(Path(__file__).resolve().parent.parent / "agent" / "main.py")


def main():
    baseline = load_agent(BASE)
    para = policy.build(policy.default_theta())
    starter = load_agent("starter")

    fails = 0
    for opp_name, opp in [("starter", starter), ("baseline", baseline)]:
        for seed in (1, 3, 42, 777, 100123):
            r_base = FastEnv(seed).run([baseline, opp])
            r_para = FastEnv(seed).run([para, opp])
            same = r_base == r_para
            fails += not same
            print(f"[{'OK ' if same else 'FAIL'}] vs {opp_name:9s} seed={seed:<7} "
                  f"baseline={r_base}  param={r_para}")
    # Sanity: a non-default theta must actually change behaviour.
    t = policy.default_theta()
    t[policy.theta_names().index("goose_target")] = 0.0
    differs = FastEnv(3).run([policy.build(t), starter]) != FastEnv(3).run([para, starter])
    print(f"[{'OK ' if differs else 'FAIL'}] perturbed theta changes behaviour")
    fails += not differs
    # theta v2: a perturbed appended param changes behaviour, and a short pre-v2
    # theta pads to the exact default behaviour (full wiring in train/test_theta_v2.py).
    t = policy.default_theta()
    t[policy.theta_names().index("sell_cap_egg")] = 1.0
    v2_differs = FastEnv(3).run([policy.build(t), starter]) != FastEnv(3).run([para, starter])
    print(f"[{'OK ' if v2_differs else 'FAIL'}] perturbed v2 tail param changes behaviour")
    fails += not v2_differs
    short = policy.build(policy.default_theta()[:47])  # 47 = pre-v2 schema length
    short_same = FastEnv(3).run([short, starter]) == FastEnv(3).run([para, starter])
    print(f"[{'OK ' if short_same else 'FAIL'}] short 47-dim theta pads to default behaviour")
    fails += not short_same
    print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
