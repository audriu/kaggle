"""Ranch-meta thetas: the top-8 ladder strategy expressed in theta-v2 dials.

Source: notes/meta_report.md (32 replays, top 8 teams, all playing RANCH+GARDEN:
9 cows + 4-6 sheep from day 0, zero geese, 12-14 re-hired hands daily, 30-40
strawberry tiles, melon one-shot ~d10, late wheat ramp, inventory-guarded
milk/wool pacing, NE d6 / SW d10 / SE never). §5 of the report maps each element
to dials; the NEEDS-CODE items (per-item milk/wool sell caps, JIT feed buying,
town/shop awareness, feed_stop_day, strawberry FERTILIZE) are NOT expressible yet
— these thetas measure how far the current policy family gets.

Measured (train/ranch_theta.py --eval, virgin seeds 950000+, both seats — see
the table printed by --eval and the numbers recorded in the research plan).

Usage:
  python train/ranch_theta.py --eval --seeds 30 --procs 16
  python train/ranch_theta.py --list
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from train.exploiters import make_theta  # noqa: E402

# Common ranch core per meta_report §5 (all within PARAMS bounds).
CORE = {
    # 1 — ranch-first opening; geese stay dead
    "sheep_day": 0.0, "sheep_target": 5.0, "sheep_cash": 500.0,
    "cow_day": 0.0, "cow_target": 9.0, "cow_req_geese": 0.0, "cow_cash": 400.0,
    "goose_target": 0.0,
    # 2 — the daily temp army (CARE + fertilizer collection come free via _unit)
    "hire_d0": 10.0, "hire_d1": 14.0, "hire_d2": 14.0, "hire_d3": 13.0,
    "hire_min": 8.0, "hire_empty_div": 1.0, "hire_cash_buffer": 0.0,
    # 3 — inventory-guard selling (floors ARE inventory stops on the glut slope)
    "sell_floor_milk": 0.60, "sell_floor_wool": 0.55, "sell_cap": 8.0,
    "dump_day": 28.0,
    # 4 — strawberry engine + melon one-shot + late wheat ramp; no carrot/tomato
    "straw_day": 4.0, "straw_seed_max": 12.0,
    "melon_day": 0.0, "melon_seed_max": 12.0, "hold_melon_until": 10.0,
    "plant_stop_day": 24.0,
    "wheat_per_animal": 3.0, "wheat_plus": 8.0, "wheat_base_p2_add": 8.0,
    "sell_floor_wheat": 0.0,   # early wheat IS the cash engine; the $40 premium arrives on its own
    "carrot_buy_cap": 0.0, "tomato_day": 31.0,
    # feed reserve sized for ~14 animals (no JIT buying dial yet)
    "feed_res_mult": 2.0, "feed_res_base": 4.0,
    # land: cheap trigger, low buffer; stop before SE's $4k window
    "land_empty_thresh": 10.0, "land_cash_buffer": 100.0, "land_stop_day": 14.0,
}

RANCH = {
    "ranch_v1": {},                                            # the report verbatim
    "ranch_lean": {"sheep_target": 4.0, "hire_d0": 6.0,        # slower opening
                   "hire_d1": 12.0, "hire_min": 4.0},
    "ranch_cows": {"sheep_target": 4.0, "cow_target": 10.0},   # yarn-less tilt
    "ranch_straw": {"straw_day": 3.0, "wheat_plus": 4.0,       # berry emphasis
                    "sell_cap": 6.0},
    "ranch_se": {"land_stop_day": 31.0},                       # allow SE quadrant
    "ranch_slowherd": {"sheep_cash": 1500.0, "cow_cash": 900.0},  # throttle the buy-pickup churn
    "ranch_wheatfloor": {"sell_floor_wheat": 0.9},             # mild floor once premium exists
}


def get_ranch_thetas():
    return [(name, make_theta({**CORE, **ov})) for name, ov in RANCH.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=950000)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--names", default=None, help="comma list; default all")
    args = ap.parse_args()

    if args.list or not args.eval:
        for name, ov in RANCH.items():
            print(f"{name}: CORE + {ov}")
        if not args.eval:
            return

    from multiprocessing import Pool
    from sim.arena import evaluate, file_spec, builtin_spec, params_spec
    from train.exploiters import get_exploiters

    cem4 = file_spec(ROOT / "dist" / "main_cem4.py", "cem4")
    crash = params_spec("agent.policy:build", dict(get_exploiters())["crasher"], "crasher")
    opps = [cem4, builtin_spec("starter"), crash]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    names = args.names.split(",") if args.names else list(RANCH)

    with Pool(args.procs) as p:
        for name in names:
            theta = make_theta({**CORE, **RANCH[name]})
            cand = params_spec("agent.policy:build", theta, name)
            rep = evaluate(cand, opps, seeds, pool=p)
            print(f"== {name}")
            print(rep)
            mirror = evaluate(cand, [cand], seeds, pool=p)
            print(f"    mirror income {mirror.mean:8.0f} +- {mirror.se:.0f} SE")


if __name__ == "__main__":
    main()
