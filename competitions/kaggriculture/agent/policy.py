"""Parameterised kaggriculture policy — the thing the trainer learns.

This is `agent/main.py` refactored so that every strategic constant is an entry in a
flat theta vector: hire schedule, portfolio targets and start days, land triggers,
sell thresholds, feed reserves. `build(theta)` returns an `agent(obs)` closure; the
default theta reproduces the baseline heuristic decision-for-decision (verified by
episode-reward equality in train/test_policy.py), so training starts from known-good
behaviour rather than from noise.

It also exposes capabilities the baseline never uses -- melon/strawberry planting,
sheep, fertilizer purchase -- gated behind params whose defaults disable them, so the
optimiser can discover them without any code change.

Self-contained on purpose: scripts/export_policy.py concatenates this file with a
trained theta to produce the submitted main.py.
"""

from __future__ import annotations

CROPS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL"},
}
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
              "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
DIRS = [("NORTH", 0, -1), ("SOUTH", 0, 1), ("EAST", 1, 0), ("WEST", -1, 0)]
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))

# name, init, sigma (CEM exploration scale), lo, hi
# Inits replicate agent/main.py exactly; params marked NEW unlock behaviour the
# baseline doesn't have, at settings that keep it disabled.
PARAMS = [
    # hiring (baseline: target = 8 if day < 10 else 10, floor 4, work counts empty//2)
    ("hire_d0",            8.0, 2.0, 0.0, 14.0),   # days 0-9
    ("hire_d1",           10.0, 2.0, 0.0, 14.0),   # days 10-16
    ("hire_d2",           10.0, 2.0, 0.0, 14.0),   # days 17-23
    ("hire_d3",           10.0, 2.0, 0.0, 14.0),   # days 24-29
    ("hire_min",           4.0, 1.0, 0.0, 10.0),
    ("hire_empty_div",     2.0, 0.7, 1.0, 8.0),
    ("hire_cash_buffer", 100.0, 50., 0.0, 1000.0),
    # planting portfolio
    ("wheat_base",         3.0, 1.0, 0.0, 12.0),
    ("wheat_per_animal",   1.0, 0.3, 0.0, 3.0),
    ("wheat_plus",         2.0, 1.0, 0.0, 8.0),
    ("wheat_buy_cap",      6.0, 2.0, 0.0, 20.0),
    ("carrot_buy_cap",    10.0, 3.0, 0.0, 30.0),
    ("tomato_day",        10.0, 3.0, 0.0, 31.0),
    ("tomato_seed_max",    2.0, 1.0, 0.0, 10.0),
    ("melon_day",         31.0, 4.0, 0.0, 31.0),   # NEW; 31 = never
    ("melon_seed_max",     0.0, 1.5, 0.0, 12.0),   # NEW
    ("straw_day",         31.0, 4.0, 0.0, 31.0),   # NEW; 31 = never
    ("straw_seed_max",     0.0, 1.5, 0.0, 12.0),   # NEW
    ("plant_stop_day",    31.0, 2.0, 20.0, 31.0),  # NEW: no one-time planting after this day
    # animals (baseline: geese from day 4 target 6; cows day 14 target 3 if 4+ geese)
    ("goose_day",          4.0, 1.5, 0.0, 31.0),
    ("goose_target",       6.0, 2.0, 0.0, 40.0),
    ("goose_cash",       500.0, 100., 300.0, 3000.0),
    ("goose_min_tiles",    8.0, 2.0, 0.0, 30.0),
    ("cow_day",           14.0, 3.0, 0.0, 31.0),
    ("cow_target",         3.0, 1.5, 0.0, 20.0),
    ("cow_req_geese",      4.0, 1.5, 0.0, 15.0),
    ("cow_cash",         900.0, 150., 400.0, 4000.0),
    ("sheep_day",         31.0, 4.0, 0.0, 31.0),   # NEW; 31 = never
    ("sheep_target",       0.0, 1.5, 0.0, 20.0),   # NEW
    ("sheep_cash",      1100.0, 200., 500.0, 4000.0),  # NEW
    # land (baseline: buy when empty<=6, keeping $600)
    ("land_empty_thresh",  6.0, 2.0, 0.0, 25.0),
    ("land_cash_buffer", 600.0, 200., 0.0, 5000.0),
    # selling (baseline: staples always; premium when price>=40 or day>=27)
    ("sell_floor_wheat",   0.0, 0.15, 0.0, 1.5),   # fraction of base price
    ("sell_floor_carrot",  0.0, 0.15, 0.0, 1.5),
    ("sell_floor_tomato",  0.0, 0.15, 0.0, 1.5),
    ("sell_floor_egg",     0.0, 0.15, 0.0, 1.5),
    ("sell_floor_fert",    0.0, 0.15, 0.0, 1.5),
    ("sell_floor_straw", 0.3333, 0.15, 0.0, 1.5),  # 40/120
    ("sell_floor_melon",  0.16, 0.15, 0.0, 1.5),   # 40/250
    ("sell_floor_milk",   0.25, 0.15, 0.0, 1.5),   # 40/160
    ("sell_floor_wool",    0.2, 0.15, 0.0, 1.5),   # 40/200
    ("sell_cap",         999.0, 100., 1.0, 999.0), # units per item per turn
    ("dump_day",          27.0, 1.5, 20.0, 31.0),
    # feed reserve (baseline: animals*2 + 2)
    ("feed_res_mult",      2.0, 0.5, 0.0, 6.0),
    ("feed_res_base",      2.0, 1.0, 0.0, 10.0),
    # fertilizer purchase (NEW; baseline never buys)
    ("fert_buy_max",       0.0, 1.0, 0.0, 10.0),
    ("fert_cash",        400.0, 100., 100.0, 2000.0),
]

SELL_FLOOR_KEY = {
    "WHEAT": "sell_floor_wheat", "CARROT": "sell_floor_carrot", "TOMATO": "sell_floor_tomato",
    "EGG": "sell_floor_egg", "FERTILIZER": "sell_floor_fert", "STRAWBERRY": "sell_floor_straw",
    "MELON": "sell_floor_melon", "MILK": "sell_floor_milk", "WOOL": "sell_floor_wool",
}


def default_theta():
    return [p[1] for p in PARAMS]


def theta_names():
    return [p[0] for p in PARAMS]


def clip_theta(theta):
    return [min(max(v, lo), hi) for v, (_, _, _, lo, hi) in zip(theta, PARAMS)]


def sigmas():
    return [p[2] for p in PARAMS]


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _tiles(farm):
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            yield x, y, t


def _step_toward(fx, fy, tx, ty, size=10):
    if (fx, fy) == (tx, ty):
        return None
    best, best_d = None, abs(fx - tx) + abs(fy - ty)
    for name, dx, dy in DIRS:
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < size and 0 <= ny < size:
            d = abs(nx - tx) + abs(ny - ty)
            if d < best_d:
                best, best_d = name, d
    return best


def _inv(private, idx):
    invs = private.get("inventories") or []
    return invs[idx] if idx < len(invs) else {}


def _scan(farm, day):
    """Collect tile coords by needed action. Mirrors agent/main.py exactly."""
    need_water, need_harvest, need_feed, need_care = [], [], [], []
    need_fert, weeds, empty, empty_coop, empty_pasture = [], [], [], [], []
    n_geese = n_cows = n_sheep = n_wheat = n_carrot = 0

    for x, y, t in _tiles(farm):
        if t is None:
            empty.append((x, y))
            continue
        if not isinstance(t, dict):
            continue
        kind = t.get("kind")
        if kind == "WEED":
            weeds.append((x, y))
        elif kind == "PLANT":
            crop = t["crop"]
            if crop == "WHEAT":
                n_wheat += 1
            elif crop == "CARROT":
                n_carrot += 1
            age = day - t["planted_day"]
            info = CROPS.get(crop)
            if info:
                if info["ongoing"]:
                    if t.get("yield_units", 0) > 0:
                        need_harvest.append((x, y))
                    elif not t.get("watered_today"):
                        need_water.append((x, y))
                else:
                    if age >= info["max_yield_day"]:
                        need_harvest.append((x, y))
                    elif not t.get("watered_today"):
                        need_water.append((x, y))
                    elif age >= info["first_yield_day"] and t.get("yield_units", 0) >= (
                        4 if crop == "WHEAT" else 3
                    ):
                        need_harvest.append((x, y))
        elif kind in ("COOP", "PASTURE"):
            animal = t.get("animal")
            if not animal:
                (empty_coop if kind == "COOP" else empty_pasture).append((x, y))
            else:
                if animal == "GOOSE":
                    n_geese += 1
                elif animal == "COW":
                    n_cows += 1
                elif animal == "SHEEP":
                    n_sheep += 1
                if not t.get("fed_today"):
                    need_feed.append((x, y))
                elif t.get("yield_units", 0) > 0:
                    need_harvest.append((x, y))
                elif not t.get("cared_today"):
                    need_care.append((x, y))
                if t.get("fertilizer_available"):
                    need_fert.append((x, y))

    return {
        "water": need_water, "harvest": need_harvest, "feed": need_feed,
        "care": need_care, "fert": need_fert, "weeds": weeds, "empty": empty,
        "empty_coop": empty_coop, "empty_pasture": empty_pasture,
        "n_geese": n_geese, "n_cows": n_cows, "n_sheep": n_sheep,
        "n_wheat": n_wheat, "n_carrot": n_carrot,
    }


def _unit(p, pos, role, farm, private, day, hour, scan, claimed, goals):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    inv = _inv(private, role)
    at_shed = (fx, fy) in SHED_TILES

    # --- act on current tile ---
    if isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "WEED":
            return ["DIG"]
        if kind == "PLANT":
            crop = tile["crop"]
            info = CROPS.get(crop)
            age = day - tile["planted_day"]
            if info:
                if info["ongoing"] and tile.get("yield_units", 0) > 0:
                    return ["HARVEST"]
                if not info["ongoing"] and age >= info["max_yield_day"]:
                    return ["HARVEST"]
            if not tile.get("watered_today"):
                return ["WATER"]
            if (
                inv.get("FERTILIZER", 0) > 0
                and info
                and not info["ongoing"]
                and tile.get("fertilized_until_day", -1) < day
                and age >= (info["max_yield_day"] + 1) // 2
            ):
                return ["FERTILIZE"]
        if tile.get("animal"):
            if not tile.get("fed_today") and inv.get("WHEAT", 0) > 0:
                return ["FEED"]
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"]
            if not tile.get("cared_today"):
                return ["CARE"]
            if tile.get("fertilizer_available"):
                return ["COLLECT_FERTILIZER"]
        if kind in ("COOP", "PASTURE") and not tile.get("animal"):
            for animal, meta in ANIMALS.items():
                if meta["structure"] == kind and inv.get(animal, 0) > 0:
                    return ["PLACE", animal]

    if tile is None:
        geese_wait = shed.get("GOOSE", 0) + inv.get("GOOSE", 0)
        cows_wait = shed.get("COW", 0) + inv.get("COW", 0)
        sheep_wait = shed.get("SHEEP", 0) + inv.get("SHEEP", 0)
        if geese_wait > len(scan["empty_coop"]) and goals.get("allow_build"):
            claimed.add((fx, fy))
            return ["BUILD_COOP"]
        if cows_wait + sheep_wait > len(scan["empty_pasture"]) and goals.get("allow_build"):
            claimed.add((fx, fy))
            return ["BUILD_PASTURE"]
        # Plant
        want_wheat = scan["n_wheat"] < goals["wheat_target"]
        one_time_ok = day < p["plant_stop_day"]
        if want_wheat and seeds.get("WHEAT", 0) > 0 and one_time_ok and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "WHEAT"]
        if seeds.get("CARROT", 0) > 0 and one_time_ok and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "CARROT"]
        if seeds.get("MELON", 0) > 0 and one_time_ok and day >= p["melon_day"] and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "MELON"]
        if seeds.get("WHEAT", 0) > 0 and one_time_ok and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "WHEAT"]
        if day >= p["tomato_day"] and seeds.get("TOMATO", 0) > 0 and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "TOMATO"]
        if day >= p["straw_day"] and seeds.get("STRAWBERRY", 0) > 0 and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "STRAWBERRY"]

    # Dump cargo at shed (except feed wheat / animals being placed).
    if at_shed and inv:
        keep = set()
        if goals.get("need_feed_wheat"):
            keep.add("WHEAT")
        for a in ANIMALS:
            if inv.get(a, 0):
                keep.add(a)
        if any(k not in keep for k in inv) or inv.get("WHEAT", 0) > 3:
            if not any(inv.get(a, 0) for a in ANIMALS) and not (
                goals.get("need_feed_wheat") and inv.get("WHEAT", 0)
            ):
                return ["DROP"]

    # Pickup at shed when needed.
    if at_shed:
        if goals.get("need_feed_wheat") and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0:
            return ["PICKUP", "WHEAT", min(5, shed["WHEAT"])]
        for animal in ("GOOSE", "COW", "SHEEP"):
            if shed.get(animal, 0) > 0 and inv.get(animal, 0) == 0:
                if animal == "GOOSE" and scan["empty_coop"]:
                    return ["PICKUP", animal, 1]
                if animal in ("COW", "SHEEP") and scan["empty_pasture"]:
                    return ["PICKUP", animal, 1]
        if shed.get("FERTILIZER", 0) > 0 and inv.get("FERTILIZER", 0) == 0 and scan["water"]:
            return ["PICKUP", "FERTILIZER", 1]

    # Navigate to work.
    priority_lists = [
        scan["feed"] if inv.get("WHEAT", 0) > 0 else [],
        scan["harvest"],
        scan["water"],
        scan["care"],
        scan["fert"],
        scan["weeds"],
    ]
    if inv.get("GOOSE", 0) > 0:
        priority_lists.append(scan["empty_coop"])
    if inv.get("COW", 0) > 0 or inv.get("SHEEP", 0) > 0:
        priority_lists.append(scan["empty_pasture"])
    geese_wait = shed.get("GOOSE", 0) + inv.get("GOOSE", 0)
    pasture_wait = (shed.get("COW", 0) + inv.get("COW", 0)
                    + shed.get("SHEEP", 0) + inv.get("SHEEP", 0))
    if geese_wait > len(scan["empty_coop"]):
        priority_lists.append(scan["empty"])
    elif pasture_wait > len(scan["empty_pasture"]):
        priority_lists.append(scan["empty"])
    elif any(seeds.get(c, 0) > 0 for c in ("CARROT", "WHEAT", "TOMATO", "MELON", "STRAWBERRY")):
        priority_lists.append(scan["empty"])

    if (goals.get("need_feed_wheat") and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0) or (
        shed.get("GOOSE", 0) > 0 and scan["empty_coop"] and inv.get("GOOSE", 0) == 0
    ):
        target = min(SHED_TILES, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        move = _step_toward(fx, fy, target[0], target[1])
        return [move] if move else ["PASS"]

    for lst in priority_lists:
        options = [t for t in lst if t not in claimed]
        if not options:
            continue
        tx, ty = min(options, key=lambda t: abs(t[0] - fx) + abs(t[1] - fy))
        claimed.add((tx, ty))
        if (fx, fy) == (tx, ty):
            return ["PASS"]
        move = _step_toward(fx, fy, tx, ty)
        return [move] if move else ["PASS"]

    move = _step_toward(fx, fy, 4, 4)
    return [move] if move else ["PASS"]


def _market(p, obs, farm, private, scan):
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    prices = (obs.get("market") or {}).get("prices") or {}
    shed = dict(private.get("shed") or {})
    seeds = private.get("seeds") or {}
    money = farm["money"]
    orders = []

    # Sell produce (keep wheat reserve for feed).
    animals = scan["n_geese"] + scan["n_cows"] + scan["n_sheep"]
    wheat_reserve = int(animals * p["feed_res_mult"] + p["feed_res_base"])
    sell_cap = int(round(p["sell_cap"]))
    for item, qty in list(shed.items()):
        if qty <= 0 or item in ANIMALS:
            continue
        sell_qty = qty
        if item == "WHEAT":
            sell_qty = max(0, qty - wheat_reserve)
        if sell_qty <= 0:
            continue
        price = prices.get(item, 0)
        floor = BASE_PRICE.get(item, 0) * p[SELL_FLOOR_KEY.get(item, "sell_floor_fert")]
        if day >= p["dump_day"]:
            orders.append(["SELL", item, sell_qty])
        elif price >= floor - 1e-9:
            orders.append(["SELL", item, min(sell_qty, sell_cap)])

    # Hire early each day.
    if hour <= 8:
        hires = farm.get("hires_today", 0)
        bucket = 0 if day < 10 else (1 if day < 17 else (2 if day < 24 else 3))
        target = int(round(p[("hire_d0", "hire_d1", "hire_d2", "hire_d3")[bucket]]))
        work = (len(scan["water"]) + len(scan["harvest"]) + len(scan["feed"])
                + max(0, int(len(scan["empty"]) // max(1, round(p["hire_empty_div"])))))
        target = min(target, max(int(round(p["hire_min"])), work))
        while hires < target:
            cost = _fib(hires)
            if money < cost + p["hire_cash_buffer"]:
                break
            orders.append(["HIRE"])
            money -= cost
            hires += 1

    # Expand land when crowded.
    unlocked = set(farm.get("unlocked_quadrants") or [])
    if len(scan["empty"]) <= p["land_empty_thresh"]:
        for quad, price in zip(LAND_ORDER, LAND_PRICES):
            if quad in unlocked:
                continue
            if money >= price + p["land_cash_buffer"]:
                orders.append(["BUY_LAND"])
                money -= price
            break

    # Animals only when previous purchases are already placed on the farm.
    geese_waiting = shed.get("GOOSE", 0)
    cows_waiting = shed.get("COW", 0)
    sheep_waiting = shed.get("SHEEP", 0)
    geese_total = scan["n_geese"] + geese_waiting
    cows_total = scan["n_cows"] + cows_waiting
    sheep_total = scan["n_sheep"] + sheep_waiting
    if (
        day >= p["goose_day"]
        and geese_waiting == 0
        and geese_total < p["goose_target"]
        and money >= p["goose_cash"]
        and scan["n_carrot"] + len(scan["empty"]) >= p["goose_min_tiles"]
    ):
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        money -= 300
        geese_total += 1
    if (
        day >= p["cow_day"]
        and cows_waiting == 0
        and cows_total < p["cow_target"]
        and geese_total >= p["cow_req_geese"]
        and scan["n_geese"] >= p["cow_req_geese"] - 1
        and money >= p["cow_cash"]
    ):
        orders.append(["BUY_ANIMAL", "COW", 1])
        money -= 400
    if (
        day >= p["sheep_day"]
        and sheep_waiting == 0
        and sheep_total < p["sheep_target"]
        and money >= p["sheep_cash"]
    ):
        orders.append(["BUY_ANIMAL", "SHEEP", 1])
        money -= 500

    # Seeds for empty land.
    wheat_target = max(p["wheat_base"], animals * p["wheat_per_animal"] + p["wheat_plus"])
    need_wheat = max(0, int(round(wheat_target)) - scan["n_wheat"] - seeds.get("WHEAT", 0))
    need_carrot = max(0, len(scan["empty"]) - need_wheat - seeds.get("CARROT", 0))
    for _ in range(min(need_wheat, int(round(p["wheat_buy_cap"])))):
        if money < 40:
            break
        orders.append(["BUY_SEED", "WHEAT", 1])
        money -= 10
    for _ in range(min(need_carrot, int(round(p["carrot_buy_cap"])))):
        if money < 50:
            break
        orders.append(["BUY_SEED", "CARROT", 1])
        money -= 20
    if day >= p["tomato_day"] and seeds.get("TOMATO", 0) < p["tomato_seed_max"] and money >= 200 and len(scan["empty"]) > 4:
        orders.append(["BUY_SEED", "TOMATO", 1])
        money -= 50
    if day >= p["melon_day"] and seeds.get("MELON", 0) < p["melon_seed_max"] and money >= 300 and len(scan["empty"]) > 2:
        orders.append(["BUY_SEED", "MELON", 1])
        money -= 80
    if day >= p["straw_day"] and seeds.get("STRAWBERRY", 0) < p["straw_seed_max"] and money >= 350 and len(scan["empty"]) > 2:
        orders.append(["BUY_SEED", "STRAWBERRY", 1])
        money -= 100

    # Fertilizer purchase (disabled at default).
    if p["fert_buy_max"] >= 1 and shed.get("FERTILIZER", 0) < p["fert_buy_max"] and money >= p["fert_cash"]:
        orders.append(["BUY_PRODUCT", "FERTILIZER", 1])

    # Emergency feed wheat from market.
    if animals > 0 and shed.get("WHEAT", 0) + seeds.get("WHEAT", 0) + scan["n_wheat"] == 0:
        if money >= 50:
            orders.append(["BUY_PRODUCT", "WHEAT", max(2, animals)])

    return orders[:10]


def build(theta):
    """Return an agent(obs) closure for this parameter vector."""
    p = dict(zip(theta_names(), clip_theta(theta)))

    def agent(obs):
        farms = obs.get("farms") or []
        player = obs.get("player", 0)
        private = obs.get("private") or {}
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farm = farms[player]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        scan = _scan(farm, day)
        animals = scan["n_geese"] + scan["n_cows"] + scan["n_sheep"]
        goals = {
            "wheat_target": max(p["wheat_base"], animals * p["wheat_per_animal"] + p["wheat_plus"]),
            "need_feed_wheat": len(scan["feed"]) > 0,
            "allow_build": True,
        }

        market = _market(p, obs, farm, private, scan)
        claimed = set()
        farmer = _unit(p, tuple(farm["farmer"]), 0, farm, private, day, hour, scan, claimed, goals)
        hands = [
            _unit(p, tuple(h), i + 1, farm, private, day, hour, scan, claimed, goals)
            for i, h in enumerate(farm.get("hands") or [])
        ]
        return {"farmer": farmer, "hands": hands, "market": market}

    return agent


agent = build(default_theta())
