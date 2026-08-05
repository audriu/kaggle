"""Kaggriculture heuristic agent.

Beat the built-in starter by farming many carrot tiles with hired hands,
keeping everything watered, then adding geese for stable egg income.
"""

from __future__ import annotations

CROPS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "product": "MILK"},
}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
DIRS = [("NORTH", 0, -1), ("SOUTH", 0, 1), ("EAST", 1, 0), ("WEST", -1, 0)]
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))


def _fib(n: int) -> int:
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
    """Collect tile coords by needed action."""
    need_water, need_harvest, need_feed, need_care = [], [], [], []
    need_fert, weeds, empty, empty_coop, empty_pasture = [], [], [], [], []
    n_geese = n_cows = n_wheat = n_carrot = 0

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
                if not t.get("fed_today"):
                    need_feed.append((x, y))
                elif t.get("yield_units", 0) > 0:
                    need_harvest.append((x, y))
                elif not t.get("cared_today"):
                    need_care.append((x, y))
                if t.get("fertilizer_available"):
                    need_fert.append((x, y))

    return {
        "water": need_water,
        "harvest": need_harvest,
        "feed": need_feed,
        "care": need_care,
        "fert": need_fert,
        "weeds": weeds,
        "empty": empty,
        "empty_coop": empty_coop,
        "empty_pasture": empty_pasture,
        "n_geese": n_geese,
        "n_cows": n_cows,
        "n_wheat": n_wheat,
        "n_carrot": n_carrot,
    }


def _claim(targets, claimed):
    for t in sorted(targets, key=lambda p: (p[0] + p[1])):
        if t not in claimed:
            claimed.add(t)
            return t
    return None


def _unit(pos, role, farm, private, day, hour, scan, claimed, goals):
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
        # Prefer build when animals are waiting in shed/inventory.
        geese_wait = shed.get("GOOSE", 0) + inv.get("GOOSE", 0)
        cows_wait = shed.get("COW", 0) + inv.get("COW", 0)
        if geese_wait > len(scan["empty_coop"]) and goals.get("allow_build"):
            claimed.add((fx, fy))
            return ["BUILD_COOP"]
        if cows_wait > len(scan["empty_pasture"]) and goals.get("allow_build"):
            claimed.add((fx, fy))
            return ["BUILD_PASTURE"]
        # Plant
        want_wheat = scan["n_wheat"] < goals["wheat_target"]
        if want_wheat and seeds.get("WHEAT", 0) > 0 and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "WHEAT"]
        if seeds.get("CARROT", 0) > 0 and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "CARROT"]
        if seeds.get("WHEAT", 0) > 0 and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "WHEAT"]
        if day >= 10 and seeds.get("TOMATO", 0) > 0 and (fx, fy) not in claimed:
            claimed.add((fx, fy))
            return ["PLANT", "TOMATO"]

    # Dump cargo at shed (except feed wheat / animals being placed).
    if at_shed and inv:
        keep = set()
        if goals.get("need_feed_wheat"):
            keep.add("WHEAT")
        for a in ANIMALS:
            if inv.get(a, 0):
                keep.add(a)
        if any(k not in keep for k in inv) or inv.get("WHEAT", 0) > 3:
            # DROP dumps everything — only if nothing critical, else PLACE items.
            if not any(inv.get(a, 0) for a in ANIMALS) and not (
                goals.get("need_feed_wheat") and inv.get("WHEAT", 0)
            ):
                return ["DROP"]

    # Pickup at shed when needed.
    if at_shed:
        if goals.get("need_feed_wheat") and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0:
            return ["PICKUP", "WHEAT", min(5, shed["WHEAT"])]
        for animal in ("GOOSE", "COW"):
            if shed.get(animal, 0) > 0 and inv.get(animal, 0) == 0:
                if animal == "GOOSE" and scan["empty_coop"]:
                    return ["PICKUP", animal, 1]
                if animal == "COW" and scan["empty_pasture"]:
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
    # Place animals onto empty structures.
    if inv.get("GOOSE", 0) > 0:
        priority_lists.append(scan["empty_coop"])
    if inv.get("COW", 0) > 0:
        priority_lists.append(scan["empty_pasture"])
    # Build then plant.
    geese_wait = shed.get("GOOSE", 0) + inv.get("GOOSE", 0)
    cows_wait = shed.get("COW", 0) + inv.get("COW", 0)
    if geese_wait > len(scan["empty_coop"]):
        priority_lists.append(scan["empty"])
    elif cows_wait > len(scan["empty_pasture"]):
        priority_lists.append(scan["empty"])
    elif any(seeds.get(c, 0) > 0 for c in ("CARROT", "WHEAT", "TOMATO")):
        priority_lists.append(scan["empty"])

    # Go to shed for pickups.
    if (goals.get("need_feed_wheat") and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0) or (
        shed.get("GOOSE", 0) > 0 and scan["empty_coop"] and inv.get("GOOSE", 0) == 0
    ):
        target = min(SHED_TILES, key=lambda p: abs(p[0] - fx) + abs(p[1] - fy))
        move = _step_toward(fx, fy, target[0], target[1])
        return [move] if move else ["PASS"]

    for lst in priority_lists:
        # Prefer nearest unclaimed
        options = [p for p in lst if p not in claimed]
        if not options:
            continue
        tx, ty = min(options, key=lambda p: abs(p[0] - fx) + abs(p[1] - fy))
        claimed.add((tx, ty))
        if (fx, fy) == (tx, ty):
            return ["PASS"]
        move = _step_toward(fx, fy, tx, ty)
        return [move] if move else ["PASS"]

    move = _step_toward(fx, fy, 4, 4)
    return [move] if move else ["PASS"]


def _market(obs, farm, private, scan):
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    prices = (obs.get("market") or {}).get("prices") or {}
    shed = dict(private.get("shed") or {})
    seeds = private.get("seeds") or {}
    money = farm["money"]
    orders = []

    # Sell produce (keep wheat reserve for feed).
    animals = scan["n_geese"] + scan["n_cows"]
    wheat_reserve = animals * 2 + 2
    for item, qty in list(shed.items()):
        if qty <= 0 or item in ANIMALS:
            continue
        sell_qty = qty
        if item == "WHEAT":
            sell_qty = max(0, qty - wheat_reserve)
        if sell_qty <= 0:
            continue
        # Always liquidate staples/eggs; hold premium only if price not crashed.
        price = prices.get(item, 0)
        if item in ("WHEAT", "CARROT", "TOMATO", "EGG", "FERTILIZER"):
            orders.append(["SELL", item, sell_qty])
        elif price >= 40 or day >= 27:
            orders.append(["SELL", item, sell_qty])

    # Hire early each day.
    if hour <= 8:
        hires = farm.get("hires_today", 0)
        target = 8 if day < 10 else 10
        work = len(scan["water"]) + len(scan["harvest"]) + len(scan["feed"]) + max(0, len(scan["empty"]) // 2)
        target = min(target, max(4, work))
        while hires < target:
            cost = _fib(hires)
            if money < cost + 100:
                break
            orders.append(["HIRE"])
            money -= cost
            hires += 1

    # Expand land when crowded.
    unlocked = set(farm.get("unlocked_quadrants") or [])
    if len(scan["empty"]) <= 6:
        for quad, price in zip(LAND_ORDER, LAND_PRICES):
            if quad in unlocked:
                continue
            if money >= price + 600:
                orders.append(["BUY_LAND"])
                money -= price
            break

    # Animals only when previous purchases are already placed on the farm.
    geese_waiting = shed.get("GOOSE", 0)
    cows_waiting = shed.get("COW", 0)
    geese_total = scan["n_geese"] + geese_waiting
    cows_total = scan["n_cows"] + cows_waiting
    if (
        day >= 4
        and geese_waiting == 0
        and geese_total < 6
        and money >= 500
        and scan["n_carrot"] + len(scan["empty"]) >= 8
    ):
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        money -= 300
        geese_total += 1
    if (
        day >= 14
        and cows_waiting == 0
        and cows_total < 3
        and geese_total >= 4
        and scan["n_geese"] >= 3
        and money >= 900
    ):
        orders.append(["BUY_ANIMAL", "COW", 1])
        money -= 400

    # Seeds for empty land.
    wheat_target = max(3, animals + 2)
    need_wheat = max(0, wheat_target - scan["n_wheat"] - seeds.get("WHEAT", 0))
    need_carrot = max(0, len(scan["empty"]) - need_wheat - seeds.get("CARROT", 0))
    for _ in range(min(need_wheat, 6)):
        if money < 40:
            break
        orders.append(["BUY_SEED", "WHEAT", 1])
        money -= 10
    for _ in range(min(need_carrot, 10)):
        if money < 50:
            break
        orders.append(["BUY_SEED", "CARROT", 1])
        money -= 20
    if day >= 10 and seeds.get("TOMATO", 0) < 2 and money >= 200 and len(scan["empty"]) > 4:
        orders.append(["BUY_SEED", "TOMATO", 1])
        money -= 50

    # Emergency feed wheat from market.
    if animals > 0 and shed.get("WHEAT", 0) + seeds.get("WHEAT", 0) + scan["n_wheat"] == 0:
        if money >= 50:
            orders.append(["BUY_PRODUCT", "WHEAT", max(2, animals)])

    return orders[:10]


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
    animals = scan["n_geese"] + scan["n_cows"]
    goals = {
        "wheat_target": max(3, animals + 2),
        "need_feed_wheat": len(scan["feed"]) > 0,
        "allow_build": True,
    }

    market = _market(obs, farm, private, scan)
    claimed = set()

    farmer = _unit(
        tuple(farm["farmer"]), 0, farm, private, day, hour, scan, claimed, goals
    )
    hands = [
        _unit(tuple(h), i + 1, farm, private, day, hour, scan, claimed, goals)
        for i, h in enumerate(farm.get("hands") or [])
    ]
    return {"farmer": farmer, "hands": hands, "market": market}
