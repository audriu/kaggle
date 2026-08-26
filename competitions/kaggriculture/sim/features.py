"""State featurisation for the value net: observation -> flat float vector.

The week-4 stage (research plan section 6.4a) regresses (state features, day) ->
final money and uses the net to truncate inference-time rollouts. That means this
module will eventually run *inside the submitted agent* on Kaggle, so it obeys two
hard constraints:

  1. Information hygiene: it reads ONLY what the agent legitimately sees at act
     time -- the observation dict FastEnv.observation(i) / Kaggle hand the agent:
     own farm + private shed/seeds/inventories, the shared market and town, and
     the opponent's PUBLIC farm (farms[] is public for both players; the
     opponent's `private` is not present and is never touched).
  2. Zero dependencies and zero mutation: stdlib-only (so scripts/export_policy.py
     can inline it into dist/main.py), and strictly read-only on the observation --
     FastEnv hands out the LIVE state object, so a single stray write would corrupt
     the episode (train/test_features.py deep-compares before/after to enforce this).

The vector is ~112 floats in a fixed order (FEATURE_NAMES is the parallel name
list): time header, own-farm tile sweep (counts by crop/animal, pending yield,
unwatered/unfed), private shed/seeds/carried per item, market prices + inventory
deviation from I0=10000 for all 9 priced items, town shop demand per product, and
the opponent's public mirror (money, tile counts, pending yield). Raw counts and
dollars on purpose -- the trainer standardises from train-split stats, and keeping
units physical makes the learned weights auditable. Two 10x10 tile sweeps per call
(~200 tile visits) cost microseconds next to the 1 s/turn Kaggle budget.
"""

import math

# Kept in sync with kaggle_environments.envs.kaggriculture (same tables policy.py
# copies). PRODUCTS order is the env's canonical one; it fixes the feature order.
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
CARRY_ITEMS = PRODUCTS + ANIMALS  # everything a shed/inventory slot can hold

MARKET_I0 = 10000  # shared starting inventory; price deviations hinge on inv-I0
TURNS_PER_DAY = 24
SEASON_STEPS = 720.0

# Town shop demand table (env SHOPS). A single-product shop consumes 2 units per
# tick, multi-product shops 1 of each -- we fold that multiplier into the count so
# `town_<item>` is "units demanded per shop tick", a direct scarcity-pressure signal
# (town demand drains inventory and pushes prices UP, e.g. STRAWBERRY 120->308).
SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
TOWN_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]

# ---------------------------------------------------------------------------
# Feature order. Built once at import; extract() appends in exactly this order.
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    # time / own header
    ["day", "hour", "season_frac", "money", "log_money",
     "hands", "hires_today", "quadrants"]
    # own farm sweep
    + ["empty", "weeds", "locked"]
    + [f"crop_{c}" for c in CROPS]
    + [f"crop_yield_{c}" for c in CROPS]
    + [f"crop_dry_{c}" for c in CROPS]
    + ["coop_empty", "pasture_empty"]
    + [f"animal_{a}" for a in ANIMALS]
    + [f"animal_yield_{a}" for a in ANIMALS]
    + [f"animal_unfed_{a}" for a in ANIMALS]
    + ["fert_ready", "uncared"]
    # own private
    + [f"shed_{it}" for it in CARRY_ITEMS]
    + [f"seeds_{c}" for c in CROPS]
    + [f"carried_{it}" for it in CARRY_ITEMS]
    # shared market
    + [f"price_{it}" for it in PRODUCTS]
    + [f"mkt_sold_{it}" for it in PRODUCTS]   # inventory - I0: positive = glut
    # shared town
    + ["shops"]
    + [f"town_{it}" for it in TOWN_PRODUCTS]
    # opponent public mirror
    + ["opp_money", "opp_log_money", "opp_hands", "opp_hires_today",
       "opp_quadrants", "opp_empty", "opp_weeds"]
    + [f"opp_crop_{c}" for c in CROPS]
    + [f"opp_animal_{a}" for a in ANIMALS]
    + ["opp_crop_yield", "opp_animal_yield"]
)

N_FEATURES = len(FEATURE_NAMES)

_CROP_IDX = {c: i for i, c in enumerate(CROPS)}
_ANIMAL_IDX = {a: i for i, a in enumerate(ANIMALS)}


def _sweep(farm):
    """One read-only pass over a farm's tiles -> aggregate counters.

    Works on either farm (tiles are public for both players). Tiles are None
    (empty unlocked), "LOCKED", or a dict/Struct with a "kind" -- same idioms as
    policy._scan so a Kaggle Struct and a plain dict both work.
    """
    n = len(CROPS)
    m = len(ANIMALS)
    s = {
        "empty": 0, "weeds": 0, "locked": 0,
        "crop_n": [0] * n, "crop_yield": [0] * n, "crop_dry": [0] * n,
        "coop_empty": 0, "pasture_empty": 0,
        "animal_n": [0] * m, "animal_yield": [0] * m, "animal_unfed": [0] * m,
        "fert_ready": 0, "uncared": 0,
    }
    for row in farm.get("tiles") or []:
        for t in row:
            if t is None:
                s["empty"] += 1
                continue
            if not isinstance(t, dict):        # "LOCKED"
                s["locked"] += 1
                continue
            kind = t.get("kind")
            if kind == "WEED":
                s["weeds"] += 1
            elif kind == "PLANT":
                i = _CROP_IDX.get(t.get("crop"))
                if i is None:
                    continue
                s["crop_n"][i] += 1
                s["crop_yield"][i] += t.get("yield_units", 0) or 0
                if not t.get("watered_today"):
                    s["crop_dry"][i] += 1
            elif kind in ("COOP", "PASTURE"):
                animal = t.get("animal")
                if not animal:
                    s["coop_empty" if kind == "COOP" else "pasture_empty"] += 1
                    continue
                j = _ANIMAL_IDX.get(animal)
                if j is None:
                    continue
                s["animal_n"][j] += 1
                s["animal_yield"][j] += t.get("yield_units", 0) or 0
                if not t.get("fed_today"):
                    s["animal_unfed"][j] += 1
                if not t.get("cared_today"):
                    s["uncared"] += 1
                if t.get("fertilizer_available"):
                    s["fert_ready"] += 1
    return s


def extract(obs):
    """Observation dict (as FastEnv.observation(i) / Kaggle deliver it) -> list
    of N_FEATURES floats, ordered exactly as FEATURE_NAMES. Read-only."""
    player = obs.get("player", 0)
    farms = obs.get("farms") or []
    farm = farms[player] if player < len(farms) else {}
    opp = farms[1 - player] if len(farms) > 1 and 1 - player < len(farms) else {}
    private = obs.get("private") or {}
    market = obs.get("market") or {}
    town = obs.get("town") or {}
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)

    f = []

    # --- time / own header ---
    money = farm.get("money", 0.0) or 0.0
    f.append(float(day))
    f.append(float(hour))
    f.append((day * TURNS_PER_DAY + hour) / SEASON_STEPS)
    f.append(float(money))
    f.append(math.log1p(max(0.0, float(money))))
    f.append(float(len(farm.get("hands") or [])))
    f.append(float(farm.get("hires_today", 0) or 0))
    f.append(float(len(farm.get("unlocked_quadrants") or [])))

    # --- own farm sweep ---
    s = _sweep(farm)
    f.append(float(s["empty"]))
    f.append(float(s["weeds"]))
    f.append(float(s["locked"]))
    f.extend(float(v) for v in s["crop_n"])
    f.extend(float(v) for v in s["crop_yield"])
    f.extend(float(v) for v in s["crop_dry"])
    f.append(float(s["coop_empty"]))
    f.append(float(s["pasture_empty"]))
    f.extend(float(v) for v in s["animal_n"])
    f.extend(float(v) for v in s["animal_yield"])
    f.extend(float(v) for v in s["animal_unfed"])
    f.append(float(s["fert_ready"]))
    f.append(float(s["uncared"]))

    # --- own private ---
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    f.extend(float(shed.get(it, 0) or 0) for it in CARRY_ITEMS)
    f.extend(float(seeds.get(c, 0) or 0) for c in CROPS)
    carried = {}
    for inv in private.get("inventories") or []:
        if not isinstance(inv, dict):
            continue
        for it, q in inv.items():
            carried[it] = carried.get(it, 0) + (q or 0)
    f.extend(float(carried.get(it, 0)) for it in CARRY_ITEMS)

    # --- shared market ---
    prices = market.get("prices") or {}
    inventory = market.get("inventory") or {}
    f.extend(float(prices.get(it, 0) or 0) for it in PRODUCTS)
    f.extend(float((inventory.get(it, MARKET_I0) or 0) - MARKET_I0) for it in PRODUCTS)

    # --- shared town ---
    unlocked = town.get("unlocked_shops") or []
    f.append(float(len(unlocked)))
    demand = {}
    for shop in unlocked:
        prods = SHOPS.get(shop)
        if not prods:
            continue
        mult = 2 if len(prods) == 1 else 1
        for it in prods:
            demand[it] = demand.get(it, 0) + mult
    f.extend(float(demand.get(it, 0)) for it in TOWN_PRODUCTS)

    # --- opponent public mirror ---
    o_money = opp.get("money", 0.0) or 0.0
    f.append(float(o_money))
    f.append(math.log1p(max(0.0, float(o_money))))
    f.append(float(len(opp.get("hands") or [])))
    f.append(float(opp.get("hires_today", 0) or 0))
    f.append(float(len(opp.get("unlocked_quadrants") or [])))
    so = _sweep(opp)
    f.append(float(so["empty"]))
    f.append(float(so["weeds"]))
    f.extend(float(v) for v in so["crop_n"])
    f.extend(float(v) for v in so["animal_n"])
    f.append(float(sum(so["crop_yield"])))
    f.append(float(sum(so["animal_yield"])))

    return f
