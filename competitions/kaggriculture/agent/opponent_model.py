"""Infer the opponent's hidden stock from public observations.

Why: search rollouts price candidate plans on the SHARED market, and the search-v1
post-mortem (plan §9) measured that an empty opponent-private estimate collapses
candidate rank fidelity against heavy sellers (Spearman ρ 0.087 worst cell, mirror
play, K=2d) while the true private restores ρ=1.000. The true private is hidden --
but nearly all of it leaves a public footprint:

- Their farm tiles are fully public, including `yield_units`: a plant/animal whose
  yield drops to 0 (or a ripe one-time crop whose tile becomes None) was HARVESTED,
  and we saw exactly how many units they got. `fertilizer_available` True→False is a
  fertilizer pickup; `fed_today` False→True consumes exactly 1 WHEAT; a rising
  `fertilized_until_day` on a plant is a FERTILIZE (−1 from their stock). All from
  the engine's `_apply_unit_action`.
- Market inventory is public and its dynamics are deterministic given the obs:
  town drain fires on fixed step intervals over the public `town.unlocked_shops`
  (each shop instance drains its product list every townShopSellInterval=4 steps,
  ×2 for single-product shops; every town-center product −1 every 24 steps -- the
  engine applies both at the PRE-advance step counter with the shop list as it
  stood before any end-of-day unlock), a SELL at price>1 adds exactly +1, a
  BUY_PRODUCT −1 (`_commit_unit`). Subtract our own net trade and the town drain
  from the per-item inventory delta and the residual is the opponent's net trade.

Our own net trade is NOT taken from our orders: order commits can partially fail
(unit actions -- e.g. our own shed PICKUPs -- run before market commits in the same
interpreter step, so a SELL can find fewer units than ordered; measured +27 phantom
opponent-buy units of FERTILIZER in 10 days when trusting orders). Instead we run
the same farm-side accounting on OURSELVES, where everything is visible: our pooled
stock P = shed + carried inventories obeys
    P_now = P_prev + production − consumption − sold + bought,
so our committed net trade is P_prev + production − consumption − P_now, exact by
construction (capacity-blocked drops, failed commits and all).

Two observability limits, both handled:
- Sales at the $1 floor add no market inventory (engine only adds supply at
  price>1), so a player dumping a crashed item is invisible; stock estimates for
  floor-priced items are decayed ×0.5/turn instead of accumulated -- at the floor
  the item has no market impact anyway.
- Opponent BUY_SEED and BUY_ANIMAL never touch market inventory; seeds are
  tracked only downward (their plantings), animals ignored (their pending shed
  animals only delay placement, invisible and second-order).

Measured (train/test_opponent_model.py): see the test's printed table -- section A
is stock accuracy vs the true private (value-weighted $), section B reruns the
verifier's failing mirror rank-fidelity cells with empty/tracker/true privates.

Self-contained: stdlib + (on the eval worker, always present) kaggle_environments
for the authoritative SHOPS table, with a verbatim fallback copy. Exec-safe with
no __file__; inlined into the packaged submission by scripts/package_search.py.

Usage inside an agent, every turn BEFORE acting:
    tracker.update(obs)
    ...
    sim = make_sim(obs, opponent_private=tracker.estimate())
"""

TOWN_SHOP_INTERVAL = 4
TOWN_CENTER_INTERVAL = 24
TURNS_PER_DAY = 24
# The engine's own tables when available (the eval worker always has them);
# verbatim fallback copies otherwise. Getting these wrong is catastrophic-but-
# quiet: un-modelled town drain masquerades as opponent purchases (measured
# $68k phantom stock with a wrong table).
try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import (  # type: ignore
        SHOPS, TOWN_CENTER_PRODUCTS)
except Exception:  # pragma: no cover - fallback mirrors kaggriculture.py 1.32.7
    SHOPS = {
        "BAKERY": ["EGG", "WHEAT"],
        "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE": ["WOOL"],
        "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE": ["CARROT"],
        "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
        "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
    }
    TOWN_CENTER_PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                            "EGG", "MILK", "WOOL"]
PRODUCE = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
           "EGG", "MILK", "WOOL", "FERTILIZER"}
PRODUCT_OF = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _town_drain(shops, step):
    """Deterministic market drain the engine applied at interpreter step `step`
    (the PRE-advance counter; pass the previous observation's step and shops)."""
    drain = {}
    if step % TOWN_SHOP_INTERVAL == 0:
        for shop in shops or []:
            products = SHOPS.get(shop, [])
            mult = 2 if len(products) == 1 else 1
            for item in products:
                drain[item] = drain.get(item, 0) + mult
    if step % TOWN_CENTER_INTERVAL == 0:
        for item in TOWN_CENTER_PRODUCTS:
            drain[item] = drain.get(item, 0) + 1
    return drain


def _tile_key(t):
    """Compact snapshot of one tile for diffing between consecutive turns."""
    if not isinstance(t, dict):
        return t  # None
    if t.get("kind") == "PLANT":
        return ("P", t.get("crop"), t.get("yield_units", 0), t.get("planted_day"),
                t.get("fertilized_until_day", -1))
    if "animal" in t and t.get("animal"):
        return ("A", t["animal"], t.get("yield_units", 0),
                bool(t.get("fed_today")), bool(t.get("fertilizer_available")))
    return ("K", t.get("kind"))


def _snap(farm):
    return [[_tile_key(t) for t in row] for row in farm["tiles"]]


def _farm_events(prev, now):
    """(production, consumption, seeds_planted) implied by two consecutive tile
    snapshots of one farm. Exact per tile: the engine applies at most one unit
    action per tile per step, and daily refreshes touch fields we ignore."""
    prod, cons, planted = {}, {}, {}
    for y, row in enumerate(now):
        prow = prev[y]
        for x, nk in enumerate(row):
            pk = prow[x]
            if pk == nk:
                continue
            if isinstance(pk, tuple) and pk[0] == "P":
                _, crop, y_units, planted_day, fert_until = pk
                if nk is None and y_units > 0:
                    prod[crop] = prod.get(crop, 0) + y_units      # ripe one-time cut
                elif isinstance(nk, tuple) and nk[0] == "P" and nk[1] == crop \
                        and nk[3] == planted_day:
                    if y_units > 0 and nk[2] == 0:
                        prod[crop] = prod.get(crop, 0) + y_units  # ongoing picked
                    if nk[4] > fert_until:
                        cons["FERTILIZER"] = cons.get("FERTILIZER", 0) + 1
            elif isinstance(pk, tuple) and pk[0] == "A":
                _, animal, y_units, fed, fert = pk
                if isinstance(nk, tuple) and nk[0] == "A" and nk[1] == animal:
                    if y_units > 0 and nk[2] == 0:
                        p = PRODUCT_OF[animal]
                        prod[p] = prod.get(p, 0) + y_units
                    if not fed and nk[3]:
                        cons["WHEAT"] = cons.get("WHEAT", 0) + 1  # FEED: 1 wheat
                    if fert and not nk[4]:
                        prod["FERTILIZER"] = prod.get("FERTILIZER", 0) + 1
            if isinstance(nk, tuple) and nk[0] == "P" and (
                    not isinstance(pk, tuple) or pk[0] != "P"):
                planted[nk[1]] = planted.get(nk[1], 0) + 1
    return prod, cons, planted


def _own_pool(private):
    pool = {}
    for k, v in dict(private.get("shed") or {}).items():
        if k in PRODUCE and v:
            pool[k] = pool.get(k, 0) + v
    for inv in private.get("inventories") or []:
        for k, v in dict(inv or {}).items():
            if k in PRODUCE and v:
                pool[k] = pool.get(k, 0) + v
    return pool


class OppTracker:
    """Streaming estimator of the opponent's private stock. O(grid) per update."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._stock = {}          # item -> estimated opponent units (shed+carried)
        self._seeds = {}
        self._prev_opp = None     # opponent tile snapshot
        self._prev_own = None     # own tile snapshot
        self._prev_pool = None    # own produce pool (shed + inventories)
        self._prev_inv = None     # market inventory
        self._prev_step = None
        self._prev_shops = []     # town.unlocked_shops as of the previous obs

    def note_own_orders(self, orders, obs=None):
        """Kept for API compatibility; own trade is now derived exactly from our
        own pool delta (orders can partially fail -- see module docstring)."""

    def update(self, obs):
        """Feed every observation, in order, before acting. Never mutates obs."""
        try:
            self._update(obs)
        except Exception:
            # An estimator must never take the agent down; on any surprise keep
            # the previous stock estimate and resynchronise the snapshots.
            try:
                self._resync(obs)
            except Exception:
                self.reset()

    def estimate(self):
        """Opponent private in reconstruct.make_sim's opponent_private shape."""
        shed = {k: int(v) for k, v in self._stock.items() if v >= 1}
        seeds = {k: int(v) for k, v in self._seeds.items() if v >= 1}
        return {"shed": shed, "seeds": seeds, "inventories": []}

    # -- internals ---------------------------------------------------------

    def _resync(self, obs):
        me = obs.get("player", 0)
        farms = obs.get("farms") or []
        self._prev_opp = _snap(farms[1 - me]) if len(farms) > 1 else None
        self._prev_own = _snap(farms[me]) if len(farms) > me else None
        self._prev_pool = _own_pool(obs.get("private") or {})
        self._prev_inv = dict((obs.get("market") or {}).get("inventory") or {})
        self._prev_step = obs.get("day", 0) * TURNS_PER_DAY + obs.get("hour", 0)
        self._prev_shops = list((obs.get("town") or {}).get("unlocked_shops") or [])

    def _update(self, obs):
        step = obs.get("day", 0) * TURNS_PER_DAY + obs.get("hour", 0)
        if self._prev_step is None or step != self._prev_step + 1:
            # First call, duplicate turn, new episode (regression), or a gap we
            # cannot attribute turn-by-turn: resynchronise, inventing nothing.
            if self._prev_step is not None and step < self._prev_step:
                self.reset()
            if self._prev_step is None or step != self._prev_step:
                self._resync(obs)
            return

        me = obs.get("player", 0)
        farms = obs.get("farms") or []
        opp_farm = farms[1 - me] if len(farms) > 1 else None
        inv = dict((obs.get("market") or {}).get("inventory") or {})
        prices = (obs.get("market") or {}).get("prices") or {}

        # 1. opponent farm events: production into / consumption from their stock
        if opp_farm is not None and self._prev_opp is not None:
            opp_now = _snap(opp_farm)
            prod, cons, planted = _farm_events(self._prev_opp, opp_now)
            for item, n in prod.items():
                self._bump(item, n)
            for item, n in cons.items():
                self._bump(item, -n)
            for crop, n in planted.items():
                self._seeds[crop] = max(0, self._seeds.get(crop, 0) - n)
        else:
            opp_now = None

        # 2. our own committed net trade, exact from our pool delta
        own_net_sold = {}
        own_now = _snap(farms[me]) if len(farms) > me else None
        pool_now = _own_pool(obs.get("private") or {})
        if self._prev_own is not None and self._prev_pool is not None and own_now is not None:
            prod, cons, _ = _farm_events(self._prev_own, own_now)
            for item in set(self._prev_pool) | set(pool_now) | set(prod) | set(cons):
                own_net_sold[item] = (self._prev_pool.get(item, 0)
                                      + prod.get(item, 0) - cons.get(item, 0)
                                      - pool_now.get(item, 0))

        # 3. market attribution: delta - our net trade - town drain = their trade
        if self._prev_inv is not None:
            drain = _town_drain(self._prev_shops, self._prev_step)
            for item in set(inv) | set(self._prev_inv):
                if prices.get(item, 999) <= 1:
                    continue  # floor-priced: flows invisible, handled in 4.
                resid = (inv.get(item, 0) - self._prev_inv.get(item, 0)
                         + drain.get(item, 0)
                         - own_net_sold.get(item, 0))
                self._bump(item, -resid)  # resid>0: they sold; <0: they bought

        # 4. floor-priced items: their sales add no inventory, so the estimate
        # can only grow; decay it instead -- at the floor the item carries no
        # market impact and heavy sellers are dumping it anyway.
        for item in list(self._stock):
            if prices.get(item, 999) <= 1:
                self._stock[item] *= 0.5

        self._prev_opp = opp_now
        self._prev_own = own_now
        self._prev_pool = pool_now
        self._prev_inv = inv
        self._prev_step = step
        self._prev_shops = list((obs.get("town") or {}).get("unlocked_shops") or [])

    def _bump(self, item, n):
        if n:
            self._stock[item] = max(0.0, self._stock.get(item, 0.0) + n)
