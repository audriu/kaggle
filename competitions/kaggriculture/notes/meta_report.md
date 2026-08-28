# Top-ladder meta report: the RANCH+GARDEN template (synthesized 2026-08-28)

Sources: 4 miner passes over 32 local replays covering the top 8 ladder teams —
rank 1 カワシギ (team 16677252, 4 eps incl. 93459936/93454366), rank 2 HealthStone
(16672202, 4 eps), rank 3 THUNDER THUNDER (16622198, 7 eps), rank 4 Kaito Fukami
(16627032, 4 eps, fully deterministic open-loop script), rank 5 Seb (allegedly)
(16630516, 4 eps incl. 91588122), rank 6 Ezzzzzekki (16670569, 4 eps), rank 7
Elzandi Irfan Zikra (16664164, 5 eps), rank 8 researchstudio.site (16676904, 4 eps).
Engine references verified against
`.venv/lib/python3.12/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`.
All eight teams play the **same archetype**; ranks 1–4 finish ~90–100k, ranks 5–8
~50–90k. The differences between them are execution polish (selling discipline,
wheat self-sufficiency, adaptive sheep counts), not strategy.

---

## 1. Why ranching works: the town-drain price mechanism (engine numbers)

Our `notes/research_plan.md` §1 price table ("MILK/WOOL crash to $1 at ~100–150
units sold") is **wrong for real games**: it maps units-sold → price assuming
inventory = I0 + units sold. In a live episode the town continuously *removes*
inventory, so what matters is **standing inventory relative to I0=10,000**, i.e.
(both players' cumulative sales) − (cumulative town drain). Paced selling at or
below drain capacity keeps inventory pinned near I0 and prices near base forever.

**Drain rates (engine `_town_consume` + `SHOPS`, defaults):**
- Each unlocked shop *instance* drains 1 of each of its products every 4 steps
  = **6/day** per product; single-product shops (YARN_STORE→WOOL, PET_CAFE→CARROT)
  drain ×2 = **12/day**.
- Town center drains 1/day of every product except FERTILIZER.
- One shop instance unlocks (drawn **with replacement** from the 8 types) at the
  end of every 3rd day (`townShopUnlockInterval=3`), capped at
  `MAX_SHOP_INSTANCES=8` → k ≈ day/3 instances, 8 by day 24.

Product exposure in `SHOPS` (8 types) and the resulting **expected** town drain
with all 8 instances up (~day 24), plus the observed range:

| product | shops containing it | E[drain]/day @8 inst (+1 town ctr) | notes |
|---|---|---|---|
| WHEAT | 5/8 | ~31/day | why wheat sits at $40–50 all game |
| STRAWBERRY | 4/8 | ~25/day | absorbs a 30–40-tile berry patch |
| MILK | 3/8 (PIZZA, ICE_CREAM, SMOOTHIE) | ~19/day | ~10/day mid-game (4 inst) |
| CARROT | PET_CAFE(×2) + FARMERS_MARKET | ~19/day | unused by the meta |
| EGG / TOMATO | 2/8 each | ~13/day | EGG untouched by every team |
| WOOL | YARN_STORE only (×2) | ~13/day *expected*, but binomial: P(0 yarn stores in 8 draws)=(7/8)^8≈34% → drain 1/day | pure draw luck |
| MELON | **none** | 1/day (town center only) | ⇒ one-shot dump only |

**Price slopes at the margin (from `MARKET_PARAMS`/`market_price`):**
- MILK glut is linear at **$2.10/unit** (1.6·160/122): inventory I0+40 → $76,
  I0+60 → $34, I0+76 → floor. Scarcity is sqrt: I0−100 → $247.
- WOOL glut is *quadratic* ($200 − 0.0581·x²): +40 → $107, +50 → $55, +59 → $1.
  Scarcity log: −100 → $240.
- WHEAT scarcity: 25 + √deficit → deficit 400 = $45 (exactly the observed band).
- EGG scarcity (hinge): −300 → $68 with nobody selling.
- MELON glut sq (amp 0.01/unit²): a 72-unit dump only costs ~$52 off base.

**The balance that makes 9 cows the right number** (rank-1/2 games, Miner 1):
9 cows with daily FEED+CARE produce ~10 milk/day/player (engine: a production
day yields 1 + accrued care bonus, so a fed+cared cow cycles ~3 milk/2 days,
fleet-realized ~1.0–1.1/day; naive unfed rate would be 0.5/day). Two mirror
ranchers ≈ 20 milk/day vs town drain 10–19/day → inventory drifts only ±80 of
I0 across カワシギ's whole games and MILK floats **$95–265, never near $1**.
The knife edge: in Seb/Ezz games (Miner 3) standing MILK inventory of just
**+70** put the price at $10–14 from day 17 on. The entire difference between
"milk earns $150/unit" and "milk earns $12/unit" is ~40 standing units — one
sloppy afternoon of overselling.

Same mechanism, opposite sign, is free money elsewhere: EGG drifts to
9,500–9,800 (price 59–81) with zero sellers in all 32 replays, and WHEAT's
5/8-shop exposure plus every player's feed-buying holds it at $40–50 vs base 25.

---

## 2. The canonical schedule (reproducible)

Consensus of all 8 teams; day numbers are decision-time days ((t−1)//24).

**Day 0 — all-in conversion of the $3,000 (end the day with <$150).**
Hire 4–5 hands (fib cost ~$7–12 total; カワシギ hires 5 for $12). Buy the seed
herd: 4 SHEEP + 1 COW (THUNDER/Kaito/HealthStone) or 2+2 (カワシギ) — first COW
*and* first SHEEP bought and placed day 0 in every one of the 32 replays. Build
pastures, plant ~5 WHEAT + ~5 MELON (melon matures d10–12), buy wheat seeds.
カワシギ money at d5: $146; THUNDER d0 end: $16.

**Days 1–6 — skeleton crew, compounding starts.**
1–3 hands/day only. Add cows d3–7 (カワシギ d3,5,7,7). First FERTILIZER sales
d1–2 (collected off animals, never bought — all 8 teams). First WOOL sale d6
(sheep first_yield_day=6), strawberry planting starts d5–10. **Day 6: buy NE
($1,000)** — every replay of every team except Seb (d4).

**Days 7–12 — ramp to full ranch.**
Hires jump to 8–14/day. Herd completes: **9 COW + 5 SHEEP** (THUNDER exactly
9+5 every game; カワシギ 9+6; rank 8 adapts 6–10 cows). **ZERO GEESE — no team
ever buys one.** First MILK sale d8–9 (cow first_yield_day=8). **SW quadrant
d9–11** (THUNDER/Kaito d10, HealthStone d10, rank 8 d9). **SE ($4,000) is
essentially never bought** — only Seb (always, d10; the weakest of the eight)
and カワシギ once (d12, its 136k boom game): 75 tiles is enough. MELON one-shot:
カワシギ sells all 72 on d10–11 @ ~$207.

**Adaptive sheep (the rank-1 refinement):** read `town.unlocked_shops`. If a
YARN_STORE appears early, wool drain is 12/day/instance and sheep scale:
カワシギ ep 93459936 (YARN d6) → 12 sheep / 6 cows, 201 wool @ $148, 136k final;
its no-yarn games stop at 4–6 sheep / 9–10 cows. HealthStone ep 91555945
(YARN d3) → 6 sheep, 180 wool @ $218.

**Days 10–27 — steady state: a daily temp-labor army servicing the ranch.**
The engine **fires all hands every night and resets the fib hire counter**, so
the meta re-hires 12–14 hands *every morning* (12 hires = $376/day, 14 = $986;
season totals ~276–300 hires, $6–11k — カワシギ 277 hires/$5,977). Every animal,
every day: FEED (1 wheat) + CARE + COLLECT_FERTILIZER (~290–375 of each verb
per game). CARE is the yield engine: the care bonus banks +1/day and is paid out
on fed production days → ~1 milk/cow/day and ~1.3 wool/sheep/day, ~3× the
unfed-uncared rate. Every animal also drops 1 collectible FERTILIZER/day →
~300 collected/game, ~60–140 used on crops, the rest sold (~220–300 @ $49–72 =
**$12–15k/game of pure by-product**; THUNDER $14.3k). Crops around the ranch:
30–40 STRAWBERRY tiles (peak d15–20), WHEAT 13–16 tiles ramping to **40–46 by
d25** for the endgame dump; ~1,000 WATER ops/game; weeds kept near zero until
the last days.

**Feed logistics — the wheat double game.** ~10–14 wheat/day fed. Teams *grow*
360–500/game AND *buy* just-in-time from the market (BUY_PRODUCT WHEAT is the
only product ever bought): THUNDER 246/game @ $40, カワシギ 753 @ $43.6, rank 8
905 @ ~$40 while *simultaneously selling* 40–85/day — buy quotes at post-buy
inventory make a round-trip ~costless, so **the market doubles as a feed
warehouse past the 100-cap shed**. Everyone also sells their harvest into the
same $40–46 town-driven premium (HealthStone finishes with a 42-tile wheat
factory dumped d28–29 @ $44).

**Days 27–29 — controlled liquidation.**
Stop feeding (sheep escape after 2 unfed days — free feed savings; wool
interval 3 means no yield lost; カワシギ sheep 6.0→4.8, rank 8 releases 5→1.5
while keeping cows), tolerate weeds, plant nothing but wheat after ~d24, dump
the shed d28–29 (~100–234 WHEAT + 27–54 MILK + 5–31 WOOL + straw + fert) at
decaying but never-floor prices.

**Money curve (rank-1 band, day-start):** d0 3,000 → d5 ~150 → d10 1.6–3.3k →
d15 15–22k → d20 41–52k → d25 71–77k → final 90–100k (カワシギ mean 92.5k, best
136k; THUNDER 98.2k).

---

## 3. Selling discipline per item (volumes vs drain capacity)

| item | start day | pace | evidence |
|---|---|---|---|
| MILK | d8–9 | **production-paced trickle ~10/day** (chunks 3/6/9, 1–3 orders/day, ~34 sell-events/game), every day, no dumping | カワシギ 184/game @ ~$130; THUNDER 227 @ $152; inventory stays I0±80 all game. Counter-example: Seb/Ezz sell the same *volume* but let inventory sit +70 → $10–14 from d17 (milk revenue $8–9k vs THUNDER's $34.7k) |
| WOOL | d6 | 4–16 units on the 3-day shear cycle (~16 events/game); target **inventory ≤ I0** (sq glut is brutal: +40–50 pins $45–58) | Ezz holds inv at 9,903 → wool $174–185 late, $25.4k/game; Seb drifts to 10,049 → $125. Yarn-store draw dominates: Kaito avg 2.0 yarn stores → wool $219 mean, $33.6k; THUNDER avg 0.43 → $130 |
| EGG | never | **nobody sells**; price climbs $50→$68–81 on town drain alone | all 32 replays; unexploited niche |
| WHEAT | d0–5 | sell everything above feed reserve continuously @ $38–46; ramp late tiles and dump d28–29 | THUNDER ~302 sold @ $42.7; rank 8 1,145 @ $41.6 = $47.6k gross (biggest single earner) |
| STRAWBERRY | d13–15 | steady 5–14/turn over 30–46 selling turns; drain ~25/day absorbs it | THUNDER 256 @ $138; rank 8 trickles 5.6/turn → $184 avg vs rank 7's 14.4/turn chunks → $102 |
| MELON | d10–11 | **one-shot dump** (no shop demand, 1/day drain, but sq curve is shallow to ~100 units) | カワシギ 72 @ $207 in 2 days; HealthStone spreads 94 @ $183 |
| FERTILIZER | d1–2 | continuous trickle of collected surplus | 220–300/game @ $49–72, all teams |

---

## 4. Contrast with our cem4 line

cem4 (plan §8–9; lineage cem2→cem3→cem4, fitness 50.2k local, ~500 Elo vs the
top's ~3,200):

| dimension | cem4 | top meta |
|---|---|---|
| animals | cows day ~3–5, target ~7.6; geese ~0; **no sheep ever** (sheep_day still 31) | 9 COW + 5 SHEEP from d0–10, sheep→12 on yarn draw; 0 geese (this part we match) |
| labor | CEM pushed hiring **down** from 8–10 | 12–14 re-hires **every day** ($376–986/day) — the workforce is what makes CARE+fert+40 crop tiles possible |
| husbandry | feed yes; CARE/fert-collection possible in the ladder but with few animals and few hands it's marginal | FEED+CARE+COLLECT on every animal daily → 3× product rate + $12–15k fert by-product |
| crops | wheat-heavy staples, token melon (~2 seeds, d28), **no strawberries** | 30–40 strawberry tiles ($22–37k/game), 12–21 melon one-shot ($11–17k), wheat ramped late into the $45 premium |
| selling | raised floors (scarcity posture learned from crasher), dump d29.7 | inventory-pinned pacing: milk at production rate, wool under I0, melon one-shot, eggs never |
| feed | grow + emergency-buy only at zero stock | grow AND routinely buy 220–900/game @ $40; market-as-warehouse round-trips |
| land | crowding-triggered | NE d6, SW d10, SE never |
| income | ~50k local (softer opposition) | 90–100k vs top opposition |

The core gap is not one trick: it is **sheep + strawberries + a 12–14-hand army
+ care/fertilizer husbandry**, four mutually-reinforcing revenue lines our theta
has switched off, plus selling discipline calibrated to town drain instead of
fixed floors.

---

## 5. Five highest-leverage changes for our bot (theta-v2 dial mapping)

1. **Ranch-first opening: sheep from day 0, cows to 9, geese stay dead.**
   Dials (all expressible): `sheep_day` 31→0, `sheep_target` 0→5 (6+ if change
   #5's shop-reading lands), `sheep_cash`→500 (its lower bound); `cow_day`→0,
   `cow_target`→9, `cow_req_geese`→0 (kills the goose gate), `cow_cash`→400;
   `goose_target`→0. Evidence: every top-8 team, e.g. THUNDER 4 SHEEP+1 COW d0,
   9+5 steady. Caveat: `_market` buys 1 animal/species/turn and only while none
   waits unplaced — the day-0 all-in spreads over ~a day; acceptable, not
   NEEDS-CODE.

2. **Field the daily temp army.** Dials: `hire_d0`→10, `hire_d1..d3`→13–14
   (hi=14 just covers the meta's 12–14), `hire_min`→8 (+`hire_min_p2_add`),
   `hire_empty_div`→1, `hire_cash_buffer`→~0–100. CARE and COLLECT_FERTILIZER
   already sit in `_unit`'s priority ladder, so headcount alone unlocks the 3×
   animal yield and the ~$13k fertilizer line (`sell_floor_fert` default 0
   already sells it). CEM has been pushing hiring *down* — that was correct for
   a crop bot and is wrong for a ranch; seed cem6 with hiring clamped high.
   Minor mismatch flagged NEEDS-CODE-lite: the meta ramps 4→8+ at day 7 but our
   d0-bucket spans days 0–9 (one dial); a `hire_d0b` split or moving the bucket
   edge would express it exactly.

3. **Inventory-guard selling for MILK/WOOL via floors + caps.** Because glut
   price is a deterministic function of standing inventory, a price floor IS an
   inventory stop: `sell_floor_milk`→~0.60 (halt below $96 ⇔ inventory >
   I0+30 on the $2.10/unit slope), `sell_floor_wool`→~0.55 (halt below $110 ⇔
   > ~I0+39 on the sq curve), global `sell_cap`→6–9/turn (matches the meta's
   3–9-unit chunks), `dump_day`→28. Evidence: THUNDER milk $34.7k vs Seb $9.1k
   on identical volume. **NEEDS-CODE (small):** per-item caps exist only for
   WHEAT/EGG (`SELL_ITEM_CAP_KEY`) — append `sell_cap_milk`/`sell_cap_wool`/
   `sell_cap_straw` dials (PARAMS is append-only, cheap) so wheat-dumping and
   milk-pacing can use different caps; true *per-day* quotas (X/day, not
   X/turn) would also need a day-counter in `_market`.

4. **Strawberry engine + melon one-shot + late wheat ramp.** Dials:
   `straw_day`→3–5, `straw_seed_max`→10–12; `melon_day`→0, `melon_seed_max`→
   10–12, `hold_melon_until`→10 (matures d10–12, dump into base-price market);
   `plant_stop_day`→24; late wheat via `wheat_per_animal`→3 + `wheat_plus`→8
   and `wheat_base_p2_add`→+8 (14 animals → target ~50, matching the meta's
   40–46 d25 tiles); `sell_floor_wheat`→~1.4 (only sell ≥$35 — the town premium
   makes this nearly always true). Evidence: strawberries are the #1 revenue
   line for ranks 5–6 ($33–37k) and $22–35k for ranks 1–4. **NEEDS-CODE:**
   `_unit` only FERTILIZEs non-ongoing crops (policy.py ~line 312, `not
   info["ongoing"]`) — fertilizing strawberries (rank 8: 137 FERTILIZE/ep,
   doubles ongoing yield) requires lifting that gate for STRAWBERRY.

5. **Feed logistics + town/shop awareness — NEEDS-CODE (the policy family
   cannot express either).** (a) Routine just-in-time feed buying: the only
   wheat purchase today is the zero-stock emergency (`_market` "Emergency feed"
   block); the meta buys 220–900 wheat/game @ ~$40 as a running warehouse
   (rank 8 buys 30–80/day while selling the same days ~costlessly). Add a
   `feed_buy_target` dial: top up shed wheat toward the reserve whenever below
   it. (b) The policy never reads `obs["town"]`: adaptive sheep scaling on
   YARN_STORE unlocks (カワシギ's 136k game; wool drain 12/day/instance vs
   1/day without) and milk pacing on milk-shop count need a shop-count input —
   e.g. `sheep_per_yarn_store` and a wool-floor modifier. (c) End-game feed
   cutoff (stop feeding d27+, let sheep escape — all top teams) has no dial;
   feeding is priority-1 in `_unit` whenever a unit carries wheat. Add a
   `feed_stop_day`.

Optional probe (not in the top five because no top team validates it): EGG is
structurally underpriced by the meta — town drain alone lifts it to $68–81 and
nobody sells; 4 geese (max_held 4, interval 1, $300 each) plus a coop might be
positive-margin now, but treat as an experiment, not a copy.

**Training note:** these dial settings should be the *seed mean* of the next CEM
run (cem6 from cem5's checkpoint format), with sigma widened on the touched
dials — cem5's current basin was shaped by an economy model (§1 price table)
that we now know was measuring the wrong variable. Local eval must also change:
arena opponents should include a ranch-meta scripted bot, because every price in
this report is a *two-rancher* market; our crop-bot league produces the very
milk crash that made ranching look unplayable.
