# Mega Lucario ex: Heuristic Control with Shallow CABT Search

**Subtitle:** A Fighting-type prize race agent for the PTCG AI Battle Challenge — deck thesis, decision policy, and local simulation evidence.

## 1. Overview

Our Simulation entry is a **Mega Lucario ex** agent: the official TPC sample deck paired with an enhanced rule-based policy and a shallow look-ahead on CABT search APIs. The thesis is simple: in a large Standard pool with a hard 10-minute clock, a **transparent, matchup-aware heuristic** with light search beats opaque end-to-end learning for stability and debugability — and that clarity is what Strategy judging rewards.

Simulation agent path (sibling repo folder): `competitions/pokemon-tcg-ai-battle/agent/` (`main.py` + `deck.csv`). This writeup documents why that deck and policy were chosen, how decisions are made per CABT context, and what local evidence we have so far.

## 2. Deck concept

### Win condition

Close prizes by evolving **Riolu → Mega Lucario ex** and attacking with **Aura Jab** / **Mega Brave**, supported by Fighting energy acceleration and disruption. The deck is not a toolbox — it is a focused Fighting beatdown with fixed roles:

| Role | Cards (IDs) | Job |
| --- | --- | --- |
| Attacker | Riolu 677, Mega Lucario ex 678 | Prize pressure; Mega Brave for big turns |
| Setup / damage | Makuhita 673, Hariyama 674 | Early damage / pivot lines |
| Consistency | Lunatone 675, Solrock 676 | Ability utility / board presence |
| Search | Dusk Ball 1102, Fighting Gong 1142, Pokégear-style lines via Pad 1152 | Find evolve pieces and energy |
| Disruption | Boss's Orders 1182, Carmine 1192, Lillie's Determination 1227 | Gust, draw, reset |
| Tech | Premium Power Pro 1141, Hero Cape 1159, Gravity Mountain 1252, Switch 1123 | +30 damage, HP, Stage-2 HP tax, pivots |
| Fuel | Basic Fighting Energy ×12 (ID 6) | Attach density for Lucario attacks |

### Why this archetype

1. **Known-good skeleton** — TPC published this list; legality and engine support are proven.
2. **Clear plan vs the ~2k card pool** — Fighting weakness/resistance interactions are explicit; we encode them rather than hope a model rediscovers them.
3. **Paired policy** — deck and agent share the same card ID constants; every heuristic weight maps to a real line (attach, evolve, Boss, stadium).
4. **Strategy fit** — judges score deck–strategy alignment (20%). A single coherent win condition reads better than a meta-soup list with opaque card choices.

Trade-off accepted: weaker into resistant walls and some Stage-2 stacks. Gravity Mountain and Boss lines exist specifically to tax those boards rather than diluting the 60 with random tech.

## 3. Agent logic

CABT exposes legal options each step. The agent never invents moves — it **scores** `obs_dict["select"]["option"]` and returns indices in `[minCount, maxCount]`.

### Pipeline

1. **Deck return** — `select is None` → return the 60 IDs from `deck.csv` (Kaggle-safe path: `deck.csv` or `/kaggle_simulations/agent/deck.csv`).
2. **Belief sync** — track remaining copies via serials seen in hand/board/discard/looking so search/discard contexts prefer scarce pieces.
3. **Attack planning** — build an `AttackPlan` (attacker, target, attack index, whether an energy attach is still needed) using damage fidelity: Fighting W/R, Premium Power Pro (+30), Gravity Mountain (−30 HP on Stage 2), Cosmic Beam ignoring W/R where applicable.
4. **Context heuristics** — setup active/bench, MAIN (play / attach / evolve / ability / retreat / attack / end), switches, searches each get typed scores (e.g. evolve ~9000+, Hero Cape ~7000, ability spikes, negative scores to skip bad multi-picks).
5. **Shallow search (MAIN, turn ≥ 2)** — `search_begin` / `search_step` / `search_end` with a small node budget (~72) and time budget (~0.40s). Search may override the top heuristic pick when clearly better under `evaluate_state` (prize differential, board HP/energy, Lucario presence).

### Design principles

- **Legality first** — only indices into engine options.
- **Clock-aware** — search is shallow; the 10-minute match clock punishes deep rollouts.
- **Deterministic enough to debug** — failures map to a score table, not a black box.
- **Kaggle load parity** — no reliance on `__file__`; packaging validates via `exec()` smoke tests before submit.

## 4. Simulation evidence

Local self-play (same deck, alternating seats) under `kaggle-environments` / CABT:

| Matchup | Result |
| --- | --- |
| Ours vs random legal play | ~39–1 / 40 |
| Ours vs official Lucario sample (`baselines/official_lucario.py`) | ~27–23 / 50 |

Interpretation: the heuristic+search stack is far above random, and modestly above the TPC rule baseline it extends. The edge comes from damage fidelity and selective search overrides, not from a different deck.

Ladder episodes remain the external stress test (hidden information, diverse opponents, timeouts). We treat local tables as **stability checks** before burning daily Simulation submits, and we will attach ladder snapshots in the media gallery as they accumulate.

## 5. Stability and failure modes

What we measure:

- Validation episode success (Kaggle-style `exec` without `__file__`).
- Self-play win rate vs fixed baselines across seat swaps.
- Timeout risk — keep search budgets small; prefer ending turn to hanging on deep trees.

Known failure modes:

1. **Resistant / high-HP actives** — plan may stall; Boss + Gravity Mountain mitigate but do not delete the matchup.
2. **Brick opens** — no Riolu/energy; setup scores bias basics and Gong/Dusk Ball, but variance remains.
3. **Over-attachment to Lucario** — retreat/switch scores try to pivot when the plan marks a bench attacker, but multi-prize swings can still punish greed.
4. **Search myopia** — 0.40s look-ahead misses long traps; we accept that for clock safety.

Changes that mattered: (a) explicit W/R and stadium modifiers in damage, (b) search refine only on MAIN after a scored shortlist, (c) packaging/validation so path bugs fail locally.

## 6. Conclusion

We optimize for a **readable Fighting prize race**: one deck thesis, a scored legal-action policy, and light CABT search under the match clock. Strategy Category scoring weights model clarity and stability (70%) and deck alignment (20%); this writeup mirrors that priority order. Next steps before finals: broader baseline suite, better hidden-card beliefs, and ladder-backed figures without bloating search past the time limit.

---

*Agent code: Simulation Category submission (`pokemon-tcg-ai-battle`). This Strategy writeup explains that agent; it is not a second executable.*
