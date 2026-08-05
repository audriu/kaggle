# Pokémon TCG AI Battle Challenge — Simulation

Local workspace for [The Pokémon Company – PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview).

Prize / finals writeup work lives in the sibling folder:
[`../pokemon-tcg-ai-battle-challenge-strategy`](../pokemon-tcg-ai-battle-challenge-strategy/).

## Competition summary

Build an AI **agent** that plays Pokémon TCG under the official **CABT** simulator. You do not implement rules — you only choose among legal options the engine exposes each step.

| Track | What you submit | Prizes | Deadline (approx.) |
| --- | --- | --- | --- |
| **Simulation** (this folder) | Agent bundle (`submission.tar.gz`) on a skill ladder | None (qualifying ground) | **16–17 Aug 2026** |
| **Strategy** | Writeup: deck concept, logic, sim results | $30k × top 8 → Japan finals | mid Sep 2026 |
| **Finals** | Live tournament (top Strategy teams) | +$50k / $30k | Sep 2026 (Japan) |

- Card pool: ~2,000 Standard-format cards (organizer list only).
- Match clock: 10 minutes per player; timeout = loss.
- Simulation: up to **5 submissions/day**; leaderboard tracks your **latest 2**.
- Engine docs: [matsuoinstitute.github.io/cabt](https://matsuoinstitute.github.io/cabt/)

**Status today (5 Aug 2026):** ~11 days left on Simulation submissions.

## Current agent

`agent/main.py` is a **Mega Lucario ex** policy (same deck as the official TPC sample):

1. **Rule-based scoring** — attack planning (Aura Jab / Mega Brave / Hariyama / Solrock), energy attach, evolve, Boss/Switch lines, setup active/bench, search/discard contexts.
2. **Damage fidelity** — Fighting weakness/resistance, Premium Power Pro (+30), Gravity Mountain (−30 HP on Stage 2), Cosmic Beam ignoring W/R.
3. **Shallow look-ahead** — on MAIN (turn ≥ 2), `search_begin` / `search_step` may override the heuristic when clearly better.
4. **Kaggle-safe load** — works under `exec()` without `__file__` (validated by `package.py` / `submit.py`).

Local self-play (alternating seats, same deck):

| Matchup | Result |
| --- | --- |
| Ours vs random | ~39–1 / 40 |
| Ours vs official Lucario sample (`baselines/official_lucario.py`) | ~27–23 / 50 |

## How an agent works

```text
engine → agent(obs_dict) → list[int]
```

1. If `obs_dict["select"] is None` → return your **60 card IDs** (deck).
2. Otherwise return indices into `obs_dict["select"]["option"]` (length between `minCount` and `maxCount`).
3. Contexts include setup active, main phase (play / attach / evolve / ability / retreat / attack / end), switches, searches, etc.

Submission archive layout (files at archive **root**, not nested):

```text
submission.tar.gz
├── main.py      # defines agent()
├── deck.csv     # 60 lines of card IDs
└── cg/          # official SDK from competition Data (do not redistribute)
```

## Repo layout

```text
agent/                 # what you edit and package
  main.py              # Lucario heuristic + CABT search
  deck.csv             # Mega Lucario ex (official sample list)
baselines/
  official_lucario.py  # TPC rule-based sample (eval opponent)
decks/                 # extra deck lists
scripts/
  setup_data.py        # kaggle competitions download → data/
  package.py           # build dist/submission.tar.gz (+ validate)
  validate_submission.py  # Kaggle-style exec smoke test
  self_play.py         # local match via kaggle-environments (needs cg/)
  submit.py            # validate then kaggle competitions submit
data/                  # competition zip + sample_submission (gitignored)
```

## One-time setup

```bash
cd /home/a/code/kaggle
source .venv/bin/activate   # already created
pip install -r requirements.txt

cd competitions/pokemon-tcg-ai-battle
# Kaggle API token → ~/.kaggle/kaggle.json (chmod 600)
# Accept rules on the competition page, then:
python scripts/setup_data.py
```

Without `~/.kaggle/kaggle.json`, you can manually download **Data** from the competition page and unpack into `data/` so that `data/sample_submission/cg/` exists.

## Day-to-day

```bash
cd competitions/pokemon-tcg-ai-battle

# edit agent/main.py and/or agent/deck.csv (or --deck decks/...)
python scripts/package.py          # builds + Kaggle-style validates
python scripts/self_play.py --games 5
python scripts/validate_submission.py   # optional standalone check

# Upload dist/submission.tar.gz (CLI or Kaggle Submit Agent UI)
python scripts/submit.py -m "Mega Lucario heuristic + shallow CABT search"
```

`package.py` / `submit.py` load the agent via `exec()` **without** `__file__` (same as Kaggle) so deck-path bugs fail locally instead of on the validation episode.

## Suggested improvement path

1. ~~Valid submission~~ / ~~rule-based Lucario + search~~ — current `agent/main.py`.
2. **Eval loop** — pit against `baselines/official_lucario.py` and ladder-style decks before burning daily submits.
3. **Deck craft** — tune archetype vs meta; keep deck and policy paired.
4. **Deeper search / beliefs** — better hidden-card guesses, longer rollouts under the 10‑minute clock.
5. **Strategy writeup** — see sibling [`../pokemon-tcg-ai-battle-challenge-strategy`](../pokemon-tcg-ai-battle-challenge-strategy/).

## Notes

- Competition card CSVs / `cg/` binaries are **Competition Use only** — keep them under `data/` (gitignored).
- The old root `sim.py` used a non-CABT observation shape; the real entrypoint is `agent/main.py`.
- Viewer / local UI option: [cabt-viewer](https://github.com/charlielockyer-rice/cabt-viewer) (still needs your local `sample_submission`).
- Official samples: [Mega Lucario notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck), CABT docs above.
