# Pokémon TCG AI Battle Challenge

Local workspace for [The Pokémon Company – PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview).

## Competition summary

Build an AI **agent** that plays Pokémon TCG under the official **CABT** simulator. You do not implement rules — you only choose among legal options the engine exposes each step.

| Track | What you submit | Prizes | Deadline (approx.) |
| --- | --- | --- | --- |
| **Simulation** | Agent bundle (`submission.tar.gz`) on a skill ladder | None (qualifying ground) | **16–17 Aug 2026** |
| **Strategy** | Writeup: deck concept, logic, sim results | $30k × top 8 → Japan finals | mid Sep 2026 |
| **Finals** | Live tournament (top Strategy teams) | +$50k / $30k | Sep 2026 (Japan) |

- Card pool: ~2,000 Standard-format cards (organizer list only).
- Match clock: 10 minutes per player; timeout = loss.
- Simulation: up to **5 submissions/day**; leaderboard tracks your **latest 2**.
- Engine docs: [matsuoinstitute.github.io/cabt](https://matsuoinstitute.github.io/cabt/)

**Status today (5 Aug 2026):** ~11 days left on Simulation submissions.

## How an agent works

```text
engine → agent(obs_dict) → list[int]
```

1. If `obs_dict["select"] is None` → return your **60 card IDs** (deck).
2. Otherwise return indices into `obs_dict["select"]["option"]` (length between `minCount` and `maxCount`).
3. Contexts include setup active, main phase (play / attach / evolve / ability / retreat / attack / end), switches, searches, etc.

Official random baseline:

```python
import random

def agent(obs_dict: dict) -> list[int]:
    return random.sample(
        list(range(len(obs_dict["select"]["option"]))),
        obs_dict["select"]["maxCount"],
    )
```

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
  main.py              # random agent (correct CABT API)
  deck.csv             # Mega Lucario ex starter (from official examples)
decks/                 # extra deck lists
scripts/
  setup_data.py        # kaggle competitions download → data/
  package.py           # build dist/submission.tar.gz
  self_play.py         # local match via kaggle-environments (needs cg/)
  submit.py            # kaggle competitions submit dist/submission.tar.gz
data/                  # competition zip + sample_submission (gitignored)
```

## One-time setup

```bash
cd /home/a/code/kaggle
source .venv/bin/activate   # already created
pip install -r requirements.txt

# Kaggle API token → ~/.kaggle/kaggle.json (chmod 600)
# Accept rules on the competition page, then:
python scripts/setup_data.py
```

Without `~/.kaggle/kaggle.json`, you can manually download **Data** from the competition page and unpack into `data/` so that `data/sample_submission/cg/` exists.

## Day-to-day

```bash
# edit agent/main.py and/or agent/deck.csv (or --deck decks/...)
python scripts/package.py
python scripts/self_play.py --games 5

# Upload dist/submission.tar.gz (CLI or Kaggle Submit Agent UI)
python scripts/submit.py -m "random agent + lucario deck"
```


## Suggested improvement path

1. **Valid submission** — random agent + legal 60-card deck (this repo’s starting point).
2. **Rule-based policy** — score options by context (official Mega Lucario / Mega Abomasnow notebooks on the Code tab).
3. **Deck craft** — tune archetype vs meta; keep deck and policy paired.
4. **Search / RL** — CABT `search_begin` / `search_step` for look-ahead; self-play eval before burning daily submits.
5. **Strategy writeup** — needed for prize track; document deck thesis + decision logic + ladder evidence.

## Notes

- Competition card CSVs / `cg/` binaries are **Competition Use only** — keep them under `data/` (gitignored).
- The old root `sim.py` used a non-CABT observation shape; the real entrypoint is `agent/main.py`.
- Viewer / local UI option: [cabt-viewer](https://github.com/charlielockyer-rice/cabt-viewer) (still needs your local `sample_submission`).
