# Kaggriculture

Local workspace for [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) (Google × Kaggle, $50k, deadline ~30 Sep 2026).

Build an AI **agent** that manages a virtual farm for one 30-day season (720 turns) and maximizes banked coins against an opponent.

## Competition summary

| | |
| --- | --- |
| Type | Simulation (2-player) |
| Submit | `main.py` or `submission.tar.gz` with `agent(obs)` |
| Board | 10×10 farm, four 5×5 quadrants (NW free; buy NE/SW/SE) |
| Start | $3000 |
| Act timeout | 1s / turn |

Docs (also shipped inside `kaggle-environments`):

- [`data/AGENTS.md`](data/AGENTS.md) — agent API, local test, CLI submit
- [`data/README.md`](data/README.md) — full rules, prices, town/market

## Current agent

`agent/main.py` — multi-tile heuristic:

1. Hire cheap farm hands each morning; carpet unlocked land with **carrots** (+ some **wheat** for feed).
2. Priority loop: feed → harvest → water → dig weeds → plant / build.
3. Mid-season: buy **geese**, build coops, place/feed/harvest eggs; optional cows later.
4. Buy land when empty tiles run low; sell produce when prices are decent.

Local check (3 seeds): typically beats `"starter"` by a wide margin (~10k–23k vs ~3.3k), with some seed variance.

## Layout

```text
agent/main.py     # edit this
scripts/
  setup_data.py   # copy docs from kaggle-environments (+ optional Kaggle download)
  package.py      # dist/main.py (+ optional tar.gz)
  submit.py       # package then kaggle competitions submit
data/             # AGENTS.md / README.md (gitignored except .gitkeep)
dist/             # built submission
```

## Setup

```bash
cd /home/a/code/kaggle
source .venv/bin/activate
pip install -r requirements.txt

cd competitions/kaggriculture
python scripts/setup_data.py
```

**Join the competition** (required before download/submit):  
https://www.kaggle.com/competitions/kaggriculture → **Join Competition**

## Local test

```bash
python -c "
from kaggle_environments import make
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1})
env.run(['agent/main.py', 'starter'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
```

## Submit

```bash
python scripts/submit.py -m "carrot+goose heuristic v1"
```
