# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A workspace for Kaggle competitions, one folder per competition under
[competitions/](competitions/): three **agent/simulation** tracks plus one **analytics writeup**
hackathon (NFL Big Data Bowl 2027). There is no library, no test suite, and no linter config — each
folder is a self-contained "edit the agent, package it, submit it" loop driven by small
`scripts/*.py` CLIs. Shared deps (`kaggle`, `kaggle-environments`) live at the repo root.

| Folder | Deliverable |
| --- | --- |
| [competitions/kaggriculture/](competitions/kaggriculture/) | single-file farming agent (`main.py`) |
| [competitions/pokemon-tcg-ai-battle/](competitions/pokemon-tcg-ai-battle/) | agent bundle `submission.tar.gz` (`main.py` + `deck.csv` + `cg/`) |
| [competitions/pokemon-tcg-ai-battle-challenge-strategy/](competitions/pokemon-tcg-ai-battle-challenge-strategy/) | Kaggle **Writeup** (markdown), not code |
| [competitions/nfl-big-data-bowl-2027/](competitions/nfl-big-data-bowl-2027/) | Kaggle **Writeup** (≤2000 words, <10 figures) + attached **public Kaggle Notebook** |

The two Pokémon folders are one effort: the Strategy writeup describes the Simulation agent in
[competitions/pokemon-tcg-ai-battle/agent/main.py](competitions/pokemon-tcg-ai-battle/agent/main.py)
and requires Simulation participation.

## Environment

```bash
cd /home/a/code/kaggle
source .venv/bin/activate     # already created; python3.12
pip install -r requirements.txt
```

All commands below run from inside the relevant competition folder. Kaggle API calls need
`~/.kaggle/kaggle.json` (chmod 600) and accepted competition rules; `setup_data.py` / `submit.py`
print recovery instructions when either is missing.

`data/`, `dist/`, `tmp_logs/` are gitignored. Competition card data and the `cg/` SDK are
**Competition Use only** — never copy them out of `data/` into tracked paths.

## Common commands

PTCG Simulation (`competitions/pokemon-tcg-ai-battle`):

```bash
python scripts/setup_data.py            # kaggle download → data/ (needs sample_submission/cg)
python scripts/package.py               # build dist/submission.tar.gz + validate
python scripts/package.py --deck decks/mega_lucario_ex.csv
python scripts/validate_submission.py [dist/submission.tar.gz | agent/]
python scripts/self_play.py --games 5 [--html out.html]
python scripts/submit.py -m "message"   # validates, then kaggle competitions submit
```

Kaggriculture (`competitions/kaggriculture`):

```bash
python scripts/setup_data.py [--kaggle]   # copies AGENTS.md/README.md out of kaggle_environments
python scripts/package.py [--tar]         # dist/main.py
python scripts/submit.py -m "message"
python -c "
from kaggle_environments import make
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1})
env.run(['agent/main.py', 'starter'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
```

PTCG Strategy (`competitions/pokemon-tcg-ai-battle-challenge-strategy`):

```bash
python scripts/setup_data.py
python scripts/package_writeup.py       # dist/writeup.md + word_count.txt + writeup_bundle.zip
python scripts/submit.py --open         # packages, prints steps, opens the Writeup UI
```

Final Strategy submission happens **in the Kaggle Writeup UI** (one writeup per team, ≤2000
words); `submit.py --file-submit` is a non-official fallback that is expected to fail.

NFL Big Data Bowl 2027 (`competitions/nfl-big-data-bowl-2027`):

```bash
pip install -r requirements.txt             # per-competition data-science deps (pandas/polars/sklearn/…)
python scripts/setup_data.py --parquet      # ~2.2 GiB kaggle download → data/ + data/parquet cache
python scripts/data_summary.py [--small] [--out notes/data_summary.md]
python scripts/package_writeup.py           # dist/writeup.md, word + figure count gates
python scripts/submit.py --open             # prints Writeup UI steps; final submit is in the UI
```

Analysis code imports `src/bdb` (`paths`, `load`, `tracking`); scripts add `src/` to `sys.path`
themselves, notebooks do `sys.path.insert(0, "../src")`. Loaders prefer the parquet cache and use
polars for the 1 GB tracking CSVs. Rubric, column dictionary and plan live in `notes/`
(`rubric.md`, `data_dictionary.md`, `research_plan.md`); keep `notes/progress_log.md` current.

## Architecture notes

### Kaggle-style loading is the critical invariant (PTCG)

Kaggle loads `main.py` via `exec()` with **no `__file__`**, cwd = agent directory. Everything in
this folder reproduces that on purpose, so path bugs fail locally instead of on the graded
validation episode:

- [scripts/validate_submission.py](competitions/pokemon-tcg-ai-battle/scripts/validate_submission.py)
  owns `load_agent_like_kaggle()` + `smoke_test_agent()` + archive-layout checks.
- [scripts/package.py](competitions/pokemon-tcg-ai-battle/scripts/package.py),
  [scripts/submit.py](competitions/pokemon-tcg-ai-battle/scripts/submit.py) and
  [scripts/self_play.py](competitions/pokemon-tcg-ai-battle/scripts/self_play.py) all load that
  module by path (`importlib.util.spec_from_file_location`) rather than importing it, and refuse
  to proceed on failure. Keep new tooling on that same path — do not add a plain `import main`.
- The agent therefore resolves `deck.csv` relative to cwd with a
  `/kaggle_simulations/agent/` fallback. Preserve that pattern when touching the top of
  `agent/main.py`.
- Archive layout must be flat: `main.py`, `deck.csv`, `cg/` at the tar root. `cg/` is located by
  globbing `data/` for a directory containing `api.py`.

### Agent contracts

PTCG (`agent(obs_dict) -> list[int]`): if `obs_dict["select"] is None`, return the 60 card IDs;
otherwise return indices into `obs_dict["select"]["option"]`, count within
`[minCount, maxCount]`, unique and in range. The engine enforces all rules — the agent only
picks among exposed legal options.

`agent/main.py` layers, roughly in file order: card-ID constants and deck tracking
(`reset_deck_track` / `consume_card` / `sync_deck_track`, plus `fill_hidden` for unseen cards) →
static evaluation (`pokemon_score`, `evaluate_state`) → a shallow look-ahead
(`search_refine`, using the engine's `search_begin`/`search_step`/`search_end`/`search_release`,
bounded by `SEARCH_NODE_BUDGET` / `SEARCH_TIME_BUDGET_S`) → `agent()`, a large per-`OptionType`
heuristic that `search_refine` may override on MAIN from turn 2. Module-level mutable state
(`plan`, `pre_turn`, `ability_used`, `deck_remaining`) persists across calls within a match;
it is reset by turn transitions, so re-check that when adding state.

Kaggriculture (`agent(obs) -> {"farmer": [...], "hands": [[...], ...], "market": [...]}`):
one command list per unit plus market orders, 1 s per turn, 720 turns. `agent()` in
[agent/main.py](competitions/kaggriculture/agent/main.py) is a thin shell: `_scan()` sweeps the
10×10 grid once into per-need tile lists, `_market()` decides buys/sells, and `_unit()` runs the
priority ladder (feed → harvest → water → weeds → plant/build) for the farmer and each hired
hand. `claimed` is a shared set threaded through the `_unit()` calls so two units never target
the same tile — pass it along when adding unit logic. The API/rules reference is
`data/AGENTS.md` + `data/README.md`, copied out of the installed `kaggle_environments` package.

### Eval before submitting

Big Data Bowl is judged on a rubric (Football 30 / Data Science 30 / Writeup 20 / Viz 20) with no
leaderboard and one submission per team, so the loop is EDA → metric → outcome model → figures →
writeup, with `package_writeup.py` as the only gate. Must use Combine *tracking* data and attach a
public Kaggle notebook or it is not scored.

Simulation allows 5 submissions/day and scores the latest 2, so prefer local evidence:
`scripts/self_play.py` (kaggle-environments `cabt` env, needs `data/sample_submission/cg`) and
`baselines/official_lucario.py` as the reference opponent. Results get logged in
[competitions/pokemon-tcg-ai-battle-challenge-strategy/notes/eval_log.md](competitions/pokemon-tcg-ai-battle-challenge-strategy/notes/eval_log.md),
which is the source of the Strategy writeup's results section.

### Dead code

`competitions/pokemon-tcg-ai-battle/sim.py` is a leftover stub using a non-CABT observation
shape. The real entrypoint is `agent/main.py`.
