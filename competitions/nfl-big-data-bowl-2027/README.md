# NFL Big Data Bowl 2027

Local workspace for [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
(National Football League, **$100,000**, Featured **hackathon**, deadline **6 Jan 2027 23:59 UTC**).

**Theme: Combine → NFL performance.** Build an analytical framework that uncovers non-obvious,
actionable linkages between 10 Hz Combine sensor tracking data and regular-season NFL game
performance, for the 510 rookies drafted/signed 2023–2025.

## What you submit

A **Kaggle Writeup** (judged by NFL team analytics staff), not a model or predictions file.

| Item | Detail |
| --- | --- |
| Deliverable | Writeup ≤ **2,000 words**, < **10** tables/figures, readable markdown, embedded visuals |
| Hard requirement | Must explicitly link Combine **tracking** data to regular-season NFL performance (PASS/FAIL) |
| Hard requirement | Must attach a **public Kaggle Notebook** (PASS/FAIL) |
| Tracks | Open ($45k) · University ($45k, undergrads only) · Grand Prize $10k for finalists presenting at the 2027 Combine |
| Limit | **One** writeup submission per team (hackathon); team size ≤ 5 |
| Judging | Football 30% · Data Science 30% · Writeup 20% · Data Viz 20% (each 0–10) |
| Timeline | Start 6 Oct 2026 · Deadline 6 Jan 2027 · Judging 7–25 Jan · Results ~26 Jan 2027 |
| Data license | CC BY-NC 4.0 — keep it inside `data/` (gitignored), never commit |
| Winner license | Open source (OSI-approved) for the writeup + code |

Full rubric and rule notes: [`notes/rubric.md`](notes/rubric.md).
Column dictionary: [`notes/data_dictionary.md`](notes/data_dictionary.md).

## Data (≈2.2 GiB, 9 CSVs)

| File | Rows | Keys | What |
| --- | --- | --- | --- |
| `players.csv` | 510 | `nfl_id` | prospect bio, college, draft slot |
| `combine_results.csv` | 510 | `nfl_id` | measurables, 40/10-split, 3-cone, shuttle, NGS scores |
| `combine_tracking.csv` | 463k | `event_id, nfl_id, time` | **10 Hz Combine drill tracking** (6,310 attempts) |
| `player_career_successes.csv` | 510 | `nfl_id` | career snaps, starts, All-Pro, Pro Bowl |
| `player_play.csv` | 316k | `game_id, play_id, nfl_id` | per-snap NGS metrics (separation, pressure, get-off, EPA …) |
| `games.csv` | 1,002 | `game_id` | schedule 2023–2025 |
| `game_tracking_{2023,2024,2025}.csv` | 1.4M / 3.0M / 4.3M | `game_id, play_id, nfl_id, time` | **10 Hz in-game tracking** for the cohort only |

## Layout

```text
writeup/draft.md     # body you paste into the Kaggle Writeup (rubric-shaped skeleton)
notebooks/           # analysis notebooks; the final one must be published as a public Kaggle Notebook
src/bdb/             # small helper package: paths, loaders, parquet cache, tracking utils
notes/               # rubric, data dictionary, research plan, progress log
assets/              # figures for the writeup (≤ 10 figures/tables in the final)
scripts/
  setup_data.py      # kaggle download → data/ (+ --parquet cache)
  data_summary.py    # shapes / nulls / key coverage sanity report
  package_writeup.py # word + figure count → dist/writeup.md + bundle zip
  submit.py          # package, print Writeup UI steps (--open)
data/                # competition CSVs + parquet cache (gitignored)
dist/                # packaged writeup outputs (gitignored)
```

## Setup

```bash
cd /home/a/code/kaggle
source .venv/bin/activate
pip install -r requirements.txt
pip install -r competitions/nfl-big-data-bowl-2027/requirements.txt

cd competitions/nfl-big-data-bowl-2027
python scripts/setup_data.py --parquet     # ~2.2 GiB download, then CSV → parquet
python scripts/data_summary.py             # sanity check
```

Rules are already accepted on this account (`kaggle competitions list -s nfl-big-data-bowl-2027` shows `userHasEntered True`).

## Day-to-day

```bash
# explore / build metrics
python -c "from bdb import load; print(load.players().head())"   # with src/ on PYTHONPATH (see src/bdb/__init__.py)
jupyter lab notebooks/

# writeup
python scripts/package_writeup.py     # checks ≤ 2000 words, < 10 figures
python scripts/submit.py --open       # opens the Writeup page + paste steps
```

Final submit is in the Kaggle Writeup UI: paste `dist/writeup.md`, attach figures, link the public
notebook, pick the **Open** track, then **Submit** (one per team).
