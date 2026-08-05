# Pokémon TCG AI Battle Challenge — Strategy

Local workspace for [The Pokémon Company – PTCG AI Battle Challenge Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy).

Playable agent + CABT packaging: [`../pokemon-tcg-ai-battle`](../pokemon-tcg-ai-battle/).

## What you submit

A **Kaggle Writeup** (hackathon), not an agent `tar.gz`:

| Item | Detail |
| --- | --- |
| Prerequisite | Same team must also play [Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) |
| Deliverable | Writeup ≤ ~2000 words + optional media gallery |
| Official submit UI | [New Writeup / projects](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/projects) |
| Limit | **One** writeup submission per team (rules) |
| Judging | Model ~70% · Deck ~20% · Report ~10% |
| Deadline | **13 Sep 2026** (approx.) |

## Layout

```text
writeup/draft.md   # body you paste into the Kaggle Writeup
notes/             # eval tables, scratch
assets/            # figures for the media gallery
scripts/
  setup_data.py       # download card CSVs/PDFs → data/
  package_writeup.py  # word-count + dist/writeup.md + zip
  submit.py           # package + print Writeup UI steps (--open)
data/              # competition card metadata (gitignored)
dist/              # packaged writeup outputs (gitignored)
```

## Day-to-day

```bash
cd /home/a/code/kaggle
source .venv/bin/activate

cd competitions/pokemon-tcg-ai-battle-challenge-strategy
python scripts/setup_data.py          # once
# edit writeup/draft.md
python scripts/package_writeup.py     # checks ≤2000 words
python scripts/submit.py --open       # opens Writeup page + paste instructions
```

Paste `dist/writeup.md` into the Writeup editor, attach gallery assets if any, select a Track, then **Submit** in the UI.
