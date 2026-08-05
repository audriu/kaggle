# Kaggle competitions workspace

Local workspace for active Kaggle competition tracks.

| Folder | Competition | What you submit |
| --- | --- | --- |
| [`competitions/kaggriculture`](competitions/kaggriculture/) | [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) | Farming agent (`main.py` / `submission.tar.gz`) |
| [`competitions/pokemon-tcg-ai-battle`](competitions/pokemon-tcg-ai-battle/) | [PTCG Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) | Agent bundle (`submission.tar.gz`) on the skill ladder |
| [`competitions/pokemon-tcg-ai-battle-challenge-strategy`](competitions/pokemon-tcg-ai-battle-challenge-strategy/) | [PTCG Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy) | Kaggle Writeup (deck + logic + sim evidence) → Japan finals |

PTCG Strategy requires Simulation participation. Shared deps live at the repo root.

## Setup

```bash
cd /home/a/code/kaggle
source .venv/bin/activate
pip install -r requirements.txt
```

Then work inside the competition folder you care about (see that folder’s README).

- **Kaggriculture** — `cd competitions/kaggriculture && python scripts/submit.py`
- **PTCG Simulation** — `cd competitions/pokemon-tcg-ai-battle && python scripts/submit.py`
- **PTCG Strategy** — `cd competitions/pokemon-tcg-ai-battle-challenge-strategy && python scripts/submit.py --open` (final Submit is in the Kaggle Writeup UI)
