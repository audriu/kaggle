#!/usr/bin/env python3
"""Download Strategy competition card data into ./data.

Requires ~/.kaggle/kaggle.json and accepted rules:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/rules

Note: Strategy has no CABT engine bundle — only card metadata CSVs/PDFs.
The playable agent lives in ../pokemon-tcg-ai-battle/.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPETITION = "pokemon-tcg-ai-battle-challenge-strategy"


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print(
            "Missing ~/.kaggle/kaggle.json\n"
            "1. Create an API token at https://www.kaggle.com/settings\n"
            "2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "3. chmod 600 ~/.kaggle/kaggle.json\n"
            f"4. Accept rules: https://www.kaggle.com/competitions/{COMPETITION}/rules",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {COMPETITION} → {DATA}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(DATA),
        ],
        check=True,
    )

    for zpath in DATA.glob("*.zip"):
        print(f"Extracting {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(DATA)

    csvs = sorted(DATA.glob("*Card*Data*.csv")) + sorted(DATA.glob("*Card Data.csv"))
    print("OK")
    if csvs:
        for p in csvs:
            print(f"  {p.name}")
    else:
        print("  (no card CSVs found yet — check data/)", file=sys.stderr)
    print("Next: draft writeup/draft.md → python scripts/package_writeup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
