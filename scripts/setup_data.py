#!/usr/bin/env python3
"""Download Pokémon TCG AI Battle competition data into ./data.

Requires a Kaggle API token at ~/.kaggle/kaggle.json
(Create one at https://www.kaggle.com/settings → API).

Accept competition rules first:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPETITION = "pokemon-tcg-ai-battle"


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print(
            "Missing ~/.kaggle/kaggle.json\n"
            "1. Create an API token at https://www.kaggle.com/settings\n"
            "2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "3. chmod 600 ~/.kaggle/kaggle.json\n"
            "4. Accept rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules",
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

    sample = _find_sample_submission(DATA)
    if sample is None:
        print("Warning: sample_submission/ not found after extract.", file=sys.stderr)
        return 1

    cg = sample / "cg"
    if not cg.is_dir():
        print(f"Warning: {cg} missing — packaging will fail until present.", file=sys.stderr)
        return 1

    print("OK")
    print(f"  sample_submission: {sample}")
    print(f"  engine (cg/):      {cg}")
    print("Next: python scripts/package.py && python scripts/self_play.py")
    return 0


def _find_sample_submission(root: Path) -> Path | None:
    direct = root / "sample_submission"
    if (direct / "cg").is_dir():
        return direct
    for path in root.rglob("sample_submission"):
        if path.is_dir() and (path / "cg").is_dir():
            return path
    return None


if __name__ == "__main__":
    raise SystemExit(main())
