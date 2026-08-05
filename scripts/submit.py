#!/usr/bin/env python3
"""Submit dist/submission.tar.gz to the Pokémon TCG AI Battle competition.

Requires a Kaggle API token at ~/.kaggle/kaggle.json
(Create one at https://www.kaggle.com/settings → API).

Accept competition rules first:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

Usage:
  python scripts/submit.py
  python scripts/submit.py -m "random agent + lucario deck"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "dist" / "submission.tar.gz"
COMPETITION = "pokemon-tcg-ai-battle"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--message",
        default="submission",
        help='Submission message (default: "submission")',
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=SUBMISSION,
        help=f"Archive to upload (default: {SUBMISSION})",
    )
    args = parser.parse_args()

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

    path = args.file.resolve()
    if not path.is_file():
        print(
            f"Missing {path}\n"
            "Build it first: python scripts/package.py",
            file=sys.stderr,
        )
        return 1

    print(f"Submitting {path} → {COMPETITION}")
    print(f"Message: {args.message}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            str(path),
            "-m",
            args.message,
        ],
        check=True,
    )
    print("OK — check status on the competition Submissions tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
