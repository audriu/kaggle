#!/usr/bin/env python3
"""Submit agent/main.py to the Kaggriculture competition.

Requires a Kaggle API token and accepted competition rules:
  https://www.kaggle.com/competitions/kaggriculture

Usage:
  python scripts/submit.py
  python scripts/submit.py -m "carrot+goose v1"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
COMPETITION = "kaggriculture"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--message",
        default="carrot+goose heuristic",
        help="Submission message",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        help="File to upload (default: package agent/main.py → dist/main.py)",
    )
    args = parser.parse_args()

    path = args.file
    if path is None:
        rc = subprocess.run(
            [sys.executable, str(SCRIPTS / "package.py")],
            check=False,
        ).returncode
        if rc != 0:
            return rc
        path = ROOT / "dist" / "main.py"

    path = path.resolve()
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1

    print(f"Submitting {path} → {COMPETITION}")
    print(f"Message: {args.message}")
    try:
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
    except subprocess.CalledProcessError as exc:
        print(
            "Submit failed. If you have not joined yet, accept rules at:\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}\n"
            "Then verify: kaggle competitions list --group entered -s kaggriculture",
            file=sys.stderr,
        )
        return exc.returncode

    print(f"OK — status: kaggle competitions submissions -c {COMPETITION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
