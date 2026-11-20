#!/usr/bin/env python3
"""Validate and submit a CSV to Santa 2026 - The Elf Calculator.

Usage:
  python scripts/submit.py -f dist/submission.csv -m "baseline v1"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
COMPETITION = "santa-2026"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=ROOT / "dist" / "submission.csv",
        help="Submission CSV to validate and upload (default: dist/submission.csv)",
    )
    parser.add_argument(
        "-m",
        "--message",
        default="santa-2026 submission",
        help="Submission message",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip local validation before upload",
    )
    args = parser.parse_args()

    path = args.file.resolve()
    if not path.is_file():
        print(f"Missing submission file: {path}", file=sys.stderr)
        return 1

    if not args.skip_validate:
        rc = subprocess.run(
            [sys.executable, str(SCRIPTS / "validate.py"), str(path)],
            check=False,
        ).returncode
        if rc != 0:
            print("Refusing to submit invalid CSV.", file=sys.stderr)
            return rc

    kaggle_bin = shutil.which("kaggle")
    cmd = (
        [kaggle_bin, "competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", args.message]
        if kaggle_bin
        else [
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
        ]
    )
    print(f"Submitting {path} -> {COMPETITION}")
    print(f"Message: {args.message}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "Submit failed. Check ~/.kaggle/kaggle.json and accepted rules at:\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules",
            file=sys.stderr,
        )
        return exc.returncode

    print(f"OK — check status with: kaggle competitions submissions -c {COMPETITION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
