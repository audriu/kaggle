#!/usr/bin/env python3
"""Prepare the Big Data Bowl 2027 submission.

Official deliverable is a **Kaggle Writeup** (hackathon, one per team) with an attached public
Kaggle Notebook — there is no file upload / leaderboard:

  https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/writeups

This script:
  1. Runs package_writeup.py (word + figure count, dist/writeup.md bundle)
  2. Prints the paste-ready steps and opens the Writeups page with --open

Usage:
  python scripts/submit.py
  python scripts/submit.py --open
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DIST = ROOT / "dist"
WRITEUP_MD = DIST / "writeup.md"
COMPETITION = "nfl-big-data-bowl-2027"
WRITEUPS_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/writeups"


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "package_writeup", SCRIPTS / "package_writeup.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load package_writeup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="open the Writeups page")
    parser.add_argument("--skip-package", action="store_true", help="do not rebuild dist/")
    parser.add_argument(
        "--force", action="store_true", help="print steps even if packaging reports problems"
    )
    args = parser.parse_args()

    if not args.skip_package:
        rc = _load_package().main()
        if rc != 0 and not args.force:
            print("Fix the warnings above (or pass --force).", file=sys.stderr)
            return rc

    if not WRITEUP_MD.is_file():
        print(f"Missing {WRITEUP_MD} — run package_writeup.py first", file=sys.stderr)
        return 1

    wc = (DIST / "word_count.txt").read_text(encoding="utf-8").strip()
    print()
    print("Big Data Bowl 2027 is a Writeup hackathon — no file submission, no leaderboard.")
    print(f"Packaged: {WRITEUP_MD}\n{wc}")
    print()
    print("Steps:")
    print(f"  1. Publish the analysis notebook as a PUBLIC Kaggle Notebook (required, PASS/FAIL)")
    print(f"  2. Open {WRITEUPS_URL}")
    print('  3. "New Writeup" → Track: Open (University only if all undergrads)')
    print("  4. Paste title/subtitle + body from dist/writeup.md; upload figures from assets/")
    print("  5. Attach the public notebook; check ≤ 2000 words and < 10 figures/tables")
    print('  6. "Submit" — ONE submission per team; deadline 2027-01-06 23:59 UTC')

    if args.open:
        webbrowser.open(WRITEUPS_URL)
        print(f"Opened {WRITEUPS_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
