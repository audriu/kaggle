#!/usr/bin/env python3
"""Prepare / guide Strategy competition submission.

Official deliverable is a **Kaggle Writeup** (one per team), not an agent archive:

  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/projects

This script:
  1. Runs package_writeup.py (word-count + dist/writeup.md bundle)
  2. Prints paste-ready steps and the Submit URL
  3. With --file-submit, attempts `kaggle competitions submit` of the markdown
     (usually rejected for hackathon/writeup comps — use only to probe)

Usage:
  python scripts/submit.py
  python scripts/submit.py --open
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DIST = ROOT / "dist"
WRITEUP_MD = DIST / "writeup.md"
COMPETITION = "pokemon-tcg-ai-battle-challenge-strategy"
PROJECTS_URL = (
    f"https://www.kaggle.com/competitions/{COMPETITION}/projects"
)


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
    parser.add_argument(
        "-m",
        "--message",
        default="Mega Lucario strategy writeup",
        help="Message if using --file-submit",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the New Writeup / projects page in a browser",
    )
    parser.add_argument(
        "--file-submit",
        action="store_true",
        help=(
            "Attempt kaggle competitions submit of dist/writeup.md "
            "(not the official Writeup path; may fail or not be judged)"
        ),
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Do not rebuild dist/ first",
    )
    args = parser.parse_args()

    if not args.skip_package:
        pkg = _load_package()
        rc = pkg.main()
        if rc != 0:
            return rc

    if not WRITEUP_MD.is_file():
        print(f"Missing {WRITEUP_MD} — run package_writeup.py first", file=sys.stderr)
        return 1

    wc = (DIST / "word_count.txt").read_text(encoding="utf-8").strip()
    print("Strategy submission is a Kaggle Writeup (hackathon), not an agent tar.gz.")
    print(f"Packaged: {WRITEUP_MD}  ({wc})")
    print()
    print("Steps:")
    print(f"  1. Open {PROJECTS_URL}")
    print('  2. Click "New Writeup" and select a Track')
    print("  3. Paste title/subtitle + body from dist/writeup.md")
    print("  4. Optionally attach assets from dist/writeup_bundle.zip / assets/")
    print('  5. Click "Submit" (teams get ONE writeup submission — rules)')
    print()
    print(f"Simulation agent to reference: ../pokemon-tcg-ai-battle/agent/")

    if args.open:
        webbrowser.open(PROJECTS_URL)
        print(f"Opened {PROJECTS_URL}")

    if args.file_submit:
        creds = Path.home() / ".kaggle" / "kaggle.json"
        if not creds.exists():
            print("Missing ~/.kaggle/kaggle.json", file=sys.stderr)
            return 1
        print()
        print(
            f"WARNING: file-submitting to {COMPETITION} — official path is Writeup UI.",
            file=sys.stderr,
        )
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
                    str(WRITEUP_MD),
                    "-m",
                    args.message,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                "CLI file submit failed (expected for this hackathon).\n"
                "Use the Writeup UI steps above — judges only score submitted Writeups.",
                file=sys.stderr,
            )
            return exc.returncode
        print("CLI upload finished — verify on the competition Submissions / Writeups tab.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
