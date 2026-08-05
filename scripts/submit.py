#!/usr/bin/env python3
"""Submit dist/submission.tar.gz to the Pokémon TCG AI Battle competition.

Validates the archive with a Kaggle-style load (exec without __file__) first.

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
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
SUBMISSION = ROOT / "dist" / "submission.tar.gz"
COMPETITION = "pokemon-tcg-ai-battle"


def _load_validate():
    spec = importlib.util.spec_from_file_location(
        "validate_submission", SCRIPTS / "validate_submission.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load validate_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip Kaggle-style validation (not recommended)",
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

    if not args.no_validate:
        validate = _load_validate()
        try:
            validate.validate_archive(path)
        except Exception as exc:
            print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
            print("Refusing to submit. Fix the agent or pass --no-validate.", file=sys.stderr)
            return 1
        print("Validated (Kaggle-style exec + smoke test)")

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
    print("OK — check status with: kaggle competitions submissions -c pokemon-tcg-ai-battle")
    print("Logs: kaggle competitions episodes <submission_id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
