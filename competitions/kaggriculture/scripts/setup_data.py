#!/usr/bin/env python3
"""Populate data/ with Kaggriculture docs; optionally download from Kaggle."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPETITION = "kaggriculture"


def copy_from_package() -> None:
    import kaggle_environments

    src = (
        Path(kaggle_environments.__file__).resolve().parent
        / "envs"
        / "kaggriculture"
    )
    if not src.is_dir():
        raise FileNotFoundError(f"kaggriculture env not found at {src}")
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("AGENTS.md", "README.md"):
        shutil.copy2(src / name, DATA / name)
        print(f"Copied {name} ← {src / name}")


def download_kaggle() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(DATA),
        "-o",
    ]
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "Download failed (have you joined the competition?)\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}\n"
            "Docs were still copied from kaggle-environments.",
            file=sys.stderr,
        )
        return exc.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Also download competition files via Kaggle API",
    )
    args = parser.parse_args()
    copy_from_package()
    if args.kaggle:
        return download_kaggle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
