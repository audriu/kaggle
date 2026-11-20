#!/usr/bin/env python3
"""Download or extract Santa 2026 - The Elf Calculator data into ./data/.

Requires ~/.kaggle/kaggle.json (or KAGGLE_USERNAME+KAGGLE_KEY / KAGGLE_API_TOKEN)
and accepted rules at:
  https://www.kaggle.com/competitions/santa-2026/rules

Usage:
  python scripts/setup_data.py
  python scripts/setup_data.py --from-zip ~/Downloads/santa-2026.zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPETITION = "santa-2026"
EXPECTED_FILES = ("puzzles.csv", "practice_puzzles.csv", "sample_submission.csv")


def has_kaggle_creds() -> bool:
    if (Path.home() / ".kaggle" / "kaggle.json").is_file():
        return True
    if (Path.home() / ".kaggle" / "access_token").is_file():
        return True
    if (Path.home() / ".config" / "kaggle" / "kaggle.json").is_file():
        return True
    if (Path.home() / ".config" / "kaggle" / "access_token").is_file():
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def extract_zip(zpath: Path, delete_after: bool = False) -> None:
    print(f"Extracting {zpath} -> {DATA}")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(DATA)
    if delete_after:
        zpath.unlink()
    nested = DATA / COMPETITION
    if nested.is_dir():
        for src in sorted(nested.iterdir()):
            dst = DATA / src.name
            if dst.exists():
                dst.unlink()
            src.rename(dst)
        nested.rmdir()


def download() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if not has_kaggle_creds():
        print(
            "Missing Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN).\n"
            "1. Create an API token at https://www.kaggle.com/settings\n"
            "2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
            f"3. Accept rules at https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
            "Or download santa-2026.zip in your browser and run:\n"
            "  python scripts/setup_data.py --from-zip ~/Downloads/santa-2026.zip",
            file=sys.stderr,
        )
        return 1

    kaggle_bin = shutil.which("kaggle")
    cmd = (
        [kaggle_bin, "competitions", "download", "-c", COMPETITION, "-p", str(DATA)]
        if kaggle_bin
        else [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA)]
    )
    print(f"Downloading {COMPETITION} -> {DATA}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "Download failed. Ensure you have accepted competition rules at:\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules",
            file=sys.stderr,
        )
        return exc.returncode

    for zpath in sorted(DATA.glob("*.zip")):
        extract_zip(zpath, delete_after=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-zip",
        type=Path,
        default=None,
        help="Extract an already-downloaded competition zip into data/",
    )
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if args.from_zip is not None:
        zpath = args.from_zip.expanduser().resolve()
        if not zpath.is_file():
            print(f"Zip not found: {zpath}", file=sys.stderr)
            return 1
        extract_zip(zpath, delete_after=False)
    else:
        rc = download()
        if rc != 0:
            return rc

    present = [f for f in EXPECTED_FILES if (DATA / f).is_file()]
    missing = [f for f in EXPECTED_FILES if f not in present]
    print(f"OK — {len(present)}/{len(EXPECTED_FILES)} files in {DATA}: {', '.join(present)}")
    if missing:
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
