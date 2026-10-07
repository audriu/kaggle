#!/usr/bin/env python3
"""Download NFL Big Data Bowl 2027 data into ./data (≈2.2 GiB, 9 CSVs).

Requires ~/.kaggle/kaggle.json and accepted rules:
  https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules

Usage:
  python scripts/setup_data.py              # download + unzip
  python scripts/setup_data.py --parquet    # also build data/parquet/*.parquet cache
  python scripts/setup_data.py --parquet --skip-download
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdb import paths  # noqa: E402

DATA = paths.DATA
COMPETITION = paths.COMPETITION


def download() -> int:
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

    print(f"Downloading {COMPETITION} → {DATA}  (≈2.2 GiB)")
    try:
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
    except subprocess.CalledProcessError as exc:
        print(
            "Download failed (have you joined the competition?)\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}",
            file=sys.stderr,
        )
        return exc.returncode

    for zpath in sorted(DATA.glob("*.zip")):
        print(f"Extracting {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(DATA)
        zpath.unlink()
    flatten()
    return 0


def flatten() -> None:
    """The zip nests files under data/<competition>/; move them up to data/."""
    nested = DATA / COMPETITION
    if not nested.is_dir():
        return
    for src in sorted(nested.iterdir()):
        dst = DATA / src.name
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        print(f"  moved {src.name} → data/")
    nested.rmdir()


def build_parquet() -> int:
    try:
        import polars as pl
    except ImportError:
        print("polars not installed: pip install -r requirements.txt", file=sys.stderr)
        return 1

    paths.CACHE.mkdir(parents=True, exist_ok=True)
    for name in paths.CSV_FILES:
        src = paths.csv(name)
        dst = paths.parquet(name)
        if not src.is_file():
            print(f"  skip {name} (missing)", file=sys.stderr)
            continue
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"  up to date {dst.name}")
            continue
        print(f"  {name} → {dst.name}")
        pl.scan_csv(
            src, null_values=paths.NULL_VALUES, infer_schema_length=paths.SCHEMA_ROWS
        ).sink_parquet(dst)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="store_true", help="build data/parquet cache")
    parser.add_argument("--skip-download", action="store_true", help="only build cache")
    args = parser.parse_args()

    if not args.skip_download:
        rc = download()
        if rc != 0:
            return rc
    else:
        flatten()

    present = [n for n in paths.CSV_FILES if paths.csv(n).is_file()]
    missing = [n for n in paths.CSV_FILES if n not in present]
    print(f"OK — {len(present)}/{len(paths.CSV_FILES)} CSVs in {DATA}")
    for n in missing:
        print(f"  missing: {n}", file=sys.stderr)

    if args.parquet:
        print("Building parquet cache")
        rc = build_parquet()
        if rc != 0:
            return rc

    print("Next: python scripts/data_summary.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
