#!/usr/bin/env python3
"""Build submission.tar.gz from agent/ + competition cg/ engine."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
DIST = ROOT / "dist"


def find_cg(data_root: Path) -> Path:
    candidates = [
        data_root / "sample_submission" / "cg",
        *data_root.glob("**/sample_submission/cg"),
        *data_root.glob("**/cg"),
    ]
    for path in candidates:
        if path.is_dir() and (path / "api.py").exists():
            return path
    raise FileNotFoundError(
        "Could not find cg/ (need data/sample_submission/cg).\n"
        "Run: python scripts/setup_data.py"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck",
        type=Path,
        default=None,
        help="Optional deck.csv to copy into the bundle (default: agent/deck.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DIST / "submission.tar.gz",
        help="Output archive path",
    )
    args = parser.parse_args()

    main_py = AGENT / "main.py"
    deck_csv = args.deck or (AGENT / "deck.csv")
    if not main_py.exists():
        print(f"Missing {main_py}", file=sys.stderr)
        return 1
    if not deck_csv.exists():
        print(f"Missing {deck_csv}", file=sys.stderr)
        return 1

    deck_ids = [ln.strip() for ln in deck_csv.read_text().splitlines() if ln.strip()]
    if len(deck_ids) != 60:
        print(f"Deck must have 60 cards, got {len(deck_ids)} in {deck_csv}", file=sys.stderr)
        return 1

    cg = find_cg(ROOT / "data")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copy2(main_py, staging / "main.py")
        shutil.copy2(deck_csv, staging / "deck.csv")
        shutil.copytree(cg, staging / "cg")

        with tarfile.open(args.out, "w:gz") as tar:
            for name in ("main.py", "deck.csv", "cg"):
                tar.add(staging / name, arcname=name)

    with tarfile.open(args.out, "r:gz") as tar:
        names = tar.getnames()
    top = {n.split("/")[0] for n in names}
    print(f"Created {args.out.resolve()}")
    print(f"Top-level entries: {sorted(top)}")
    print(f"Files: {len(names)}  cg source: {cg}")
    if "main.py" not in top or "deck.csv" not in top or "cg" not in top:
        print("ERROR: archive layout invalid", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
