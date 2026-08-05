#!/usr/bin/env python3
"""Copy agent/main.py into dist/ for Kaggle submit (single-file or tar.gz)."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent" / "main.py"
DIST = ROOT / "dist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tar",
        action="store_true",
        help="Also write dist/submission.tar.gz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DIST / "main.py",
        help="Output path for the agent file",
    )
    args = parser.parse_args()

    if not AGENT.is_file():
        print(f"Missing {AGENT}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGENT, args.out)
    print(f"Wrote {args.out.resolve()}")

    if args.tar:
        tar_path = DIST / "submission.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname="main.py")
        print(f"Wrote {tar_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
