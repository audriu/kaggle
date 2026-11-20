#!/usr/bin/env python3
"""Merge two or more valid Santa 2026 submission CSVs, keeping the shortest valid program per puzzle.

Usage:
  python scripts/merge_best.py dist/out.csv sub1.csv sub2.csv [--puzzles data/puzzles.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Merged output CSV")
    parser.add_argument("inputs", nargs="+", type=Path, help="Candidate submission CSVs")
    parser.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.csv")
    args = parser.parse_args()

    csv.field_size_limit(10 * 1024 * 1024)
    puzzles: dict[str, tuple[list[int], list[int], int, int]] = {}
    order: list[str] = []
    with args.puzzles.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            order.append(pid)
            puzzles[pid] = (
                json.loads(row["starting_values"]),
                json.loads(row["targets"]),
                int(row["value_limit"]),
                int(row["registers"]),
            )

    best_prog: dict[str, str] = {}
    best_len: dict[str, int] = {}
    best_src: dict[str, str] = {}

    for path in args.inputs:
        if not path.is_file():
            print(f"Skipping missing {path}", file=sys.stderr)
            continue
        wins = 0
        total = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = row["id"]
                if pid not in puzzles:
                    continue
                starts, targets, vlim, nreg = puzzles[pid]
                ok, steps, err = validate.validate_program(pid, starts, targets, vlim, nreg, row["program"])
                if not ok:
                    print(f"WARN invalid in {path}: {err}", file=sys.stderr)
                    continue
                total += steps
                if pid not in best_len or steps < best_len[pid]:
                    best_len[pid] = steps
                    best_prog[pid] = row["program"]
                    best_src[pid] = path.name
                    wins += 1
        print(f"Loaded {path.name}: total={total:,}, new_bests={wins}")

    missing = [pid for pid in order if pid not in best_prog]
    if missing:
        print(f"Missing {len(missing)} puzzles!", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in order:
            w.writerow([pid, best_prog[pid]])

    print(f"Merged -> {args.output}: total_instructions = {sum(best_len.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
