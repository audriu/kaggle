#!/usr/bin/env python3
"""Exact local validator and scorer for Santa 2026 - The Elf Calculator.

Usage:
  python scripts/validate.py dist/submission.csv
  python scripts/validate.py data/sample_submission.csv --puzzles data/puzzles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_INSTRUCTIONS = 4096
MAX_PROGRAM_BYTES = 131072
VALID_OPS = frozenset(("+", "-", "*", "/"))


def validate_program(
    puzzle_id: str,
    starting_values: list[int],
    targets: list[int],
    value_limit: int,
    num_registers: int,
    program_str: str,
) -> tuple[bool, int, str]:
    raw_bytes = len(program_str.encode("utf-8"))
    if raw_bytes > MAX_PROGRAM_BYTES:
        return False, 0, f"{puzzle_id}: program exceeds {MAX_PROGRAM_BYTES} bytes ({raw_bytes})"

    try:
        program: Any = json.loads(program_str)
    except json.JSONDecodeError as exc:
        return False, 0, f"{puzzle_id}: invalid JSON ({exc})"

    if not isinstance(program, list):
        return False, 0, f"{puzzle_id}: program must be a JSON list"
    if len(program) > MAX_INSTRUCTIONS:
        return False, 0, f"{puzzle_id}: too many instructions ({len(program)} > {MAX_INSTRUCTIONS})"

    regs: list[int | None] = [None] * num_registers
    for idx, val in enumerate(starting_values):
        regs[idx] = val

    remaining = set(targets) - set(starting_values)

    for step, inst in enumerate(program):
        if not isinstance(inst, list) or len(inst) != 4:
            return False, 0, f"{puzzle_id} step {step}: instruction must be [op, i, j, d]"
        op, i, j, d = inst
        if op not in VALID_OPS:
            return False, 0, f"{puzzle_id} step {step}: invalid op {op!r}"
        if (
            not isinstance(i, int)
            or isinstance(i, bool)
            or not isinstance(j, int)
            or isinstance(j, bool)
            or not isinstance(d, int)
            or isinstance(d, bool)
        ):
            return False, 0, f"{puzzle_id} step {step}: register indices must be ints"
        if not (0 <= i < num_registers and 0 <= j < num_registers and 0 <= d < num_registers):
            return False, 0, f"{puzzle_id} step {step}: register index out of range 0..{num_registers - 1}"

        a = regs[i]
        b = regs[j]
        if a is None or b is None:
            return False, 0, f"{puzzle_id} step {step}: read from undefined register ({i}={a}, {j}={b})"

        if op == "+":
            res = a + b
        elif op == "-":
            if a <= b:
                return False, 0, f"{puzzle_id} step {step}: non-positive subtraction ({a} - {b})"
            res = a - b
        elif op == "*":
            res = a * b
        else:  # "/"
            if a % b != 0:
                return False, 0, f"{puzzle_id} step {step}: non-exact division ({a} / {b})"
            res = a // b

        if res < 1 or res > value_limit:
            return False, 0, f"{puzzle_id} step {step}: value {res} out of [1, {value_limit}]"

        regs[d] = res
        remaining.discard(res)

    if remaining:
        sample_missing = sorted(remaining)[:5]
        return False, 0, f"{puzzle_id}: missed {len(remaining)} targets (e.g. {sample_missing})"

    return True, len(program), ""


def validate_submission(submission_path: Path, puzzles_path: Path) -> tuple[bool, int, list[str]]:
    csv.field_size_limit(10 * 1024 * 1024)
    puzzles: dict[str, tuple[list[int], list[int], int, int]] = {}
    with puzzles_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzles[row["id"]] = (
                json.loads(row["starting_values"]),
                json.loads(row["targets"]),
                int(row["value_limit"]),
                int(row["registers"]),
            )

    errors: list[str] = []
    seen: set[str] = set()
    total_steps = 0

    with submission_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "program"]:
            return False, 0, [f"Invalid header {reader.fieldnames!r}; expected ['id', 'program']"]
        for row_idx, row in enumerate(reader, start=2):
            pid = row["id"]
            if pid in seen:
                errors.append(f"Row {row_idx}: duplicate id {pid}")
                continue
            if pid not in puzzles:
                errors.append(f"Row {row_idx}: unknown id {pid}")
                continue
            seen.add(pid)
            starts, targets, vlim, nreg = puzzles[pid]
            ok, steps, err = validate_program(pid, starts, targets, vlim, nreg, row["program"])
            if not ok:
                errors.append(err)
            else:
                total_steps += steps

    missing_ids = sorted(set(puzzles) - seen)
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} puzzle IDs (e.g. {missing_ids[:5]})")

    return len(errors) == 0, total_steps, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to submission CSV")
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=ROOT / "data" / "puzzles.csv",
        help="Path to puzzles.csv (default: data/puzzles.csv)",
    )
    args = parser.parse_args()

    ok, total, errors = validate_submission(args.submission.resolve(), args.puzzles.resolve())
    if not ok:
        print(f"INVALID ({len(errors)} error(s)):", file=sys.stderr)
        for err in errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"VALID — {args.submission}: total_instructions = {total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
