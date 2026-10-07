#!/usr/bin/env python3
"""Sanity report over the downloaded data: shapes, nulls, drill catalogue, key coverage.

Usage:
  python scripts/data_summary.py             # all tables (game tracking read lazily with polars)
  python scripts/data_summary.py --small     # skip the three game_tracking files
  python scripts/data_summary.py --out notes/data_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdb import paths  # noqa: E402


def _scan(name: str):
    import polars as pl

    pq = paths.parquet(name)
    if pq.is_file():
        return pl.scan_parquet(pq)
    csv = paths.csv(name)
    if not csv.is_file():
        return None
    return pl.scan_csv(csv, null_values=paths.NULL_VALUES, infer_schema_length=paths.SCHEMA_ROWS)


def table_report(name: str) -> list[str]:
    import polars as pl

    lf = _scan(name)
    if lf is None:
        return [f"## {name}", "", "_missing — run scripts/setup_data.py_", ""]
    schema = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    nulls = lf.select([pl.col(c).null_count().alias(c) for c in schema]).collect().row(0)
    lines = [f"## {name}", "", f"rows: {n:,} · cols: {len(schema)}", "", "| column | dtype | nulls |", "| --- | --- | ---: |"]
    for (col, dtype), nn in zip(schema.items(), nulls):
        lines.append(f"| {col} | {dtype} | {nn:,} |")
    lines.append("")
    return lines


def drill_catalogue() -> list[str]:
    import polars as pl

    lf = _scan("combine_tracking")
    if lf is None:
        return []
    cat = (
        lf.filter(pl.col("entity_type") == "PLAYER")
        .group_by("drill_type", "drill_name")
        .agg(
            attempts=pl.col("event_id").n_unique(),
            players=pl.col("nfl_id").n_unique(),
            frames=pl.len(),
        )
        .sort("drill_type", "drill_name")
        .collect()
    )
    lines = ["## Combine drill catalogue", "", "| drill_type | drill_name | attempts | players | frames |", "| --- | --- | ---: | ---: | ---: |"]
    for r in cat.iter_rows():
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:,} | {r[4]:,} |")
    lines.append("")
    return lines


def position_coverage() -> list[str]:
    import polars as pl

    players = _scan("players")
    ct = _scan("combine_tracking")
    pp = _scan("player_play")
    if players is None or ct is None or pp is None:
        return []
    tracked = ct.filter(pl.col("entity_type") == "PLAYER").select("nfl_id").unique()
    played = pp.group_by("nfl_id").agg(snaps=pl.len())
    cov = (
        players.select("nfl_id", "nfl_position", "draft_year")
        .join(tracked.with_columns(has_combine_tracking=pl.lit(True)), on="nfl_id", how="left")
        .join(played, on="nfl_id", how="left")
        .with_columns(
            pl.col("has_combine_tracking").fill_null(False),
            pl.col("snaps").fill_null(0),
        )
        .group_by("nfl_position")
        .agg(
            players=pl.len(),
            with_combine_tracking=pl.col("has_combine_tracking").sum(),
            with_snaps=(pl.col("snaps") > 0).sum(),
            median_snaps=pl.col("snaps").median(),
        )
        .sort("players", descending=True)
        .collect()
    )
    lines = ["## Position coverage (players.csv × combine_tracking × player_play)", "", "| position | players | w/ combine tracking | w/ snaps | median snaps |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in cov.iter_rows():
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.0f} |")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true", help="skip game_tracking_* tables")
    parser.add_argument("--out", type=Path, default=None, help="write markdown report here")
    args = parser.parse_args()

    try:
        import polars  # noqa: F401
    except ImportError:
        print("polars not installed: pip install -r requirements.txt", file=sys.stderr)
        return 1

    names = [n for n in paths.CSV_FILES if not (args.small and n.startswith("game_tracking"))]
    lines = ["# Data summary", ""]
    for n in names:
        lines += table_report(n)
    lines += drill_catalogue()
    lines += position_coverage()
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
