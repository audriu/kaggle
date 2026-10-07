"""Project paths. Data stays in data/ (gitignored, CC BY-NC 4.0 — never commit)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "parquet"
ASSETS = ROOT / "assets"
DIST = ROOT / "dist"
NOTES = ROOT / "notes"

COMPETITION = "nfl-big-data-bowl-2027"

CSV_FILES = (
    "players.csv",
    "combine_results.csv",
    "combine_tracking.csv",
    "player_career_successes.csv",
    "player_play.csv",
    "games.csv",
    "game_tracking_2023.csv",
    "game_tracking_2024.csv",
    "game_tracking_2025.csv",
)

TRACKING_SEASONS = (2023, 2024, 2025)

# The CSVs write missing values as the literal string "NA" (pandas handles it, polars does not).
NULL_VALUES = ["NA", ""]
SCHEMA_ROWS = 100_000


def csv(name: str) -> Path:
    """Path to a raw competition CSV (``name`` with or without ``.csv``)."""
    if not name.endswith(".csv"):
        name += ".csv"
    return DATA / name


def parquet(name: str) -> Path:
    """Path to the parquet cache of a CSV."""
    return CACHE / (name.removesuffix(".csv") + ".parquet")
