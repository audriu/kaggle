"""Loaders for the nine competition tables.

Each loader returns a pandas DataFrame. Parquet cache is used when present (``setup_data.py
--parquet``); otherwise the CSV is read directly. ``polars`` is used for the big tracking files
when installed because it is several times faster than pandas on 1 GB CSVs.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from . import paths

_SMALL = {
    "players",
    "combine_results",
    "player_career_successes",
    "games",
}


def _read(name: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else None
    pq = paths.parquet(name)
    if pq.is_file():
        return pd.read_parquet(pq, columns=cols)
    csv = paths.csv(name)
    if not csv.is_file():
        raise FileNotFoundError(
            f"{csv} missing — run scripts/setup_data.py (data is ~2.2 GiB)"
        )
    if name in _SMALL:
        return pd.read_csv(csv, usecols=cols)
    try:
        import polars as pl

        lf = pl.scan_csv(csv, null_values=paths.NULL_VALUES, infer_schema_length=paths.SCHEMA_ROWS)
        if cols:
            lf = lf.select(cols)
        return lf.collect().to_pandas()
    except ImportError:
        return pd.read_csv(csv, usecols=cols)


def players() -> pd.DataFrame:
    return _read("players")


def combine_results() -> pd.DataFrame:
    return _read("combine_results")


def career() -> pd.DataFrame:
    return _read("player_career_successes")


def games() -> pd.DataFrame:
    return _read("games")


def player_play(columns: Iterable[str] | None = None) -> pd.DataFrame:
    return _read("player_play", columns)


def combine_tracking(
    columns: Iterable[str] | None = None,
    players_only: bool = True,
) -> pd.DataFrame:
    """10 Hz Combine drill tracking. Drops BALL rows by default."""
    df = _read("combine_tracking", columns)
    if players_only and "entity_type" in df.columns:
        df = df[df["entity_type"] == "PLAYER"].copy()
    df = _parse_time(df)
    return df


def game_tracking(
    seasons: Iterable[int] = paths.TRACKING_SEASONS,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """10 Hz in-game tracking for the rookie cohort, concatenated over ``seasons``."""
    frames = []
    for season in seasons:
        df = _read(f"game_tracking_{season}", columns)
        df["season"] = season
        frames.append(df)
    return _parse_time(pd.concat(frames, ignore_index=True))


def rookies() -> pd.DataFrame:
    """players + combine_results + career successes, one row per nfl_id."""
    df = players().merge(
        combine_results().drop(columns=["draft_year"]), on="nfl_id", how="left"
    )
    return df.merge(career(), on="nfl_id", how="left")


def regular_season_plays(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """player_play restricted to REG season games, with season/week attached."""
    g = games()[["game_id", "season", "season_type", "week"]]
    pp = player_play(columns)
    pp = pp.merge(g, on="game_id", how="left")
    return pp[pp["season_type"] == "REG"].copy()


def _parse_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    return df
