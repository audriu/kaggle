"""Small utilities for 10 Hz tracking frames (Combine and in-game share x/y/s/a/dis/dir)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FRAME_DT = 0.1  # seconds between frames


def add_frame_index(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Add ``frame`` (0-based) and ``t`` (seconds from first frame) within each group."""
    df = df.sort_values(keys + ["time"]).copy()
    first = df.groupby(keys)["time"].transform("min")
    df["t"] = (df["time"] - first).dt.total_seconds()
    df["frame"] = (df["t"] / FRAME_DT).round().astype(int)
    return df


def velocity_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``vx``/``vy`` from speed and heading. NFL ``dir``: 0° = +y, clockwise."""
    rad = np.deg2rad(df["dir"].to_numpy())
    df = df.copy()
    df["vx"] = df["s"] * np.sin(rad)
    df["vy"] = df["s"] * np.cos(rad)
    return df


def heading_change(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Signed per-frame change in ``dir`` (degrees, wrapped to [-180, 180]) within groups."""
    d = df.groupby(keys)["dir"].diff()
    return ((d + 180) % 360) - 180


def finite_difference_accel(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Acceleration recomputed from ``s`` (useful to cross-check the provided ``a``)."""
    return df.groupby(keys)["s"].diff() / FRAME_DT


def attempt_summary(ct: pd.DataFrame) -> pd.DataFrame:
    """One row per Combine drill attempt: duration, peak speed/accel/decel, distance."""
    keys = ["nfl_id", "drill_type", "drill_name", "attempt", "event_id"]
    g = ct.groupby(keys, sort=False)
    out = g.agg(
        frames=("time", "size"),
        duration_s=("time", lambda s: (s.max() - s.min()).total_seconds()),
        peak_speed=("s", "max"),
        peak_accel=("a", "max"),
        peak_decel=("a", "min"),
        distance=("dis", "sum"),
    )
    return out.reset_index()
