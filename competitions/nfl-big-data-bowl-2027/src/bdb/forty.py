"""Distance-indexed speed curves from 40-yard-dash tracking.

The tracking clock does not start on the timer, so time-based splits are unreliable. Speed as
a function of distance covered, s(d), is immune to that offset, and the official 40 time pins the
clock when a time is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT = 0.1
GRID = np.arange(0.0, 40.5, 0.5)  # yards
CHECKPOINTS = (2.5, 5, 10, 15, 20, 30, 40)


def _smooth(v: np.ndarray, k: int = 3) -> np.ndarray:
    return pd.Series(v).rolling(k, center=True, min_periods=1).mean().to_numpy()


def speed_by_distance(g: pd.DataFrame) -> np.ndarray | None:
    """Speed (yd/s) sampled on ``GRID`` yards of straight-line displacement from the start."""
    g = g.sort_values("time")
    x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), _smooth(g["s"].to_numpy())
    d = np.hypot(x - x[0], y - y[0])
    if d[-1] < 39.0:
        return None
    # displacement must be monotone for interpolation
    d = np.maximum.accumulate(d)
    ok = np.concatenate([[True], np.diff(d) > 0])
    return np.interp(GRID, d[ok], s[ok], right=np.nan)


def analyse_run(g: pd.DataFrame, forty_official: float | None = None) -> dict | None:
    sd = speed_by_distance(g)
    if sd is None:
        return None
    top = float(np.nanmax(sd))
    d90 = float(GRID[np.argmax(sd >= 0.9 * top)])
    d95 = float(GRID[np.argmax(sd >= 0.95 * top)])
    out = {
        "event_id": g["event_id"].iloc[0],
        "nfl_id": int(g["nfl_id"].iloc[0]),
        "draft_year": int(g["draft_year"].iloc[0]),
        "attempt": int(g["attempt"].iloc[0]),
        "top_speed": top,
        "d_90pct": d90,
        "d_95pct": d95,
        "mean_s_0_10": float(np.nanmean(sd[(GRID > 0) & (GRID <= 10)])),
        "mean_s_10_20": float(np.nanmean(sd[(GRID > 10) & (GRID <= 20)])),
        "mean_s_20_40": float(np.nanmean(sd[(GRID > 20) & (GRID <= 40)])),
    }
    for c in CHECKPOINTS:
        out[f"s_at_{c:g}"] = float(sd[np.searchsorted(GRID, c)])
    # Sensor-derived splits in the official clock: time from d1 to d2 = integral of 1/s.
    with np.errstate(divide="ignore"):
        inv = 1.0 / np.where(sd > 0.3, sd, np.nan)
    seg = 0.5 * (inv[1:] + inv[:-1]) * np.diff(GRID)  # trapezoid
    cum = np.concatenate([[0.0], np.nancumsum(seg)])
    out["t_10_40_sensor"] = float(cum[np.searchsorted(GRID, 40)] - cum[np.searchsorted(GRID, 10)])
    out["t_20_40_sensor"] = float(cum[np.searchsorted(GRID, 40)] - cum[np.searchsorted(GRID, 20)])
    if forty_official is not None and not np.isnan(forty_official):
        out["ten_sensor"] = forty_official - out["t_10_40_sensor"]
    return out


def run_table(ct: pd.DataFrame, combine_results: pd.DataFrame | None = None) -> pd.DataFrame:
    f = ct[ct["drill_type"] == "FORTY_YARD_DASH"]
    forty = {}
    if combine_results is not None:
        forty = combine_results.set_index("nfl_id")["forty"].to_dict()
    rows = []
    for _, g in f.groupby("event_id", sort=False):
        r = analyse_run(g, forty.get(int(g["nfl_id"].iloc[0])))
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def curves_table(ct: pd.DataFrame) -> pd.DataFrame:
    """Long table of (event_id, nfl_id, d, s) for plotting."""
    f = ct[ct["drill_type"] == "FORTY_YARD_DASH"]
    rows = []
    for ev, g in f.groupby("event_id", sort=False):
        sd = speed_by_distance(g)
        if sd is None:
            continue
        rows.append(pd.DataFrame({"event_id": ev, "nfl_id": int(g["nfl_id"].iloc[0]), "d": GRID, "s": sd}))
    return pd.concat(rows, ignore_index=True)


def player_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Best run per player (fastest sensor 10–40 time), plus number of runs."""
    best = runs.sort_values("t_10_40_sensor").groupby("nfl_id").first()
    best["n_runs"] = runs.groupby("nfl_id").size()
    return best.reset_index()
