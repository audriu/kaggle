"""First-step burst metrics from Combine defensive-line drills that start from a stance.

The recordings begin at or just before the first movement (no snap cue is tracked), so every
quantity is measured from *movement onset*, not from a reaction signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT = 0.1
STANCE_DRILLS = ["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL", "RUN_AND_CLUB_DRILL"]
ONSET_SPEED = 1.0  # yd/s


def _smooth(v: np.ndarray, k: int = 3) -> np.ndarray:
    return pd.Series(v).rolling(k, center=True, min_periods=1).mean().to_numpy()


def analyse_start(g: pd.DataFrame, horizon_s: float = 1.5) -> dict | None:
    """Burst metrics for the first ``horizon_s`` seconds after movement onset of one attempt."""
    g = g.sort_values("time")
    s = _smooth(g["s"].to_numpy())
    x = g["x"].to_numpy()
    y = g["y"].to_numpy()
    n = len(s)
    on = np.where(s >= ONSET_SPEED)[0]
    if len(on) == 0:
        return None
    i0 = int(on[0])
    # step back to the last frame before onset where speed was still rising from its minimum
    while i0 > 0 and s[i0 - 1] < s[i0] and s[i0 - 1] > 0.3:
        i0 -= 1
    h = int(round(horizon_s / DT))
    if i0 + h >= n:
        return None
    disp = np.hypot(x - x[i0], y - y[i0])  # straight-line displacement from the start point
    ds = np.gradient(s, DT)

    def t_to(d: float) -> float:
        idx = np.where(disp[i0:] >= d)[0]
        return float(idx[0] * DT) if len(idx) else np.nan

    return {
        "event_id": g["event_id"].iloc[0],
        "nfl_id": int(g["nfl_id"].iloc[0]),
        "drill_name": g["drill_name"].iloc[0],
        "attempt": int(g["attempt"].iloc[0]),
        "onset_frame": i0,
        "t_1yd": t_to(1.0),
        "t_3yd": t_to(3.0),
        "t_5yd": t_to(5.0),
        "speed_0p5s": float(s[i0 + 5]),
        "speed_1s": float(s[i0 + 10]),
        "dist_1s": float(disp[i0 + 10]),
        "peak_accel_1s": float(ds[i0 : i0 + 11].max()),
        "mean_accel_1s": float((s[i0 + 10] - s[i0]) / 1.0),
        "top_speed": float(s.max()),
        "frames": n,
    }


def attempt_table(ct: pd.DataFrame, drills: list[str] | None = None) -> pd.DataFrame:
    names = drills or STANCE_DRILLS
    sub = ct[ct["drill_name"].isin(names)]
    rows = [r for _, g in sub.groupby("event_id", sort=False) if (r := analyse_start(g))]
    return pd.DataFrame(rows)


BETTER_LOW = {"t_1yd", "t_3yd", "t_5yd"}


def zscore_within_drill(att: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = att.copy()
    for c in cols:
        v = z[c] * (-1 if c in BETTER_LOW else 1)
        z[c + "_z"] = v.groupby(z["drill_name"]).transform(lambda s: (s - s.mean()) / s.std())
    return z


def player_scores(att: pd.DataFrame, cols: list[str], min_attempts: int = 1) -> pd.DataFrame:
    z = zscore_within_drill(att, cols)
    zc = [c + "_z" for c in cols]
    per_drill = z.groupby(["nfl_id", "drill_name"])[zc].mean().reset_index()
    out = per_drill.groupby("nfl_id").agg(
        n_drills=("drill_name", "nunique"), **{c: (c, "mean") for c in zc}
    )
    out["n_attempts"] = att.groupby("nfl_id").size()
    return out[out["n_attempts"] >= min_attempts].reset_index()
