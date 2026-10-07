"""Break detection and stop/go metrics for Combine route drills.

A route "break" is the point where the receiver changes direction. For each drill attempt we
locate it from the tracking alone (smoothed speed + position-derived heading), then measure how
the player stops into it and re-accelerates out of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT = 0.1

# Route drills with at least one deliberate change of direction, by position group.
ROUTE_DRILLS = {
    "WR": [
        "COMEBACK_ROUTE_RIGHT",
        "CURL_ROUTE_RIGHT",
        "DAGGER_ROUTE_LEFT",
        "POST_CORNER_ROUTE_LEFT",
        "POST_CORNER_ROUTE_RIGHT",
        "SLANT_ROUTE_LEFT",
        "SLOT_SAIL_ROUTE_RIGHT",
        "SLOT_STRIKE_ROUTE_LEFT",
        "SPEED_OUT_LEFT",
        "SPEED_OUT_ROUTE_LEFT",
        "CIRCUS_ROUTE_LEFT",
    ],
    "TE": [
        "CORNER_ROUTE",
        "FLAT_ROUTE_LEFT",
        "HOOK_ROUTE_RIGHT",
        "IN_ROUTE_LEFT",
        "WHEEL_ROUTE_LEFT",
    ],
}
# Spelling variants in drill_name → canonical
DRILL_ALIASES = {
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "LINE": "LINE_DRILL",
}
ALL_ROUTE_DRILLS = sorted({DRILL_ALIASES.get(d, d) for v in ROUTE_DRILLS.values() for d in v})


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(v).rolling(k, center=True, min_periods=1).mean().to_numpy()


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def analyse_attempt(
    g: pd.DataFrame,
    min_turn_deg: float = 30.0,
    min_entry_speed: float = 4.0,
    window_s: float = 0.4,
) -> dict | None:
    """Locate the main break of one drill attempt and measure stop/go quantities.

    ``g`` is one event's PLAYER frames sorted by time. Returns None if no break is found.
    """
    if len(g) < 20:
        return None
    s = _smooth(g["s"].to_numpy(), 5)
    x = _smooth(g["x"].to_numpy(), 3)
    y = _smooth(g["y"].to_numpy(), 3)
    n = len(s)
    w = int(round(window_s / DT))

    # Heading from the path itself (the dir column is noisy at low speed).
    head = np.degrees(np.arctan2(np.gradient(y), np.gradient(x)))
    turn = np.zeros(n)
    turn[w:-w] = np.abs(_wrap(head[2 * w :] - head[: -2 * w]))

    # The break must come after the player is up to speed and before the final slow-down.
    fast = np.where(s >= min_entry_speed)[0]
    if len(fast) == 0:
        return None
    lo = fast[0] + w
    hi = max(lo + 1, n - int(1.0 / DT))
    seg = np.arange(lo, hi)
    if len(seg) == 0:
        return None
    cand = seg[turn[seg] >= min_turn_deg]
    if len(cand) == 0:
        return None
    # Among turning frames prefer the deepest speed trough (sharp break), tie-break by turn size.
    score = turn[cand] / turn[cand].max() - s[cand] / s.max()
    i_turn = int(cand[np.argmax(score)])
    lo_t, hi_t = max(0, i_turn - 5), min(n, i_turn + 6)
    i_trough = lo_t + int(np.argmin(s[lo_t:hi_t]))

    pre_lo = max(0, i_trough - 15)
    i_entry = pre_lo + int(np.argmax(s[pre_lo : i_trough + 1]))
    post_hi = min(n, i_trough + 11)
    i_exit = i_trough + int(np.argmax(s[i_trough:post_hi]))

    ds = np.gradient(s, DT)
    entry, trough, exit_ = s[i_entry], s[i_trough], s[i_exit]
    if entry < min_entry_speed:
        return None
    decel_peak = float(ds[i_entry : i_trough + 1].min()) if i_trough > i_entry else 0.0
    accel_out = float(ds[i_trough : post_hi].max())
    t_stop = (i_trough - i_entry) * DT
    t_go = (i_exit - i_trough) * DT
    return {
        "event_id": g["event_id"].iloc[0],
        "nfl_id": int(g["nfl_id"].iloc[0]),
        "drill_name": DRILL_ALIASES.get(g["drill_name"].iloc[0], g["drill_name"].iloc[0]),
        "attempt": int(g["attempt"].iloc[0]),
        "frames": n,
        "t_break": i_trough * DT,
        "turn_deg": float(turn[i_turn]),
        "entry_speed": float(entry),
        "trough_speed": float(trough),
        "exit_speed": float(exit_),
        "speed_loss": float(entry - trough),
        "t_stop": t_stop,
        "t_go": t_go,
        "decel_peak": decel_peak,  # yd/s², negative
        "accel_out": accel_out,  # yd/s²
        "reaccel": float(exit_ - trough),
        "top_speed": float(s.max()),
    }


def attempt_table(ct: pd.DataFrame, drills: list[str] | None = None) -> pd.DataFrame:
    """Run ``analyse_attempt`` over every attempt of the given route drills."""
    names = set(drills or ALL_ROUTE_DRILLS)
    canon = ct["drill_name"].map(lambda d: DRILL_ALIASES.get(d, d))
    sub = ct[canon.isin(names)].sort_values(["event_id", "time"])
    rows = []
    for _, g in sub.groupby("event_id", sort=False):
        r = analyse_attempt(g)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


def player_scores(
    att: pd.DataFrame,
    cols: tuple[str, ...] = ("decel_peak", "accel_out", "speed_loss", "t_stop", "reaccel"),
    min_drills: int = 2,
) -> pd.DataFrame:
    """Within-drill z-scores averaged per player (so route type is controlled for).

    Signs are flipped where needed so that higher = better for every column.
    """
    z = att.copy()
    better_low = {"decel_peak", "t_stop", "t_go"}  # more negative decel / shorter stop = better
    for c in cols:
        v = z[c] * (-1 if c in better_low else 1)
        z[c + "_z"] = v.groupby(z["drill_name"]).transform(lambda s: (s - s.mean()) / s.std())
    per_drill = z.groupby(["nfl_id", "drill_name"])[[c + "_z" for c in cols]].mean().reset_index()
    out = per_drill.groupby("nfl_id").agg(
        n_drills=("drill_name", "nunique"), **{c + "_z": (c + "_z", "mean") for c in cols}
    )
    out["n_attempts"] = att.groupby("nfl_id").size()
    return out[out["n_drills"] >= min_drills].reset_index()
