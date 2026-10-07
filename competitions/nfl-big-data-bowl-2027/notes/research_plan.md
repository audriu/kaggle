# Research plan

Status: **analysis done, writeup drafted** (2026-10-07). Deadline 2027-01-06. Chosen angle: the 40-yd dash as a speed-by-distance curve, linked to position-specific outcomes (see `results.md`). Angles 1 and 3 below were tried and dropped.

## Guiding constraints

- One submission, judged on a rubric, no leaderboard → iterate on *clarity + football usefulness*,
  not on a metric. Pick **one narrow question** (one position group × one drill family).
- Must explicitly link *Combine tracking* → *regular-season* outcomes. Stopwatch-only analysis fails the gate.
- ≤ 2,000 words, < 10 figures, public Kaggle notebook attached.

## Candidate angles (pick one after EDA)

1. **Receivers: deceleration into the break.** From WR/TE skill drills (routes, gauntlet) derive
   peak deceleration, time-to-re-accelerate, and break sharpness; relate to
   `separation_at_pass_forward` on in-game routes of the same family (`route_ran`), controlling
   for coverage (`team_coverage_man_zone`, `cushion`). Output: a "break efficiency" metric and a
   per-prospect comparison vs. 40/3-cone.
2. **DBs: transition quality.** `BACK_PEDAL_AND_TRANSITION_*` drills → hip-flip latency
   (time between direction reversal and reaching X% of top speed) → in-game separation allowed
   / targets when in MAN coverage.
3. **Trench: first-step get-off.** DL/OL drill tracking → time to first yard, accel profile in
   first 0.5 s → `player_get_off`, `quick_pressure`, `time_to_pressure` (DL) or
   `pressure_allowed`, `peak_pressure_probability_allowed` (OL).
4. **Drill translation (meta-angle).** Where does the sensor curve add information beyond the
   official split? e.g. 40-yd dash acceleration shape vs. `ten_yd_split`/`forty`, residualised
   against draft slot, predicting snap share / starts.

## Methodology checklist (Data Science 30%)

- Define outcomes per snap and per season; account for sample size (empirical-Bayes shrinkage).
- Control for obvious confounders: draft slot, position, age, snaps, coverage type, team context.
- Validate: leave-one-draft-class-out; show calibration/uncertainty, not just correlations.
- Report effect sizes a scout can act on (e.g. "1 SD better break deceleration ≈ +0.3 yd separation").

## Viz checklist (20%)

- Drill trajectory small-multiples (x/y with speed colour) for best/worst examples.
- Metric vs. outcome scatter with shrinkage and position facets.
- One "scouting card" table per highlighted prospect.
- Colour-blind safe palette, readable at Kaggle writeup width.

## Milestones

| When | What |
| --- | --- |
| Oct wk 2 | data downloaded, parquet cache, EDA of drill types / coverage per position |
| Oct wk 3–4 | choose angle, build metric v1, first outcome link |
| Nov | model + validation, figures, draft writeup v1 |
| early Dec | polish, public notebook published, internal review vs rubric |
| ≤ 2027-01-05 | submit writeup (buffer one day) |
