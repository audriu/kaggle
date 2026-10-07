# Data dictionary (condensed)

Condensed from the competition Data page (2026-10-07). 510 rookies (draft classes 2023–2025),
9 CSVs, ≈2.2 GiB. `nfl_id` joins everything. Tracking is 10 Hz; `x` is field length 0–120 yd,
`y` is width 0–53.33 yd, `s` yd/s, `a` yd/s², `dis` yd since last frame, `dir` motion heading
0–360°, `o` body orientation 0–360° (game tracking only).

## Entity relationships

```
players 1─1 combine_results
players 1─* combine_tracking        (event_id = one drill attempt; BALL rows have nfl_id null)
players 1─1 player_career_successes
players 1─* player_play *─1 games
player_play 1─* game_tracking_{2023,2024,2025}
```

## players.csv (510 × 10)

`nfl_id`, `display_name`, `draft_year` (2023/24/25), `nfl_position`, `birth_date`,
`college_name`, `college_conference`, `draft_round`, `draft_pick_within_round`,
`draft_overall_pick` (last three null for UDFAs).

## combine_results.csv (510 × 18)

`draft_year`, `nfl_id`, `combine_position`, `combine_height` (in), `combine_weight` (lb),
`hand_size`, `arm_length`, `wing_span` (in), `ten_yd_split`, `forty` (s), `vertical` (in),
`broad_jump` (in), `three_cone`, `short_shuttle` (s, null on opt-out), `bench_reps` (null on
opt-out), `ngs_athleticism_score` (0–100), `ngs_college_production_score`, `ngs_final_score`.

## combine_tracking.csv (463,189 × 14) — the required input

`draft_year`, `event_id` (one continuous drill attempt), `nfl_id` (null for BALL),
`entity_type` (PLAYER/BALL), `time` (ISO 8601, 0.1 s steps), `drill_type`, `drill_name`,
`attempt`, `x`, `y`, `s`, `a`, `dis`, `dir`.

`drill_type` values seen in the description: FORTY_YARD_DASH, THREE_CONE_DRILL, SHORT_SHUTTLE,
SKILL_DRILLS_WR / _DB / _OL / _DL / _TE. `drill_name` examples: GAUNTLET_DRILL,
RUN_THE_HOOP_DRILL, BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION, SLOT_STRIKE_ROUTE_LEFT,
PASS_PRO_MIRROR_DRILL, OL_PULL_DRILL_FOLD… 6,310 attempts total. Enumerate the real set with
`scripts/data_summary.py`.

## player_career_successes.csv (510 × 9)

`nfl_id`, `career_offensive_snaps`, `career_defensive_snaps`, `career_special_teams_snaps`,
`career_games_active`, `career_games_started`, `ap_all_pro_1st_team`, `ap_all_pro_2nd_team`,
`pro_bowl_original_ballot`.

## player_play.csv (316,338 × 64) — per-snap NGS outcomes

- Context: `game_id`, `play_id`, `team_abbr`, `nfl_id`, `lined_up_position`, `play_description`,
  `quarter`, `down`, `yards_to_go`, `possession_team`, `yardline_side`, `yardline_number`,
  `game_clock`, `pre_snap_home_score`, `pre_snap_visitor_score`, `play_direction`.
- WP/EP: `pre_snap_home_team_win_probability`, `pre_snap_visitor_team_win_probability`,
  `home_team_win_probability_added`, `visitor_team_win_probility_added` (sic),
  `expected_points_added`, `expected_points`.
- Scheme/result: `offense_formation`, `receiver_alignment`, `pass_result` (C/I/S/IN/R),
  `pass_length`, `play_nullified_by_penalty`, `penalty_yards`, `pre_penalty_yards_gained`,
  `yards_gained`, `team_coverage_man_zone`, `team_coverage_type`, `in_motion_at_ball_snap`.
- Receiver/ball-carrier: `target`, `rec_yards`, `rush_yards`, `yards_after_catch`, `fumble`,
  `fumble_lost`, `route_ran`, `separation_at_pass_forward`, `cushion`,
  `expected_yards_after_catch`.
- OL / pass pro: `dropback_duration`, `extended_sack_allowed`, `pass_rushers_encountered`,
  `peak_pressure_probability_allowed`, `pressure_allowed`, `sack_allowed`,
  `time_to_pressure_allowed`.
- Defense: `blitzing`, `player_get_off`, `sack`, `tackle`, `assist`, `caused_forced_fumble`,
  `recovered_fumble`, `tackle_for_loss`, `time_to_pressure`, `time_to_qb_hurry`,
  `coverage_assignment`, `coverage_assignment_at_snap`, `quick_pressure`, `unblocked_pressure`.

## games.csv (1,002 × 7)

`game_id` (YYYYMMDD##), `game_key`, `season`, `season_type` (REG/POST), `week`,
`home_team_abbr`, `visitor_team_abbr`.

## game_tracking_YYYY.csv (1.42M / 3.04M / 4.34M × 12)

`game_id`, `play_id`, `nfl_id`, `time`, `x`, `y`, `s`, `a`, `dis`, `o`, `dir`,
`event` (ball_snap, pass_forward, pass_arrived, tackle, touchdown, first_contact,
out_of_bounds, … or null). Cohort players only, not all 22 on the field.

## Gotchas (verified on the downloaded data, 2026-10-07)

- **Missing values are the literal string `NA`.** pandas handles it; polars needs
  `null_values=["NA", ""]` or every numeric column with an opt-out becomes a string
  (`forty` has 86 nulls, `three_cone` 330, `bench_reps` 318). `bdb.paths.NULL_VALUES` is used by
  every scan in this repo.
- **`a` sign convention is inconsistent.** Combine tracking and game_tracking_2023/2024 have
  `a >= 0` (magnitude only, max 44.6 in Combine → outliers); game_tracking_2025 goes negative
  (min −14.4). Derive signed accel/decel from `s` yourself (`bdb.tracking.finite_difference_accel`)
  and smooth it: the raw finite difference shows ±240 yd/s² spikes, so check for frame gaps or
  duplicate timestamps within an `event_id` before trusting it.
- **Game tracking is larger than the Data page says.** 2023: 3.67M rows / 156 players / 333
  games; 2024: 7.83M / 318 / 334; 2025: 11.16M / 448 / 334 (page claims 1.4M / 3.0M / 4.3M).
  Parquet cache for everything is ~0.4 GB; loading one season into pandas takes ~12 s.
- Combine tracking: 431k PLAYER rows after dropping BALL, 6,310 attempts, every one of the 510
  players has at least one drill; all but a handful have NFL snaps (see `data_summary.md`).
- `drill_name` has spelling variants to merge: `PASS_PRO-MIRROR_DRILL` / `PASS_PRO_MIRROR_DRILL`,
  `OVER_SHOULDER_ADJUST` / `OVER_THE_SHOULDER_ADJUST`, `LINE` / `LINE_DRILL`,
  `SPEED_OUT_LEFT` / `SPEED_OUT_ROUTE_LEFT`. LB skill drills exist but only for 4 players.
- Cohort is skewed: WR 106, CB 60, G 51, T 50, DT 47, TE 42, DE 36 … RB/FB/MLB 1 each.

## Gotchas to check early

- Combine tracking has no `o` (orientation) column; game tracking does.
- Combine `x`/`y` are on a field grid too — verify drill start position / direction conventions
  per `drill_name` before computing cut angles.
- `player_play` is the rookie's own row per snap; teammates/opponents are not present, so
  "separation" etc. are precomputed NGS fields, not derivable from tracking here.
- Only REG season counts for the prompt; `games.season_type` filters out POST.
- 2025 rookies have one season, 2023 rookies have three → normalise outcomes per snap / per season.
