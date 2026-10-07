# Data summary

## players.csv

rows: 510 · cols: 10

| column | dtype | nulls |
| --- | --- | ---: |
| nfl_id | Int64 | 0 |
| display_name | String | 0 |
| draft_year | Int64 | 0 |
| nfl_position | String | 0 |
| birth_date | String | 0 |
| college_name | String | 0 |
| college_conference | String | 0 |
| draft_round | Int64 | 127 |
| draft_pick_within_round | Int64 | 127 |
| draft_overall_pick | Int64 | 127 |

## combine_results.csv

rows: 510 · cols: 18

| column | dtype | nulls |
| --- | --- | ---: |
| draft_year | Int64 | 0 |
| nfl_id | Int64 | 0 |
| combine_position | String | 0 |
| combine_height | Float64 | 0 |
| combine_weight | Int64 | 0 |
| hand_size | Float64 | 0 |
| arm_length | Float64 | 0 |
| wing_span | Float64 | 0 |
| ten_yd_split | Float64 | 85 |
| forty | Float64 | 86 |
| vertical | Float64 | 72 |
| broad_jump | Int64 | 87 |
| three_cone | Float64 | 330 |
| short_shuttle | Float64 | 309 |
| bench_reps | Int64 | 318 |
| ngs_athleticism_score | Int64 | 0 |
| ngs_college_production_score | Int64 | 1 |
| ngs_final_score | Int64 | 1 |

## combine_tracking.csv

rows: 463,189 · cols: 14

| column | dtype | nulls |
| --- | --- | ---: |
| draft_year | Int64 | 0 |
| event_id | String | 0 |
| nfl_id | Int64 | 0 |
| entity_type | String | 0 |
| time | String | 0 |
| drill_type | String | 0 |
| drill_name | String | 0 |
| attempt | Int64 | 0 |
| x | Float64 | 0 |
| y | Float64 | 0 |
| s | Float64 | 0 |
| a | Float64 | 0 |
| dis | Float64 | 0 |
| dir | Float64 | 0 |

## player_career_successes.csv

rows: 510 · cols: 9

| column | dtype | nulls |
| --- | --- | ---: |
| nfl_id | Int64 | 0 |
| career_offensive_snaps | Int64 | 0 |
| career_defensive_snaps | Int64 | 0 |
| career_special_teams_snaps | Int64 | 0 |
| career_games_active | Int64 | 0 |
| career_games_started | Int64 | 0 |
| ap_all_pro_1st_team | Int64 | 0 |
| ap_all_pro_2nd_team | Int64 | 0 |
| pro_bowl_original_ballot | Int64 | 0 |

## player_play.csv

rows: 314,197 · cols: 64

| column | dtype | nulls |
| --- | --- | ---: |
| game_id | Int64 | 0 |
| play_id | Int64 | 0 |
| team_abbr | String | 0 |
| nfl_id | Int64 | 0 |
| lined_up_position | String | 1,383 |
| play_description | String | 0 |
| quarter | Int64 | 0 |
| down | Int64 | 0 |
| yards_to_go | Int64 | 0 |
| possession_team | String | 0 |
| yardline_side | String | 4,102 |
| yardline_number | Int64 | 0 |
| game_clock | String | 0 |
| pre_snap_home_score | Int64 | 0 |
| pre_snap_visitor_score | Int64 | 0 |
| play_direction | String | 50 |
| pre_snap_home_team_win_probability | Float64 | 3 |
| pre_snap_visitor_team_win_probability | Float64 | 3 |
| home_team_win_probability_added | Float64 | 6 |
| visitor_team_win_probility_added | Float64 | 6 |
| expected_points_added | Float64 | 0 |
| expected_points | Float64 | 0 |
| offense_formation | String | 5,335 |
| receiver_alignment | String | 4,597 |
| pass_result | String | 103,636 |
| pass_length | Int64 | 129,458 |
| play_nullified_by_penalty | String | 0 |
| penalty_yards | Int64 | 293,447 |
| pre_penalty_yards_gained | Int64 | 0 |
| yards_gained | Int64 | 0 |
| team_coverage_man_zone | String | 38,919 |
| team_coverage_type | String | 38,919 |
| in_motion_at_ball_snap | Boolean | 158,231 |
| target | Boolean | 0 |
| rec_yards | Int64 | 307,937 |
| rush_yards | Int64 | 314,175 |
| yards_after_catch | Int64 | 307,937 |
| fumble | Int64 | 314,124 |
| fumble_lost | Int64 | 314,124 |
| route_ran | String | 260,889 |
| separation_at_pass_forward | Float64 | 267,460 |
| cushion | Float64 | 210,445 |
| expected_yards_after_catch | Float64 | 305,221 |
| dropback_duration | Float64 | 253,394 |
| extended_sack_allowed | Float64 | 253,394 |
| pass_rushers_encountered | Int64 | 253,394 |
| peak_pressure_probability_allowed | Float64 | 256,169 |
| pressure_allowed | Boolean | 253,394 |
| sack_allowed | Float64 | 253,394 |
| time_to_pressure_allowed | Float64 | 309,227 |
| quick_pressure | Boolean | 310,434 |
| unblocked_pressure | Boolean | 227,359 |
| blitzing | Boolean | 113,470 |
| player_get_off | Float64 | 270,195 |
| sack | Float64 | 227,362 |
| tackle | Int64 | 302,301 |
| assist | Int64 | 302,301 |
| caused_forced_fumble | Int64 | 302,301 |
| recovered_fumble | Boolean | 311,983 |
| tackle_for_loss | Boolean | 302,301 |
| time_to_pressure | Float64 | 310,393 |
| time_to_qb_hurry | Float64 | 310,318 |
| coverage_assignment | String | 241,545 |
| coverage_assignment_at_snap | String | 242,072 |

## games.csv

rows: 1,002 · cols: 7

| column | dtype | nulls |
| --- | --- | ---: |
| game_id | Int64 | 0 |
| game_key | Int64 | 0 |
| season | Int64 | 0 |
| season_type | String | 0 |
| week | Int64 | 0 |
| home_team_abbr | String | 0 |
| visitor_team_abbr | String | 0 |

## game_tracking_2023.csv

rows: 3,665,443 · cols: 12

| column | dtype | nulls |
| --- | --- | ---: |
| game_id | Int64 | 0 |
| play_id | Int64 | 0 |
| nfl_id | Int64 | 0 |
| time | String | 0 |
| x | Float64 | 0 |
| y | Float64 | 0 |
| s | Float64 | 0 |
| a | Float64 | 0 |
| dis | Float64 | 0 |
| o | Float64 | 0 |
| dir | Float64 | 0 |
| event | String | 3,422,066 |

## game_tracking_2024.csv

rows: 7,827,944 · cols: 12

| column | dtype | nulls |
| --- | --- | ---: |
| game_id | Int64 | 0 |
| play_id | Int64 | 0 |
| nfl_id | Int64 | 0 |
| time | String | 0 |
| x | Float64 | 0 |
| y | Float64 | 0 |
| s | Float64 | 0 |
| a | Float64 | 0 |
| dis | Float64 | 0 |
| o | Float64 | 0 |
| dir | Float64 | 0 |
| event | String | 7,306,249 |

## game_tracking_2025.csv

rows: 11,160,903 · cols: 12

| column | dtype | nulls |
| --- | --- | ---: |
| game_id | Int64 | 0 |
| play_id | Int64 | 0 |
| nfl_id | Int64 | 0 |
| time | String | 0 |
| x | Float64 | 0 |
| y | Float64 | 0 |
| s | Float64 | 0 |
| a | Float64 | 0 |
| dis | Float64 | 0 |
| o | Float64 | 0 |
| dir | Float64 | 0 |
| event | String | 10,425,238 |

## Combine drill catalogue

| drill_type | drill_name | attempts | players | frames |
| --- | --- | ---: | ---: | ---: |
| FORTY_YARD_DASH | FORTY_YARD_DASH | 793 | 418 | 41,500 |
| SHORT_SHUTTLE | SHORT_SHUTTLE | 223 | 170 | 11,690 |
| SKILL_DRILLS_DB | BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL | 124 | 122 | 8,422 |
| SKILL_DRILLS_DB | BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION | 241 | 122 | 21,569 |
| SKILL_DRILLS_DB | BOX_DRILL | 125 | 122 | 9,439 |
| SKILL_DRILLS_DB | GAUNTLET_DRILL | 123 | 119 | 12,253 |
| SKILL_DRILLS_DB | LINE | 50 | 33 | 3,552 |
| SKILL_DRILLS_DB | LINE_DRILL | 167 | 85 | 11,895 |
| SKILL_DRILLS_DB | TERYL_AUSTIN_DRILL_1 | 103 | 101 | 7,765 |
| SKILL_DRILLS_DB | TERYL_AUSTIN_DRILL_2 | 113 | 111 | 9,514 |
| SKILL_DRILLS_DB | W_DRILL | 124 | 121 | 10,214 |
| SKILL_DRILLS_DL | BACK_PEDAL_AND_REACT | 58 | 55 | 6,057 |
| SKILL_DRILLS_DL | BODY_CONTROL_DRILL | 113 | 111 | 6,611 |
| SKILL_DRILLS_DL | FOUR_BAG_AGILITY_DRILL | 112 | 112 | 11,709 |
| SKILL_DRILLS_DL | FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION | 112 | 112 | 12,845 |
| SKILL_DRILLS_DL | PASS_RUSH_DRILL | 217 | 114 | 9,847 |
| SKILL_DRILLS_DL | RUN_AND_CLUB_DRILL | 113 | 112 | 8,180 |
| SKILL_DRILLS_DL | RUN_THE_HOOP_DRILL | 119 | 109 | 9,120 |
| SKILL_DRILLS_DL | SHORT_ZONE_BREAKS | 105 | 36 | 7,895 |
| SKILL_DRILLS_DL | SHORT_ZONE_BREAK_FLAT_WHEEL | 20 | 20 | 2,374 |
| SKILL_DRILLS_DL | SHORT_ZONE_BREAK_FORWARD | 20 | 20 | 1,256 |
| SKILL_DRILLS_DL | SHORT_ZONE_BREAK_INSIDE | 18 | 18 | 1,534 |
| SKILL_DRILLS_LB | BACK_PEDAL_AND_REACT | 4 | 4 | 448 |
| SKILL_DRILLS_LB | FOUR_BAG_SHUFFLE_DRILL | 3 | 3 | 365 |
| SKILL_DRILLS_LB | PASS_RUSH_DRILL | 8 | 4 | 438 |
| SKILL_DRILLS_LB | SHORT_ZONE_BREAKS | 14 | 4 | 1,149 |
| SKILL_DRILLS_LB | SHUFFLE_SPRINT_AND_COD_DRILL | 4 | 4 | 386 |
| SKILL_DRILLS_LB | WAVE_DRILL | 4 | 4 | 477 |
| SKILL_DRILLS_OL | FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE | 120 | 120 | 18,087 |
| SKILL_DRILLS_OL | OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT | 98 | 98 | 4,412 |
| SKILL_DRILLS_OL | OL_PULL_DRILL_FOLD_BLOCK_RIGHT | 119 | 119 | 4,706 |
| SKILL_DRILLS_OL | OL_PULL_DRILL_LONG_PULL_RIGHT | 93 | 93 | 4,720 |
| SKILL_DRILLS_OL | PASS_PRO-MIRROR_DRILL | 84 | 84 | 7,020 |
| SKILL_DRILLS_OL | PASS_PRO_MIRROR_DRILL | 37 | 37 | 3,401 |
| SKILL_DRILLS_OL | PASS_RUSH_DROP | 239 | 120 | 10,752 |
| SKILL_DRILLS_OL | SCREEN_DRILL | 120 | 119 | 8,031 |
| SKILL_DRILLS_TE | BLOCK_EXPLOSION | 40 | 39 | 2,330 |
| SKILL_DRILLS_TE | CORNER_ROUTE | 61 | 30 | 4,233 |
| SKILL_DRILLS_TE | END_ZONE_FADE_RIGHT | 35 | 28 | 1,476 |
| SKILL_DRILLS_TE | FLAT_ROUTE_LEFT | 42 | 42 | 2,223 |
| SKILL_DRILLS_TE | GAUNTLET_DRILL | 81 | 41 | 7,710 |
| SKILL_DRILLS_TE | HOOK_ROUTE_RIGHT | 45 | 41 | 2,895 |
| SKILL_DRILLS_TE | IN_ROUTE_LEFT | 43 | 40 | 2,753 |
| SKILL_DRILLS_TE | OVER_SHOULDER_ADJUST | 34 | 30 | 2,193 |
| SKILL_DRILLS_TE | OVER_THE_SHOULDER_ADJUST | 12 | 12 | 762 |
| SKILL_DRILLS_TE | WHEEL_ROUTE_LEFT | 45 | 40 | 2,527 |
| SKILL_DRILLS_WR | CIRCUS_ROUTE_LEFT | 43 | 37 | 2,551 |
| SKILL_DRILLS_WR | COMEBACK_ROUTE_RIGHT | 110 | 98 | 6,647 |
| SKILL_DRILLS_WR | CURL_ROUTE_RIGHT | 110 | 103 | 6,210 |
| SKILL_DRILLS_WR | DAGGER_ROUTE_LEFT | 115 | 107 | 7,551 |
| SKILL_DRILLS_WR | GAUNTLET_DRILL | 200 | 107 | 18,265 |
| SKILL_DRILLS_WR | GO_ROUTE_RIGHT | 105 | 98 | 7,659 |
| SKILL_DRILLS_WR | OVER_SHOULDER_ADJUST | 101 | 100 | 6,604 |
| SKILL_DRILLS_WR | POST_CORNER_ROUTE_LEFT | 80 | 70 | 4,289 |
| SKILL_DRILLS_WR | POST_CORNER_ROUTE_RIGHT | 111 | 96 | 7,938 |
| SKILL_DRILLS_WR | RED_ZONE_FADE_RIGHT | 112 | 98 | 5,063 |
| SKILL_DRILLS_WR | SLANT_ROUTE_LEFT | 110 | 107 | 4,326 |
| SKILL_DRILLS_WR | SLOT_SAIL_ROUTE_RIGHT | 106 | 98 | 7,462 |
| SKILL_DRILLS_WR | SLOT_STRIKE_ROUTE_LEFT | 102 | 101 | 6,339 |
| SKILL_DRILLS_WR | SPEED_OUT_LEFT | 37 | 31 | 1,675 |
| SKILL_DRILLS_WR | SPEED_OUT_ROUTE_LEFT | 87 | 76 | 3,829 |
| THREE_CONE_DRILL | THREE_CONE_DRILL | 203 | 160 | 15,951 |

## Position coverage (players.csv × combine_tracking × player_play)

| position | players | w/ combine tracking | w/ snaps | median snaps |
| --- | ---: | ---: | ---: | ---: |
| WR | 106 | 106 | 105 | 254 |
| CB | 60 | 60 | 58 | 540 |
| G | 51 | 51 | 50 | 568 |
| T | 50 | 50 | 50 | 574 |
| DT | 47 | 47 | 47 | 296 |
| TE | 42 | 42 | 38 | 151 |
| DE | 36 | 36 | 35 | 502 |
| OLB | 30 | 30 | 28 | 399 |
| SS | 28 | 28 | 28 | 640 |
| FS | 25 | 25 | 24 | 248 |
| C | 20 | 20 | 20 | 342 |
| NT | 5 | 5 | 5 | 397 |
| DB | 5 | 5 | 5 | 30 |
| ILB | 2 | 2 | 1 | 194 |
| RB | 1 | 1 | 1 | 86 |
| FB | 1 | 1 | 1 | 36 |
| MLB | 1 | 1 | 1 | 282 |
