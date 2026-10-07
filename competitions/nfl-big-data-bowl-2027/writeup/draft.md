<!--
Kaggle Writeup body for NFL Big Data Bowl 2027. Limits: ≤ 2,000 words, < 10 tables/figures.
Figures live in ../assets/ (fig1_curve.png … fig4_rushers.png); upload them in the Writeup editor.
Companion notebook: notebooks/bdb27_analysis.ipynb (publish as a public Kaggle Notebook and link it below).
Word/figure count: python scripts/package_writeup.py
-->

> Paste into the Kaggle Writeup editor: title + subtitle above, this body below.

# Reading the 40 by the yard

**Subtitle:** The 40-yard dash is two tests in one. Sensor tracking separates them, and which one matters depends on the position.

## The question

Every spring, two numbers from the 40-yard dash get copied onto every draft board: the 10-yard split and the finish. They are almost the same number. Across the 424 prospects here who ran, the official 10 and 40 correlate at 0.95, so the stopwatch cannot tell a scout whether a player is quick off the line or simply fast at the end.

The sensor can. A 40 recorded at 10 Hz is about fifty speed measurements instead of two, every position runs the identical task, and it is the one drill where the sensor can be checked against an official clock. We asked one narrow question: **which yards of the 40 carry information about regular-season performance, and does the answer change by position?**

The short version: for pass rushers the signal lives in the first five yards; for receivers in the top-speed phase; for offensive linemen only in the first step or two; and for safeties and linemen the finish of the 40 has no measurable relationship with how fast they ever run in a game.

## Turning a sprint into a curve

The tracking clock does not start on the timer. Recording begins somewhere around the first movement, so time-based sensor splits are unreliable (our first attempt correlated 0.24 with the official 10-yard split). We therefore index speed by **distance covered**. Speed at the 5-yard mark does not care when the recording started. When a time is needed, the official 40 pins the clock: the time from 10 to 40 yards is the integral of 1/speed along the curve, and the sensor 10-yard split is the official 40 minus that segment.

This checks out. The sensor 10-yard split correlates 0.89 with the official split (bias 0.045 s, spread 0.053 s), and the sensor 10-to-40 segment correlates 0.97 with the official difference. Figure 1 shows the curve, the validation, and run-to-run repeatability by yard: 0.92 at 10 yards and 0.98 from 20 on, 0.69 at 5 yards, and only 0.42 in the first two and a half yards, the price of measuring the start.

![Figure 1. The 40 as a speed-by-distance curve (A), sensor versus stopwatch (B), and run-to-run repeatability by yard (C).](https://raw.githubusercontent.com/audriu/kaggle/main/competitions/nfl-big-data-bowl-2027/assets/fig1_curve.png)

The curve has two phases: acceleration up to about 10 yards, then top speed. The stopwatch cannot separate them; the curve does. That is the whole analysis: for each outcome we correlate speed at every half-yard with the outcome and read where the correlation peaks.

## What we linked it to

All outcomes come from 2023 to 2025 regular-season games, adjusted for context before averaging per player.

* **Pass rushers** (DE, DT, NT, OLB aligned on the edge or interior): pressure rate and sack rate per pass-rush snap, and get-off time adjusted for alignment, down, distance and season. Minimum 50 snaps, 64 players.
* **Receivers** (WR, TE): separation at the throw, adjusted for route, man or zone coverage, cushion, pass length, alignment, down and season, split into break routes and vertical routes. Minimum 100 routes, 75 players.
* **Offensive linemen**: pressure allowed, adjusted for spot, down, distance, rushers faced and season. Minimum 100 pass-protection snaps, 52 players.
* **Game speed** for everyone with at least 100 tracked plays: the 95th percentile of per-play top speed from the game tracking files.

The outcomes are noisier than the Combine inputs (split-half reliability 0.80 for get-off, 0.57 for pressure rate, 0.54 for pressure allowed, 0.45 for separation), so every correlation below is attenuated and would grow with more seasons.

## Which yards matter

![Figure 2. Correlation between speed at each yard of the 40 and the NFL outcome, with bootstrap 95% bands. Shaded area is the acceleration phase.](https://raw.githubusercontent.com/audriu/kaggle/main/competitions/nfl-big-data-bowl-2027/assets/fig2_corr_by_yard.png)

**Pass rushers.** The correlation with pressure rate rises steeply through the first yards, peaks at exactly 5 yards (r = 0.52) and then drifts down to 0.43 at the finish. Sack rate has the same shape with a lower ceiling. A rusher's 40 is a 5-yard test with 35 yards of cool-down.

**Receivers.** The opposite of what a route-running intuition suggests. Separation on break routes is unrelated to the first 10 yards and only becomes related in the top-speed phase (peak r = 0.27 near 35 yards); separation on vertical routes barely relates to the 40 at all. One reading is that a corner has to respect long speed, and that respect buys cushion on the hitch and the out. The band includes zero, so this is a lead, not a finding.

**Offensive linemen.** Only the first step carries anything (r = −0.31 at 1 yard, −0.07 by the finish). A lineman's finish time says nothing about pressure allowed.

**Game speed, by position.** Figure 3 asks a simpler question: does the 40 predict how fast a player runs on Sundays?

![Figure 3. Which yard of the 40 predicts game-day top speed by position (A), and the share of Combine top speed that each position reaches in games (B).](https://raw.githubusercontent.com/audriu/kaggle/main/competitions/nfl-big-data-bowl-2027/assets/fig3_speed_used.png)

Receivers reach 82% of their Combine top speed in games, and the finish of the 40 predicts it (r = 0.61 at 40 yards, 0.09 at 5). Defensive linemen reach 65%, predicted by the start (r = 0.62 at 2.5 yards). Offensive linemen reach 54% and safeties 75%, and for both no yard of the 40 predicts game speed (r ≈ 0). A safety's 4.45 and a tackle's 5.10 describe something that never happens on the field.

## The first five yards of a pass rusher

The pass-rush result is strong enough for a draft board, so we tested it properly.

![Figure 4. Speed at the 5-yard mark against pressure rate (A), and out-of-class R² for pressure rate from models fitted on two draft classes and scored on the third (B).](https://raw.githubusercontent.com/audriu/kaggle/main/competitions/nfl-big-data-bowl-2027/assets/fig4_rushers.png)

*Does the sensor add anything beyond the stopwatch?* Controlling for the official 10, the official 40, weight, draft slot and edge-versus-interior alignment, speed at 5 yards still carries a partial correlation of 0.26 with pressure rate (bootstrap 95% CI 0.02 to 0.48). The early burst beyond what the player's 40 time implies does the same (0.28, CI 0.05 to 0.50). For comparison, the official 10 split, given weight and slot, has a partial correlation of −0.20 with a CI that crosses zero.

*Does it hold out of sample?* Fitting on two draft classes and scoring the third, a base model of draft slot, weight and alignment explains 30% of the variance in pressure rate. Adding the official 10 split lowers that to 27%, adding the 40 leaves it at 30%, and adding both collapses it to 14% because the two times are collinear and the sample is 64 players. Replacing them with the single sensor number, speed at 5 yards, lifts it to 36%.

*How big is it?* One standard deviation of 5-yard speed is 0.33 yd/s, roughly a 6.6 against a 6.9. That is worth 1.3 points of pressure rate (p = 0.006) against a rookie mean of 6.3% and a player-to-player spread of 3.0: about 0.4 standard deviations of pressure rate from one Combine number.

*Is it just get-off?* Mostly not. Five-yard speed shaves 0.018 s per standard deviation off adjusted in-game get-off (p = 0.13), a quarter of the spread between players. It predicts pressure beyond the snap reaction, which points at the second and third step rather than the first.

**Table 1** lists the rushers whose first five yards most outran and most trailed what their 40 time implied. The six biggest over-performers averaged a 7.9% pressure rate; the four biggest under-performers averaged 4.4%; the group mean is 6.3%. Cedric Johnson, pick 214, carried the second-highest pressure rate in the cohort at 12.5% on a 4.63 forty nobody circled; the sensor had him at 7.30 yd/s five yards in, the fifth-largest early burst in the sample.

| Player | Pos | Class | Pick | 40 | 10 split | Speed at 5 yd | Burst vs 40 | Pressure % | Rush snaps |
|---|---|---|---|---|---|---|---|---|---|
| Mekhi Wingo | DT | 2024 | 189 | 4.85 | 1.64 | 7.20 | +0.49 | 6.8 | 132 |
| Eric Watts | DE | 2024 | UDFA | 4.67 | 1.65 | 7.37 | +0.47 | 3.6 | 225 |
| Justin Eboigbe | DE | 2024 | 105 | 5.18 | 1.80 | 6.73 | +0.36 | 7.1 | 253 |
| Byron Murphy | DT | 2024 | 16 | 4.87 | 1.69 | 7.05 | +0.36 | 8.9 | 682 |
| Cedric Johnson | DE | 2024 | 214 | 4.63 | 1.61 | 7.30 | +0.36 | 12.5 | 216 |
| Adetomiwa Adebawore | DT | 2023 | 110 | 4.49 | 1.61 | 7.38 | +0.29 | 8.6 | 511 |
| Tyrion Ingram-Dawkins | DE | 2025 | 139 | 4.86 | 1.69 | 6.38 | −0.33 | 5.3 | 113 |
| Tyler Baron | DE | 2025 | 176 | 4.62 | 1.61 | 6.63 | −0.32 | 4.4 | 68 |
| Nazir Stackhouse | DT | 2025 | UDFA | 5.15 | 1.80 | 6.11 | −0.30 | 1.4 | 71 |
| Barryn Sorrell | DE | 2025 | 124 | 4.68 | 1.65 | 6.60 | −0.29 | 6.7 | 119 |

*Table 1. Burst vs 40 is speed at 5 yards minus the value predicted from the player's official 40 within his position group, in yd/s.*

## What a team can do with this

1. **Report the 40 as three numbers, not two.** Speed at 5 yards, at 10 yards, and top speed, from the sensor. The first is a cleaner 10-split (no reaction time, two runs averaged), the third is almost perfectly repeatable, and together they separate the phases the stopwatch cannot.
2. **Weight the phases by position.** Pass rushers: the first five yards, discount the finish. Receivers and corners: the finish. Offensive line and safeties: the first steps for linemen, and for both, stop paying for long speed that is never used.
3. **Pass-rush boards.** Replace the 10 split with speed at 5 yards. On this cohort that substitution moved out-of-class explained variance in pressure rate from 27% to 36%, and it flagged a seventh-round edge who became a top-three pressure producer.
4. **Development.** The acceleration phase is repeatable enough to be a target (0.69 for one run, 0.82 for the mean of two). A rusher whose 5-yard speed trails his 40 is a candidate for first-step work; one who out-accelerates his 40 is a candidate the stopwatch undersells.

## Limitations

The samples are small (64 rushers, 75 receivers, 52 linemen) and limited to rookies who earned snaps, so every link is estimated on survivors. Pressure rate is only moderately reliable over one to three seasons, which shrinks every correlation. Draft slot is a strong confounder; we controlled for it, but it may stand in for traits the 40 does not measure. The receiver and lineman bands touch zero and should be read as leads. Nothing here is causal: a fast first five yards may mark a trait rather than make a rusher.

## Appendix

* Notebook (every figure and number): https://www.kaggle.com/code/vienas/reading-the-40-by-the-yard-bdb-2027
* Code: https://github.com/audriu/kaggle
* Methods: curves use straight-line displacement from the first frame, smoothed over three frames, on a half-yard grid; two runs averaged. Outcome adjustments are play-level OLS or logistic residuals, averaged per player. Partial correlations use OLS residuals on the control set with 4,000-sample bootstraps over players. Cross-validation fits on two draft classes and scores the third.
