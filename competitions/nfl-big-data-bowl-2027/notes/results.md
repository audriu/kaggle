# Results log — "Reading the 40 by the yard"

All numbers reproduced by `notebooks/bdb27_analysis.py` (= `bdb27_analysis.ipynb`), 2026-10-07.

## Angles tried and dropped

| Angle | What happened |
| --- | --- |
| WR route-drill break metrics (decel into / accel out of the break) → in-game separation | Within-drill test-retest fine (0.84–0.89) but cross-drill consistency ≈ 0.1–0.3 and no link to adjusted separation (r ≈ 0, some wrong-signed). Dropped. Code kept in `src/bdb/breaks.py`. |
| DL pass-rush drill first-step burst → in-game get-off / pressure | Drill's first second is a controlled move on a bag (speed plateaus ~4 yd/s at 1 s), test-retest 0.4, r ≈ 0 with get-off. Dropped. Code in `src/bdb/getoff.py`. |
| 40-yd dash speed-by-distance curve → position outcomes | **Kept.** Clean, validated against the official clock, strong DL result. Code in `src/bdb/forty.py` and the notebook. |

## Measurement

- 789 runs, 417 players (424 ran; 7 without tracking). Sensor 10-split vs official r = 0.89, bias +0.045 s, sd 0.053 s; sensor 10→40 vs official r = 0.97.
- Test-retest by yard (300 players × 2 runs): 1 yd 0.46 · 2.5 yd 0.42 · 5 yd 0.69 · 10 yd 0.92 · 20 yd 0.98 · 40 yd 0.99.
- Official ten vs forty correlate 0.95 (n = 424).

## Outcome reliability (split-half, snaps)

get-off 0.80 (≥50 rush snaps, n 81) · pressure rate 0.57 · OL pressure allowed 0.54 (≥100) · receiver separation 0.45 (≥100 routes) · game p95 speed season-to-season 0.91–0.93.

## Correlation by yard (peak)

| Outcome | n | r@5 | r@40 | peak |
| --- | --- | --- | --- | --- |
| DL/OLB pressure rate | 64 | +0.52 | +0.43 | +0.52 @ 5 yd |
| DL/OLB sack rate | 64 | +0.35 | +0.26 | +0.35 @ 4.5 |
| WR/TE separation, break routes | 75 | +0.06 | +0.26 | +0.27 @ 35.5 |
| WR/TE separation, vertical | 75 | +0.11 | +0.11 | +0.13 @ 34 |
| OL pressure allowed (adj) | 52 | −0.20 | −0.07 | −0.31 @ 1 |
| WR game p95 speed | 56 | +0.09 | +0.61 | +0.61 @ 39.5 |
| DL game p95 speed | 49 | +0.60 | +0.53 | +0.62 @ 2.5 |
| CB game p95 speed | 34 | +0.20 | +0.21 | +0.31 @ 34 |
| S game p95 speed | 26 | −0.07 | +0.05 | ≈ 0 |
| OL game p95 speed | 55 | +0.19 | +0.10 | ≈ 0 |

Share of Combine top speed reached in games (p95): WR 82 % · CB 77 % · TE 77 % · S 75 % · DL 65 % · OL 54 %.

## Pass rushers (n = 64, ≥50 rush snaps, official ten & forty available)

- Partial r with pressure rate | ten + forty + weight + log pick + edge: s_at_5 +0.26 [0.02, 0.48]; acc_resid +0.28 [0.05, 0.50]. For comparison | weight + pick + edge: forty −0.14 [−0.38, 0.12], ten −0.20 [−0.40, 0.05].
- Leave-one-draft-class-out R²: base (pick, weight, edge) 0.296 · +ten 0.274 · +forty 0.302 · +both 0.136 · +s_at_5 0.363.
- Effect: 1 SD s_at_5 = 0.33 yd/s → +1.3 pts pressure rate (mean 6.3 %, sd 3.0; p = 0.006). → get-off −0.018 s per SD (p = 0.13; sd 0.070).
- Table 1: top-6 acc_resid mean pressure 7.9 % vs bottom-4 4.4 % vs all 6.3 %.

## Writeup status

`writeup/draft.md` ≈ 1,950 words, 4 figures + 1 table. Notebook executes clean locally (nbconvert). Submitted 2026-10-07 (Open track); see `progress_log.md` for links.
