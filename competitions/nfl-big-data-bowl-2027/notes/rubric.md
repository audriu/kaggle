# NFL Big Data Bowl 2027 — rubric & rules cheat-sheet

Source: competition Overview / Rules pages, captured 2026-10-07.

## Prompt (one sentence)

Uncover **non-obvious, actionable** linkages between 10 Hz Combine sensor tracking and
regular-season NFL game performance that help coaches/evaluators find hidden talent, avoid draft
busts, and design position-specific development protocols.

Organisers explicitly suggest focusing on **one aspect of movement, one position group, or one
drill** rather than everything. Example angles they list:

- WR/DB: change of direction, deceleration into breaks, acceleration out of cuts → in-game separation
- OL/DL: first-step quickness, lateral agility, burst in agility drills → pass-rush / run-block output
- Drill translation: sensor-derived metrics vs. stopwatch times — where does tracking add context?
- Position-specific mechanics: body control, acceleration profiles, pursuit angles → rookie performance

## Gates (PASS/FAIL)

1. **Combine-to-NFL linkage** — explicitly links Combine *tracking* data to regular-season NFL game performance.
2. **Writeup completeness** — readable markdown, embedded visuals, and an attached **public Kaggle Notebook**.

Any writeup that does not use the player tracking data is not scored.

## Scored components (0–10 each, weighted)

| Weight | Component | Judges ask |
| --- | --- | --- |
| 30% | Football | Usable by teams week-to-week? Accounts for what makes football data messy? Unique ideas? |
| 30% | Data science | Correct? Claims backed by data? Appropriate statistical models? Innovative applications? |
| 20% | Writeup | Well written, easy to follow, motivation (metric / player evaluation) clearly defined? |
| 20% | Data viz | Charts/tables accessible, accurate, innovative? |

Judges: analytics staff from the 32 NFL teams and tracking vendors. Weighted average of scores.

## Format limits

- ≤ 2,000 words; fewer than 10 tables/figures (over → may be penalised).
- Code mostly in an appendix or the linked notebook; show snippets only where they aid understanding.

## Rules that matter for us

- Hackathon: **one submission per team**; team size max 5; mergers allowed before the deadline.
- Data: CC BY-NC 4.0, non-commercial, do not redistribute → stays in `data/` (gitignored).
- External data allowed if free and equally accessible to all (e.g. public college stats, draft data).
- Winner license: open source (OSI) for the writeup and the code that produced it.
- No private code sharing outside the team; public sharing must be on the Kaggle forum/notebooks.
- No leaderboard. Judging 7–25 Jan 2027, results ~26 Jan 2027.
- Finalists (3 per track) are invited to present at the 2027 Combine; one wins the extra $10k.

## Timeline

| Date | |
| --- | --- |
| 2026-10-06 | start |
| 2027-01-06 23:59 UTC | final submission deadline (shows as 7 Jan 01:59 local) |
| 2027-01-07 → 01-25 | judging |
| 2027-01-26 | results (anticipated) |
