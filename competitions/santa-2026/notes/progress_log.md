# Progress log — Santa 2026 (The Elf Calculator)

- **2026-11-20 — Investigation & environment setup**
  - Competition started 2026-11-18; entry/merger deadline 2027-01-21; final deadline 2027-01-28 (23:59 UTC); $50k prize pool ($12k / $10k / $10k / $8k + $10k Rudolph Prize for longest time at #1).
  - **100% public leaderboard** (no private split), 100 submissions/day, 2 final selections, ties broken by earliest submission.
  - Auth configured via `~/.kaggle/access_token` (new Kaggle token format; works natively with `kaggle==2.2.5`).
  - Environment quirk: created repo `.venv` via `uv venv .venv` (Python 3.12.3); `kaggle==2.2.5` omits `six` from its declared dependencies (`ModuleNotFoundError: No module named 'six'`), fixed by installing `six` into `.venv`.
  - Host hardware: 16 CPU threads (`nproc=16`), `g++ 13.3.0` (`-O3 -march=native -fopenmp -std=c++20`).

- **2026-11-20 — Data download & EDA**
  - Downloaded `santa-2026.zip` (788 KiB) → `data/{puzzles.csv,practice_puzzles.csv,sample_submission.csv}`.
  - `puzzles.csv`: 1,000 scored puzzles (`scored_0001`..`scored_1000`), `registers=48` always, `n_starts` 1–8 (median 3), `n_targets` 1–32 (median 30, total **26,902** targets, total `lower_bound` **26,898**), total target bits **1,008,549** (avg **37.49** bits/target, max 60 bits / $10^{18}$), `value_limit / max(targets)` ranges `1.000`–`5.000` (p10 = `1.025`, so subtraction intermediates `q*m = T+d` must always check `<= value_limit`), 16 puzzles have `max_target <= 1000`.
  - `practice_puzzles.csv`: 200 unscored puzzles with `family` (`shared`: 51, `mixed`: 48, `independent`: 45, `multiplicative`: 40, `tiny`: 16) and `design_tier` (`hard`: 170, `easy`: 16, `medium`: 14).
  - Structural findings by family:
    - `independent` (avg 50.4 bits/target): high-entropy targets, rare inter-target relations (2/285).
    - `mixed` (avg 43.7 bits/target): occasional close multiples/diffs (23/299) and large targets up to $10^{18}$ with tight `value_limit`.
    - `shared` (avg 36.5 bits/target): dense linear/scaled clusters (`t_j = p*t_i ± delta`, 73/243 close multiples, many small diffs like 5–600).
    - `multiplicative` (avg 23.8 bits/target): heavily near-smooth (212/299) with many integer/rational relations (`q*t_j = p*t_i ± delta`, 127/299).
    - `tiny` (1–4 targets, `max_t <= 16`): exact BFS solves optimally.
  - Official `sample_submission.csv`: validated locally at **1,490,379** total instructions (per-puzzle range 3–2,873, median 1,596; binary double-and-add per target).
  - Day-2 benchmark landscape (2026-11-20): top public notebooks `~138,374–138,485`; LB #1 (`sekai013`) `131,054`, top-5 `131,054–133,534`.

- **2026-11-20 — Harness & v1 C++ solver (`solver/solve.cpp`)**
  - Built `scripts/{setup_data.py,validate.py,merge_best.py,submit.py}` and 16-thread C++20 portfolio solver `solver/solve.cpp` with internal exact `u128` verifier + dead-code elimination (DCE).
  - Prototype v0 (K=41, 6 profiles, max_exp=150): fixed leaf-register collision bug (`dst_reg == scratch1` during `EDGE_MUL_P_*` overwritten by `d_chain` → separated `scratch0=45, scratch1=46, scratch2=47`), scored **40,639** on 200 practice puzzles (avg 203.2) in 6.5s.
  - Upgraded v1 (+K=46 zero-parent-reserve profiles for `independent`/`mixed`, rational edges `(p*T_i ± d)/q` for `q in {2,3,4,5,6,8}`, larger `mod_E1` tables `3400 × 36`, `max_exp=240`):
    - Practice (200 puzzles, 16.2s): **40,176** (avg 200.88) — `independent` 289.6 (`4.97` bits/op), `mixed` 243.4, `shared` 186.7, `multiplicative` 147.3, `tiny` 3.125.
    - Scored (1,000 puzzles, 1m13s): **198,252** (avg 198.252) — **7.52× smaller** than `sample_submission.csv` (`1,490,379`), verified valid by both C++ internal simulator and `scripts/validate.py`.
  - **Submission v1** uploaded via Kaggle CLI: ref **58613555** (`dist/submission.csv`, 3.14 MB, local exact score **198,252**, 99 submissions left today).
