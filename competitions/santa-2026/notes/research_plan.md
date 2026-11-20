# Research plan — Santa 2026 (The Elf Calculator)

## Objective & Exact Rules
Minimize total instruction count $\sum_{p=1}^{1000} |\text{program}_p|$ across all 1,000 puzzles in `data/puzzles.csv` (100% public leaderboard; local exact score == LB score).

- **Registers:** 48 (`0..47`), `starting_values` occupy `0..len-1`, rest undefined.
- **Instruction:** `["op", i, j, d]` with `op ∈ {"+", "-", "*", "/"}`. Source registers must be defined; reads happen before write (`d == i`, `d == j`, `i == j` all legal; initial registers may be overwritten).
- **Legality:** All intermediate values must be positive integers in `[1, value_limit]` (`value_limit <= 10^18`); subtraction requires `a > b`; division must be exact (`a % b == 0`).
- **Coverage & limits:** Every integer in `targets` must appear in any register at least once (remains covered if overwritten). Max 4,096 instructions/puzzle and 131,072 UTF-8 bytes per CSV cell. Any invalid puzzle rejects the whole submission.

## Score Targets (as of 2026-11-20)
- `sample_submission.csv`: `1,490,379`
- **Our v1 (`solve.cpp`, ref 58613555)**: **`198,252`** (~200.9/puzzle on practice)
- Public SOTA notebooks (Day 2): `~138,374`
- Leaderboard #1 (Day 2, `sekai013`): `131,054` (top 5: `131,054`–`133,534`)

## Core Optimization Roadmap (198k → <135k)

1. **Per-puzzle monotone portfolio (`best_programs.csv` via `scripts/merge_best.py`)**
   - Every puzzle is independent; maintain a persistent per-puzzle best store and merge every new solver/post-processor monotonically.

2. **Target-tailored / joint-prefix dictionary & multi-target recipes (biggest lever: ~35k–45k gain)**
   - Currently `solve.cpp` builds generic coprime dictionaries and solves each target independently (~10.6 ops for 60-bit targets, ~4.97 bits/op).
   - **Joint prefix / shared intermediate synthesis:** Instead of independent chains for 30 targets, extract shared high-order prefixes / common quotients / shared factor bases across the 30 targets (e.g., beam/meet-in-the-middle finding a small set of 10–25 puzzle-specific anchors `A` that simultaneously reduce many targets in 1–2 ops).
   - **Custom dictionary optimization (simulated annealing / greedy dictionary selection):** Score candidate dictionary entries directly by how many `(target -> reduced)` hits they create on that specific puzzle's 30 targets instead of generic coprimality! Even optimizing the last 15–20 registers of `V` for the specific puzzle can shave 1–2 ops/target (~30–50 ops/puzzle).

3. **Richer backward transitions & base cases (~15k–20k gain)**
   - Add **3-op base cases** (`e1_a * e1_b`, `e1_a ± e1_b`, `(e1 op v1) op v2`) and **1-op multiplier steps** (`q * e1 ± v` where `e1 ∈ E1`) so 20–24 bit remainders finish in 2–3 ops instead of 4.
   - Add **exact factorization / smooth-neighbor search** (`T ± d` with small `d` factored into smooth factors, Pollard-rho for `<= 10^18`) specifically for `multiplicative` and `shared` puzzles.
   - Expand inter-target DAG beyond single-parent trees: allow **multi-parent target combinations** (`t_k = t_i + t_j`, `t_k = t_i * t_j`, `t_k = a*t_i + b*t_j ± d`) and exact Chu-Liu/Edmonds minimum arborescence + optimal Sethi-Ullman register scheduling.

4. **DAG Superoptimization & Arithmetic Graph Rewrites (~5k–15k gain)**
   - Convert best programs to value-DAGs (SSA form), run:
     - Global common subexpression elimination (CSE) & value numbering.
     - Local cone / window resynthesis (replace $k$-instruction sub-DAGs with $\le k-1$ instructions via exact BFS / SAT).
     - Register-packing & liveness-aware re-allocation (graph coloring / linear scan over 48 registers) so intermediate values survive longer for downstream reuse.
