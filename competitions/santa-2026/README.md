# Santa 2026 - The Elf Calculator

Optimization solver and submission harness for [Santa 2026 - The Elf Calculator](https://www.kaggle.com/competitions/santa-2026).

## Metric
Minimize total instruction count across all 1,000 puzzles in `data/puzzles.csv` (100% public leaderboard, no private split).

## Quick Commands

```bash
# 1. Download/extract data (uses ~/.kaggle/access_token or ~/.kaggle/kaggle.json)
../../.venv/bin/python scripts/setup_data.py

# 2. Compile C++ solver
g++ -O3 -march=native -fopenmp -std=c++20 solver/solve.cpp -o solver/solve

# 3. Benchmark on 200 practice puzzles
./solver/solve data/practice_puzzles.csv dist/practice_out.csv
python3 scripts/validate.py dist/practice_out.csv --puzzles data/practice_puzzles.csv

# 4. Solve all 1,000 scored puzzles + validate
./solver/solve data/puzzles.csv dist/submission.csv
python3 scripts/validate.py dist/submission.csv

# 5. Merge per-puzzle best across multiple runs
python3 scripts/merge_best.py dist/best.csv dist/submission.csv ...

# 6. Submit (validates locally first)
../../.venv/bin/python scripts/submit.py -f dist/submission.csv -m "message"
```
