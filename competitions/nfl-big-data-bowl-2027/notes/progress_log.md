# Progress log

- 2026-10-07 — Subproject created. Competition joined, rules accepted. Scaffold: README, notes
  (rubric, data dictionary, research plan), `src/bdb` loaders, scripts (setup/summary/package/submit),
  writeup skeleton. No analysis yet.
- 2026-10-07 — Data downloaded (2.2 GiB, 9 CSVs) and parquet cache built (`setup_data.py
  --parquet`, 13 s). `data_summary.md` generated (drill catalogue, position coverage). Fixed two
  loader bugs found on real data: zip extracts into a nested folder (now flattened) and `NA`
  strings were not parsed as nulls by polars. Logged `a`-sign and row-count quirks in
  `data_dictionary.md`. Next: EDA notebook, pick angle (see research_plan.md).
- 2026-10-07 — Full analysis + writeup drafted. Tried WR break metrics and DL drill burst
  (both null, see `results.md`), settled on 40-yd-dash speed-by-distance curves linked to
  position-specific outcomes. `notebooks/bdb27_analysis.py` (jupytext-style) is the single source of
  truth; converted to `.ipynb` and executed clean. Figures in `assets/`, writeup in
  `writeup/draft.md` (under 2,000 words, 5 figures/tables). Remaining: publish notebook on
  Kaggle, fill links, submit via Writeup UI.
