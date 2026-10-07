# Notebooks

`bdb27_analysis.py` is the single source of truth for the analysis: a jupytext-style script with
`# %%` cell markers. It reads the competition CSVs from `/kaggle/input/nfl-big-data-bowl-2027` when
run on Kaggle, otherwise from `../data` (override with `BDB_DATA`), and writes figures and tables to
`/kaggle/working` or `../assets` (override with `BDB_OUT`). It runs in about 20 s locally.

```bash
cd notebooks && python bdb27_analysis.py          # regenerate assets/ and print every number in the writeup
python scripts/build_notebook.py --execute        # (from the competition root) rebuild + run the .ipynb
```

## Publishing on Kaggle (required: the writeup must attach a public notebook)

1. Kaggle → Code → New Notebook → File → Import Notebook → upload `bdb27_analysis.ipynb`.
2. Add Data → competition `nfl-big-data-bowl-2027`. polars, statsmodels, scikit-learn and
   matplotlib are all in the default Kaggle image.
3. Run all (a few minutes; the game-tracking CSVs are about 2 GB). Save version → Share → Public.
4. Paste the notebook URL into the Appendix of `writeup/draft.md` and attach it in the Writeup editor.

Exploratory code for the angles that were tried and dropped lives in `../src/bdb/breaks.py` and
`../src/bdb/getoff.py`; see `../notes/results.md` for what happened to them.
