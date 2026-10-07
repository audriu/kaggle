#!/usr/bin/env python3
"""Rebuild notebooks/bdb27_analysis.ipynb from the jupytext-style script and (optionally) execute it.

Usage:
  python scripts/build_notebook.py            # convert only
  python scripts/build_notebook.py --execute  # convert, then run with nbconvert (writes outputs into the .ipynb)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "bdb27_analysis.py"
NB = ROOT / "notebooks" / "bdb27_analysis.ipynb"


def convert() -> int:
    cells = []
    for chunk in re.split(r"^# %%", SRC.read_text(encoding="utf-8"), flags=re.M):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            lines = chunk.split("\n")[1:]
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines).strip()
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}
    nbformat.write(nbformat.v4.new_notebook(cells=cells, metadata=meta), NB)
    print(f"Wrote {NB} ({len(cells)} cells)")
    return len(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the notebook after converting")
    args = parser.parse_args()
    convert()
    if args.execute:
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--ExecutePreprocessor.timeout=1800", str(NB), "--output", NB.name]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=NB.parent)
        nb = nbformat.read(NB, 4)
        errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
        print("execution errors:", len(errs))
        return 1 if errs else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
