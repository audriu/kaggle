#!/usr/bin/env python3
"""Validate and package the Big Data Bowl writeup for the Kaggle Writeup UI.

Builds:
  dist/writeup.md          — cleaned body ready to paste
  dist/writeup_bundle.zip  — writeup + assets/ for the media gallery
  dist/word_count.txt      — word count vs the 2,000-word limit and figure count vs < 10

Exit code 1 if over limits or the draft is still a skeleton.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup" / "draft.md"
ASSETS = ROOT / "assets"
DIST = ROOT / "dist"
WORD_LIMIT = 2000
FIGURE_LIMIT = 10  # "fewer than 10 tables or figures"


def _strip_front_matter(text: str) -> str:
    """Drop HTML comments and the leading 'Paste into…' blockquote guidance."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    out: list[str] = []
    skipping_tip = False
    for line in text.splitlines():
        if line.strip().startswith(">") and "Paste into" in line:
            skipping_tip = True
            continue
        if skipping_tip:
            if line.strip() == "" or line.strip().startswith(">"):
                continue
            skipping_tip = False
        out.append(line)
    body = "\n".join(out).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", body)


def word_count(text: str) -> int:
    no_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    no_img = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", no_code)   # image embeds are figures, not prose
    no_url = re.sub(r"https?://\S+", "url", no_img)
    return len(re.findall(r"[A-Za-z0-9']+", no_url))


def figure_count(text: str) -> tuple[int, int]:
    """(images, markdown tables) — both count toward the <10 budget."""
    images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text))
    tables = len(re.findall(r"^\s*\|\s*-{3,}", text, flags=re.MULTILINE))
    return images, tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writeup", type=Path, default=WRITEUP)
    parser.add_argument("--out-dir", type=Path, default=DIST)
    parser.add_argument("--limit", type=int, default=WORD_LIMIT)
    args = parser.parse_args()

    if not args.writeup.is_file():
        print(f"Missing {args.writeup}", file=sys.stderr)
        return 1

    body = _strip_front_matter(args.writeup.read_text(encoding="utf-8"))
    n = word_count(body)
    images, tables = figure_count(body)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md_out = args.out_dir / "writeup.md"
    md_out.write_text(body, encoding="utf-8")
    (args.out_dir / "word_count.txt").write_text(
        f"{n} words (limit {args.limit})\n"
        f"{images} images + {tables} tables = {images + tables} (must be < {FIGURE_LIMIT})\n",
        encoding="utf-8",
    )

    bundle = args.out_dir / "writeup_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_out, arcname="writeup.md")
        if ASSETS.is_dir():
            for path in sorted(ASSETS.rglob("*")):
                if path.is_file() and path.name != ".gitkeep":
                    zf.write(path, arcname=f"assets/{path.relative_to(ASSETS)}")

    print(f"Wrote {md_out}")
    print(f"Wrote {bundle}")
    print(f"Word count: {n} / {args.limit}")
    print(f"Figures: {images} images + {tables} tables = {images + tables} (< {FIGURE_LIMIT})")

    rc = 0
    if n > args.limit:
        print(f"WARNING: over the {args.limit}-word limit — trim.", file=sys.stderr)
        rc = 1
    if images + tables >= FIGURE_LIMIT:
        print(f"WARNING: {images + tables} figures/tables — must be fewer than {FIGURE_LIMIT}.", file=sys.stderr)
        rc = 1
    if n < 200:
        print("WARNING: writeup looks like the skeleton — fill writeup/draft.md first.", file=sys.stderr)
        rc = 1
    if "<link>" in body:
        print("WARNING: notebook/code links still placeholders.", file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
