#!/usr/bin/env python3
"""Validate and package the Strategy writeup for Kaggle Writeup paste/upload.

Strategy submissions are Kaggle Writeups (not agent tar.gz). This builds:
  dist/writeup.md          — cleaned body ready to paste
  dist/writeup_bundle.zip  — writeup + assets/ for the media gallery
  dist/word_count.txt      — word count vs the ~2000 word limit
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


def _strip_front_matter(text: str) -> str:
    """Drop HTML comments and the leading 'paste into…' blockquote guidance."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    lines = text.splitlines()
    out: list[str] = []
    skipping_tip = False
    for line in lines:
        if line.strip().startswith(">") and "Paste into" in line:
            skipping_tip = True
            continue
        if skipping_tip:
            if line.strip() == "" or line.strip().startswith(">"):
                continue
            skipping_tip = False
        out.append(line)
    return "\n".join(out).strip() + "\n"


def word_count(text: str) -> int:
    # Count words in visible prose (ignore fenced code lightly)
    no_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    return len(re.findall(r"[A-Za-z0-9']+", no_code))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--writeup",
        type=Path,
        default=WRITEUP,
        help=f"Source markdown (default: {WRITEUP})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DIST,
        help=f"Output directory (default: {DIST})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=WORD_LIMIT,
        help=f"Soft word limit (default: {WORD_LIMIT})",
    )
    args = parser.parse_args()

    if not args.writeup.is_file():
        print(f"Missing {args.writeup}", file=sys.stderr)
        return 1

    body = _strip_front_matter(args.writeup.read_text(encoding="utf-8"))
    n = word_count(body)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md_out = args.out_dir / "writeup.md"
    md_out.write_text(body, encoding="utf-8")
    (args.out_dir / "word_count.txt").write_text(
        f"{n} words (limit {args.limit})\n", encoding="utf-8"
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
    if n > args.limit:
        print(
            f"WARNING: over the ~{args.limit} word limit — trim before submitting.",
            file=sys.stderr,
        )
        return 1
    if n < 200:
        print(
            "WARNING: writeup looks too short — fill writeup/draft.md first.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
