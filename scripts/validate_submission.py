#!/usr/bin/env python3
"""Validate a submission the way Kaggle's cabt env loads it.

Catches failures that local importlib / self-play miss, especially:
  - NameError: __file__ is not defined (agent loaded via exec())
  - missing deck.csv / non-60-card deck
  - missing agent() or bad return shapes

Usage:
  python scripts/validate_submission.py
  python scripts/validate_submission.py dist/submission.tar.gz
  python scripts/validate_submission.py agent/
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "dist" / "submission.tar.gz"
REQUIRED_TOP = ("main.py", "deck.csv", "cg")


def load_agent_like_kaggle(agent_dir: Path):
    """Load main.py the way kaggle_environments does: exec() with no __file__."""
    main_py = agent_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"Missing {main_py}")

    code = main_py.read_text(encoding="utf-8")
    # Mimic Kaggle: cwd is the agent directory; __file__ is absent.
    prev_cwd = Path.cwd()
    prev_path = list(sys.path)
    try:
        os.chdir(agent_dir)
        # agent dir first so `import cg` works when present
        sys.path.insert(0, str(agent_dir.resolve()))
        env: dict = {"__name__": "__kaggle_agent__"}
        exec(compile(code, str(main_py), "exec"), env)  # noqa: S102 — intentional
        agent = env.get("agent")
        if not callable(agent):
            raise TypeError("main.py must define a callable agent(obs_dict)")
        return agent
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def smoke_test_agent(agent) -> None:
    deck = agent({"select": None})
    if not isinstance(deck, list) or len(deck) != 60:
        raise ValueError(
            f"agent(select=None) must return 60 card IDs, got {type(deck).__name__} "
            f"len={len(deck) if isinstance(deck, list) else 'n/a'}"
        )
    if not all(isinstance(x, int) for x in deck):
        raise TypeError("deck card IDs must be ints")

    action = agent(
        {
            "select": {
                "option": ["a", "b", "c", "d"],
                "minCount": 1,
                "maxCount": 2,
            }
        }
    )
    if not isinstance(action, list):
        raise TypeError(f"agent(select=...) must return list[int], got {type(action)}")
    if not action:
        raise ValueError("agent returned empty action for non-empty options")
    if len(action) > 2:
        raise ValueError(f"action length {len(action)} exceeds maxCount=2")
    if len(set(action)) != len(action):
        raise ValueError("action indices must be unique")
    if not all(isinstance(i, int) and 0 <= i < 4 for i in action):
        raise ValueError(f"action indices out of range: {action}")


def check_archive_layout(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    top = {n.split("/")[0] for n in names}
    missing = [name for name in REQUIRED_TOP if name not in top]
    if missing:
        raise ValueError(
            f"Archive top-level missing {missing}; got {sorted(top)}. "
            "Files must be at archive root (not nested in a folder)."
        )
    if any("/" not in n and n.endswith(".py") and n != "main.py" for n in names):
        # warn-only style: allow extra files, but flag accidental root scripts
        pass
    return names


def extract_archive(archive: Path, dest: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    # If someone nested under a single folder, detect and fail clearly
    main = dest / "main.py"
    if main.is_file():
        return dest
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "main.py").is_file():
        raise ValueError(
            f"Archive nests files under {children[0].name}/ — "
            "Kaggle expects main.py/deck.csv/cg at archive root"
        )
    raise FileNotFoundError("main.py not found after extracting archive")


def validate_agent_dir(agent_dir: Path) -> None:
    agent_dir = agent_dir.resolve()
    deck_csv = agent_dir / "deck.csv"
    if not deck_csv.is_file():
        raise FileNotFoundError(f"Missing {deck_csv}")
    lines = [ln.strip() for ln in deck_csv.read_text().splitlines() if ln.strip()]
    if len(lines) != 60:
        raise ValueError(f"{deck_csv} must have 60 cards, got {len(lines)}")

    agent = load_agent_like_kaggle(agent_dir)
    smoke_test_agent(agent)


def validate_archive(archive: Path) -> None:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Missing archive: {archive}")
    check_archive_layout(archive)
    with tempfile.TemporaryDirectory() as tmp:
        agent_dir = extract_archive(archive, Path(tmp))
        validate_agent_dir(agent_dir)


def validate_path(path: Path) -> None:
    path = path.resolve()
    if path.is_dir():
        validate_agent_dir(path)
    elif path.is_file() and path.name.endswith((".tar.gz", ".tgz")):
        validate_archive(path)
    else:
        raise ValueError(f"Expected agent directory or .tar.gz, got {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Archive or agent dir (default: {DEFAULT_ARCHIVE})",
    )
    args = parser.parse_args()
    try:
        validate_path(args.path)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK — Kaggle-style load + smoke test passed for {args.path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
