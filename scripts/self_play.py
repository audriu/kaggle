#!/usr/bin/env python3
"""Run a local self-play match with the CABT / kaggle-environments cabt env.

Requires competition data (sample_submission/cg) via scripts/setup_data.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("submission_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure agent dir (deck.csv) and cg package are importable
    agent_dir = path.parent
    cg_parent = _find_cg_parent()
    sys.path.insert(0, str(agent_dir))
    if cg_parent:
        sys.path.insert(0, str(cg_parent))
    spec.loader.exec_module(mod)
    return mod.agent


def _find_cg_parent() -> Path | None:
    data = ROOT / "data"
    for pattern in ("sample_submission", "**/sample_submission"):
        for path in ([data / "sample_submission"] if pattern == "sample_submission" else data.glob(pattern)):
            if (path / "cg" / "api.py").exists():
                return path
    return None


def _read_deck(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=ROOT / "agent" / "main.py")
    parser.add_argument("--deck", type=Path, default=ROOT / "agent" / "deck.csv")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--html", type=Path, default=None, help="Optional replay HTML path")
    args = parser.parse_args()

    cg_parent = _find_cg_parent()
    if cg_parent is None:
        print(
            "Missing data/sample_submission/cg — run: python scripts/setup_data.py",
            file=sys.stderr,
        )
        return 1

    try:
        from kaggle_environments import make
    except ImportError:
        print("Install deps: pip install -r requirements.txt", file=sys.stderr)
        return 1

    deck = _read_deck(args.deck)
    if len(deck) != 60:
        print(f"Deck must be 60 cards, got {len(deck)}", file=sys.stderr)
        return 1

    agent_fn = _load_agent(args.agent)
    wins = [0, 0]
    for i in range(args.games):
        env = make("cabt", configuration={"decks": [deck, deck]}, debug=True)
        env.run([agent_fn, agent_fn])
        rewards = env.steps[-1][0].get("reward"), env.steps[-1][1].get("reward")
        print(f"game {i + 1}: rewards={rewards}")
        if rewards[0] is not None and rewards[1] is not None:
            if rewards[0] > rewards[1]:
                wins[0] += 1
            elif rewards[1] > rewards[0]:
                wins[1] += 1
        if args.html and i == args.games - 1:
            args.html.write_text(env.render(mode="html"))
            print(f"wrote {args.html}")

    print(f"wins seat0/seat1: {wins[0]}/{wins[1]} over {args.games} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
