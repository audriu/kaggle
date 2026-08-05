"""CABT agent entrypoint for the Pokémon TCG AI Battle Challenge.

Contract (official cabt engine):
  - First call often has select=None → return a legal 60-card deck (card IDs).
  - Later calls: return up to select.maxCount distinct indices into select.option.
"""

from __future__ import annotations

import random
from pathlib import Path


def _deck_candidates() -> list[Path]:
    # Kaggle loads agents via exec(), so __file__ is often undefined — never
    # evaluate it while building the search list.
    paths = [
        Path("deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
    ]
    try:
        paths.insert(1, Path(__file__).resolve().parent / "deck.csv")
    except NameError:
        pass
    return paths


def _load_deck() -> list[int]:
    for path in _deck_candidates():
        if path.exists():
            lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            deck = [int(x) for x in lines]
            if len(deck) != 60:
                raise ValueError(f"{path} must contain exactly 60 card IDs, got {len(deck)}")
            return deck
    raise FileNotFoundError("deck.csv not found next to main.py")


MY_DECK = _load_deck()


def agent(obs_dict: dict) -> list[int]:
    """Decide the next action(s) for the current observation."""
    select = obs_dict.get("select")
    if select is None:
        # Deck submission / setup phase
        return MY_DECK

    options = select.get("option") or []
    max_count = int(select.get("maxCount", 1))
    if not options:
        return []

    k = min(max_count, len(options))
    return random.sample(list(range(len(options))), k)
