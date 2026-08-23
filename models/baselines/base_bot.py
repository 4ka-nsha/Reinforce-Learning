"""Define the common interface for scripted opponents.
This module also contains tactical helpers shared across OpenSpiel games,
including immediate-win checks, threat detection, and action-style labels.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyspiel

RULE_BASED = "rule_based"
FIXED_STRATEGY = "fixed_strategy"


class OpponentBot(ABC):
    """Interface for scripted opponents."""

    name: str = "base"
    archetype: str = RULE_BASED

    def reset(self) -> None:
        """Clear episode-local state."""
        return None

    @abstractmethod
    def act(self, state: "pyspiel.State", player_id: int) -> int:
        """Choose a legal action for the player."""
        raise NotImplementedError

    def on_episode_end(self, reward: float) -> None:
        """Handle the bot's final episode reward."""
        return None


def fast_choice(rng: np.random.Generator, options: List):
    """Choose uniformly from a small sequence."""
    return options[int(rng.integers(0, len(options)))]


def fast_weighted_choice(rng: np.random.Generator, options: List, weights: List[float]):
    """Choose from options using inverse-CDF sampling."""
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for option, w in zip(options, weights):
        cumulative += w
        if r <= cumulative:
            return option
    return options[-1]  

# Shared tactical helpers.

def find_winning_action(
    state: "pyspiel.State", player_id: int, legal_actions: Optional[List[int]] = None
) -> Optional[int]:
    """Return an immediate winning action, if one exists."""
    for a in legal_actions if legal_actions is not None else state.legal_actions():
        clone = state.clone()
        clone.apply_action(a)
        if clone.is_terminal() and clone.returns()[player_id] > 0:
            return a
    return None


def opponent_has_winning_reply(
    state: "pyspiel.State", action: int, my_player_id: int
) -> bool:
    """Return whether the opponent has an immediate winning reply."""
    clone = state.clone()
    clone.apply_action(action)
    if clone.is_terminal() or clone.is_chance_node():
        return False
    opp_id = clone.current_player()
    if opp_id == my_player_id:
        return False
    return find_winning_action(clone, opp_id) is not None


_AGGRESSIVE_WORDS = ("bet", "raise", "call")
_DEFENSIVE_WORDS = ("fold", "pass", "check")

_STYLE_CACHE: Dict[Tuple[int, int, int], str] = {}


def classify_action_style(state: "pyspiel.State", player_id: int, action: int) -> str:
    """Classify an action from its human-readable label."""
    key = (id(state.get_game()), player_id, action)
    cached = _STYLE_CACHE.get(key)
    if cached is not None:
        return cached

    label = state.action_to_string(player_id, action).lower()
    if any(w in label for w in _AGGRESSIVE_WORDS):
        style = "aggressive"
    elif any(w in label for w in _DEFENSIVE_WORDS):
        style = "defensive"
    else:
        style = "neutral"

    _STYLE_CACHE[key] = style
    return style


def center_biased_action(rng: np.random.Generator, legal) -> int:
    """Prefer legal actions near the numeric center."""
    legal = list(legal)
    center = sum(legal) / len(legal)
    weights = [1.0 / (1.0 + abs(a - center)) for a in legal]
    return fast_weighted_choice(rng, legal, weights)


def last_opponent_action(state: "pyspiel.State", my_player_id: int) -> Optional[int]:
    """Return the other player's most recent non-chance action."""
    for play in reversed(state.full_history()):
        if play.player != my_player_id and play.player != pyspiel.PlayerId.CHANCE:
            return play.action
    return None
