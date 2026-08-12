"""
Shared interface and tactical helper functions for opponent bots used across
the training curriculum (rule-based competency bots + fixed-strategy
"personality" bots).

Every bot implements the same three-method contract so the environment
wrapper (environments/env_wrapper.py) can drive *any* bot identically,
regardless of which OpenSpiel game is currently active:

    bot.reset()                        -> clear episode-local memory
    bot.act(state, player_id) -> int   -> choose a legal action
    bot.on_episode_end(reward)         -> optional hook for bots whose
                                           behaviour depends on past outcomes

Design note on game-agnosticism:
Kuhn/Leduc Poker expose human-readable action strings ("Bet", "Fold", ...)
so aggressive/defensive intent can be read directly off `action_to_string`.
Connect Four's actions are just column indices ("x0".."x6"), so aggressive/
defensive intent instead comes from lightweight game-tree lookahead
(`find_winning_action`, `opponent_has_winning_reply`). Bots use whichever
signal a given game actually supports and fall back gracefully.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyspiel

RULE_BASED = "rule_based"
FIXED_STRATEGY = "fixed_strategy"


class OpponentBot(ABC):
    """Base class every hand-scripted opponent must implement."""

    name: str = "base"
    archetype: str = RULE_BASED

    def reset(self) -> None:
        """Called at the start of every episode. Clear *episode-local* memory
        only - persistent, cross-episode state (e.g. ExploitativeBot's mode,
        PeriodicBot's cycle position) belongs in __init__, not here, since
        those bots are defined by behaviour that carries across episodes."""
        return None

    @abstractmethod
    def act(self, state: "pyspiel.State", player_id: int) -> int:
        """Return a legal action for `player_id` given the current OpenSpiel state."""
        raise NotImplementedError

    def on_episode_end(self, reward: float) -> None:
        """Optional hook fired once per episode with this bot's own final
        return (not the agent's). Default is a no-op; stateful bots like
        ExploitativeBot override it."""
        return None


def fast_choice(rng: np.random.Generator, options: List):
    """Uniform random pick from a small sequence."""
    return options[int(rng.integers(0, len(options)))]


def fast_weighted_choice(rng: np.random.Generator, options: List, weights: List[float]):
    """Weighted random pick via inverse-CDF sampling - same distribution as
    `rng.choice(options, p=weights)`, without its per-call setup cost."""
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for option, w in zip(options, weights):
        cumulative += w
        if r <= cumulative:
            return option
    return options[-1]  

# ---------------------------------------------------------------------------
# Shared tactical helpers - all take a live pyspiel.State and are game-agnostic.
# ---------------------------------------------------------------------------

def find_winning_action(
    state: "pyspiel.State", player_id: int, legal_actions: Optional[List[int]] = None
) -> Optional[int]:
    """Return a legal action that ends the game with a win for `player_id`,
    if one exists this turn. Pass `legal_actions` if the caller already has
    it (every bot does) to skip a redundant state.legal_actions() call -
    cheap individually, but this runs on every decision for several bots.

    Hidden-information caveat: `state.clone()` clones OpenSpiel's *true*
    internal state, private cards included. In Connect Four (perfect
    information) that's completely fair. In Kuhn/Leduc Poker, checking
    `clone.returns()` after a Call/Fold effectively lets a bot "see" the
    showdown result before a real player could - i.e. these bots are a bit
    more clairvoyant than a blind rule-based player would be. That's an
    acceptable simplification for a Phase 1 competency bar, just worth
    knowing when characterizing exactly how strong that bar is."""
    for a in legal_actions if legal_actions is not None else state.legal_actions():
        clone = state.clone()
        clone.apply_action(a)
        if clone.is_terminal() and clone.returns()[player_id] > 0:
            return a
    return None


def opponent_has_winning_reply(
    state: "pyspiel.State", action: int, my_player_id: int
) -> bool:
    """Two-ply lookahead: if I play `action` now, can my opponent immediately
    win on their very next turn? Generic - works on any sequential OpenSpiel
    game via clone()/apply_action(), and is what gives DefensiveBot real
    blocking behaviour in Connect Four. Same hidden-information caveat as
    `find_winning_action` applies for the poker games.

    Reuses `find_winning_action` for the opponent's reply rather than
    re-walking their legal actions with separate logic - same cost, one
    fewer place for the win-check logic to drift out of sync."""
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
    """Classify an action as 'aggressive' / 'defensive' / 'neutral' from its
    human-readable label. Works for Kuhn/Leduc Poker; Connect Four's column
    labels ("x0", "x1", ...) don't carry this signal and will always come
    back 'neutral' by design - Connect Four bots lean on the lookahead
    helpers above instead. Results are memoized per (game, player, action)."""
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
    """Weight legal actions toward the middle of their numeric range. This is
    the 'aggressive' bias for column-style action spaces (Connect Four center
    control); harmless elsewhere since Kuhn/Leduc bots reach for the verbal
    style classifier first and only fall back to this when that's unavailable."""
    legal = list(legal)
    center = sum(legal) / len(legal)
    weights = [1.0 / (1.0 + abs(a - center)) for a in legal]
    return fast_weighted_choice(rng, legal, weights)


def last_opponent_action(state: "pyspiel.State", my_player_id: int) -> Optional[int]:
    """Walk the action history backwards and return the most recent action
    taken by the *other* player (skipping chance events). Powers MirrorBot."""
    for play in reversed(state.full_history()):
        if play.player != my_player_id and play.player != pyspiel.PlayerId.CHANCE:
            return play.action
    return None