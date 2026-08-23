"""Implement the repository's scripted opponent personalities.
Rule-based bots provide simple competency baselines.
Fixed-strategy bots expose recognizable behaviors for opponent profiling.
"""

from typing import Optional

import numpy as np

from .base_bot import (
    OpponentBot,
    RULE_BASED,
    FIXED_STRATEGY,
    find_winning_action,
    opponent_has_winning_reply,
    classify_action_style,
    center_biased_action,
    last_opponent_action,
    fast_choice,
)


class RandomBot(OpponentBot):
    """Choose a uniformly random legal action."""

    name = "random"
    archetype = RULE_BASED

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def act(self, state, player_id):
        return fast_choice(self._rng, state.legal_actions())


class GreedyBot(OpponentBot):
    """Take an immediate win; otherwise choose randomly."""

    name = "greedy"
    archetype = RULE_BASED

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def act(self, state, player_id):
        legal = state.legal_actions()
        winning = find_winning_action(state, player_id, legal)
        if winning is not None:
            return winning
        return fast_choice(self._rng, legal)


class AggressiveBot(OpponentBot):
    """Prefer aggressive actions and center columns after taking free wins."""

    name = "aggressive"
    archetype = FIXED_STRATEGY

    def __init__(self, seed: Optional[int] = None, aggression: float = 0.85):
        self._rng = np.random.default_rng(seed)
        self.aggression = aggression  # Probability of choosing an aggressive action.

    def act(self, state, player_id):
        legal = state.legal_actions()

        winning = find_winning_action(state, player_id, legal)
        if winning is not None:
            return winning

        styled = [a for a in legal if classify_action_style(state, player_id, a) == "aggressive"]
        if styled:
            if self._rng.random() < self.aggression:
                return fast_choice(self._rng, styled)
            return fast_choice(self._rng, legal)

        return center_biased_action(self._rng, legal)


class DefensiveBot(OpponentBot):
    """Take free wins, block immediate threats, and prefer safe actions."""

    name = "defensive"
    archetype = FIXED_STRATEGY

    def __init__(self, seed: Optional[int] = None, caution: float = 0.85):
        self._rng = np.random.default_rng(seed)
        self.caution = caution

    def act(self, state, player_id):
        legal = state.legal_actions()

        winning = find_winning_action(state, player_id, legal)
        if winning is not None:
            return winning

        safe = [a for a in legal if not opponent_has_winning_reply(state, a, player_id)]
        pool = safe if safe else legal

        styled = [a for a in pool if classify_action_style(state, player_id, a) == "defensive"]
        if styled and self._rng.random() < self.caution:
            return fast_choice(self._rng, styled)

        return fast_choice(self._rng, pool)


class MirrorBot(OpponentBot):
    """Replay the opponent's latest action when legal; otherwise choose randomly."""

    name = "mirror"
    archetype = FIXED_STRATEGY

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def act(self, state, player_id):
        legal = state.legal_actions()
        mirrored = last_opponent_action(state, player_id)
        if mirrored is not None and mirrored in legal:
            return mirrored
        return fast_choice(self._rng, legal)


class PeriodicBot(OpponentBot):
    """Cycle through a fixed action-style pattern across episodes."""

    name = "periodic"
    archetype = FIXED_STRATEGY
    _CYCLE = ("aggressive", "defensive", "aggressive", "neutral", "defensive")

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def act(self, state, player_id):
        legal = state.legal_actions()
        style = self._CYCLE[self._step % len(self._CYCLE)]
        self._step += 1

        if style == "neutral":
            return fast_choice(self._rng, legal)

        styled = [a for a in legal if classify_action_style(state, player_id, a) == style]
        if styled:
            return fast_choice(self._rng, styled)

        if style == "aggressive":
            return center_biased_action(self._rng, legal)

        safe = [a for a in legal if not opponent_has_winning_reply(state, a, player_id)]
        return fast_choice(self._rng, safe if safe else legal)


class ExploitativeBot(OpponentBot):
    """Switch between aggressive and defensive modes after losses."""

    name = "exploitative"
    archetype = FIXED_STRATEGY

    def __init__(self, seed: Optional[int] = None):
        self._aggressive = AggressiveBot(seed=seed)
        self._defensive = DefensiveBot(seed=seed)
        self._mode = "aggressive"

    def act(self, state, player_id):
        policy = self._aggressive if self._mode == "aggressive" else self._defensive
        return policy.act(state, player_id)

    def on_episode_end(self, reward: float) -> None:
        if reward < 0:
            self._mode = "defensive" if self._mode == "aggressive" else "aggressive"


RULE_BASED_BOTS = {
    "random": RandomBot,
    "greedy": GreedyBot,
}

FIXED_STRATEGY_BOTS = {
    "aggressive": AggressiveBot,
    "defensive": DefensiveBot,
    "mirror": MirrorBot,
    "periodic": PeriodicBot,
    "exploitative": ExploitativeBot,
}

ALL_BOTS = {**RULE_BASED_BOTS, **FIXED_STRATEGY_BOTS}
