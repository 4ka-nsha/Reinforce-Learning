"""
Hand-scripted opponent bots.

Two tiers, matching the training-strategy docs:

  RULE_BASED      - simple, cheap opponents used to sanity-check the agent's
                     basic competency before anything harder (Phase 1).
  FIXED_STRATEGY  - distinct, recognizable "personalities" the opponent
                     profiler is meant to learn to tell apart (Phase 2). Each
                     one deliberately embodies one of the behavioural patterns
                     called out in the project README's Motivation section.

    RandomBot        -> rule_based        uniform random baseline
    GreedyBot        -> rule_based        take a free win, else random
    AggressiveBot    -> fixed_strategy    "over-aggressive opening sequences"
    DefensiveBot     -> fixed_strategy    blocks threats, plays it safe
    MirrorBot        -> fixed_strategy    "reactive tit-for-tat patterns"
    PeriodicBot      -> fixed_strategy    "periodic structural trap placements"
    ExploitativeBot  -> fixed_strategy    "exploitative shifts after a loss"

Every act() fetches state.legal_actions() exactly once and threads it into
the helper calls that need it, and every random pick goes through
fast_choice()/fast_weighted_choice() rather than rng.choice() - see
base_bot.py for why (both are measured, not stylistic, wins).
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
    """Uniform-random legal action. The floor every other bot should beat."""

    name = "random"
    archetype = RULE_BASED

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def act(self, state, player_id):
        return fast_choice(self._rng, state.legal_actions())


class GreedyBot(OpponentBot):
    """Takes an immediate win when one is on the table, otherwise plays
    randomly. Deliberately simple 'hardcoded logic' - a slightly-better-
    than-random bar for the Phase 1 competency check."""

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
    """Over-aggressive opener: grabs free wins, otherwise leans hard into
    bet/raise/call in poker or center-column play in Connect Four."""

    name = "aggressive"
    archetype = FIXED_STRATEGY

    def __init__(self, seed: Optional[int] = None, aggression: float = 0.85):
        self._rng = np.random.default_rng(seed)
        self.aggression = aggression  # P(taking the most aggressive legal action)

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
    """Purely defensive: takes a free win if handed one, otherwise blocks the
    opponent's immediate threats (2-ply lookahead) or leans on fold/pass/
    check in poker."""

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
    """Tit-for-tat: replays the opponent's most recent action whenever it's
    still legal, otherwise plays randomly."""

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
    """Cycles through a fixed style pattern independent of what's actually
    happening in the game - a deterministic 'structural trap' the profiler
    should learn to recognize as a period-N pattern rather than a reactive
    one. The cycle position is persistent across episodes (NOT cleared by
    reset()), so the periodicity is visible across a whole run of matches
    against this bot rather than resetting every hand."""

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
    """Starts aggressive; flips its whole sub-strategy to defensive the
    moment it loses an episode, and flips back the moment defensive loses
    too - i.e. it exploits whatever isn't currently losing. Mode is
    persistent across episodes by design (NOT cleared by reset())."""

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