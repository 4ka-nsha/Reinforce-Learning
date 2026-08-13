"""
Opponent sampling. Wraps the hand-scripted bots in models/baselines/ into a
pool the environment can draw from, filtered by tier ("rule_based" /
"fixed_strategy") so callers can independently control curriculum-game
difficulty (curriculum.py) and opponent-personality difficulty - e.g. Phase 1
trains against rule_based only, Phase 2 opens up fixed_strategy, and both
can move forward on their own schedule.

Each archetype gets exactly one persistent instance for the lifetime of the
pool, not a fresh one per episode: bots like ExploitativeBot and PeriodicBot
are designed to carry state *across* episodes (that's the whole point of
"exploitative shifts after a loss"), while `OpponentBot.reset()` is called
once per episode to clear only episode-local memory (e.g. MirrorBot's
last-seen action).
"""

from typing import Dict, Iterable, List, Optional

import numpy as np

from models.baselines import ALL_BOTS, OpponentBot

class OpponentPool:
    def __init__(self, registry: Optional[Dict[str, type]] = None, seed: Optional[int] = None):
        registry = registry if registry is not None else ALL_BOTS
        self._instances: Dict[str, OpponentBot] = {
            name: bot_cls(seed=seed) for name, bot_cls in registry.items()
        }

    @property
    def names(self) -> List[str]:
        return list(self._instances.keys())

    def tiers(self, tiers: Optional[Iterable[str]] = None) -> List[str]:
        """Names of bots belonging to any of the given tiers (None/empty = all)."""
        if not tiers:
            return self.names
        tier_set = set(tiers)
        return [name for name, bot in self._instances.items() if bot.archetype in tier_set or name in tier_set]

    def get(self, name: str) -> OpponentBot:
        return self._instances[name]

    def sample(self, tiers: Optional[Iterable[str]], rng: np.random.Generator) -> OpponentBot:
        candidates = self.tiers(tiers)
        if not candidates:
            raise ValueError(f"No registered opponents match tiers={list(tiers) if tiers else tiers}. "
                              f"Available: {self.names}")
        name = candidates[int(rng.integers(0, len(candidates)))]
        return self._instances[name]