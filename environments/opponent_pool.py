"""Manage the scripted opponents available to the environment.
Bots are stored as persistent instances so selected strategies can retain
cross-episode state while still resetting episode-local state.
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
        """Return bot names matching any requested tier or name."""
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
