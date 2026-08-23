"""Expose curriculum games through a single-agent Gymnasium interface.
The wrapper resolves chance and scripted-opponent turns internally.
Observations and action masks are padded to shared curriculum dimensions.
"""

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pyspiel
from gymnasium import spaces

from environments.curriculum import Curriculum, CurriculumStage
from environments.opponent_pool import OpponentPool
from models.baselines.base_bot import fast_choice, fast_weighted_choice


class AdaptiveOpponentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        curriculum: Optional[Curriculum] = None,
        opponent_pool: Optional[OpponentPool] = None,
        opponent_tiers: Optional[Sequence[str]] = ("rule_based",),
        opponent_selection: str = "curriculum",
        seed: Optional[int] = None,
    ):
        super().__init__()
        if opponent_selection not in ("curriculum", "random"):
            raise ValueError("opponent_selection must be 'curriculum' or 'random'.")

        self.curriculum = curriculum or Curriculum()
        self.opponent_pool = opponent_pool or OpponentPool(seed=seed)
        self.opponent_tiers = tuple(opponent_tiers) if opponent_tiers else None
        self.opponent_selection = opponent_selection

        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.curriculum.max_action_dim)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.curriculum.max_observation_dim,), dtype=np.float32
        )

        self._game: Optional["pyspiel.Game"] = None
        self._state: Optional["pyspiel.State"] = None
        self._stage: Optional[CurriculumStage] = None
        self._agent_player_id: Optional[int] = None
        self._opponent_player_id: Optional[int] = None
        self._opponent_bot = None
        self._cached_mask: Optional[np.ndarray] = None

    def set_curriculum_stage(self, stage) -> None:
        self.curriculum.set_stage(stage)

    def set_opponent_tiers(self, tiers: Optional[Sequence[str]]) -> None:
        self.opponent_tiers = tuple(tiers) if tiers else None

    # Gymnasium API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._stage = (
            self.curriculum.sample_stage(self._rng)
            if self.opponent_selection == "random"
            else self.curriculum.stage
        )
        self._game = self.curriculum.game_for(self._stage)
        self._state = self._game.new_initial_state()

        self._agent_player_id = int(self._rng.integers(0, self._game.num_players()))
        self._opponent_player_id = 1 - self._agent_player_id

        self._opponent_bot = self.opponent_pool.sample(self.opponent_tiers, self._rng)
        self._opponent_bot.reset()

        self._last_opponent_action = None
        self._play_until_agent_turn()
        self._refresh_mask_cache()

        return self._observation(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.is_terminal():
            raise RuntimeError("Episode has already ended; call reset() to start a new one.")

        legal = self._state.legal_actions()
        if action not in legal:
            warnings.warn(
                f"Action {action} illegal in this state (legal: {legal}); substituting a random "
                f"legal action. Check that your policy is applying action_masks().",
                stacklevel=2,
            )
            action = fast_choice(self._rng, legal)

        self._state.apply_action(action)
        self._play_until_agent_turn()
        self._refresh_mask_cache()

        terminated = self._state.is_terminal()
        reward = float(self._state.returns()[self._agent_player_id]) if terminated else 0.0

        if terminated:
            opponent_reward = float(self._state.returns()[self._opponent_player_id])
            self._opponent_bot.on_episode_end(opponent_reward)

        return self._observation(), reward, terminated, False, self._info()

    def action_masks(self) -> np.ndarray:
        """Return a copy of the cached legal-action mask."""
        return self._cached_mask.copy()

    # Internals
    def _play_until_agent_turn(self) -> None:
        """Resolve chance and opponent turns until the agent can act."""
        while not self._state.is_terminal() and self._state.current_player() != self._agent_player_id:
            if self._state.is_chance_node():
                outcomes = self._state.chance_outcomes()
                actions, probs = zip(*outcomes)
                self._state.apply_action(int(fast_weighted_choice(self._rng, actions, probs)))
            else:
                opp_action = self._opponent_bot.act(self._state, self._state.current_player())
                self._last_opponent_action = opp_action
                self._state.apply_action(opp_action)

    def _raw_tensor(self):
        """Return the current information or observation tensor."""
        if self.curriculum.uses_information_state_tensor(self._stage.game_name):
            return self._state.information_state_tensor(self._agent_player_id)
        return self._state.observation_tensor(self._agent_player_id)

    def _observation(self) -> np.ndarray:
        raw = self._raw_tensor()
        obs = np.zeros(self.curriculum.max_observation_dim, dtype=np.float32)
        obs[: len(raw)] = raw
        return obs

    def _compute_mask(self) -> np.ndarray:
        mask = np.zeros(self.curriculum.max_action_dim, dtype=bool)
        if not self._state.is_terminal() and self._state.current_player() == self._agent_player_id:
            base = self._state.legal_actions_mask(self._agent_player_id)
            mask[: len(base)] = base
        return mask

    def _refresh_mask_cache(self) -> None:
        """Refresh the cached legal-action mask."""
        self._cached_mask = self._compute_mask()

    def _info(self) -> Dict[str, Any]:
        return {
            "stage": self._stage.name,
            "game_name": self._stage.game_name,
            "opponent": self._opponent_bot.name if self._opponent_bot else None,
            "agent_player_id": self._agent_player_id,
            "action_mask": self._cached_mask.copy(),
            "last_opponent_action": self._last_opponent_action,
        }
