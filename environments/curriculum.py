"""Define the OpenSpiel game ladder used by training and evaluation.
The curriculum caches loaded games and tracks the active stage.
It also derives shared observation and action dimensions for the models.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pyspiel


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    game_name: str                     # OpenSpiel short name, e.g. "kuhn_poker"
    game_params: dict = field(default_factory=dict)


DEFAULT_STAGES: List[CurriculumStage] = [
    CurriculumStage("tic_tac_toe", "tic_tac_toe"),
    CurriculumStage("kuhn_poker", "kuhn_poker"),
    CurriculumStage("leduc_poker", "leduc_poker"),
    CurriculumStage("connect_four", "connect_four"),
]


class Curriculum:
    """Track stages, cache OpenSpiel games, and derive shared tensor sizes."""

    def __init__(self, stages: Optional[List[CurriculumStage]] = None):
        self.stages: List[CurriculumStage] = list(stages) if stages else list(DEFAULT_STAGES)
        if not self.stages:
            raise ValueError("Curriculum needs at least one stage.")

        self._games: Dict[str, "pyspiel.Game"] = {}
        self._uses_info_state: Dict[str, bool] = {}
        self._stage_idx = 0

        for stage in self.stages:
            game = self.load_game(stage.game_name, stage.game_params)
            if game.num_players() != 2:
                raise ValueError(
                    f"Stage '{stage.name}' uses '{stage.game_name}', which has "
                    f"{game.num_players()} players; this curriculum is 2-player only."
                )
            if game.get_type().dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
                raise ValueError(
                    f"Stage '{stage.name}' uses '{stage.game_name}', which is not a "
                    f"sequential-move game (see the scope note in this module's docstring)."
                )

        self.max_action_dim = max(g.num_distinct_actions() for g in self._games.values())
        self.max_observation_dim = max(self._observation_dim(name) for name in self._games)

    def load_game(self, game_name: str, game_params: Optional[dict] = None) -> "pyspiel.Game":
        if game_name not in self._games:
            game = pyspiel.load_game(game_name, game_params) if game_params else pyspiel.load_game(game_name)
            self._games[game_name] = game
            self._uses_info_state[game_name] = game.get_type().provides_information_state_tensor
        return self._games[game_name]

    def uses_information_state_tensor(self, game_name: str) -> bool:
        """Return whether the game exposes an information-state tensor."""
        return self._uses_info_state[game_name]

    def _observation_dim(self, game_name: str) -> int:
        game = self._games[game_name]
        return (
            game.information_state_tensor_size()
            if self._uses_info_state[game_name]
            else game.observation_tensor_size()
        )

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self._stage_idx]

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    def set_stage(self, stage: Union[int, str]) -> CurriculumStage:
        if isinstance(stage, str):
            names = [s.name for s in self.stages]
            if stage not in names:
                raise KeyError(f"Unknown curriculum stage '{stage}'. Options: {names}")
            self._stage_idx = names.index(stage)
        else:
            if not (0 <= stage < len(self.stages)):
                raise IndexError(f"Stage index {stage} out of range (0..{len(self.stages) - 1}).")
            self._stage_idx = stage
        return self.stage

    def advance(self) -> CurriculumStage:
        """Advance one stage, stopping at the final stage."""
        self._stage_idx = min(self._stage_idx + 1, len(self.stages) - 1)
        return self.stage

    def sample_stage(self, rng) -> CurriculumStage:
        """Sample a stage uniformly at random."""
        idx = int(rng.integers(0, len(self.stages)))
        return self.stages[idx]

    def game_for(self, stage: CurriculumStage) -> "pyspiel.Game":
        return self.load_game(stage.game_name, stage.game_params)

    def __repr__(self) -> str:
        marker = lambda i: ">" if i == self._stage_idx else " "
        lines = [f"{marker(i)} [{i}] {s.name} ({s.game_name})" for i, s in enumerate(self.stages)]
        return "Curriculum:\n" + "\n".join(lines)
