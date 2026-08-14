"""
Game-ladder curriculum: Kuhn Poker -> Leduc Poker -> Connect Four.

Resolves the "Game Progression" half of Part 1 onto OpenSpiel only. The
README/ideation doc also name PettingZoo, Gym, and Kaggle Environments as
candidates, and describe an early ladder of Tic-Tac-Toe -> Connect Four ->
Rock-Paper-Scissors -> "more complex games". Both of those have since been
superseded: the project standardized on OpenSpiel as the single framework
for the whole curriculum (it already covers every game type here, plus the
PSRO/self-play tooling later phases need, so a second framework would only
add API friction), and the ladder itself was narrowed to Kuhn/Leduc Poker
for fast iteration plus Connect Four as the curriculum-complexity stage.

This module owns *only* the game ladder. Which opponents are available at a
given point in training is a separate, independent concern - see
opponent_pool.py / models/baselines/bots.py - so the two can be advanced on
different schedules by whichever training loop drives this environment.

Scope note: all three stages are sequential (turn-based) games, so
env_wrapper.py only implements sequential-turn handling. The "most complex
target game" slot mentioned in the README is still an open decision; if that
eventually lands on a simultaneous-move game (e.g. iterated Rock-Paper-
Scissors), this module and env_wrapper.py's turn-resolution loop would both
need a simultaneous-move code path added - deliberately not built now, to
avoid adding real complexity for a game that hasn't been chosen yet.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pyspiel


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    game_name: str                     # pyspiel short name, e.g. "kuhn_poker"
    game_params: dict = field(default_factory=dict)


DEFAULT_STAGES: List[CurriculumStage] = [
    CurriculumStage("tic_tac_toe", "tic_tac_toe"),
    CurriculumStage("kuhn_poker", "kuhn_poker"),
    CurriculumStage("leduc_poker", "leduc_poker"),
    CurriculumStage("connect_four", "connect_four"),
]


class Curriculum:
    """Tracks the current game-ladder stage and hands out cached, loaded
    OpenSpiel Game objects. Also derives the fixed observation/action
    dimensions the rest of the pipeline standardizes on (see env_wrapper.py),
    and precomputes, per game, whether to read observations off the
    information-state tensor or the observation tensor.
    """

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
        """True if this game's observations should come from
        information_state_tensor (bet-history aware - matters for the poker
        games) rather than observation_tensor (Connect Four's only option,
        since it doesn't implement an information-state tensor at all)."""
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
        """Move to the next stage, if any. Stays on the final (hardest) stage
        once reached rather than wrapping. Advancing is a training-loop
        decision (e.g. a win-rate threshold, or the profiler's t-SNE
        cluster-separation exit criterion) - this class only tracks *which*
        stage is active, not *when* to move on."""
        self._stage_idx = min(self._stage_idx + 1, len(self.stages) - 1)
        return self.stage

    def sample_stage(self, rng) -> CurriculumStage:
        """Pick a stage uniformly at random, independent of `stage_index`.
        Backs the environment's 'random' opponent-selection mode."""
        idx = int(rng.integers(0, len(self.stages)))
        return self.stages[idx]

    def game_for(self, stage: CurriculumStage) -> "pyspiel.Game":
        return self.load_game(stage.game_name, stage.game_params)

    def __repr__(self) -> str:
        marker = lambda i: ">" if i == self._stage_idx else " "
        lines = [f"{marker(i)} [{i}] {s.name} ({s.game_name})" for i, s in enumerate(self.stages)]
        return "Curriculum:\n" + "\n".join(lines)