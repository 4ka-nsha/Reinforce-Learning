"""Exercise every registered bot across every curriculum stage.
The tests cover episode completion, fixed spaces, seating, tier filtering,
and the environment's handling of illegal actions.
"""

import warnings

import numpy as np
import pytest

from environments import AdaptiveOpponentEnv, Curriculum
from environments.curriculum import DEFAULT_STAGES
from environments.opponent_pool import OpponentPool
from models.baselines import ALL_BOTS, RULE_BASED_BOTS

N_EPISODES = 10
MAX_STEPS = 200
STAGE_NAMES = [s.name for s in DEFAULT_STAGES]


@pytest.fixture(scope="module")
def curriculum():
    return Curriculum()


@pytest.fixture(scope="module")
def full_pool():
    return OpponentPool(seed=0)


def _play_episode(env, rng):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

    done = False
    steps = 0
    ep_return = 0.0
    while not done:
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)
        legal = np.flatnonzero(mask)
        assert len(legal) > 0, "no legal actions offered on a non-terminal step"

        action = int(rng.choice(legal))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat mask violations as failures.
            obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        ep_return += reward
        steps += 1
        done = terminated or truncated
        assert steps < MAX_STEPS, "episode ran suspiciously long - possible infinite loop"

    return ep_return, steps


@pytest.mark.parametrize("stage_name", STAGE_NAMES)
@pytest.mark.parametrize("bot_name", list(ALL_BOTS))
def test_bot_vs_stage_runs_cleanly(curriculum, stage_name, bot_name):
    # Use an isolated pool containing only the selected bot.
    solo_pool = OpponentPool(registry={bot_name: ALL_BOTS[bot_name]}, seed=0)
    env = AdaptiveOpponentEnv(curriculum=curriculum, opponent_pool=solo_pool, opponent_tiers=None, seed=0)
    env.set_curriculum_stage(stage_name)

    rng = np.random.default_rng(0)
    for _ in range(N_EPISODES):
        _play_episode(env, rng)


def test_observation_and_action_space_constant_across_stages(curriculum, full_pool):
    env = AdaptiveOpponentEnv(curriculum=curriculum, opponent_pool=full_pool, seed=0)
    for stage_name in STAGE_NAMES:
        env.set_curriculum_stage(stage_name)
        obs, _ = env.reset()
        assert obs.shape == (curriculum.max_observation_dim,)
        assert env.action_space.n == curriculum.max_action_dim


def test_agent_seat_is_randomized(curriculum, full_pool):
    env = AdaptiveOpponentEnv(curriculum=curriculum, opponent_pool=full_pool, seed=0)
    seats_seen = {info["agent_player_id"] for info in (env.reset()[1] for _ in range(50))}
    assert seats_seen == {0, 1}, "expected both seats to come up over 50 resets"


def test_opponent_tier_filtering_is_respected(curriculum, full_pool):
    env = AdaptiveOpponentEnv(
        curriculum=curriculum, opponent_pool=full_pool, opponent_tiers=("rule_based",), seed=0
    )
    opponents_seen = {env.reset()[1]["opponent"] for _ in range(50)}
    assert opponents_seen, "no opponents were sampled at all"
    assert opponents_seen <= set(RULE_BASED_BOTS), (
        f"opponent_tiers=('rule_based',) leaked non-rule-based opponents: "
        f"{opponents_seen - set(RULE_BASED_BOTS)}"
    )


def test_illegal_action_degrades_instead_of_crashing(curriculum, full_pool):
    env = AdaptiveOpponentEnv(curriculum=curriculum, opponent_pool=full_pool, seed=0)
    env.set_curriculum_stage("kuhn_poker")  # Only actions 0 and 1 are valid.
    env.reset()
    with pytest.warns(UserWarning, match="illegal"):
        env.step(6)  # Deliberately invalid action.
