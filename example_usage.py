"""Show the basic environment interaction loop.
The example starts with rule-based opponents, then advances the curriculum
and enables the fixed-strategy tier.
"""

import numpy as np

from environments import AdaptiveOpponentEnv, Curriculum, OpponentPool


def main():
    curriculum = Curriculum()               # Load the default game ladder.
    opponent_pool = OpponentPool(seed=0)

    env = AdaptiveOpponentEnv(
        curriculum=curriculum,
        opponent_pool=opponent_pool,
        opponent_tiers=("rule_based",),      # Use Random and Greedy bots.
        opponent_selection="curriculum",
        seed=0,
    )

    obs, info = env.reset()
    print(f"stage={info['stage']:<12} opponent={info['opponent']:<10} obs.shape={obs.shape}")

    done = False
    episode_return = 0.0
    while not done:
        legal_actions = np.flatnonzero(env.action_masks())
        action = int(np.random.choice(legal_actions))   # Replace with the agent policy.

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        done = terminated or truncated

    print(f"episode finished, agent return={episode_return:+.1f}")

    # Advance the ladder and enable fixed-strategy opponents.
    env.set_curriculum_stage("connect_four")
    env.set_opponent_tiers(("rule_based", "fixed_strategy"))
    obs, info = env.reset()
    print(f"stage={info['stage']:<12} opponent={info['opponent']:<10} obs.shape={obs.shape}")

    done = False
    episode_return = 0.0
    while not done:
        legal_actions = np.flatnonzero(env.action_masks())
        action = int(np.random.choice(legal_actions))

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        done = terminated or truncated

    print(f"episode finished, agent return={episode_return:+.1f}")


if __name__ == "__main__":
    main()
