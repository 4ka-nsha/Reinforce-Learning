"""
Minimal usage example for Role 2 (profiler) / Role 3 (policy/PPO): this is
the entire surface area you need to know about this module.

Run from the repo root:  python example_usage.py
"""

import numpy as np

from environments import AdaptiveOpponentEnv, Curriculum, OpponentPool


def main():
    curriculum = Curriculum()               # Kuhn Poker -> Leduc Poker -> Connect Four
    opponent_pool = OpponentPool(seed=0)

    env = AdaptiveOpponentEnv(
        curriculum=curriculum,
        opponent_pool=opponent_pool,
        opponent_tiers=("rule_based",),      # Phase 1: competency check vs Random/Greedy
        opponent_selection="curriculum",
        seed=0,
    )

    obs, info = env.reset()
    print(f"stage={info['stage']:<12} opponent={info['opponent']:<10} obs.shape={obs.shape}")

    done = False
    episode_return = 0.0
    while not done:
        legal_actions = np.flatnonzero(env.action_masks())
        action = int(np.random.choice(legal_actions))   # <-- Role 2/3's policy goes here

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        done = terminated or truncated

    print(f"episode finished, agent return={episode_return:+.1f}")

    # Advance the game ladder and unlock the personality tier - e.g. once
    # Phase 1's baseline PPO clears its competency bar against rule_based bots.
    env.set_curriculum_stage("connect_four")
    env.set_opponent_tiers(("rule_based", "fixed_strategy"))
    obs, info = env.reset()
    print(f"stage={info['stage']:<12} opponent={info['opponent']:<10} obs.shape={obs.shape}")


if __name__ == "__main__":
    main()