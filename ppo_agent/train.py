import torch
import torch.optim as optim
import numpy as np
import random
import os
import argparse

from environments import AdaptiveOpponentEnv, Curriculum, OpponentPool
from models.baselines.self_play_bot import SelfPlayBot
from models.profiler.tracker import TrajectoryTracker
from models.profiler.models import OpponentProfiler
from ppo_agent.model import PPOActorCritic
from ppo_agent.buffer import Memory
from ppo_agent.model_snapshot import ModelSnapshot
from ppo_agent.ppo import compute_advantages, ppo_update
from ppo_agent.helpers import save_joint_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30, help="Total episodes to train")
    parser.add_argument("--rollout_size", type=int, default=128, help="PPO rollout steps per update batch")
    parser.add_argument("--stage", type=str, default="kuhn_poker", choices=["kuhn_poker", "leduc_poker", "connect_four"], help="Active curriculum stage")
    args = parser.parse_args()

    # 1. Instantiate the curriculum and opponent pool
    curriculum = Curriculum()
    curriculum.set_stage(args.stage)
    opponent_pool = OpponentPool(seed=42)

    # 2. Instantiate Gymnasium wrapper
    env = AdaptiveOpponentEnv(
        curriculum=curriculum,
        opponent_pool=opponent_pool,
        opponent_tiers=("rule_based", "fixed_strategy"),
        opponent_selection="curriculum",
        seed=42
    )

    # 3. Initialize models and tracking utilities
    # Input size: 126 state dims + 32 latent dims = 158
    # Action dimension: 7
    policy = PPOActorCritic(input_size=158, action_dim=7, hidden_dim=128)
    profiler = OpponentProfiler(input_dim=7, z_dim=32, hidden_dim=64)
    
    # Combined optimizer to train both networks together
    optimizer = optim.Adam(list(policy.parameters()) + list(profiler.parameters()), lr=0.001)
    
    bfr = Memory()
    tracker = TrajectoryTracker(window_size=10, feature_dim=7)
    snap = ModelSnapshot(max_pool_size=10)

    print(f"Starting joint adaptive training on stage: {args.stage}")
    print(f"Policy params: {sum(p.numel() for p in policy.parameters())}")
    print(f"Profiler params: {sum(p.numel() for p in profiler.parameters())}")

    step_count = 0
    checkpoint_path = f"models/checkpoints/joint_{args.stage}_model.pt"

    for ep in range(args.episodes):
        # Phase 3 self-play matchmaking check
        opponent_type = snap.define_opponent(current_ppo_frac=ep / args.episodes)
        if opponent_type == "self_play":
            print(f"[Episode {ep+1}] Self-Play Matchmaking triggered.")
            # Retrieve past policy checkpoint
            past_policy = snap.get_self_play_opponent((158, 7))
            # Wrap as a bot
            sp_bot = SelfPlayBot(
                policy_model=past_policy,
                profiler_model=profiler,
                curriculum=curriculum,
                stage=env._stage,
                seed=ep
            )
            # Inject into opponent pool
            opponent_pool._instances["self_play_bot"] = sp_bot
            env.set_opponent_tiers(("self_play_bot",))
        else:
            # Play against rule-based/fixed-strategy bot pool
            env.set_opponent_tiers(("rule_based", "fixed_strategy"))

        obs, info = env.reset()
        tracker.reset()
        episode_return = 0.0
        done = False
        
        while not done:
            # Check opponent's action at previous step (if any) to update profile tracker
            opp_action = info["last_opponent_action"]
            if opp_action is not None:
                action_one_hot = np.zeros(7, dtype=np.float32)
                action_one_hot[opp_action] = 1.0
            else:
                action_one_hot = np.zeros(7, dtype=np.float32)
                
            tracker.update(action_one_hot)
            history_tensor = tracker.get_history_tensor()

            # Generate latent profile vector (no gradient backprop during rollout step collection)
            with torch.no_grad():
                z_opp = profiler(history_tensor).squeeze(0) # shape: (32,)

            obs_t = torch.tensor(obs, dtype=torch.float32)
            action_mask = env.action_masks()
            mask_t = torch.tensor(action_mask, dtype=torch.bool)

            # Policy forward pass
            with torch.no_grad():
                value, action_probs = policy(obs_t, z_opp, mask_t)

            # Sample action index
            probs_np = action_probs.squeeze(0).numpy()
            dist = torch.distributions.Categorical(action_probs.squeeze(0))
            action = dist.sample().item()
            action_prob = probs_np[action]

            # Step the Gym environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_count += 1

            # Store actual opponent action response (target for auxiliary loss classification)
            next_opp_action = next_info["last_opponent_action"]
            opp_target = next_opp_action if next_opp_action is not None else -1

            bfr.store(
                states=obs_t,
                z_opp=z_opp,
                actions_taken=action,
                actions_prob=action_prob,
                action_mask=mask_t,
                value=value.item(),
                reward=reward,
                game_ended=done,
                history_tensor=history_tensor,
                opponent_action=opp_target
            )

            obs = next_obs
            info = next_info

            # Execute Joint PPO and Profiler parameter updates
            if len(bfr.states) >= args.rollout_size:
                advantages, returns = compute_advantages(bfr.rewards, bfr.value, bfr.game_ended)
                ppo_update(
                    policy=policy,
                    profiler=profiler,
                    optimizer=optimizer,
                    memory=bfr,
                    advantages=advantages,
                    returns=returns,
                    epochs=4,
                    lambda_aux=0.5
                )
                bfr.clear()
                
                # Checkpoint snapshot for future self-play iterations
                snap.save_snap(policy)

        print(f"[Episode {ep+1}/{args.episodes}] opponent={info['opponent']:<12} return={episode_return:+.1f}")

    # Save joint checkpoint on training completion
    save_joint_checkpoint(policy, profiler, checkpoint_path)


if __name__ == "__main__":
    main()
