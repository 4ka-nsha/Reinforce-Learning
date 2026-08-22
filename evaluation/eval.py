import os
import torch
import numpy as np
import argparse

from environments import AdaptiveOpponentEnv, Curriculum, OpponentPool
from models.profiler.tracker import TrajectoryTracker
from models.profiler.models import OpponentProfiler
from models.profiler.analysis import LatentVisualizer
from ppo_agent.model import PPOActorCritic
from ppo_agent.helpers import load_joint_checkpoint

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_per_bot", type=int, default=50, help="Number of games to evaluate per bot archetype")
    parser.add_argument("--stage", type=str, default="kuhn_poker", help="Active curriculum stage")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()

    curriculum = Curriculum()
    curriculum.set_stage(args.stage)
    opponent_pool = OpponentPool(seed=100)

    # Initialize policy and profiler.
    input_size = curriculum.max_observation_dim + 32
    action_dim = curriculum.max_action_dim
    policy = PPOActorCritic(input_size=input_size, action_dim=action_dim, hidden_dim=128)
    profiler = OpponentProfiler(input_dim=action_dim, z_dim=32, hidden_dim=64)

    checkpoint_path = args.checkpoint or f"models/checkpoints/joint_{args.stage}_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        load_joint_checkpoint(policy, profiler, checkpoint_path)
    else:
        print(f"No checkpoint found at '{checkpoint_path}'. Evaluating using default initialized weights.")

    policy.eval()
    profiler.eval()

    tracker = TrajectoryTracker(window_size=10, feature_dim=action_dim)
    visualizer = LatentVisualizer()

    # Evaluate each registered bot archetype.
    bot_archetypes = ["random", "greedy", "aggressive", "defensive", "mirror", "periodic", "exploitative"]
    evaluation_results = {}

    print(f"\n--- Starting Evaluation on stage: {args.stage} ---")

    for bot_name in bot_archetypes:
        if bot_name not in opponent_pool.names:
            print(f"Skipping bot archetype '{bot_name}' (not registered).")
            continue
            
        env = AdaptiveOpponentEnv(
            curriculum=curriculum,
            opponent_pool=opponent_pool,
            opponent_tiers=None, # Disable tier filtering.
            opponent_selection="curriculum",
            seed=100
        )
        
        # Force this environment to select one bot.
        env.opponent_tiers = (bot_name,)
        
        returns = []
        wins = 0
        draws = 0
        losses = 0

        for episode in range(args.games_per_bot):
            obs, info = env.reset()
            tracker.reset()
            episode_return = 0.0
            done = False
            
            while not done:
                opp_action = info["last_opponent_action"]
                if opp_action is not None:
                    action_one_hot = np.zeros(action_dim, dtype=np.float32)
                    action_one_hot[opp_action] = 1.0
                else:
                    action_one_hot = np.zeros(action_dim, dtype=np.float32)
                    
                tracker.update(action_one_hot)
                history_tensor = tracker.get_history_tensor()

                # Generate the opponent profile.
                with torch.no_grad():
                    z_opp = profiler(history_tensor).squeeze(0)
                    
                # Record the profile for clustering.
                visualizer.record(z_opp, opponent_label=bot_name)

                obs_t = torch.tensor(obs, dtype=torch.float32)
                action_mask = env.action_masks()
                mask_t = torch.tensor(action_mask, dtype=torch.bool)

                # Evaluate the policy.
                with torch.no_grad():
                    _, action_probs = policy(obs_t, z_opp, mask_t)

                action_probs_np = action_probs.squeeze(0).numpy()
                legal_actions = np.flatnonzero(action_mask)
                
                # Renormalize over legal actions.
                probs = action_probs_np[legal_actions]
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(len(legal_actions)) / len(legal_actions)
                    
                action = int(np.random.choice(legal_actions, p=probs))

                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                
                obs = next_obs
                info = next_info

            returns.append(episode_return)
            if episode_return > 0:
                wins += 1
            elif episode_return < 0:
                losses += 1
            else:
                draws += 1

        mean_return = np.mean(returns)
        win_rate = (wins / args.games_per_bot) * 100
        evaluation_results[bot_name] = {
            "mean_return": mean_return,
            "win_rate": win_rate,
            "wins": wins,
            "draws": draws,
            "losses": losses
        }
        print(f"Bot: {bot_name:<15} Win Rate: {win_rate:>5.1f}% | Wins: {wins:<3} Draws: {draws:<3} Losses: {losses:<3} | Mean Return: {mean_return:>+4.2f}")

    # Plot profile clusters.
    os.makedirs("results", exist_ok=True)
    plot_path = f"results/{args.stage}_opponent_strategy_clusters.png"
    print(f"\nGenerating clustering diagrams...")
    # Save the t-SNE plot.
    visualizer.plot_clusters(method="tsne", save_path=plot_path)
    print(f"Clusters saved successfully to {plot_path}")

    print("\n--- Zero-Shot Adaptation Benchmarking Complete ---")


if __name__ == "__main__":
    run_evaluation()
