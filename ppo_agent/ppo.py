"""Implement advantage estimation and the joint PPO optimization step.
The update trains the policy, value head, and opponent profiler together.
"""

import torch

def compute_advantages(rewards, values, game_ended, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages and critic return targets."""
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae_lam = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - float(game_ended[t])
            next_val = next_value
        else:
            next_non_terminal = 1.0 - float(game_ended[t])
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
    values_tensor = torch.tensor(values, dtype=torch.float32)
    returns = advantages + values_tensor
    return advantages, returns


def ppo_update(
    policy: torch.nn.Module, 
    profiler: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    memory, 
    advantages: torch.Tensor, 
    returns: torch.Tensor, 
    clip_eps: float = 0.2, 
    epochs: int = 4,
    lambda_aux: float = 0.5
) -> None:
    """Run a joint PPO and profiler optimization step."""
    states = torch.stack(memory.states)
    actions_taken = torch.tensor(memory.actions_taken, dtype=torch.int32)
    actions_prob = torch.tensor(memory.actions_prob, dtype=torch.float32)
    action_masks = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in memory.action_masks])
    
    # Combine histories into the rollout batch.
    history_tensors = torch.cat(memory.history_tensors, dim=0)
    opponent_actions_taken = torch.tensor(memory.opponent_actions_taken, dtype=torch.long)
    
    # Normalize advantages before the policy update.
    advantages_ = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for run in range(epochs):
        optimizer.zero_grad()
        
        # Recompute opponent profiles with gradients enabled.
        new_z_opp = profiler(history_tensors)
        
        # Evaluate the policy and value function.
        value_, moves_ = policy(states, new_z_opp, action_masks)    
        
        # PPO clipped-surrogate loss.
        new_action_probs = moves_.gather(1, actions_taken.long().unsqueeze(1)).squeeze(1)
        ratio = new_action_probs / (actions_prob + 1e-10)
        option1 = ratio * advantages_
        option2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_
        policy_loss = -torch.min(option1, option2).mean()
        
        # Critic regression loss.
        value_loss = ((value_ - returns)**2).mean()
        
        # Entropy bonus for exploration.
        entropy = -(moves_ * torch.log(moves_ + 1e-10)).sum(dim=-1).mean()
        
        # Auxiliary next-action prediction loss.
        valid_mask = (opponent_actions_taken != -1)
        if valid_mask.any():
            aux_loss = profiler.get_auxiliary_loss(
                new_z_opp[valid_mask], 
                opponent_actions_taken[valid_mask], 
                is_discrete=True
            )
        else:
            aux_loss = torch.tensor(0.0, device=states.device)
            
        # Combine policy, value, entropy, and profiler objectives.
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + lambda_aux * aux_loss
        
        total_loss.backward()
        optimizer.step()
