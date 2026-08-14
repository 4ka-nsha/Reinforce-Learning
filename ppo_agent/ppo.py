import torch

def compute_advantages(rewards, values, game_ended, next_value, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and returns targets.
    
    Args:
        rewards (list): list of episode rewards, length T
        values (list): state value estimates, length T
        game_ended (list): terminal flags, length T
        next_value (float): value function estimate of the state following the final step
        gamma (float): temporal discount factor
        gae_lambda (float): GAE factor (lambda)
    """
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
            
        # Delta TD residual
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        
        # GAE accumulation
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
    """
    Performs joint PPO optimization. Gradients flow back from both the policy 
    surrogate loss and the profiler auxiliary next-action prediction loss 
    into the GRU encoder.
    """
    states = torch.stack(memory.states)
    actions_taken = torch.tensor(memory.actions_taken, dtype=torch.int32)
    actions_prob = torch.tensor(memory.actions_prob, dtype=torch.float32)
    action_masks = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in memory.action_masks])
    
    # Concatenate sequence histories: shape (T, seq_len, feature_dim)
    history_tensors = torch.cat(memory.history_tensors, dim=0)
    opponent_actions_taken = torch.tensor(memory.opponent_actions_taken, dtype=torch.long)
    
    # Normalize advantages using standard scalar reduce
    advantages_ = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for run in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass through profiler with gradient tracking
        new_z_opp = profiler(history_tensors)
        
        # 2. Forward pass through policy
        value_, moves_ = policy(states, new_z_opp, action_masks)    
        
        # 3. PPO Surrogate Clipped Loss
        new_action_probs = moves_.gather(1, actions_taken.long().unsqueeze(1)).squeeze(1)
        ratio = new_action_probs / (actions_prob + 1e-10)
        option1 = ratio * advantages_
        option2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_
        policy_loss = -torch.min(option1, option2).mean()
        
        # 4. Value function MSE loss
        value_loss = ((value_ - returns)**2).mean()
        
        # 5. Policy Entropy
        entropy = -(moves_ * torch.log(moves_ + 1e-10)).sum(dim=-1).mean()
        
        # 6. Auxiliary next-action prediction loss
        valid_mask = (opponent_actions_taken != -1)
        if valid_mask.any():
            aux_loss = profiler.get_auxiliary_loss(
                new_z_opp[valid_mask], 
                opponent_actions_taken[valid_mask], 
                is_discrete=True
            )
        else:
            aux_loss = torch.tensor(0.0, device=states.device)
            
        # Joint Total Loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + lambda_aux * aux_loss
        
        total_loss.backward()
        optimizer.step()