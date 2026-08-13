import torch

def compute_advantages(rewards, values, game_ended, gamma=0.99):
    n = len(rewards)
    advantages = []
    for i in range(n-1, -1, -1):
        next_value = values[i+1] if i+1 < n else 0
        surprise = (rewards[i] + gamma * next_value * (1 - game_ended[i])) - values[i]
        advantages.append(surprise)
        
    advantages.reverse()  
    advantages = torch.tensor(advantages, dtype=torch.float32)
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
    
    # Normalize advantages
    advantages_ = (advantages - torch.mean(advantages, -1, keepdim=True)) / (torch.std(advantages, -1, keepdim=True) + 1e-10)
    
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