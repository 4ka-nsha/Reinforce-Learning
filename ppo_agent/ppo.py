import torch

def compute_advantages(rewards, values, game_ended, gamma=0.99):
    n = len(rewards)
    advantages = []
    for i in range(n-1,-1,-1):
        next_value = values[i+1] if i+1 < n else 0
        surprise = (rewards[i] + gamma * next_value * (1 - game_ended[i])) - values[i]
        advantages.append(surprise)
        
    advantages.reverse()  
    advantages = torch.tensor(advantages, dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32)
    returns = advantages + values_tensor
        
    return advantages, returns


def ppo_update(model, optimizer, memory, advantages, returns, clip_eps=0.2, epochs=4):
    memory.states = torch.stack(memory.states)
    memory.z_opp = torch.stack(memory.z_opp)    
    memory.actions_taken = torch.tensor(memory.actions_taken, dtype = torch.int32)
    memory.actions_prob = torch.tensor(memory.actions_prob, dtype = torch.float32)
    memory.value = torch.tensor(memory.value, dtype = torch.float32)
    memory.rewards = torch.tensor(memory.rewards, dtype = torch.float32)
    memory.game_ended = torch.tensor(memory.game_ended, dtype = torch.bool)
    
    local_loss  = 0
    total_loss = 0
    advantages_ = (advantages - torch.mean(advantages,-1)) /(torch.std(advantages,-1) + 0.0000000001)
    
    for run in range(epochs):
        optimizer.zero_grad()
        value_, moves_ = model.forward(memory.states, memory.z_opp)    
        new_action_probs = moves_.gather(1, memory.actions_taken.long().unsqueeze(1)).squeeze(1)
        ratio = new_action_probs / (memory.actions_prob + 1e-10)
        option1 = ratio*advantages_
        option2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_
        policy_loss = -torch.min(option1,option2).mean()
        
        local_loss = ((value_ - returns)**2).mean()
        entropy = -(moves_ * torch.log(moves_ + 1e-10)).sum(dim=-1).mean()
        total_loss = policy_loss + 0.5 * local_loss - 0.01 * entropy
        
        total_loss.backward()
        optimizer.step()
        
        
            
        
            
            
    