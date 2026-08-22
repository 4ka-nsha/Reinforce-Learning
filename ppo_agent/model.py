"""Define the PPO actor-critic used by the training loop.
The policy combines the game observation with the opponent latent profile.
Its categorical output is masked to exclude illegal actions.
"""

import torch
import torch.nn as nn 
import numpy as np

class PPOActorCritic(nn.Module):
    """Actor-critic policy conditioned on game state and opponent profile."""
    def __init__(self, input_size: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value = nn.Linear(hidden_dim, 1)
        self.action_logits = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor, latent_vector: torch.Tensor, action_mask: torch.Tensor = None):
        """Return value estimates and masked action probabilities."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)
            
        x = torch.cat([state, latent_vector], dim=-1)
        features = self.network(x)
        
        value = self.value(features).squeeze(-1) # Value per batch item.
        logits = self.action_logits(features)    # Action logits per batch item.
        
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # Set illegal-action logits to negative infinity.
            mask = torch.where(action_mask.bool(), torch.tensor(0.0, device=logits.device), torch.tensor(float('-inf'), device=logits.device))
            logits = logits + mask
            
        action_probs = torch.softmax(logits, dim=-1)
        return value, action_probs
