import torch
import torch.nn as nn 
import numpy as np

class PPOActorCritic(nn.Module):
    """
    Game-agnostic Actor-Critic policy conditioned on both the environmental 
    state (s_t) and the inferred latent opponent profile vector (z_opp).
    Supports logits masking using action masks supplied by the Gym environment.
    """
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
        """
        Args:
            state (torch.Tensor): Board or environmental state observation, shape (batch_size, state_dim)
            latent_vector (torch.Tensor): Normalized latent profile z_opp, shape (batch_size, z_dim)
            action_mask (torch.Tensor, optional): Boolean mask for legal actions, shape (batch_size, action_dim)
        """
        # Support both single samples and batched inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)
            
        x = torch.cat([state, latent_vector], dim=-1)
        features = self.network(x)
        
        value = self.value(features).squeeze(-1) # shape: (batch_size,)
        logits = self.action_logits(features)    # shape: (batch_size, action_dim)
        
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # set illegal actions to -inf (so their probability is 0 after softmax)
            mask = torch.where(action_mask.bool(), torch.tensor(0.0, device=logits.device), torch.tensor(float('-inf'), device=logits.device))
            logits = logits + mask
            
        action_probs = torch.softmax(logits, dim=-1)
        return value, action_probs