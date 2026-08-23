"""Neural models for encoding recent opponent behavior.
The profiler uses a GRU to produce a normalized latent vector.
An auxiliary action-prediction head provides the training signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OpponentProfiler(nn.Module):
    """Encode recent opponent actions into a fixed-size latent profile."""
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super(OpponentProfiler, self).__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        
        # Encode the rolling action history.
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Project the final hidden state into the latent space.
        self.latent_proj = nn.Linear(hidden_dim, z_dim)
        
        # Predict the opponent's next action for auxiliary training.
        self.aux_head = nn.Linear(z_dim, input_dim)

    def forward(self, history_tensor: torch.Tensor) -> torch.Tensor:
        """Return normalized profiles for batched action histories."""
        # Use the final hidden state from the GRU.
        _, hidden = self.encoder(history_tensor)
        
        final_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        z_opp = self.latent_proj(final_hidden)
        z_opp = F.normalize(z_opp, p=2, dim=-1)
        
        return z_opp

    def get_auxiliary_loss(self, z_opp: torch.Tensor, target_action: torch.Tensor, is_discrete: bool = True) -> torch.Tensor:
        """Return the auxiliary next-action prediction loss."""
        predicted_logits = self.aux_head(z_opp)
        
        if is_discrete:
            # Targets are integer action indices.
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(predicted_logits, target_action)
        else:
            # Use regression for continuous targets.
            loss_fn = nn.MSELoss()
            return loss_fn(predicted_logits, target_action)
