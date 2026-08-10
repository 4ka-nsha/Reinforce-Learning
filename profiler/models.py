import torch
import torch.nn as nn
import torch.nn.functional as F

class OpponentProfiler(nn.Module):
    """
    Ingests a sequence of historical opponent interactions and outputs a fixed-size 
    latent vector (z_opp) that profiles their behavior.
    """
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super(OpponentProfiler, self).__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        
        # RNN Encoder: Processes the rolling window of opponent history.
        # batch_first=True expects shape (batch_size, seq_len, feature_dim)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Latent Projection: Maps the final RNN hidden state to the target z_opp dimension.
        self.latent_proj = nn.Linear(hidden_dim, z_dim)
        
        # Auxiliary Head: Predicts the opponent's next action from z_opp.
        # This prevents the GRU from collapsing or memorizing noise.
        # Note: Set output dimension to match action space size for cross-entropy.
        self.aux_head = nn.Linear(z_dim, input_dim)

    def forward(self, history_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent opponent profile vector.
        
        Args:
            history_tensor (torch.Tensor): Shape (batch_size, seq_len, input_dim)
                representing the rolling window of past opponent actions/states.
                
        Returns:
            z_opp (torch.Tensor): Shape (batch_size, z_dim) profiling the opponent.
        """
        # _ is the full sequence output, hidden is the state at the final time step.
        # hidden shape: (num_layers, batch_size, hidden_dim)
        _, hidden = self.encoder(history_tensor)
        
        # Extract the hidden state from the last layer of the GRU
        final_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        
        # Project to latent space
        z_opp = self.latent_proj(final_hidden)
        
        # L2 Normalization (Optional but highly recommended)
        # Bounding the latent space prevents large magnitude shifts from 
        # destabilizing the concatenated PPO policy network.
        z_opp = F.normalize(z_opp, p=2, dim=-1)
        
        return z_opp

    def get_auxiliary_loss(self, z_opp: torch.Tensor, target_action: torch.Tensor, is_discrete: bool = True) -> torch.Tensor:
        """
        Computes the auxiliary loss for next-action prediction to enforce representation learning.
        
        Args:
            z_opp (torch.Tensor): The latent vector from the forward pass. Shape (batch_size, z_dim)
            target_action (torch.Tensor): The actual next action taken by the opponent.
            is_discrete (bool): Flag to toggle between classification (CrossEntropy) and regression (MSE).
            
        Returns:
            loss (torch.Tensor): Scalar loss value to be added to the PPO objective.
        """
        predicted_logits = self.aux_head(z_opp)
        
        if is_discrete:
            # Assumes target_action contains integer class indices
            # Shape requirements: predicted_logits (N, C), target_action (N,)
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(predicted_logits, target_action)
        else:
            # For continuous action spaces (e.g., continuous control variants)
            loss_fn = nn.MSELoss()
            return loss_fn(predicted_logits, target_action)