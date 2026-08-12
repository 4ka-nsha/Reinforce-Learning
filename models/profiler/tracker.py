import torch
import numpy as np
from collections import deque

class TrajectoryTracker:
    """
    Maintains a high-performance sliding window of the opponent's most recent interactions.
    Automatically formats the data into batch-first PyTorch tensors for the OpponentProfiler.
    """
    def __init__(self, window_size: int, feature_dim: int):
        """
        Args:
            window_size (int): The number of past steps (N) to remember.
            feature_dim (int): The size of the input feature vector at each step 
                               (must match input_dim of OpponentProfiler).
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # deque automatically pops the oldest element when maxlen is reached, 
        # making it computationally cheaper than popping from a standard list.
        self.buffer = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        """
        Clears the history buffer. 
        
        Pre-fills the buffer with zero-vectors so the RNN always receives a 
        consistent sequence length (seq_len = window_size), even on step 1 of an episode.
        """
        self.buffer.clear()
        for _ in range(self.window_size):
            self.buffer.append(np.zeros(self.feature_dim, dtype=np.float32))

    def update(self, step_features):
        """
        Adds the latest opponent interaction to the rolling window.
        
        Args:
            step_features (np.ndarray, list, or torch.Tensor): 
                A 1D array of size `feature_dim`. If the environment uses discrete 
                actions, Person 1 should pass them one-hot encoded here.
        """
        # Standardize input to a float32 numpy array
        if isinstance(step_features, torch.Tensor):
            step_features = step_features.detach().cpu().numpy()
        else:
            step_features = np.array(step_features, dtype=np.float32)
            
        if step_features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Tracker expected feature_dim {self.feature_dim}, "
                f"but received vector of size {step_features.shape[0]}."
            )
            
        self.buffer.append(step_features)

    def get_history_tensor(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Converts the sliding window into a PyTorch tensor ready for the model's forward pass.
        
        Args:
            device (torch.device): The device (CPU/GPU) where the tensor should be sent.
            
        Returns:
            history_tensor (torch.Tensor): Shape (1, window_size, feature_dim).
                                           The '1' acts as the batch_size dimension.
        """
        # Stack the sequence of 1D arrays into a 2D array: (window_size, feature_dim)
        seq_array = np.stack(self.buffer)
        
        # Convert to tensor and add the batch dimension at axis 0
        history_tensor = torch.tensor(seq_array, dtype=torch.float32, device=device).unsqueeze(0)
        
        return history_tensor