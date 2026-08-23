"""Maintain the fixed-length action history consumed by the profiler.
The tracker pads new episodes with zero vectors and returns batched tensors.
"""

import torch
import numpy as np
from collections import deque

class TrajectoryTracker:
    """Maintain a fixed-length window of opponent action features."""
    def __init__(self, window_size: int, feature_dim: int):
        """Set the history length and feature width."""
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        self.buffer = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        """Clear history and restore the zero-padded window."""
        self.buffer.clear()
        for _ in range(self.window_size):
            self.buffer.append(np.zeros(self.feature_dim, dtype=np.float32))

    def update(self, step_features):
        """Append one feature vector to the history window."""
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
        """Return the window as a batched PyTorch tensor."""
        seq_array = np.stack(self.buffer)
        history_tensor = torch.tensor(seq_array, dtype=torch.float32, device=device).unsqueeze(0)
        
        return history_tensor
