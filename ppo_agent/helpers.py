import torch
import os
from typing import Optional

def save_joint_checkpoint(policy: torch.nn.Module, profiler: Optional[torch.nn.Module], path: str) -> None:
    """
    Saves a single combined checkpoint containing states of both the 
    PPO policy network and the Opponent Profiler.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    state = {
        "policy_state_dict": policy.state_dict(),
        "profiler_state_dict": profiler.state_dict() if profiler is not None else None
    }
    torch.save(state, path)
    print(f"Joint checkpoint saved successfully to {path}")


def load_joint_checkpoint(policy: torch.nn.Module, profiler: Optional[torch.nn.Module], path: str) -> None:
    """
    Loads both PPO policy and Opponent Profiler weights from a combined checkpoint file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
    state = torch.load(path, map_location="cpu")
    policy.load_state_dict(state["policy_state_dict"])
    
    if profiler is not None and "profiler_state_dict" in state and state["profiler_state_dict"] is not None:
        profiler.load_state_dict(state["profiler_state_dict"])
        
    print(f"Joint checkpoint loaded successfully from {path}")
