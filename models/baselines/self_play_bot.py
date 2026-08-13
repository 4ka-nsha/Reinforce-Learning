import torch
import numpy as np
from typing import Optional

from models.baselines.base_bot import OpponentBot, last_opponent_action
from models.profiler.tracker import TrajectoryTracker

class SelfPlayBot(OpponentBot):
    """
    Opponent bot that wraps a PPOActorCritic policy checkpoint to allow
    self-play training. Optionally profiles the main agent's behavior.
    """
    name = "self_play_bot"
    
    def __init__(
        self, 
        policy_model: torch.nn.Module, 
        profiler_model: Optional[torch.nn.Module] = None, 
        curriculum = None, 
        stage = None, 
        seed: Optional[int] = None
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.profiler_model = profiler_model
        self.curriculum = curriculum
        self.stage = stage
        self._rng = np.random.default_rng(seed)
        
        # Instantiate an internal tracker to profile the training agent
        if profiler_model is not None and curriculum is not None:
            self.tracker = TrajectoryTracker(window_size=10, feature_dim=curriculum.max_action_dim)
        else:
            self.tracker = None
            
        self.reset()

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self._last_agent_action = None

    def act(self, state, player_id: int) -> int:
        # 1. Update the tracker with the training agent's last action
        if self.tracker is not None:
            agent_action = last_opponent_action(state, player_id)
            if agent_action is not None:
                action_one_hot = np.zeros(self.curriculum.max_action_dim, dtype=np.float32)
                action_one_hot[agent_action] = 1.0
                self.tracker.update(action_one_hot)

        # 2. Extract state representation and zero-pad to max_observation_dim
        game_name = self.stage.game_name
        if self.curriculum.uses_information_state_tensor(game_name):
            raw_obs = state.information_state_tensor(player_id)
        else:
            raw_obs = state.observation_tensor(player_id)
            
        obs = np.zeros(self.curriculum.max_observation_dim, dtype=np.float32)
        obs[:len(raw_obs)] = raw_obs
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # 3. Generate the opponent's latent representation
        if self.tracker is not None and self.profiler_model is not None:
            history_tensor = self.tracker.get_history_tensor()
            with torch.no_grad():
                z_opp = self.profiler_model(history_tensor).squeeze(0)
        else:
            z_opp = torch.zeros(32, dtype=torch.float32)

        # 4. Form action mask
        mask = np.zeros(self.curriculum.max_action_dim, dtype=bool)
        base_mask = state.legal_actions_mask(player_id)
        mask[:len(base_mask)] = base_mask
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        # 5. Feed to the policy model
        with torch.no_grad():
            _, action_probs = self.policy_model(obs_tensor, z_opp, mask_tensor)
            
        # 6. Sample from legal actions
        action_probs = action_probs.squeeze(0).numpy()
        legal_actions = state.legal_actions(player_id)
        
        # Filter action probabilities over legal actions
        probs = action_probs[:len(base_mask)]
        legal_probs = probs[legal_actions]
        
        if legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            # Fallback uniform distribution
            legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
        chosen_action = int(self._rng.choice(legal_actions, p=legal_probs))
        return chosen_action
