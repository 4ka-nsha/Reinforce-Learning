from dataclasses import dataclass, field
from typing import Any

@dataclass
class Memory:
    states: list[Any] = field(default_factory=list)
    z_opp: list[Any] = field(default_factory=list)
    actions_taken: list[Any] = field(default_factory=list)
    actions_prob: list[Any] = field(default_factory=list)
    action_masks: list[Any] = field(default_factory=list)
    value: list[Any] = field(default_factory=list)
    rewards: list[Any] = field(default_factory=list)
    game_ended: list[Any] = field(default_factory=list)
    
    # Profiler training fields.
    history_tensors: list[Any] = field(default_factory=list)
    opponent_actions_taken: list[Any] = field(default_factory=list)
    
    def store(
        self, 
        states, 
        z_opp, 
        actions_taken, 
        actions_prob, 
        action_mask, 
        value, 
        reward, 
        game_ended, 
        history_tensor,
        opponent_action
    ):
        self.states.append(states)
        self.z_opp.append(z_opp)
        self.actions_taken.append(actions_taken)
        self.actions_prob.append(actions_prob)
        self.action_masks.append(action_mask)
        self.value.append(value)
        self.rewards.append(reward)
        self.game_ended.append(game_ended)
        self.history_tensors.append(history_tensor)
        self.opponent_actions_taken.append(opponent_action)
        
    def clear(self):
        self.states.clear()
        self.z_opp.clear()
        self.actions_taken.clear()
        self.actions_prob.clear()
        self.action_masks.clear()
        self.value.clear()
        self.rewards.clear()
        self.game_ended.clear()
        self.history_tensors.clear()
        self.opponent_actions_taken.clear()
