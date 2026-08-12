from dataclasses import field
from typing import Any
import random
from ppo_agent.model import PPOActorCritic

class ModelSnapshot:
    def __init__(self, max_pool_size):
        self.pool = []
        self.max_pool_size = max_pool_size 
    
    def save_snap(self,model):
        weights = model.state_dict()    
        self.pool.append(weights)
        if(len(self.pool) > self.max_pool_size):
                self.pool = self.pool[1:]
    
    def define_opponent(self, current_ppo_frac):
        self_play_chance = min(current_ppo_frac, 0.8)
        if not self.pool or random.random() > self_play_chance:
            return "fixed_bot"
        return "self_play"
    
    def get_self_play_opponent(self, architecture_args):
        weights = random.choice(self.pool)
        opponent_model = PPOActorCritic(*architecture_args)
        opponent_model.load_state_dict(weights)
        opponent_model.eval()   
        return opponent_model