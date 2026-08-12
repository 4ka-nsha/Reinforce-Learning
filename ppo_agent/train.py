from ppo_agent.model import PPOActorCritic
from ppo_agent.buffer import Memory
from ppo_agent.model_snapshot import ModelSnapshot
from ppo_agent.ppo import compute_advantages, ppo_update
import torch.optim as optim
import torch
import random

model = PPOActorCritic(input_size=41, action_dim=9, hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=0.001)
bfr = Memory()
snap = ModelSnapshot(max_pool_size=10)

ppo_update_count = 0

def fake_board():
    # random board: 0 = my move, 1 = opponent, 2 = empty
    return torch.tensor([random.choice([0.0, 1.0, 2.0]) for _ in range(9)],
        dtype=torch.float32
    )
 
def fake_z_opp():
    # state is 9 dims, concat z_opp (32 dims) to make 41 dims input_size
    return torch.randn(32)
 
def fake_reward():
    return random.choice([-1.0, 0.0, 1.0])


for episode in range(5):
    opponent_type = snap.define_opponent(current_ppo_frac=episode / 5)
    
    # 9 because 3*3 tic-tac-toe can have atmost 9 moves
    for t in range(9):
        state = fake_board()
        z_opp = fake_z_opp()
        
        # generate boolean action mask: True if slot is empty (2.0)
        action_mask = (state == 2.0)
        # if no legal moves, make all True to avoid failure
        if not action_mask.any():
            action_mask = torch.ones(9, dtype=torch.bool)
 
        value, moves = model.forward(state, z_opp, action_mask)
        # Squeeze batch dimension for Categorical
        moves_squeezed = moves.squeeze(0)
        dist = torch.distributions.Categorical(moves_squeezed)
        action = dist.sample()
        action_prob = moves_squeezed[action].item()
 
        reward = fake_reward()
        done = (t == 8)
 
        bfr.store(
            states=state,
            z_opp=z_opp,
            actions_taken=action.item(),
            actions_prob=action_prob,
            action_mask=action_mask,
            value=value.item(),
            rewards=reward,
            game_ended=done,
        )
 
        if done:
            break
        
        # After 20 moves are stored, we do some updates
        if len(bfr.states) >= 20:
            advantages, returns = compute_advantages(bfr.rewards, bfr.value, bfr.game_ended)
            ppo_update(model, optimizer, bfr, advantages, returns)
            bfr.clear()
            ppo_update_count += 1

            if ppo_update_count % 2 == 0:
                snap.save_snap(model)
