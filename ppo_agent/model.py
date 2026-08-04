import torch
import torch.nn as nn 
import numpy as np

# Consider tick-tack-toe 3*3 possibilites
# 0 denotes your move, 1 opponents, 2 empty
# Output is a value + probabilistic representation of the 9 slots
class DLLayer(nn.Module):
    def __init__(self, input_size, board_count, hidden_layer,) -> None:
        super().__init__()
        self.input_size = input_size
        self.board_count = board_count
        self.hidden_layer  = hidden_layer
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU()
        )
        
        self.value = nn.Linear(hidden_layer,1)
        self.moves = nn.Linear(hidden_layer, board_count)
        
    def forward(self,current_board,player_vector):
        is_empty = (current_board == 2)
        mask = torch.where(is_empty,torch.tensor(0.0),torch.tensor(float('-inf')))
        output = self.network(torch.cat([current_board, player_vector], dim=-1))
        
        value = self.value(output).squeeze(-1)
        
        logits = self.moves(output)
        logits = logits + mask
        
        # The best case next move is the highest probability index, given there was a 2 there        
        moves = torch.softmax(logits, dim = -1)
        
        return value,moves
        
    

        
    