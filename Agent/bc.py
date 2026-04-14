import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.helper import *  

def list_subtraction(p1,p2):
    point1 = p1.copy()
    point2 = p2.copy()
    v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
    return v[0], v[1]


class BC(nn.Module):
    
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len, player_id):
        super(BC, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.player_id = player_id
        # Embeddings for categorical features within the state
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        # Output layers for different components of the action
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        # Assuming state[:, 12] is shot_type and state[:, 17] is player_id
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)       # Shape: [batch_size, 8]
        player_embeds = self.player_embedding(player_ids)   # Shape: [batch_size, 8]
        
        # Concatenate embeddings with other state features
        # Adjust indices based on your actual state structure
        state_features = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)  # Shape: [batch_size, 32]
        return state_features

    def forward(self, x):
        x = x.float()
        embedded = self.embed_and_transform(x)  # Shape: [batch_size, 32]
        out = self.fc(embedded)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)

        land_logit = self.predict_land_area(out)    # Shape: [batch_size, 2]
        move_logit = self.predict_move_area(out)    # Shape: [batch_size, 2]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]

        return land_logit, shot_logit, move_logit
    
    def translation(self, state, info): # Calculate the full states
        return_state_list = [0]*18
        
        # match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
        # player_state = ['player_location_x', 'player_location_y']
        # ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
        # opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y']

        # match_state
        if info['rally'] < 0:
            return_state_list[0] = 1
        else:
            return_state_list[0] = info['rally']
        
        return_state_list[1] = info['round'][-1]
        return_state_list[2] = info['score'][0] 
        return_state_list[3] = info['score'][1]
        return_state_list[4] = info['score'][0] - info['score'][1] # the difference score between the player and the opponent

        # player_state
        return_state_list[5] = state[1][0]
        return_state_list[6] = state[1][1]
        
        # ball_state
        return_state_list[8], return_state_list[9] = list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = state[0]
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        if info['action'][-1] != None:
            opponent_last_x = (info['action'][-1][3][0] / 177.5)
            opponent_last_y = (info['action'][-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        return_state_list[17] = self.player_id
    
        return torch.FloatTensor(return_state_list)
        
        

class BC_Agent(BaseAgent):

    def __init__(self, player_name, model_path, player_id):
        super().__init__(player_name)
        self.model = BC(input_size=18, hidden_size=128, player_id_len=39, shot_type_len=12, player_id=player_id)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, state, info, launch):
        # encode the state
        normalized_state = self.encode_state(state)
        bc_state = self.model.translation(normalized_state, info)
        bc_land, bc_shot_logit, bc_move = self.forward(bc_state.unsqueeze(0))
        bc_shot_type_probs = F.softmax(bc_shot_logit, dim=-1)
        

        if launch == True:
            mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)
        else:
            mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(self.device)

        masked_probs = bc_shot_type_probs.masked_fill(~mask, 0.0)
        sum_probs = masked_probs.sum(dim=-1, keepdim=True)
        bc_shot_type_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
        
        bc_shot_type_dist = Categorical(bc_shot_type_probs)
        bc_shot_type = bc_shot_type_dist.sample()
        output_shot = bc_shot_type.unsqueeze(1).item()
        output_shot_dist = bc_shot_type_probs.tolist()[0]
        output_land = bc_land.tolist()[0]
        output_move = bc_move.tolist()[0]

        if output_shot == 11:
            return None
        else:
            prob_array = np.array(output_shot_dist)
            prob_array = prob_array[1:11]
            normalized_array = prob_array / prob_array.sum()
            normalized_tuple = tuple(normalized_array)

            action = (output_shot, state[-1], output_land, output_move, normalized_tuple)
            action = self.decode_action(action)
            return action

class BCDataset(Dataset):
    def __init__(self, states, landing, shot, move):
        self.states = states
        self.landing = landing
        self.shot = shot
        self.move = move

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.landing[idx],
            self.shot[idx],
            self.move[idx]
        )

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join("Data", "offline_data", "train_test_split.json"), "r") as f:
        split_content = json.load(f)
    with open(os.path.join("Data", "offline_data", "player_index_map.json"), "r") as f:
        player2index = json.load(f)
    
    df = pd.read_csv(os.path.join("Data", "input_data", "merged.csv"))
    df["player"] = df["player"].map(player2index)
    state_data, action_data = [], []

    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    for player_name, match_ids in split_content["train"].items():
        selected_df = df[(df['player'] == player2index[player_name]) & (df['match_id'].isin(match_ids))]
        state_data.extend(selected_df[state_col].values)
        action_data.extend(selected_df[action_col].values)

    # 轉 numpy
    state_np = np.array(state_data)
    action_np = np.array(action_data)
    # 轉 tensor
    state_tensor = torch.tensor(state_np, dtype=torch.float32)
    action_tensor = torch.tensor(action_np, dtype=torch.float32)

    landing = action_tensor[:, 0:2]       # landing_x, landing_y
    shot_type = action_tensor[:, 2].long()  # player_type (分類)
    move = action_tensor[:, 3:5]          # moving_x, moving_y
    
    dataset = BCDataset(state_tensor, landing, shot_type, move)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define model parameters
    input_dim = 32  # Updated from 18 to 32 to match embedded state size
    hidden_dim = 128
    player_id_len = 43
    shot_type_len = 12
    num_epochs = 100

    model = BC(input_dim, hidden_dim, player_id_len, shot_type_len, player_id=0).to(device)

    # Define loss functions and optimizer
    ce_loss = nn.CrossEntropyLoss()  # For categorical shot type
    mse_loss = nn.MSELoss()          # For continuous landing and movement positions

    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(num_epochs):
        total_loss = 0
        
        for state, land_gt, shot_gt, move_gt in loader:
            state = state.cuda()
            land_gt = land_gt.cuda()
            shot_gt = shot_gt.cuda()
            move_gt = move_gt.cuda()
            
            land_pred, shot_pred, move_pred = model(state)
            
            loss_land = mse_loss(land_pred, land_gt)
            loss_move = mse_loss(move_pred, move_gt)
            loss_shot = ce_loss(shot_pred, shot_gt)
            
            loss = loss_land + loss_move + loss_shot

                        
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    
    # save model
    model_path = "BC_weight.pth"
    torch.save(model.state_dict(), model_path)


    # # testing for in-domain performance
    # model.eval()
    # total_shot_loss = 0.0
    # total_land_loss = 0.0
    # total_move_loss = 0.0

    # with torch.no_grad():
    #     for batch_inputs, batch_labels in test_loader:
    #         true_landing_pos = batch_labels[:, :2].float()
    #         true_shot_type = batch_labels[:, 2].long()
    #         true_movement_pos = batch_labels[:, 3:].float()

    #         land_logit, shot_logit, move_logit = model(batch_inputs)

    #         # Compute losses
    #         land_loss = mse(land_logit, true_landing_pos)
    #         shot_loss = ce(shot_logit, true_shot_type)
    #         move_loss = mse(move_logit, true_movement_pos)

    #         total_land_loss += land_loss.item()
    #         total_shot_loss += shot_loss.item()
    #         total_move_loss += move_loss.item()

    # # Calculate average losses
    # avg_shot_loss = total_shot_loss / len(test_loader)
    # avg_land_loss = total_land_loss / len(test_loader)
    # avg_move_loss = total_move_loss / len(test_loader)

    # print('Evaluation Results:')
    # print(f'Shot Loss: {avg_shot_loss:.4f}')
    # print(f'Land Loss: {avg_land_loss:.4f}')
    # print(f'Move Loss: {avg_move_loss:.4f}')

    # # Saving the model
    # PATH = "BC_weight.pth"
    # torch.save(model.state_dict(), PATH)
    # print(f'Model saved to {PATH}')