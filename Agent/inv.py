import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import pickle
import os
import sys
import json

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Utils.helper import *


os.environ['CUDA_LAUNCH_BLOCKING']='1'

def make_dataset(device):
    with open("Experiments/rallynetv2/data/norm/train_player_traj.pkl", 'rb') as f:
        player_data = pickle.load(f)
    with open("Experiments/rallynetv2/data/norm/train_opponent_traj.pkl", 'rb') as f:
        opponent_data = pickle.load(f)
    with open("Experiments/rallynetv2/data/norm/train_player_mask.pkl", 'rb') as f:
        player_masks = pickle.load(f)
    with open("Experiments/rallynetv2/data/norm/train_opponent_mask.pkl", 'rb') as f:
        opponent_masks = pickle.load(f)
    with open("Data/intermediate_data/player_index_map.json", 'r') as f:
        player2index = json.load(f)


    merged_data = []
    merged_masks = []
    shot_type_set = []

    for player_rally, opponent_rally, player_mask_rally, opponent_mask_rally in zip(player_data, opponent_data, player_masks, opponent_masks):
        merged_rally = []
        merged_mask_rally = []
        player_len = len(player_rally)
        opponent_len = len(opponent_rally)
        if player_len > 0:
            max_len = max(player_len, opponent_len)
            for i in range(max_len):
                if i < player_len:
                    merged_rally.append(player_rally[i])
                    merged_mask_rally.append(player_mask_rally[i])
                    shot_type_set.append(player_rally[i][12])
                if i < opponent_len:
                    merged_rally.append(opponent_rally[i])
                    merged_mask_rally.append(opponent_mask_rally[i])
                    shot_type_set.append(opponent_rally[i][12])

            merged_data.append(merged_rally)
            merged_masks.append(merged_mask_rally)

    max_length = max(len(rally) for rally in merged_data)
    xs = pad_and_convert_to_tensor(merged_data, max_length).cuda()
    masks = pad_and_convert_to_tensor(merged_masks, max_length).cuda()

    ts = torch.linspace(0, xs.shape[0], steps=xs.shape[0], device=device)
    min_val = torch.min(ts)
    max_val = torch.max(ts)
    normalized_tensor = (ts - min_val) / (max_val - min_val)
    ts = normalized_tensor * 2

    return xs, masks, ts, len(player2index), len(set(shot_type_set))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xs, masks, ts, player_id_len, shot_type_len = make_dataset(device=device)
state = xs[:,:,:18].cuda()
action = xs[:,:,-5:].cuda()

# 将 state 和 action 数据转换为适合 MLP 输入的形式
sequence_length, batch_size, feature_dim = state.shape
action_dim = action.shape[-1]

# 准备数据集
inputs = []
labels = []
mask_flat = []

for i in range(sequence_length - 1):
    current_state = state[i].reshape(batch_size, -1)
    next_state = state[i + 1].reshape(batch_size, -1)
    combined_state = torch.cat((current_state, next_state), dim=1)
    inputs.append(combined_state)
    labels.append(action[i])
    current_mask = masks[i].reshape(batch_size, -1)
    mask_flat.append(current_mask[:, :1])

inputs = torch.stack(inputs).reshape(-1, feature_dim * 2)
labels = torch.stack(labels).reshape(-1, action_dim)
mask_flat = torch.stack(mask_flat).reshape(-1, 1)


# 過濾掉全零的樣本
non_zero_inputs = (inputs != 0).any(dim=1)
inputs = inputs[non_zero_inputs]
labels = labels[non_zero_inputs]
mask_flat = mask_flat[non_zero_inputs]


# 使用 75% 的数据进行训练，25% 的数据进行评估
train_dataset = TensorDataset(inputs, labels, mask_flat)
# train_size = int(0.75 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)
        player_embeds = self.player_embedding(player_ids)
        state = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)
        return state

    def forward(self, x):
        x = x.float()
        raw_out = torch.cat((self.embed_and_transform(x[:, :18]), self.embed_and_transform(x[:, 18:])), -1)
        out = self.fc(raw_out)
        out = self.relu(out)

        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)

        return land_logit, shot_logit, move_logit
    
input_dim = 32
hidden_dim = 128

model = MLP(input_dim, hidden_dim, player_id_len, shot_type_len).cuda()
ce = nn.CrossEntropyLoss(ignore_index = 0)
mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, batch_masks in train_loader:
        true_landing_pos = labels[:, :2].float()
        true_movement_pos = labels[:, 3:].float()
        true_shot_type = labels[:, 2].long()
        batch_masks = batch_masks.squeeze().float()

        optimizer.zero_grad()
        land_logit, shot_logit, move_logit = model(inputs)

        land_loss = mse(land_logit, true_landing_pos)
        shot_loss = ce(shot_logit, true_shot_type)
        move_loss = mse(move_logit, true_movement_pos)

        loss = (land_loss + shot_loss + move_loss) * batch_masks
        loss = loss.sum() / batch_masks.sum().clamp(min=1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# 假设模型保存在 model.pth 文件中
PATH = "INV_weight.pth"
# 保存模型
torch.save(model.state_dict(), PATH)
