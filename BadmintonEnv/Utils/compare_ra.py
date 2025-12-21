
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.patches as mpatches

from Agent.RallyNet import RallyNet
import pickle

import sys
import os

def get_data(file_path):
    feature_columns = [
        'player_location_x', 'player_location_y',
        'hit_x','hit_y',
        'opponent_location_x', 'opponent_location_y',
        'pre_type']
    label_columns = ['type']

    data = pd.read_csv(file_path)
    features = data[feature_columns].values
    labels = data[label_columns].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def plot_scatter_by_shot_type(X_train, y_train, output_path,player1):
    with open('./input_data/target_players_ids.pkl', 'rb') as f:
        target_players = pickle.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    player_name = player1
    player_id = target_players.index(player_name)

    with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
        hyperparameters = pickle.load(f)

    player_agent = RallyNet(
            data_size= hyperparameters['data_size'], 
            latent_size=hyperparameters['latent_size'],
            context_size=hyperparameters['context_size'],
            hidden_size=hyperparameters['hidden_size'],
            player_ids_len = hyperparameters['player_ids_len'],
            target_players = hyperparameters['target_players'],
            shot_type_len = hyperparameters['shot_type_len'],
            ts = hyperparameters['ts'],
            id = player_id,
            device = device,
        ).to(device)
    player_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/RallyNet_weight.trc", map_location = device, weights_only=True),strict = False)

    info_dict = {}
    

if __name__ == "__main__":
    # 指定 CSV 文件路徑與輸出圖片路徑
    csv_file = '/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/2_dataset_merge.csv'
    output_folder = '/home/adsl-1-3/Documents/Demo/from_gdwang/BadmintonEnv/Utils/scatter_img/'

    player1 = 'Kento MOMOTA'

    features, labels = get_data(csv_file)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.001, random_state=42)
    
    # 繪製並儲存不同球種的散點圖（同時包含真實與預測）
    plot_scatter_by_shot_type(X_train, y_train, output_folder,player1)