import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.patches as mpatches

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from BadmintonEnv.Agent.BC_model._model import gmm_sample
from BadmintonEnv.Agent.BC_model.s_model import SimpleNN as MLP2 

# 讀取 CSV 文件
def get_data(file_path):
    feature_columns = [
        'player_location_x', 'player_location_y',
        'hit_x','hit_y',
        'opponent_location_x', 'opponent_location_y',
        'pre_type','type'
    ]
    label_columns = ['landing_x','landing_y','moving_x','moving_y']
    data = pd.read_csv(file_path)
    features = data[feature_columns].values
    labels = data[label_columns].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# 繪製並存成 PNG 散點圖（同時顯示預測與真實）
def plot_scatter_by_shot_type(X_train, y_train, output_path):
    # 從 X_train 提取對應球種
    shot_types = X_train[:, -1].int().numpy()
    unique_types = sorted(set(shot_types))

    # 將 predictions_land 從 Tensor 轉為 numpy

    for shot_type in unique_types:
        # 篩選出對應球種的資料
        indices = [i for i, t in enumerate(shot_types) if t == shot_type]
        if len(indices) == 0:
            continue
        
        gmm = 5
        mlp_2 = MLP2(gmm)
        mlp_2.load_state_dict(torch.load(f"/home/adsl-1-3/Documents/Demo/from_gdwang/model_weights_gmm7{shot_type}_min.pth", weights_only=True))
        mlp_2.eval()

        # 取得模型預測
        # predictions_move_param, predictions_land_param = mlp_2(X_train)
        predictions_land_param = mlp_2(X_train)
        predictions_land = gmm_sample(predictions_land_param.view(-1, gmm * 5), gmm, max_std=1.0).unsqueeze(1)
        predictions_land = predictions_land.squeeze(1).detach().cpu().numpy()


        # 提取真實的 landing_x 和 landing_y
        true_landing_x = y_train[indices, 0].numpy()
        true_landing_y = y_train[indices, 1].numpy()

        # 提取預測的 landing_x 和 landing_y
        pred_landing_x = predictions_land[indices, 0]
        pred_landing_y = predictions_land[indices, 1]

        # 繪製散點圖
        plt.figure(figsize=(8, 6))

        # 真實資料
        
        plt.scatter(true_landing_x, true_landing_y, alpha=0.5, color='blue', edgecolors='k', label='True Landing')

        # 預測資料
        plt.scatter(pred_landing_x, pred_landing_y, alpha=0.5, color='red', edgecolors='k', label='Predicted Landing')

        plt.title(f'Scatter Plot of Landing Positions (Shot Type {shot_type})')
        plt.xlabel('landing_x')
        plt.ylabel('landing_y')
        plt.ylim([-1,1])
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 儲存圖片
        file_name = f'{output_path}shot_type_{shot_type}_landing_comparison.png'
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f' 散點圖已存為: {file_name}')

        plt.figure(figsize=(8, 5))
        sns.kdeplot(x=true_landing_x,y=true_landing_y, fill=True, color='blue', thresh=0.05, alpha = 1.0,label='True Landing')
        sns.kdeplot(x=pred_landing_x,y=pred_landing_y, fill=True, color='red', thresh=0.05, alpha = 0.5,label='Predicted Landing')

        blue_patch = mpatches.Patch(color='blue', alpha=1.0, label='True Landing')
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Predicted Landing')
        
        plt.legend(handles=[blue_patch, red_patch], loc='upper right')
        plt.title("Kernel Density Estimation (KDE) Plot")
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.legend(loc='upper right')
        plt.grid(True)

        file_name = f'{output_path}kde_shot_type_{shot_type}_landing_comparison.png'
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f' 散點圖已存為: {file_name}')

if __name__ == "__main__":
    # 指定 CSV 文件路徑與輸出圖片路徑
    csv_file = '/home/adsl-1-3/Documents/Demo/from_gdwang/input_data/2_dataset_merge.csv'
    output_folder = '/home/adsl-1-3/Documents/Demo/from_gdwang/BadmintonEnv/Utils/scatter_img/'

    gmm = 5

    # 讀取數據
    features, labels = get_data(csv_file)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.001, random_state=42)
    
    # 繪製並儲存不同球種的散點圖（同時包含真實與預測）
    plot_scatter_by_shot_type(X_train, y_train, output_folder)
