import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.patches as mpatches

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Agent.BC_model._model import gmm_sample
# from BadmintonEnv.Agent.BC_model.s_model import SimpleNN as MLP2 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_and_plot(data_sim, data_true, output_folder):
    # 檢查輸出資料夾是否存在，若不存在則創建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 讀取兩個 CSV 檔案
    data1 = data_sim
    data2 = data_true
    # rally_id = data1['rally_id']
    # unique_rally_id = sorted(set(rally_id))
    # print(unique_rally_id)

    # data2 = data2[data2['rally_id'] < data1['rally_id'].iloc[-1]]

    type_mapping= {'發短球': 1, '長球': 2, '推撲球': 3, 
                '殺球': 4, '接殺防守': 5, '平球': 6, 
                '接不到': 11, '網前球': 7, '挑球': 8, 
                '切球': 9, '發長球': 10}
    
    type_mapping_eng = {
            1: "Serve short",
            2: "Clear",
            3: "Push Shot",
            4: "Smash",
            5: "Smash Defence",
            6: "Drive",
            7: "Net Shot",
            8: "Lob",
            9: "Drop",
            10: "Serve long",
            11: "Missed shot",
        }
    
    data2['type'] = data2['type'].map(type_mapping).fillna(0).astype(int)

    # 確認 'type' 欄位是否存在
    if 'type' not in data1.columns or 'type' not in data2.columns:
        print("Error: 'type' column not found in one of the CSV files.")
        return

    # 移除 Type 1, 10, 11
    type_counts1 = data1['type'].value_counts().reset_index()
    type_counts1.columns = ['Type', 'Count']
    type_counts1['Dataset'] = 'Simulated Data'
    type_counts1['Count'] = type_counts1['Count'] / len(data1)
    type_counts1 = type_counts1.loc[~type_counts1['Type'].isin([1, 10, 11])]

    type_counts2 = data2['type'].value_counts().reset_index()
    type_counts2.columns = ['Type', 'Count']
    type_counts2['Dataset'] = 'True Data'
    type_counts2['Count'] = type_counts2['Count'] / len(data2)
    type_counts2 = type_counts2.loc[~type_counts2['Type'].isin([1, 10, 11])]

    # Mapping Type 數字轉為英文
    type_counts1['Type'] = type_counts1['Type'].map(type_mapping_eng)
    type_counts2['Type'] = type_counts2['Type'].map(type_mapping_eng)

    # 合併兩組統計資料
    combined_counts = pd.concat([type_counts1, type_counts2])

    # 固定 Type 的順序，使畫圖時不會自動排序
    ordered_types = [type_mapping_eng[i] for i in range(2, 10)]  # 依照 2～9 的順序
    combined_counts['Type'] = pd.Categorical(combined_counts['Type'], categories=ordered_types, ordered=True)

    # 繪製長條圖
    plt.figure(figsize=(12, 12))
    sns.barplot(x='Type', y='Count', hue='Dataset', data=combined_counts, palette='viridis')
    plt.xlabel('Type')
    plt.ylabel('Percentage')
    plt.ylim([0, 0.3])
    plt.title('Comparison of Type Counts')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    # 儲存圖片
    output_file = os.path.join(output_folder, 'type_count_comparison.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"長條圖已儲存為: {output_file}")

    return
    
    data1 = data_sim
    data2 = data_true
    # data2 = data2[data2['rally_id'] < data1['rally_id'].iloc[-1]]

    max_values1 = []
    current_max1 = 0
    max_values2 = []
    current_max2 = 0

    # print(len(data1['ball_round']))
    # print(len(data2['ball_round']))

    id_list1 = []
    id_list2 = []

    # 遍歷 ball_round 列
    for i in range(len(data1['ball_round']) - 2):
        # 更新當前的最大值
        current_max1 = max(current_max1, data1['ball_round'].iloc[i])
        
        # 如果下一個數字變小了（代表重置），記錄當前段落的最大值
        if data1['ball_round'].iloc[i + 1]==1:
            # id_list1.append(data1['rally_id'].iloc[i + 1])
            max_values1.append(current_max1)
            current_max1 = 0  # 重置最大值

    for i in range(len(data2['ball_round']) - 2):
        # 更新當前的最大值
        current_max2 = max(current_max2, data2['ball_round'].iloc[i])
        
        # 如果下一個數字變小了（代表重置），記錄當前段落的最大值
        if data2['ball_round'].iloc[i + 1]==1:
            # id_list2.append(data2['rally_id'].iloc[i + 1])
            max_values2.append(current_max2)
            current_max2 = 0  # 重置最大值
    
    # for i in range(330):
    #     print(id_list1[i])
    #     print(id_list2[i])
    #     print(i)
    #     print()

    max_values1.append(max(current_max1, data1['ball_round'].iloc[-1]))
    max_values2.append(max(current_max2, data2['ball_round'].iloc[-1]))

    print(len(max_values1))
    print(len(max_values2))

    sns.kdeplot(max_values2, fill=True, color='blue')
    sns.kdeplot(max_values1, fill=True, color='red')
    plt.xlabel('Ball round')
    # plt.ylabel('Density')
    plt.title('KDE Plot of Ballround')

    output_file = os.path.join(output_folder, 'count_comparison.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_scatter_by_shot_type(data_simulate,data_true,output_folder):

    adjust = 1

    filtered_data_true = data_true#[data_true['rally_id'] < data_simulate['rally_id'].iloc[-1]]
    shot_types = data_true['type']
    unique_types = sorted(set(shot_types))

    true_shot_types = filtered_data_true['type']
    sim_shot_types = data_simulate['type']

    type_mapping= {'發短球': 1, '長球': 2, '推撲球': 3, 
                '殺球': 4, '接殺防守': 5, '平球': 6, 
                '接不到': 11, '網前球': 7, '挑球': 8, 
                '切球': 9, '發長球': 10}
    
    type_mapping_eng = {
            1: "Serve Short",
            2: "Clear",
            3: "Push Shot",
            4: "Smash",
            5: "Smash Defence",
            6: "Drive",
            7: "Net Shot",
            8: "Lob",
            9: "Drop",
            10: "Serve Long",
            11: "Missed Shot",
        }

    for shot_type in unique_types:
        true_indices = [i for i, t in enumerate(true_shot_types) if t == shot_type]
 
        if len(true_indices) == 0:
            continue

        shot_type = type_mapping.get(shot_type, 0)
        type_name = type_mapping_eng.get(shot_type, "Unknown Type")
        sim_indices = [i for i, t in enumerate(sim_shot_types) if t == shot_type]
        
        if len(sim_indices) == 0:
            continue

        true_landing_x = filtered_data_true['landing_x'].iloc[true_indices]
        true_landing_y = filtered_data_true['landing_y'].iloc[true_indices]

        sim_landing_x = -data_simulate['landing_x'].iloc[sim_indices]/177.5
        sim_landing_y = -(data_simulate['landing_y'].iloc[sim_indices]-240)/240

        plt.figure(figsize=(8, 5))
        sns.kdeplot(x=true_landing_x,y=true_landing_y, fill=True, color='blue', thresh=0.25, alpha = 1.0, bw_adjust=adjust)
        sns.kdeplot(x=sim_landing_x,y=sim_landing_y, fill=True, color='red', thresh=0.25, alpha = 0.5, bw_adjust=adjust)

        plt.scatter([], [], alpha=1.0, color='blue', edgecolors='k', label='True Landing')
        plt.scatter([], [], alpha=0.5, color='red', edgecolors='k', label='Simulated Landing')

        plt.title(f"Landing of {type_name}")
        # plt.xlabel("Values")
        # plt.ylabel("Density")
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        plt.legend(loc='upper right')
        plt.grid(True)

        file_name = f'{output_folder}kde_shot_type_{shot_type}_landing_comparison.png'
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f' 散點圖已存為: {file_name}')

        true_moving_x = filtered_data_true['moving_x'].iloc[true_indices]
        true_moving_y = filtered_data_true['moving_y'].iloc[true_indices]

        sim_moving_x = data_simulate['player_movement_x'].iloc[sim_indices]/177.5
        sim_moving_y = (data_simulate['player_movement_y'].iloc[sim_indices]+240)/240

        # print(sim_moving_x)

        plt.figure(figsize=(8, 5))

        sns.kdeplot(x=true_moving_x, y=true_moving_y, fill=True, color='blue', thresh=0.25, alpha = 1.0,bw_adjust=adjust)
        sns.kdeplot(x=sim_moving_x, y=sim_moving_y, fill=True, color='red', thresh=0.25, alpha = 0.5,bw_adjust=adjust)

        plt.scatter([], [], alpha=1.0, color='blue', edgecolors='k', label='True Moving')
        plt.scatter([], [], alpha=0.5, color='red', edgecolors='k', label='Simulated Moving')

        plt.title(f"Moving of {type_name}")
        # plt.xlabel("Values")
        # plt.ylabel("Density")
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        plt.legend(loc='upper right')
        plt.grid(True)

        file_name = f'{output_folder}kde_shot_type_{shot_type}_moving_comparison.png'
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f' 散點圖已存為: {file_name}')

if __name__ == "__main__":
    # 指定 CSV 文件路徑與輸出圖片路徑
    csv_file = 'output_data/output_game_sim_K_V.csv'
    true_file = 'input_data/K_V_data.csv'
    output_folder = 'BadmintonEnv/Utils/compare_K/'

    data_simulate = pd.read_csv(csv_file)
    data_true = pd.read_csv(true_file)

    data_simulate = data_simulate[data_simulate['player'] == 'B']
    data_true = data_true[data_true['player'] == 'Kento MOMOTA']

    plot_scatter_by_shot_type(data_simulate,data_true,output_folder)
    compare_and_plot(data_simulate, data_true, output_folder)



    output_folder = 'BadmintonEnv/Utils/compare_V/'

    data_simulate = pd.read_csv(csv_file)
    data_true = pd.read_csv(true_file)

    data_simulate = data_simulate[data_simulate['player'] == 'A']
    data_true = data_true[data_true['player'] == 'Viktor AXELSEN']

    plot_scatter_by_shot_type(data_simulate,data_true,output_folder)
    compare_and_plot(data_simulate, data_true, output_folder)



    output_folder = 'BadmintonEnv/Utils/evaluation_img/'

    data_simulate = pd.read_csv('output_data/output_game_sim_all.csv')
    data_true = pd.read_csv('input_data/All_dataset.csv')

    plot_scatter_by_shot_type(data_simulate,data_true,output_folder)
    compare_and_plot(data_simulate, data_true, output_folder)
    