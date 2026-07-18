import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F

from bisect import bisect_left
from operator import itemgetter
from copy import copy, deepcopy

import os


def compute_shot_type_weights(data_file=None, method='sqrt_inverse', num_classes=12):
    """
    從檔案讀取資料並計算 shot type 的類別權重
    
    Args:
        data_file: 資料檔案路徑 (預設: Experiments/bc/data/norm/train_action.pkl)
        method: 'sqrt_inverse' (平方根倒數) 或 'uniform' (均勻權重)
        num_classes: 總類別數 (包含 Type 0)
    
    Returns:
        torch.Tensor: 類別權重 (shape: [num_classes])
    """
    # 預設檔案路徑
    if data_file is None:
        data_file = 'Experiments/bc/data/norm/train_action.pkl'
    
    # 讀取 pickle 檔案 (numpy array, shape: [N, 5])
    # columns: [landing_x, landing_y, shot_type, moving_x, moving_y]
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # 取得 shot_type (第 3 欄, index=2)
    shot_types = data[:, 2].astype(int)
    
    # 排除 0 (初始狀態)
    shot_types = shot_types[shot_types > 0]
    
    # 計算每個 type (1-11) 的出現次數
    class_counts = np.zeros(11, dtype=np.float32)
    for shot_type in range(1, 12):
        class_counts[shot_type - 1] = (shot_types == shot_type).sum()
    
    # 避免除以零，將 0 次數替換為 1
    class_counts = np.maximum(class_counts, 1.0)
    
    if method == 'sqrt_inverse':
        # 平方根倒數 (較溫和)
        weights = 1.0 / np.sqrt(class_counts)
    elif method == 'uniform':
        # 均勻權重
        weights = np.ones(len(class_counts))
    else:
        raise ValueError(f"不支援的 method: {method}. 請使用 'sqrt_inverse' 或 'uniform'.")
    
    # 正規化
    weights = weights / weights.sum() * len(class_counts)
    
    # 插入 Type 0 的權重 (ignore_index)
    full_weights = np.insert(weights, 0, 0.0)
    
    return torch.tensor(full_weights, dtype=torch.float32)


def edit_distance(seq1, seq2):
    """
    Calculate edit distance (Levenshtein distance) between two sequences.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                  dp[i][j-1],    # insertion
                                  dp[i-1][j-1])  # substitution
    
    return dp[m][n] / max(m, n)


def normalized_dtw_ndim_distance(seq_pred, seq_gt):
    """
    Compute n-dimensional DTW distance normalized by sqrt(warping path length).
    """
    from dtaidistance import dtw_ndim

    dtw_dist = dtw_ndim.distance(seq_pred, seq_gt)
    warping_path = dtw_ndim.warping_path(seq_pred, seq_gt)
    if not warping_path:
        return float('inf')
    return float(dtw_dist / np.sqrt(len(warping_path)))


def pad_and_convert_to_tensor(merged_data, max_length):
    """
    將合併後的資料填充到相同的長度並轉換成 PyTorch tensor。

    參數：
    merged_data: list of merged rallies
    max_length: 最大的 rally 長度

    返回：
    tensor_data: PyTorch tensor with shape (max_length, batch_size, 23)
    """
    batch_size = len(merged_data)
    padded_data = []

    for rally in merged_data:
        rally_len = len(rally)
        padded_rally = rally + [[0]*23] * (max_length - rally_len)
        padded_data.append(padded_rally)

    # Convert to tensor and transpose to (max_length, batch_size, 23)
    tensor_data = torch.tensor(padded_data).transpose(0, 1)
    return tensor_data

def find_first_zero_index(lst):
    try:
        index = lst.index(0)
        return index
    except ValueError:
        return -1

def onehot(value, depth):
    a = np.zeros([depth])
    a[int(value)] = 1
    return a

def set_seed(seed_value):
    # random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder "{folder_path}" created.')
    else:
        print(f'Folder "{folder_path}" already exists.')

match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
player_state = ['player_location_x', 'player_location_y']
ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y', "player"]
state_col = match_state + player_state + ball_state + opponent_state
action_col =  ['landing_x', 'landing_y', 'player_type','moving_x','moving_y']
state_action_dividing = len(state_col)

def coord2area(point_x, point_y):
    mistake_landing_area = 33

    point_x = (point_x * (355/2)) + (355/2)
    point_y = (point_y * 240) + 240

    area1 = [[50,150],[104,204],1]
    area2 = [[104,150],[177.5,204],2]
    area3 = [[177.5,150],[251,204],3]
    area4 = [[251,150],[305,204],4]
    row1 = [area1, area2, area3, area4]

    area5 = [[50,204],[104,258],5]
    area6 = [[104,204],[177.5,258],6]
    area7 = [[177.5,204],[251,258],7]
    area8 = [[251,204],[305,258],8]
    row2 = [area5, area6, area7, area8]

    area9 = [[50,258],[104,312],9]
    area10 = [[104,258],[177.5,312],10]
    area11 = [[177.5,258],[251,312],11]
    area12 = [[251,258],[305,312],12]
    row3 = [area9, area10, area11, area12]
    
    area13 = [[50,312],[104, 366],13]
    area14 = [[104,312],[177.5,366],14]
    area15 = [[177.5,312],[251,366],15]
    area16 = [[251,312],[305,366],16]
    row4 = [area13, area14, area15, area16]

    area17 = [[50,366],[104,423],17]
    area18 = [[104,366],[177.5,423],18]
    area19 = [[177.5,366],[251,423],19]
    area20 = [[251,366],[305,423],20]
    row5 = [area17, area18, area19, area20]

    area21 = [[50,423],[104,480],21]
    area22 = [[104,423],[177.5,480],22]
    area23 = [[177.5,423],[251,480],23]
    area24 = [[251,423],[305,480],24]
    row6 = [area21, area22, area23, area24]

    area25 = [[305,366],[355,480],25]
    area26 = [[305,204],[355,366],26]
    area27 = [[305,0],[355,204],27]
    area28 = [[177.5,0],[305,150],28]
    row7 = [area25, area26, area27, area28]

    area29 = [[0,366],[50,480],29]
    area30 = [[0,204],[50,366],30]
    area31 = [[0,0],[50,204],31]
    area32 = [[50,0],[177.5,150],32]
    row8 = [area29, area30, area31, area32]

    check_area_list = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8
    hit_area = mistake_landing_area
    for check_area in check_area_list:
        if point_x >= check_area[0][0] and point_y >= check_area[0][1] and point_x <= check_area[1][0] and point_y <= check_area[1][1]:
            hit_area = check_area[2]
    return hit_area

def list_subtraction(p1,p2):
    point1 = p1.copy()
    point2 = p2.copy()
    v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
    return v[0], v[1]


def terminate(state):
    state = state.clone().tolist()
    hit_error = False
    if coord2area(state[10], state[11]) > 24:
        hit_error = True
    if ((hit_error == True) or (state[10] == 0 and state[11] == 1) or (state[12] == 11)):
        return True
    else:
        return False

def transition(player_action, player_current_state, target_number): # Calculate the opponent's state based on state and action
    player_action = player_action.tolist()
    player_current_state = player_current_state.tolist()
    
    return_state_list = player_current_state.copy()
    opponent_score = return_state_list[2]
    player_score = return_state_list[3]
    score_status = player_score - opponent_score
    return_state_list[2] = player_score
    return_state_list[3] = opponent_score
    return_state_list[4] = score_status
    return_state_list[1] = return_state_list[1] + 1 # ball round +1
    hit_x = player_action[0]
    hit_y = player_action[1]
    opponent_type = player_action[2]
    opponent_location_x = player_action[3]
    opponent_location_y = player_action[4]
    return_state_list[5] = player_current_state[13]
    return_state_list[6] = player_current_state[14]
    return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
    return_state_list[8], return_state_list[9] = list_subtraction([player_action[0], player_action[1]],[player_current_state[13],player_current_state[14]])
    return_state_list[10] = hit_x
    return_state_list[11] = hit_y
    return_state_list[12] = opponent_type
    return_state_list[13] = opponent_location_x
    return_state_list[14] = opponent_location_y
    return_state_list[15], return_state_list[16] = list_subtraction([player_action[3], player_action[4]],[player_current_state[5],player_current_state[6]])
    return_state_list[17] = target_number
    
    return torch.FloatTensor(return_state_list)
