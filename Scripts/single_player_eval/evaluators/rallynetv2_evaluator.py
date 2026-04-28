"""
RallyNetV2 Evaluator for Single Player
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import CTCLoss
from dtaidistance import dtw_ndim

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_path)

from .base_evaluator import BaseEvaluator
from Agent.rallynetv2 import LatentSDE
from Agent.eval_rallynetv2 import build_state_tensor_from_row, build_action_tensor_from_row
from Utils.helper import state_col, action_col, transition, terminate, edit_distance


class RallyNetV2Evaluator(BaseEvaluator):
    """RallyNetV2單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 model_path=None, hyperparameters_path=None):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots
            model_path: 模型路徑（可選）
            hyperparameters_path: hyperparameters路徑（可選）
        """
        super().__init__(player_name, experiment, device, skip_first_n)
        
        # 設定模型路徑
        if model_path is None:
            self.model_path = 'Experiments/rallynetv2/weight/gen_e_30000.trc'
        else:
            self.model_path = model_path
        
        if hyperparameters_path is None:
            self.hyperparameters_path = 'Experiments/rallynetv2/hyperparameters.pkl'
        else:
            self.hyperparameters_path = hyperparameters_path
        
        # 載入player2index
        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)
        
        if self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")
        
        self.player_id = self.player2index[self.player_name]
        
        # 載入模型
        self.model = self._load_model()
        
        # 載入資料
        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        
        # 準備時間步
        self.ts = self._prepare_ts()
    
    def _load_model(self):
        """載入RallyNetV2模型"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        # 載入hyperparameters
        with open(self.hyperparameters_path, 'rb') as f:
            hyperparameters = pickle.load(f)
        
        self.hyperparameters = hyperparameters
        
        model = LatentSDE(
            data_size=hyperparameters['data_size'],
            latent_size=hyperparameters['latent_size'],
            context_size=hyperparameters['context_size'],
            hidden_size=hyperparameters['hidden_size'],
            player_ids_len=hyperparameters['player_ids_len'],
            shot_type_len=hyperparameters['shot_type_len']
        ).to(device)
        
        model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        model.eval()
        
        return model
    
    def _prepare_ts(self):
        """準備時間步tensor"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        ts = torch.linspace(0, 71, steps=71, device=device)
        min_val = torch.min(ts)
        max_val = torch.max(ts)
        normalized_tensor = (ts - min_val) / (max_val - min_val)
        return normalized_tensor * 2
    
    def get_algorithm_name(self):
        return "rallynetv2"
    
    def evaluate(self):
        """執行RallyNetV2評估"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        # 取得選手的測試比賽
        test_match_ids = self.get_player_test_matches()
        
        ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        rally_land_dtw_list = []
        rally_move_dtw_list = []
        rally_shot_ctc_list = []
        rally_edit_dist_list = []
        total_rallies = 0
        total_shots = 0
        
        for match_id in tqdm(test_match_ids, desc=f"Matches for {self.player_name}"):
            match_df = self.df[self.df['match_id'] == match_id].copy()
            if len(match_df) == 0:
                continue
            
            # 找出對手
            players_in_match = match_df['player'].unique()
            opponent_name = None
            for p in players_in_match:
                if p != self.player_name:
                    opponent_name = p
                    break
            
            if opponent_name is None or opponent_name not in self.player2index:
                continue
            
            opponent_player_id = self.player2index[opponent_name]
            
            # 分組rallies
            rallies = match_df.groupby(['set', 'rally'])
            
            for (set_num, rally_num), rally_df in rallies:
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                
                if len(rally_df) == 0:
                    continue
                
                rally_land_pred_seq = []
                rally_land_gt_seq = []
                rally_move_pred_seq = []
                rally_move_gt_seq = []
                rally_shot_logits_seq = []
                rally_shot_gt_seq = []
                rally_shot_pred_seq = []
                
                target_player_shot_count = 0
                
                # 初始化第一個state
                first_row = rally_df.iloc[0]
                current_player_name = first_row['player']
                current_player_id = self.player_id if current_player_name == self.player_name else opponent_player_id
                current_state = build_state_tensor_from_row(first_row, current_player_id).to(device)
                
                # 初始化ht tensor
                rally_length = len(rally_df)
                ht = torch.zeros(rally_length, 1, 18, device=device)
                
                rally_done = False
                shot_idx = 0
                
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == self.player_name)
                    current_player_id = self.player_id if is_target_turn else opponent_player_id
                    
                    current_state[17] = current_player_id
                    launch = (current_state[1].item() == 1)
                    
                    # Update ht
                    ht[shot_idx, :, :] = current_state
                    
                    with torch.no_grad():
                        _xs, inv_action, inv_shot_logit, mlp_action, mlp_shot_logit = self.model.sample(
                            ht=ht,
                            ts=self.ts,
                            step=shot_idx,
                            player_ids_len=self.hyperparameters['player_ids_len'],
                            shot_type_len=self.hyperparameters['shot_type_len']
                        )
                        
                        # 使用inverse model的輸出
                        land_logit = inv_action[0:2].unsqueeze(0)
                        shot_type = inv_action[2].long().unsqueeze(0)
                        move_logit = inv_action[3:5].unsqueeze(0)
                        shot_logit = mlp_shot_logit
                    
                    # Apply mask
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    shot_probs_masked = shot_logit * mask.float()
                    shot_probs_masked = shot_probs_masked / (shot_probs_masked.sum() + 1e-8)
                    shot_type_masked = torch.argmax(shot_probs_masked).unsqueeze(0)
                    
                    if is_target_turn:
                        target_player_shot_count += 1
                        if target_player_shot_count > self.skip_first_n:
                            rally_land_pred_seq.append(land_logit.squeeze(0).cpu().numpy())
                            gt_action = build_action_tensor_from_row(row)
                            rally_land_gt_seq.append(gt_action[0:2].numpy())
                            rally_move_pred_seq.append(move_logit.squeeze(0).cpu().numpy())
                            rally_move_gt_seq.append(gt_action[3:5].numpy())
                            rally_shot_logits_seq.append(shot_logit.squeeze(0).cpu())
                            rally_shot_gt_seq.append(int(gt_action[2].item()))
                            rally_shot_pred_seq.append(int(shot_type_masked.item()))
                    
                    # Build action tensor
                    action_tensor = torch.tensor([
                        land_logit[0, 0].item(), land_logit[0, 1].item(),
                        float(shot_type_masked.item()),
                        move_logit[0, 0].item(), move_logit[0, 1].item()
                    ], dtype=torch.float32)
                    
                    # State transition
                    other_player_id = opponent_player_id if is_target_turn else self.player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                    
                    if terminate(next_state.cpu()):
                        rally_done = True
                    
                    current_state = next_state.to(device)
                    shot_idx += 1
                
                # Calculate rally-level metrics
                if len(rally_land_gt_seq) > 0:
                    land_pred_array = np.array(rally_land_pred_seq)
                    land_gt_array = np.array(rally_land_gt_seq)
                    land_dtw = dtw_ndim.distance(land_pred_array, land_gt_array) / len(land_gt_array)
                    rally_land_dtw_list.append(land_dtw)
                    
                    move_pred_array = np.array(rally_move_pred_seq)
                    move_gt_array = np.array(rally_move_gt_seq)
                    move_dtw = dtw_ndim.distance(move_pred_array, move_gt_array) / len(move_gt_array)
                    rally_move_dtw_list.append(move_dtw)
                    
                    rally_shot_logits_tensor = torch.stack(rally_shot_logits_seq)
                    rally_shot_targets_tensor = torch.tensor(rally_shot_gt_seq, dtype=torch.long)
                    
                    log_probs = F.log_softmax(rally_shot_logits_tensor.unsqueeze(1), dim=-1)
                    input_length = torch.tensor([log_probs.shape[0]], dtype=torch.long)
                    target_length = torch.tensor([len(rally_shot_targets_tensor)], dtype=torch.long)
                    
                    rally_ctc = ctc_loss_fn(log_probs, rally_shot_targets_tensor, input_length, target_length)
                    rally_shot_ctc_list.append(rally_ctc.item())
                    
                    rally_edit_dist = edit_distance(rally_shot_pred_seq, rally_shot_gt_seq)
                    rally_edit_dist_list.append(rally_edit_dist)
                    
                    total_rallies += 1
                    total_shots += len(rally_shot_gt_seq)
        
        # Build results
        if total_rallies > 0:
            results = {
                self.player_name: {
                    'land_dtw': float(np.mean(rally_land_dtw_list)),
                    'move_dtw': float(np.mean(rally_move_dtw_list)),
                    'shot_ctc': float(np.mean(rally_shot_ctc_list)),
                    'edit_dist': float(np.mean(rally_edit_dist_list)),
                    'num_rallies': total_rallies,
                    'num_shots': total_shots
                }
            }
        else:
            results = {
                self.player_name: {
                    'land_dtw': float('inf'),
                    'move_dtw': float('inf'),
                    'shot_ctc': float('inf'),
                    'edit_dist': float('inf'),
                    'num_rallies': 0,
                    'num_shots': 0
                }
            }
        
        return results
