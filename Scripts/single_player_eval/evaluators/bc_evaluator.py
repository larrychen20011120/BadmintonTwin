"""
BC Evaluator for Single Player
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_path)

from .base_evaluator import BaseEvaluator
from Agent.bc import BC, build_state_tensor_from_row, build_action_tensor_from_row
from Utils.helper import state_col, action_col, transition, terminate, edit_distance
from torch.nn import CTCLoss
from dtaidistance import dtw_ndim
import torch.nn.functional as F


class BCEvaluator(BaseEvaluator):
    """BC模型單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0, 
                 model_path=None):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots
            model_path: 模型路徑（可選，預設使用config中的路徑）
        """
        super().__init__(player_name, experiment, device, skip_first_n)
        
        # 設定模型路徑
        if model_path is None:
            self.model_path = self.config['paths']['bc']['models'][experiment]
        else:
            self.model_path = model_path
        
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
    
    def _load_model(self):
        """載入BC模型"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        model = BC(
            input_size=32,  # embedded state size
            hidden_size=self.config['bc_training']['hidden_dim'],
            player_id_len=self.config['bc_training']['player_id_len'],
            shot_type_len=self.config['bc_training']['shot_type_len'],
            player_id=self.player_id
        ).to(device)
        
        model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        model.eval()
        
        return model
    
    def get_algorithm_name(self):
        return "bc"
    
    def evaluate(self):
        """執行BC模型評估"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        # 取得選手的測試比賽
        test_match_ids = self.get_player_test_matches()
        
        # 收集該選手的所有rallies
        rallies_data = []
        
        for match_id in test_match_ids:
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
                rallies_data.append({
                    'rally_df': rally_df.sort_values('ball_round').reset_index(drop=True),
                    'opponent_player_id': opponent_player_id,
                    'opponent_name': opponent_name
                })
        
        if not rallies_data:
            return {
                self.player_name: {
                    'land_dtw': float('inf'),
                    'move_dtw': float('inf'),
                    'shot_ctc': float('inf'),
                    'edit_dist': float('inf'),
                    'num_rallies': 0,
                    'num_shots': 0
                }
            }
        
        # 評估每個rally
        ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        rally_land_dtw_list = []
        rally_move_dtw_list = []
        rally_shot_ctc_list = []
        rally_edit_dist_list = []
        total_rallies = 0
        total_shots = 0
        
        for rally_info in tqdm(rallies_data, desc=f"Evaluating {self.player_name}"):
            rally_df = rally_info['rally_df']
            opponent_player_id = rally_info['opponent_player_id']
            
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
            
            for shot_idx in range(len(rally_df)):
                row = rally_df.iloc[shot_idx]
                ground_truth_player = row['player']
                is_target_turn = (ground_truth_player == self.player_name)
                
                current_player_id = self.player_id if is_target_turn else opponent_player_id
                current_state[17] = current_player_id
                
                launch = (current_state[1].item() == 1)
                
                # 模型預測
                with torch.no_grad():
                    land_logit, shot_logit, move_logit = self.model(current_state.unsqueeze(0))
                    
                    # Apply mask
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    shot_probs = F.softmax(shot_logit, dim=-1)
                    masked_probs = shot_probs.masked_fill(~mask, 0.0)
                    sum_probs = masked_probs.sum(dim=-1, keepdim=True)
                    shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
                    shot_type = torch.argmax(shot_probs, dim=-1)
                
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
                        rally_shot_pred_seq.append(int(shot_type.item()))
                
                # Build action tensor
                action_tensor = torch.tensor([
                    land_logit[0, 0].item(), land_logit[0, 1].item(),
                    float(shot_type.item()),
                    move_logit[0, 0].item(), move_logit[0, 1].item()
                ], dtype=torch.float32)
                
                # State transition
                other_player_id = opponent_player_id if is_target_turn else self.player_id
                next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                
                if terminate(next_state.cpu()):
                    break
                
                current_state = next_state.to(device)
            
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
