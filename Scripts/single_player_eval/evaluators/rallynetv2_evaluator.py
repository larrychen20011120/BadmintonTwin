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

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_path)

from .base_evaluator import BaseEvaluator
from Agent.rallynetv2 import LatentSDE
from Utils.helper import state_col, action_col, transition, terminate
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row, build_pred_shot


def build_state_tensor_from_row(row, player_id):
    """
    Build state tensor from dataframe row using state_col ordering.
    """
    state_values = [row[col] for col in state_col]
    state_values[-1] = player_id
    return torch.tensor(state_values, dtype=torch.float32)


class RallyNetV2Evaluator(BaseEvaluator):
    """RallyNetV2單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 model_path=None, hyperparameters_path=None,
                 save_trajectories=False):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            model_path: 模型路徑（可選）
            hyperparameters_path: hyperparameters路徑（可選）
        """
        super().__init__(player_name, experiment, device, skip_first_n,
                         save_trajectories=save_trajectories)
        
        # 設定模型路徑
        if model_path is None:
            self.model_path = self.config['paths']['rallynetv2']['weight'][experiment]
        else:
            self.model_path = model_path
        
        if hyperparameters_path is None:
            self.hyperparameters_path = self.config['paths']['rallynetv2']['hyperparameters'][experiment]
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

        self.print_runtime_parameters(
            "rallynetv2",
            extra_rows=[
                ("rallynetv2.model_path", self.model_path),
                ("rallynetv2.hyperparameters_path", self.hyperparameters_path),
            ],
        )
    
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
            shot_type_len=hyperparameters['shot_type_len'],
            gmm_num=hyperparameters.get('gmm_num', 5),
            min_sigma=hyperparameters.get('min_sigma', 1e-3)
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
    
    def simulate(self):
        """執行RallyNetV2模擬，收集trajectory"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        launch_mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool, device=device)
        rally_mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool, device=device)
        
        # 取得選手的測試比賽
        test_match_ids = self.get_player_test_matches()
        target_max_len = self.get_player_max_sequence_length(self.player_name)
        if target_max_len == 0:
            return []
        
        rally_trajectories = []
        
        for match_id in tqdm(test_match_ids, desc=f"Simulating {self.player_name}"):
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
                
                # 收集完整GT（雙方玩家，不受simulation影響）
                rally_gt_full_seq = []
                for _, row in rally_df.iterrows():
                    rally_gt_full_seq.append(build_gt_shot_from_row(row))
                
                # 收集預測（雙方玩家）
                rally_pred_full_seq = []
                
                target_player_won = None

                first_row = rally_df.iloc[0]
                
                # 初始化第一個state
                current_player_name = first_row['player']
                current_player_id = self.player_id if current_player_name == self.player_name else opponent_player_id
                current_state = build_state_tensor_from_row(first_row, current_player_id).to(device)
                
                max_rollout_steps = max(target_max_len * 2 + 2, len(rally_df), 1)
                ht = torch.zeros(max_rollout_steps, 1, 18, device=device)
                
                rally_done = False
                shot_idx = 0
                simulation_ended_early = False
                target_pred_count = 0
                
                while not rally_done and target_pred_count < target_max_len and shot_idx < max_rollout_steps:
                    current_player_id = int(current_state[17].item())
                    is_target_turn = current_player_id == self.player_id
                    current_player_name = self.player_name if is_target_turn else opponent_name
                    current_player_id = self.player_id if is_target_turn else opponent_player_id
                    current_state[17] = current_player_id
                    launch = (current_state[1].item() == 1)
                    ball_round = int(current_state[1].item())
                    ht[shot_idx, :, :] = current_state
                    
                    with torch.inference_mode():
                        
                        # Use model.sample to get predictions with SDE sampling
                        # Pass ht[:seq_len] which has shape [seq_len, 1, 18]
                        _xs, inv_action, inv_shot_logit, mlp_action, mlp_shot_logit = self.model.sample(
                            ht=ht,
                            ts=self.ts,
                            step=shot_idx,
                            player_ids_len=self.hyperparameters['player_ids_len'],
                            shot_type_len=self.hyperparameters['shot_type_len']
                        )
                        # Extract predictions from mlp_action (shape: [5])
                        land_logit = mlp_action[0:2].unsqueeze(0)  # [1, 2]
                        shot_type = mlp_action[2].long().unsqueeze(0)  # [1]
                        move_logit = mlp_action[3:5].unsqueeze(0)  # [1, 2]
                        # land_logit = inv_action[0:2].unsqueeze(0)  # [1, 2]
                        # shot_type = inv_action[2].long().unsqueeze(0)  # [1]
                        # move_logit = inv_action[3:5].unsqueeze(0)  # [1, 2]
                        shot_logit = mlp_shot_logit
                    
                    # Apply mask for valid shot types
                    if launch:
                        mask = launch_mask
                    else:
                        mask = rally_mask
                    
                    # Re-apply mask to shot_probs and resample if needed
                    masked_logits = shot_logit.masked_fill(mask == 0, -1e9)
                    probs = torch.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    shot_type_masked = dist.sample()
                    
                    # 收集預測（雙方玩家）
                    land_2d = [land_logit.squeeze(0)[0].item(), land_logit.squeeze(0)[1].item()]
                    move_2d = [move_logit.squeeze(0)[0].item(), move_logit.squeeze(0)[1].item()]
                    rally_pred_full_seq.append(build_pred_shot(
                        current_player_name, int(shot_type_masked.item()), land_2d, move_2d, ball_round))
                    if is_target_turn:
                        target_pred_count += 1
                    
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
                        simulation_ended_early = True
                    
                    current_state = next_state.to(device)
                    shot_idx += 1
                
                # Save rally trajectory (包含雙方玩家，GT完整長度)
                if self.save_trajectories:
                    rally_trajectories.append({
                        'match_id': match_id,
                        'set': int(set_num),
                        'rally': int(rally_num),
                        'target_player': self.player_name,
                        'opponent_player': opponent_name,
                        'gt_sequence': rally_gt_full_seq,
                        'pred_sequence': rally_pred_full_seq,
                        'target_player_won': target_player_won,
                        'rally_length_gt': len(rally_gt_full_seq),
                        'rally_length_pred': len(rally_pred_full_seq),
                        'simulation_ended_early': simulation_ended_early,
                    })
        
        # Save full trajectories
        self._save_trajectories_if_needed(rally_trajectories)
        
        return rally_trajectories
