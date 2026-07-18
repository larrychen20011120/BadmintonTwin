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
from Agent.bc import BC, build_state_tensor_from_row, sample_from_gaussian, sample_from_gmm
from Utils.helper import state_col, action_col, transition, terminate
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row, build_pred_shot
import torch.nn.functional as F


class BCEvaluator(BaseEvaluator):
    """BC模型單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0, 
                 model_path=None, save_trajectories=False,
                 distribution=None, gmm_num=None):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            model_path: 模型路徑（可選，預設使用config中的路徑）
            distribution: 輸出模式，可選 deterministic、gaussian 或 gmm
            save_trajectories: 是否儲存原始軌跡
        """
        super().__init__(player_name, experiment, device, skip_first_n,
                         save_trajectories=save_trajectories)

        if distribution is None:
            distribution = self.config['bc_training'].get('distribution', 'deterministic')

        if distribution not in ('deterministic', 'gaussian', 'gmm'):
            raise ValueError('distribution must be deterministic, gaussian, or gmm')
        self.distribution = distribution
        self.gmm_num = gmm_num or self.config['bc_training'].get('gmm_num', 5)
        
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

        self.print_runtime_parameters(
            "bc",
            distribution=self.distribution,
            gmm_num=self.gmm_num,
            extra_rows=[("bc.model_path", self.model_path)],
        )
    
    def _load_model(self):
        """載入BC模型"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        model = BC(
            input_size=32,  # embedded state size
            hidden_size=self.config['bc_training']['hidden_dim'],
            player_id_len=self.config['bc_training']['player_id_len'],
            shot_type_len=self.config['bc_training']['shot_type_len'],
            player_id=self.player_id,
            distribution=self.distribution,
            gmm_num=self.gmm_num
        ).to(device)
        
        model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        model.eval()
        
        return model
    
    def get_algorithm_name(self):
        return "bc"
    
    def simulate(self):
        """執行BC模型模擬，收集trajectory"""
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
                    'opponent_name': opponent_name,
                    'match_id': match_id,
                    'set_num': int(set_num),
                    'rally_num': int(rally_num),
                })
        
        if not rallies_data:
            return []

        target_max_len = self.get_player_max_sequence_length(self.player_name)
        if target_max_len == 0:
            return []
        
        rally_trajectories = []
        
        for rally_info in tqdm(rallies_data, desc=f"Simulating {self.player_name}"):
            rally_df = rally_info['rally_df']
            opponent_player_id = rally_info['opponent_player_id']
            opponent_name = rally_info['opponent_name']
            match_id = rally_info['match_id']
            set_num = rally_info['set_num']
            rally_num = rally_info['rally_num']
            
            # 收集完整GT（雙方玩家，不受simulation影響）
            rally_gt_full_seq = []
            for _, row in rally_df.iterrows():
                rally_gt_full_seq.append(build_gt_shot_from_row(row))
            
            # 收集預測（雙方玩家）
            rally_pred_full_seq = []
            
            # Determine target player's score at rally start
            first_row = rally_df.iloc[0]
            target_player_won = None
            
            # 初始化第一個state
            current_player_name = first_row['player']
            current_player_id = self.player_id if current_player_name == self.player_name else opponent_player_id
            current_state = build_state_tensor_from_row(first_row, current_player_id).to(device)
            
            simulation_ended_early = False
            target_pred_count = 0
            
            while target_pred_count < target_max_len:
                current_player_id = int(current_state[17].item())
                is_target_turn = current_player_id == self.player_id
                current_player_name = self.player_name if is_target_turn else opponent_name
                current_player_id = self.player_id if is_target_turn else opponent_player_id
                current_state[17] = current_player_id
                state_for_pred = current_state
                ball_round = int(state_for_pred[1].item())
                
                launch = (state_for_pred[1].item() == 1)
                
                # 模型預測
                with torch.no_grad():
                    land_logit, shot_logit, move_logit = self.model(state_for_pred.unsqueeze(0))
                    
                    # Apply mask
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    shot_probs = F.softmax(shot_logit, dim=-1)
                    masked_probs = shot_probs.masked_fill(~mask, 0.0)
                    sum_probs = masked_probs.sum(dim=-1, keepdim=True)
                    shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
                    # shot_type = torch.argmax(shot_probs, dim=-1)
                    # use categorical sampling to allow for more diverse predictions
                    shot_type = torch.multinomial(shot_probs, num_samples=1).squeeze(-1)

                
                # 收集預測（雙方玩家）
                if self.distribution == 'gmm':
                    land_2d = sample_from_gmm(
                        land_logit,
                        self.model.gmm_num,
                        self.model.min_sigma
                    )[0].tolist()
                    move_2d = sample_from_gmm(
                        move_logit,
                        self.model.gmm_num,
                        self.model.min_sigma
                    )[0].tolist()
                elif self.distribution == 'gaussian':
                    land_2d = sample_from_gaussian(land_logit)[0].tolist()
                    move_2d = sample_from_gaussian(move_logit)[0].tolist()
                else:
                    land_2d = [land_logit[0, 0].item(), land_logit[0, 1].item()]
                    move_2d = [move_logit[0, 0].item(), move_logit[0, 1].item()]
                rally_pred_full_seq.append(build_pred_shot(
                    current_player_name, int(shot_type.item()), land_2d, move_2d, ball_round))

                if is_target_turn:
                    target_pred_count += 1
                
                # Build action tensor
                action_tensor = torch.tensor([
                    land_2d[0], land_2d[1],
                    float(shot_type.item()),
                    move_2d[0], move_2d[1]
                ], dtype=torch.float32)
                
                # State transition
                other_player_id = opponent_player_id if is_target_turn else self.player_id
                next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                
                if terminate(next_state.cpu()):
                    simulation_ended_early = True
                    break
                
                current_state = next_state.to(device)
            
            # Save rally trajectory (包含雙方玩家，GT完整長度)
            if self.save_trajectories:
                rally_trajectories.append({
                    'match_id': match_id,
                    'set': set_num,
                    'rally': rally_num,
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
