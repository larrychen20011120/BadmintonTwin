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
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
    build_rollout_metadata,
)
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
import torch.nn.functional as F


class BCEvaluator(BaseEvaluator):
    """BC模型單一選手評估器"""
    
    def __init__(self, player_name=None, experiment='rally', device='cuda', skip_first_n=0,
                 model_path=None, save_trajectories=False,
                 distribution=None, gmm_num=None, all_rallies=False):
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
        self.all_rallies = all_rallies

        if distribution is None:
            distribution = self.config['bc_training'].get('distribution', 'deterministic')

        if distribution not in ('deterministic', 'gaussian', 'gmm'):
            raise ValueError('distribution must be deterministic, gaussian, or gmm')
        self.distribution = distribution
        self.gmm_num = gmm_num or self.config['bc_training'].get('gmm_num', 5)
        
        # 設定模型路徑
        if model_path is None:
            self.model_path = self.config['paths']['bc']['model']
        else:
            self.model_path = model_path
        
        # 載入player2index
        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)
        
        if not self.all_rallies and self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")
        
        self.player_id = self.player2index.get(self.player_name, 0)
        
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
        """Simulate each selected physical rally exactly once."""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        records = dataset['test']
        if not self.all_rallies:
            records = [record for record in records if self.player_name in record['players']]

        selected_rows = filter_rally_rows(self.df, records)
        rally_frames = {
            tuple(int(value) for value in key): rally_df.sort_values('ball_round').reset_index(drop=True)
            for key, rally_df in selected_rows.groupby(['match_id', 'set', 'rally'], sort=False)
        }
        missing_keys = sorted({rally_key(record) for record in records} - set(rally_frames))
        if missing_keys:
            raise ValueError(f'No model-ready rows found for test rallies: {missing_keys[:5]}')

        rally_trajectories = []
        seen = set()
        description = 'Simulating physical rallies' if self.all_rallies else f'Simulating {self.player_name}'
        for record in tqdm(records, desc=description):
            key = rally_key(record)
            if key in seen:
                raise ValueError(f'Duplicate physical test rally: {key}')
            seen.add(key)
            match_id, set_num, rally_num = key
            rally_df = rally_frames[key]

            players = list(dict.fromkeys(rally_df['player'].astype(str)))
            if len(players) != 2:
                raise ValueError(f'Expected two players for rally {key}, found {players}')
            missing_players = [player for player in players if player not in self.player2index]
            if missing_players:
                raise ValueError(f'Players missing from player map for rally {key}: {missing_players}')
            player_ids = {player: self.player2index[player] for player in players}
            id_to_player = {player_id: player for player, player_id in player_ids.items()}

            rally_gt_full_seq = [
                build_gt_shot_from_row(row) for _, row in rally_df.iterrows()
            ]
            max_rollout_steps = len(rally_gt_full_seq) + 2
            rally_pred_full_seq = []
            first_row = rally_df.iloc[0]
            current_player_name = str(first_row['player'])
            current_player_id = player_ids[current_player_name]
            current_state = build_state_tensor_from_row(first_row, current_player_id).to(device)
            termination_reason = 'max_rollout_reached'

            for _ in range(max_rollout_steps):
                current_player_id = int(current_state[17].item())
                if current_player_id not in id_to_player:
                    raise ValueError(f'Unknown player id {current_player_id} during rally {key}')
                current_player_name = id_to_player[current_player_id]
                current_state[17] = current_player_id
                state_for_pred = current_state
                ball_round = int(state_for_pred[1].item())
                launch = (state_for_pred[1].item() == 1)

                with torch.no_grad():
                    land_logit, shot_logit, move_logit = self.model(state_for_pred.unsqueeze(0))
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    shot_probs = F.softmax(shot_logit, dim=-1)
                    masked_probs = shot_probs.masked_fill(~mask, 0.0)
                    sum_probs = masked_probs.sum(dim=-1, keepdim=True)
                    shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
                    shot_type = torch.multinomial(shot_probs, num_samples=1).squeeze(-1)

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

                action_tensor = torch.tensor([
                    land_2d[0], land_2d[1],
                    float(shot_type.item()),
                    move_2d[0], move_2d[1]
                ], dtype=torch.float32)
                other_player_id = next(
                    player_id for player_id in id_to_player if player_id != current_player_id
                )
                next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                if terminate(next_state.cpu()):
                    termination_reason = 'natural_terminate'
                    break
                current_state = next_state.to(device)

            target_player = None if self.all_rallies else self.player_name
            opponent_player = None
            if target_player in players:
                opponent_player = next(player for player in players if player != target_player)
            rally_trajectories.append({
                'match_id': match_id,
                'set': set_num,
                'rally': rally_num,
                'players': players,
                'player_profiles': record.get('player_profiles', {}),
                'instruction_mode': 'no_instruct',
                'target_player': target_player,
                'opponent_player': opponent_player,
                'gt_sequence': rally_gt_full_seq,
                'pred_sequence': rally_pred_full_seq,
                'target_player_won': None,
                **build_rollout_metadata(
                    rally_gt_full_seq,
                    rally_pred_full_seq,
                    termination_reason,
                ),
            })

        self._save_trajectories_if_needed(rally_trajectories)
        return rally_trajectories
