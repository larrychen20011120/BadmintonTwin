"""
Twin (I-MSDP) Evaluator for Single Player
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
from Agent.twin import TwinAgent, TwinDatabase
from Utils.helper import state_col, action_col, transition, terminate
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row, build_pred_shot


class TwinEvaluator(BaseEvaluator):
    """Twin (I-MSDP) 單一選手評估器"""

    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 train_split_path=None, top_k=5, max_scale=10, enable_tracing=True,
                 
                 teacher_forcing=False, save_trajectories=False):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            train_split_path: train split 路徑（用於建立 database，可選）
            top_k: 軌跡段檢索數量
            max_scale: I-MSDP 最大 suffix 長度
            enable_tracing: 是否啟用 Langfuse 追蹤
            teacher_forcing: 是否使用 teacher forcing 模式
        """
        super().__init__(player_name, experiment, device, skip_first_n,
                         teacher_forcing=teacher_forcing,
                         save_trajectories=save_trajectories)

        if train_split_path is None:
            self.train_split_path = self.config['paths'][f'{experiment}_split']
        else:
            self.train_split_path = train_split_path

        self.top_k = top_k
        self.max_scale = max_scale
        self.enable_tracing = enable_tracing

        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)

        if self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")

        self.player_id = self.player2index[self.player_name]

        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)

        self.agent = self._create_agent()

    def _create_agent(self):
        """建立 Twin Agent"""

        with open(self.train_split_path, 'r') as f:
            train_split = json.load(f)

        train_matches = train_split.get('train', {}).get(self.player_name, [])

        if not train_matches:
            raise ValueError(f"No training matches found for player '{self.player_name}'")

        train_df = self.df[self.df['match_id'].isin(train_matches)].copy()

        if len(train_df) == 0:
            raise ValueError(f"No training data found for player '{self.player_name}'")

        database = TwinDatabase(train_df, self.player_name, self.player_id,
                                max_scale=self.max_scale)

        twin_config = self.config.get('twin', {})

        agent = TwinAgent(
            player_name=self.player_name,
            database=database,
            model_name=twin_config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
            api_key=twin_config.get('api_key', 'EMPTY'),
            port=twin_config.get('port', 16000),
            top_k=self.top_k,
            max_scale=self.max_scale,
            enable_tracing=self.enable_tracing,
        )

        return agent

    def get_algorithm_name(self):
        return "twin"

    def _state_to_env_format(self, state_18):
        """Convert 18-dim state to env format"""
        opponent_type = int(state_18[12].item())
        player_pos = (state_18[5].item(), state_18[6].item())
        opponent_pos = (state_18[13].item(), state_18[14].item())
        landing_pos = (state_18[10].item(), state_18[11].item())
        return [opponent_type, player_pos, opponent_pos, landing_pos]

    def _build_info(self, row):
        """Build info dict from row"""
        return {
            'rally': row['rally'],
            'round': [row['ball_round']],
            'score': [row['player_score'], row['opponent_score']],
            'action': [None]
        }

    def simulate(self):
        """執行 Twin (I-MSDP) 模擬，收集trajectory"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        test_match_ids = self.get_player_test_matches()

        rally_trajectories = []

        for match_id in tqdm(test_match_ids, desc=f"Simulating {self.player_name}"):
            match_df = self.df[self.df['match_id'] == match_id].copy()
            if len(match_df) == 0:
                continue

            players_in_match = match_df['player'].unique()
            opponent_name = None
            for p in players_in_match:
                if p != self.player_name:
                    opponent_name = p
                    break

            if opponent_name is None or opponent_name not in self.player2index:
                continue

            opponent_player_id = self.player2index[opponent_name]

            rallies = match_df.groupby(['set', 'rally'])

            for (set_num, rally_num), rally_df in rallies:
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)

                if len(rally_df) == 0:
                    continue
                
                self.agent.reset_history()

                # 收集完整GT（雙方玩家，不受simulation影響）
                rally_gt_full_seq = []
                for _, row in rally_df.iterrows():
                    rally_gt_full_seq.append(build_gt_shot_from_row(row))
                
                # 收集預測（雙方玩家）
                rally_pred_full_seq = []

                target_player_won = None

                first_row = rally_df.iloc[0]

                current_player_name = first_row['player']
                current_player_id = self.player_id if current_player_name == self.player_name else opponent_player_id

                state_values = [first_row[col] for col in state_col]
                state_values[-1] = current_player_id
                current_state = torch.tensor(state_values, dtype=torch.float32)

                info = self._build_info(first_row)

                simulation_ended_early = False

                for shot_idx in range(len(rally_df)):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == self.player_name)

                    current_player_id = self.player_id if is_target_turn else opponent_player_id

                    if self.teacher_forcing:
                        tf_values = [row[col] for col in state_col]
                        tf_values[-1] = current_player_id
                        state_for_pred = torch.tensor(tf_values, dtype=torch.float32)
                    else:
                        current_state[17] = current_player_id
                        state_for_pred = current_state

                    info['score'] = [row['player_score'], row['opponent_score']]

                    launch = (state_for_pred[1].item() == 1)

                    env_state = self._state_to_env_format(state_for_pred)
                    action, reason = self.agent.predict(env_state, info, launch)

                    if action is None:
                        simulation_ended_early = True
                        break

                    output_shot, hit, output_land, output_move, dist = action

                    land_logit = torch.tensor(output_land).unsqueeze(0)
                    move_logit = torch.tensor(output_move).unsqueeze(0)

                    # 收集預測（雙方玩家）
                    land_2d = [land_logit.squeeze(0)[0].item(), land_logit.squeeze(0)[1].item()]
                    move_2d = [move_logit.squeeze(0)[0].item(), move_logit.squeeze(0)[1].item()]
                    rally_pred_full_seq.append(build_pred_shot(
                        ground_truth_player, int(output_shot), land_2d, move_2d, int(row['ball_round'])))

                    if self.teacher_forcing:
                        continue

                    action_tensor = torch.tensor([
                        land_logit[0, 0].item(), land_logit[0, 1].item(),
                        float(output_shot),
                        move_logit[0, 0].item(), move_logit[0, 1].item()
                    ], dtype=torch.float32)

                    other_player_id = opponent_player_id if is_target_turn else self.player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)

                    if terminate(next_state.cpu()):
                        simulation_ended_early = True
                        break

                    current_state = next_state
                    info['action'] = [action]

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
