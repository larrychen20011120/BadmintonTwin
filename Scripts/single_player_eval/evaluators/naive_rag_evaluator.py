"""
NaiveRAG Evaluator for Single Player
"""

import os
import sys
import json
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
from Agent.naive_rag import NaiveRAGAgent, NaiveRAGDatabase
from Utils.helper import state_col, action_col, transition, terminate, edit_distance


class NaiveRAGEvaluator(BaseEvaluator):
    """NaiveRAG單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 train_split_path=None, top_k=5):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots
            train_split_path: train split路徑（用於建立database，可選）
            top_k: RAG檢索數量
        """
        super().__init__(player_name, experiment, device, skip_first_n)
        
        # 設定train split路徑
        if train_split_path is None:
            self.train_split_path = self.config['paths'][f'{experiment}_split']
        else:
            self.train_split_path = train_split_path
        
        self.top_k = top_k
        
        # 載入player2index
        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)
        
        if self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")
        
        self.player_id = self.player2index[self.player_name]
        
        # 載入資料
        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        
        # 建立Agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """建立NaiveRAG Agent"""
        from Utils.llm import LLM
        
        # 載入train split以建立database
        with open(self.train_split_path, 'r') as f:
            train_split = json.load(f)
        
        # 取得選手的train matches
        train_matches = train_split.get('train', {}).get(self.player_name, [])
        
        if not train_matches:
            raise ValueError(f"No training matches found for player '{self.player_name}'")
        
        # 篩選train data
        train_df = self.df[self.df['match_id'].isin(train_matches)].copy()
        
        if len(train_df) == 0:
            raise ValueError(f"No training data found for player '{self.player_name}'")
        
        # 建立database
        database = NaiveRAGDatabase(train_df, self.player_name, self.player_id)
        
        # 建立LLM client
        llm_config = self.config['naive_rag']
        llm = LLM(
            model_name=llm_config['model_name'],
            api_key=llm_config['api_key'],
            port=llm_config['port']
        )
        
        # 建立agent
        agent = NaiveRAGAgent(
            player_name=self.player_name,
            database=database,
            llm=llm,
            top_k=self.top_k
        )
        
        return agent
    
    def get_algorithm_name(self):
        return "naive_rag"
    
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
            'action': [None]  # 初始action為None
        }
    
    def evaluate(self):
        """執行NaiveRAG評估"""
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
                
                # 建立state tensor
                state_values = [first_row[col] for col in state_col]
                state_values[-1] = current_player_id
                current_state = torch.tensor(state_values, dtype=torch.float32)
                
                info = self._build_info(first_row)
                
                for shot_idx in range(len(rally_df)):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == self.player_name)
                    
                    # 更新state中的player id
                    current_player_id = self.player_id if is_target_turn else opponent_player_id
                    current_state[17] = current_player_id
                    
                    # 更新info
                    info['score'] = [row['player_score'], row['opponent_score']]
                    
                    launch = (current_state[1].item() == 1)
                    
                    # Agent預測
                    env_state = self._state_to_env_format(current_state)
                    action, reason = self.agent.action(env_state, info, launch)
                    
                    if action is None:
                        # Agent返回None，跳過這個rally
                        break
                    
                    output_shot, hit, output_land, output_move, dist = action
                    
                    # 轉換回normalized格式
                    land_logit = torch.tensor([
                        -output_land[0] / 177.5,
                        -(output_land[1] - 240) / 240
                    ]).unsqueeze(0)
                    
                    move_logit = torch.tensor([
                        output_move[0] / 177.5,
                        (output_move[1] + 240) / 240
                    ]).unsqueeze(0)
                    
                    # 建立shot logit（模擬分布）
                    shot_logit = torch.zeros(12)
                    if output_shot > 0 and output_shot <= 11:
                        shot_logit[output_shot] = 1.0
                    shot_logit = shot_logit.unsqueeze(0)
                    
                    if is_target_turn:
                        target_player_shot_count += 1
                        if target_player_shot_count > self.skip_first_n:
                            rally_land_pred_seq.append(land_logit.squeeze(0).numpy())
                            gt_action_values = [row[col] for col in action_col]
                            rally_land_gt_seq.append(np.array(gt_action_values[0:2]))
                            rally_move_pred_seq.append(move_logit.squeeze(0).numpy())
                            rally_move_gt_seq.append(np.array(gt_action_values[3:5]))
                            rally_shot_logits_seq.append(shot_logit.squeeze(0).cpu())
                            rally_shot_gt_seq.append(int(gt_action_values[2]))
                            rally_shot_pred_seq.append(int(output_shot))
                    
                    # Build action tensor for transition
                    action_tensor = torch.tensor([
                        land_logit[0, 0].item(), land_logit[0, 1].item(),
                        float(output_shot),
                        move_logit[0, 0].item(), move_logit[0, 1].item()
                    ], dtype=torch.float32)
                    
                    # State transition
                    other_player_id = opponent_player_id if is_target_turn else self.player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                    
                    if terminate(next_state.cpu()):
                        break
                    
                    current_state = next_state
                    info['action'] = [action]
                
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
