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

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_path)

from .base_evaluator import BaseEvaluator
from Agent.naive_rag import NaiveRAGAgent, NaiveRAGDatabase
from Utils.helper import state_col, action_col, transition, terminate
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row, build_pred_shot


class NaiveRAGEvaluator(BaseEvaluator):
    """NaiveRAG單一選手評估器"""
    
    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                  train_split_path=None, top_k=None, enable_tracing=True,
                  save_trajectories=False,
                 style_shift_instruction_mode='auto', verbose_style_shift=False,
                 output_experiment=None):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            train_split_path: train split路徑（用於建立database，可選）
            top_k: RAG檢索數量
            enable_tracing: 是否啟用 Langfuse 追蹤
        """
        super().__init__(player_name, experiment, device, skip_first_n,
                         save_trajectories=save_trajectories,
                         style_shift_instruction_mode=style_shift_instruction_mode,
                         verbose_style_shift=verbose_style_shift,
                         output_experiment=output_experiment)
        
        # 設定train split路徑
        if train_split_path is None:
            self.train_split_path = self.config['paths'][f'{experiment}_split']
        else:
            self.train_split_path = train_split_path
        
        llm_config = self.config.get('naive_rag', {})
        retrieval_config = self.config.get('retrieval', {})
        self.retrieval_mode = retrieval_config.get('mode', 'classic')
        self.top_k = top_k if top_k is not None else retrieval_config.get('classic_top_k', retrieval_config.get('top_k', llm_config.get('top_k', 5)))
        self.normal_top_k = retrieval_config.get('normal_top_k', self.top_k)
        self.miss_top_k = retrieval_config.get('miss_top_k', max(1, min(5, self.top_k)))
        self.enable_tracing = enable_tracing
        self.database_dir = self.config['paths']['naive_rag']['data'][experiment]
        
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
        
        # 建立/快取每位選手各自的 Agent
        self.agents = {}
        self.agent = self._get_agent(self.player_name)

        self.print_runtime_parameters(
            "naive_rag",
            top_k=self.top_k,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_mode,
            include_llm=True,
            extra_rows=[
                ("database_dir", self.database_dir),
                ("prompt_variant", self.prompt_variant),
            ],
        )
    
    def _create_agent(self, player_name, player_id):
        """建立NaiveRAG Agent"""
        db_path = self._database_path(player_name)
        print(f"[NaiveRAG] Loading database for {player_name}: {db_path}")
        database = NaiveRAGDatabase.load(db_path)
        print(f"[NaiveRAG] Database ready for {player_name}: {len(database.data)} entries")
        
        # 建立LLM client
        llm_config = self.config.get('naive_rag', {})
        
        # 建立agent
        agent = NaiveRAGAgent(
            player_name=player_name,
            database=database,
            model_name=llm_config.get('model_name'),
            api_key=llm_config.get('api_key'),
            host=llm_config.get('host'),
            port=llm_config.get('port'),
            base_url=llm_config.get('base_url'),
            host_url=llm_config.get('host_url'),
            top_k=self.top_k,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_mode,
            enable_tracing=self.enable_tracing,
            prompt_variant=self.prompt_variant,
        )
        print(f"[NaiveRAG] Agent created for {player_name}")
        
        return agent

    def _database_path(self, player_name):
        safe_name = player_name.replace(" ", "_").replace(".", "")
        return os.path.join(self.database_dir, f"{safe_name}_db.pkl")

    def _get_agent(self, player_name):
        """取得指定選手的 Agent；target/opponent 必須使用各自的 player_name。"""
        if player_name not in self.player2index:
            raise ValueError(f"Player '{player_name}' not found in player map")

        if player_name not in self.agents:
            player_id = self.player2index[player_name]
            self.agents[player_name] = self._create_agent(player_name, player_id)

        return self.agents[player_name]
    
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
            'match_id': row['match_id'],
            'rally': row['rally'],
            'round': [row['ball_round']],
            'score': [row['player_score'], row['opponent_score']],
            'action': [None]  # 初始action為None
        }

    def _build_history_stroke(self, state_for_pred, player_name, action):
        """Build one completed stroke for the shared full-rally history."""
        output_shot, _, output_land, output_move, _ = action
        return {
            'player': player_name,
            'rally': int(state_for_pred[0].item()),
            'ball_round': int(state_for_pred[1].item()),
            'player_score': int(state_for_pred[2].item()),
            'opponent_score': int(state_for_pred[3].item()),
            'score_status': int(state_for_pred[4].item()),
            'player_location_x': float(state_for_pred[5].item()),
            'player_location_y': float(state_for_pred[6].item()),
            'ball_distance': float(state_for_pred[7].item()),
            'ball_distance_x': float(state_for_pred[8].item()),
            'ball_distance_y': float(state_for_pred[9].item()),
            'hit_x': float(state_for_pred[10].item()),
            'hit_y': float(state_for_pred[11].item()),
            'opponent_type': int(state_for_pred[12].item()),
            'opponent_location_x': float(state_for_pred[13].item()),
            'opponent_location_y': float(state_for_pred[14].item()),
            'opponent_move_x': float(state_for_pred[15].item()),
            'opponent_move_y': float(state_for_pred[16].item()),
            'landing_x': float(output_land[0]),
            'landing_y': float(output_land[1]),
            'player_type': int(output_shot),
            'moving_x': float(output_move[0]),
            'moving_y': float(output_move[1]),
        }
    
    def simulate(self):
        """執行NaiveRAG模擬，收集trajectory"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
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
            opponent_agent = self._get_agent(opponent_name)
            
            # 分組rallies
            rallies = match_df.groupby(['set', 'rally'])
            
            for (set_num, rally_num), rally_df in rallies:
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                
                if len(rally_df) == 0:
                    continue
                
                self.agent.reset_history()
                opponent_agent.reset_history()
                
                # 收集完整GT（雙方玩家，不受simulation影響）
                rally_gt_full_seq = []
                for _, row in rally_df.iterrows():
                    rally_gt_full_seq.append(build_gt_shot_from_row(row))
                
                # 收集預測（雙方玩家）
                rally_pred_full_seq = []
                rally_history = []
                
                target_player_won = None
                
                # Determine target player's score at rally start
                first_row = rally_df.iloc[0]
                
                # 初始化第一個state
                current_player_name = first_row['player']
                current_player_id = self.player_id if current_player_name == self.player_name else opponent_player_id
                
                # 建立state tensor
                state_values = [first_row[col] for col in state_col]
                state_values[-1] = current_player_id
                current_state = torch.tensor(state_values, dtype=torch.float32)
                
                simulation_ended_early = False
                target_pred_count = 0
                last_action = None
                
                while target_pred_count < target_max_len:
                    current_player_id = int(current_state[17].item())
                    is_target_turn = current_player_id == self.player_id
                    current_player_name = self.player_name if is_target_turn else opponent_name
                    current_player_id = self.player_id if is_target_turn else opponent_player_id
                    current_state[17] = current_player_id
                    state_for_pred = current_state
                    info = self.build_rollout_info(match_id, rally_num, state_for_pred, last_action)
                    ball_round = int(state_for_pred[1].item())
                    
                    launch = (state_for_pred[1].item() == 1)
                    
                    # Agent預測
                    env_state = self._state_to_env_format(state_for_pred)
                    current_agent = self.agent if is_target_turn else opponent_agent
                    info['current_player_name'] = current_player_name
                    info['opponent_player_name'] = opponent_name if is_target_turn else self.player_name
                    info['rally_history'] = list(rally_history)
                    self.attach_style_shift_instruction_for_rollout(info, current_player_name, match_id, rally_num)
                    action, reason = current_agent.predict(env_state, info, launch)
                    
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
                        current_player_name, int(output_shot), land_2d, move_2d, ball_round))
                    rally_history.append(self._build_history_stroke(state_for_pred, current_player_name, action))
                    if is_target_turn:
                        target_pred_count += 1
                    
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
                        simulation_ended_early = True
                        break
                    
                    current_state = next_state
                    last_action = action
                
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
