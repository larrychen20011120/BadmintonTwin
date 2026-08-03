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
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
    build_rollout_metadata,
)
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
from Utils.artifact_names import player_database_filename


class NaiveRAGEvaluator(BaseEvaluator):
    """NaiveRAG單一選手評估器"""
    
    def __init__(self, player_name=None, experiment='rally', device='cuda', skip_first_n=0,
                  train_split_path=None, conditioned_top_k=None, enable_tracing=True,
                  save_trajectories=False,
                 style_shift_instruction_mode='auto', verbose_style_shift=False,
                 output_experiment=None, all_rallies=False,
                 instruction_mode='no_instruct'):
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
        if instruction_mode not in ('instruct', 'no_instruct'):
            raise ValueError("instruction_mode must be 'instruct' or 'no_instruct'")
        super().__init__(player_name, 'rally', device, skip_first_n,
                         save_trajectories=save_trajectories,
                         style_shift_instruction_mode='off',
                         verbose_style_shift=verbose_style_shift,
                         output_experiment=output_experiment or f'rally_{instruction_mode}')
        self.all_rallies = all_rallies
        self.instruction_mode = instruction_mode
        self.prompt_variant = 'shifted' if instruction_mode == 'instruct' else 'norm'
        
        # 設定train split路徑
        self.train_split_path = train_split_path or self.config['paths']['rally_split']
        
        llm_config = self.config.get('naive_rag', {})
        retrieval_config = self.config.get('retrieval', {})
        stage1_config = retrieval_config.get('stage1', {})
        stage2_config = retrieval_config.get('stage2', {})
        self.retrieval_architecture = retrieval_config.get('architecture', 'two_stage_v1')
        if self.retrieval_architecture != 'two_stage_v1':
            raise ValueError('NaiveRAG requires retrieval.architecture=two_stage_v1')
        self.normal_top_k = int(stage1_config.get('normal_top_k', 10))
        self.miss_top_k = int(stage1_config.get('miss_top_k', 5))
        self.conditioned_top_k = int(
            conditioned_top_k
            if conditioned_top_k is not None
            else stage2_config.get('conditioned_top_k', 10)
        )
        if min(self.normal_top_k, self.miss_top_k, self.conditioned_top_k) < 1:
            raise ValueError('Two-stage retrieval top-k values must all be positive')
        self.enable_tracing = enable_tracing
        self.database_dir = self.config['paths']['naive_rag']['data']
        
        # 載入player2index
        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)
        
        if self.player_name is not None and self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")
        
        self.player_id = self.player2index.get(self.player_name)
        
        # 載入資料
        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        
        # 建立/快取每位選手各自的 Agent
        self.agents = {}
        self.agent = self._get_agent(self.player_name) if self.player_name is not None else None

        self.print_runtime_parameters(
            "naive_rag",
            top_k=self.conditioned_top_k,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_architecture,
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
        if database.player_name != player_name or database.player_id != player_id:
            raise ValueError(f"NaiveRAG database identity mismatch: {db_path}")
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        if database.build_metadata.get('split_sha256') != dataset['split_sha256']:
            raise ValueError(f"NaiveRAG database split mismatch: {db_path}")
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
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            conditioned_top_k=self.conditioned_top_k,
            enable_tracing=self.enable_tracing,
            prompt_variant=self.prompt_variant,
        )
        print(f"[NaiveRAG] Agent created for {player_name}")
        
        return agent

    def _database_path(self, player_name):
        return os.path.join(self.database_dir, player_database_filename(player_name))

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
    
    def simulate_rally(self, record):
        """Simulate one canonical physical rally with both player-specific agents."""
        match_id, set_num, rally_num = rally_key(record)
        rally_df = filter_rally_rows(self.df, [record]).sort_values('ball_round').reset_index(drop=True)
        if rally_df.empty:
            raise ValueError(f'No model-ready rows found for rally {(match_id, set_num, rally_num)}')

        players = list(record.get('players') or [])
        if len(players) != 2:
            raise ValueError(f'Expected two players for rally {(match_id, set_num, rally_num)}: {players}')
        missing = [player for player in players if player not in self.player2index]
        if missing:
            raise ValueError(f'Players missing from player map: {missing}')

        agents = {player: self._get_agent(player) for player in players}
        for agent in agents.values():
            agent.reset_history()
        player_ids = {player: self.player2index[player] for player in players}
        id_to_player = {player_id: player for player, player_id in player_ids.items()}
        profiles = record.get('player_profiles') or {}
        instructions = {}
        for player in players:
            if self.instruction_mode != 'instruct':
                instructions[player] = ''
                continue
            profile = profiles.get(player) or {}
            label = profile.get('label')
            instruction = profile.get('instruction')
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f'Missing tactical profile label for {player!r}')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f'Missing tactical profile instruction for {player!r}')
            instructions[player] = f'{label.strip()}. {instruction.strip()}'

        rally_gt_full_seq = [build_gt_shot_from_row(row) for _, row in rally_df.iterrows()]
        extra_steps = int(self.config.get('evaluation', {}).get('rollout_extra_steps', 2))
        max_rollout_steps = len(rally_gt_full_seq) + extra_steps
        rally_pred_full_seq = []
        rally_history = []
        first_row = rally_df.iloc[0]
        current_player_name = str(first_row['player'])
        state_values = [first_row[col] for col in state_col]
        state_values[-1] = player_ids[current_player_name]
        current_state = torch.tensor(state_values, dtype=torch.float32)
        termination_reason = 'max_rollout_reached'
        last_action = None

        while len(rally_pred_full_seq) < max_rollout_steps:
            current_player_id = int(current_state[17].item())
            if current_player_id not in id_to_player:
                raise ValueError(f'Unknown player id {current_player_id} during rally')
            current_player_name = id_to_player[current_player_id]
            opponent_name = next(player for player in players if player != current_player_name)
            state_for_pred = current_state
            info = self.build_rollout_info(match_id, rally_num, state_for_pred, last_action)
            info.update({
                'current_player_name': current_player_name,
                'opponent_player_name': opponent_name,
                'rally_history': list(rally_history),
                'style_shift_instruction': instructions[current_player_name],
                'rally_profile_instruction': instructions[current_player_name],
                'instruction_mode': self.instruction_mode,
            })
            ball_round = int(state_for_pred[1].item())
            launch = state_for_pred[1].item() == 1
            current_agent = agents[current_player_name]
            action, reason = current_agent.predict(
                self._state_to_env_format(state_for_pred), info, launch
            )
            if action is None:
                termination_reason = 'action_none'
                break

            output_shot, hit, output_land, output_move, dist = action
            land_2d = [float(output_land[0]), float(output_land[1])]
            move_2d = [float(output_move[0]), float(output_move[1])]
            decision = dict(getattr(current_agent, 'last_decision_trace', None) or {})
            if not decision:
                decision = {
                    'case_ids': [],
                    'rationale': reason,
                    'action': {
                        'shot_type': int(output_shot),
                        'landing_x': land_2d[0],
                        'landing_y': land_2d[1],
                        'moving_x': move_2d[0],
                        'moving_y': move_2d[1],
                    },
                    'action_source': 'agent',
                }
            rally_pred_full_seq.append(build_pred_shot(
                current_player_name, int(output_shot), land_2d, move_2d, ball_round,
                decision=decision,
            ))
            rally_history.append(self._build_history_stroke(
                state_for_pred, current_player_name, action
            ))
            action_tensor = torch.tensor([
                land_2d[0], land_2d[1], float(output_shot), move_2d[0], move_2d[1]
            ], dtype=torch.float32)
            next_state = transition(
                action_tensor.cpu(), current_state.cpu(), player_ids[opponent_name]
            )
            if terminate(next_state.cpu()):
                termination_reason = 'natural_terminate'
                break
            current_state = next_state
            last_action = action

        return {
            'match_id': match_id,
            'set': set_num,
            'rally': rally_num,
            'players': players,
            'player_profiles': profiles,
            'applied_instructions': instructions,
            'instruction_mode': self.instruction_mode,
            'evaluation_unit': 'physical_rally',
            'target_player': None,
            'opponent_player': None,
            'gt_sequence': rally_gt_full_seq,
            'pred_sequence': rally_pred_full_seq,
            'target_player_won': None,
            **build_rollout_metadata(
                rally_gt_full_seq,
                rally_pred_full_seq,
                termination_reason,
                rollout_limit=max_rollout_steps,
            ),
        }

    def simulate(self, records=None):
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        records = list(records if records is not None else dataset['test'])
        if not self.all_rallies and self.player_name is not None:
            records = [record for record in records if self.player_name in record.get('players', [])]
        trajectories = [
            self.simulate_rally(record)
            for record in tqdm(records, desc='Simulating physical rallies')
        ]
        self._save_trajectories_if_needed(trajectories)
        return trajectories
