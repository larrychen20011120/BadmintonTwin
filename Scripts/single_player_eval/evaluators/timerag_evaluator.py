"""
TimeRAG Evaluator for Single Player
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
from Agent.timerag import TimeRAGAgent, TimeRAGDatabase
from Utils.helper import state_col, action_col, transition, terminate
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
    build_rollout_metadata,
)
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
from Utils.artifact_names import player_database_filename


class TimeRAGEvaluator(BaseEvaluator):
    """TimeRAG 單一選手評估器"""

    def __init__(self, player_name=None, experiment='rally', device='cuda', skip_first_n=0,
                  train_split_path=None, conditioned_top_k=None, window_size=None, enable_tracing=True,
                  save_trajectories=False,
                  style_shift_instruction_mode='auto', verbose_style_shift=False,
                   output_experiment=None, all_rallies=False,
                   instruction_mode='no_instruct', case_grounding=True,
                   action_architecture='two_stage_v1', save_prompt_traces=False):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            train_split_path: train split 路徑（用於建立 database，可選）
            top_k: RAG 檢索數量
            window_size: 時間窗口大小
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
        if action_architecture not in {'two_stage_v1', 'single_stage_v1'}:
            raise ValueError(f'Unsupported action architecture: {action_architecture}')
        if action_architecture == 'single_stage_v1':
            if case_grounding:
                raise ValueError('Single-stage TimeRAG supports direct output only')
        self.action_architecture = action_architecture
        self.algorithm_name = (
            'timerag_single_stage'
            if action_architecture == 'single_stage_v1' else 'timerag'
        )
        if not isinstance(case_grounding, bool):
            raise TypeError('case_grounding must be a bool')
        self.case_grounding = case_grounding
        self.save_prompt_traces = bool(save_prompt_traces)

        self.train_split_path = train_split_path or self.config['paths']['rally_split']

        timerag_config = self.config.get('timerag', {})
        retrieval_config = self.config.get('retrieval', {})
        stage1_config = retrieval_config.get('stage1', {})
        stage2_config = retrieval_config.get('stage2', {})
        configured_architecture = retrieval_config.get('architecture', 'two_stage_v1')
        if configured_architecture != 'two_stage_v1':
            raise ValueError('TimeRAG requires retrieval.architecture=two_stage_v1')
        self.retrieval_architecture = action_architecture
        self.normal_top_k = int(stage1_config.get('normal_top_k', 10))
        self.miss_top_k = int(stage1_config.get('miss_top_k', 5))
        self.conditioned_top_k = int(
            conditioned_top_k
            if conditioned_top_k is not None
            else stage2_config.get('conditioned_top_k', 10)
        )
        if min(self.normal_top_k, self.miss_top_k, self.conditioned_top_k) < 1:
            raise ValueError('Two-stage retrieval top-k values must all be positive')
        self.window_size = window_size if window_size is not None else timerag_config.get('window_size', 3)
        self.enable_tracing = enable_tracing
        self.database_dir = self.config['paths']['timerag']['data']

        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)

        if self.player_name is not None and self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")

        self.player_id = self.player2index.get(self.player_name)

        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)

        self.agents = {}
        self.agent = self._get_agent(self.player_name) if self.player_name is not None else None

        self.print_runtime_parameters(
            self.algorithm_name,
            top_k=self.conditioned_top_k,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_architecture,
            window_size=self.window_size,
            include_llm=True,
            extra_rows=[
                ("database_dir", self.database_dir),
                ("prompt_variant", self.prompt_variant),
                ("case_grounding", self.case_grounding),
            ],
        )

    def _create_agent(self, player_name, player_id):
        """建立 TimeRAG Agent"""
        db_path = self._database_path(player_name)
        print(f"[TimeRAG] Loading database for {player_name}: {db_path}")
        database = TimeRAGDatabase.load(db_path)
        if database.player_name != player_name or database.player_id != player_id:
            raise ValueError(f"TimeRAG database identity mismatch: {db_path}")
        if database.window_size != self.window_size:
            raise ValueError(
                f"TimeRAG window_size mismatch: database={database.window_size}, "
                f"configured={self.window_size}"
            )
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        if database.build_metadata.get('split_sha256') != dataset['split_sha256']:
            raise ValueError(f"TimeRAG database split mismatch: {db_path}")
        print(f"[TimeRAG] Database ready for {player_name}: {len(database.windows)} windows")

        timerag_config = self.config.get('timerag', {})

        agent = TimeRAGAgent(
            player_name=player_name,
            database=database,
            model_name=timerag_config.get('model_name'),
            api_key=timerag_config.get('api_key'),
            host=timerag_config.get('host'),
            port=timerag_config.get('port'),
            base_url=timerag_config.get('base_url'),
            host_url=timerag_config.get('host_url'),
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            conditioned_top_k=self.conditioned_top_k,
            enable_tracing=self.enable_tracing,
            prompt_variant=self.prompt_variant,
            case_grounding=self.case_grounding,
            action_architecture=self.action_architecture,
            save_prompt_traces=self.save_prompt_traces,
        )
        print(f"[TimeRAG] Agent created for {player_name}")

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
        return self.algorithm_name

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
            'action': [None]
        }

    def _build_history_stroke(self, state_for_pred, player_name, action):
        """Build a completed stroke record for TimeRAG's full-rally history."""
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
        from .naive_rag_evaluator import NaiveRAGEvaluator

        return NaiveRAGEvaluator.simulate_rally(self, record)

    def simulate(self, records=None):
        from .naive_rag_evaluator import NaiveRAGEvaluator

        return NaiveRAGEvaluator.simulate(self, records)
