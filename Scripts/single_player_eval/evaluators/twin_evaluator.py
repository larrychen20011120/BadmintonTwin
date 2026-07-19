"""
Twin Evaluator for Single Player
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
from Agent.twin import TwinAgent, TwinDatabase, TwinEmbeddingCache
from Utils.embeddings import create_embedding_client, resolve_embedding_settings
from Utils.helper import state_col, action_col, transition, terminate
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row, build_pred_shot


class TwinEvaluator(BaseEvaluator):
    """Twin aspect-memory 單一選手評估器"""

    def __init__(self, player_name, experiment='norm', device='cuda', skip_first_n=0,
                 train_split_path=None, top_k=None, lambda_weight=None, enable_tracing=True,
                 save_trajectories=False,
                 style_shift_instruction_mode='auto', verbose_style_shift=False,
                 output_experiment=None):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型
            device: 運算裝置
            skip_first_n: 跳過前n個shots (只在存檔時記錄，不影響模擬)
            train_split_path: train split 路徑（用於建立 database，可選）
            top_k: 軌跡段檢索數量
            lambda_weight: numeric/semantic retrieval mixing weight
            enable_tracing: 是否啟用 Langfuse 追蹤
        """
        super().__init__(player_name, experiment, device, skip_first_n,
                         save_trajectories=save_trajectories,
                         style_shift_instruction_mode=style_shift_instruction_mode,
                         verbose_style_shift=verbose_style_shift,
                         output_experiment=output_experiment)

        if train_split_path is None:
            self.train_split_path = self.config['paths'][f'{experiment}_split']
        else:
            self.train_split_path = train_split_path

        twin_config = self.config.get('twin', {})
        embedding_settings = resolve_embedding_settings(config=self.config)
        retrieval_config = self.config.get('retrieval', {})
        self.retrieval_mode = retrieval_config.get('mode', 'classic')
        self.top_k = top_k if top_k is not None else retrieval_config.get('classic_top_k', retrieval_config.get('top_k', twin_config.get('top_k', 5)))
        self.normal_top_k = retrieval_config.get('normal_top_k', self.top_k)
        self.miss_top_k = retrieval_config.get('miss_top_k', max(1, min(5, self.top_k)))
        self.lambda_weight = lambda_weight if lambda_weight is not None else twin_config.get('lambda_weight', 0.5)
        self.embedding_settings = embedding_settings
        self.embedding_provider = embedding_settings['provider']
        self.embedding_model_name = embedding_settings['model_name']
        self.embedding_revision = embedding_settings['revision']
        self.embedding_dimension = embedding_settings['dimension']
        self.embedding_device = embedding_settings['device']
        self.embedding_batch_size = embedding_settings['batch_size']
        self.embedding_dtype = embedding_settings['dtype']
        self.normalize_embeddings = embedding_settings['normalize_embeddings']
        self.enable_tracing = enable_tracing
        self.memory_dir = self._resolve_project_path(self.config['paths']['twin']['memory'][experiment])
        self.embedding_cache_dir = self._resolve_project_path(self.config['paths']['twin']['embeddings'][experiment])
        self.embedding_model = None

        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)

        if self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")

        self.player_id = self.player2index[self.player_name]

        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)

        self.agents = {}
        self.agent = self._get_agent(self.player_name)

        self.print_runtime_parameters(
            "twin",
            top_k=self.top_k,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_mode,
            lambda_weight=self.lambda_weight,
            include_llm=True,
            extra_rows=[
                ("memory_dir", self.memory_dir),
                ("embedding_cache_dir", self.embedding_cache_dir),
                ("embedding_provider", self.embedding_provider),
                ("embedding_model", self.embedding_model_name),
                ("embedding_revision", self.embedding_revision),
                ("embedding_dimension", self.embedding_dimension),
                ("embedding_device", self.embedding_device),
                ("embedding_batch_size", self.embedding_batch_size),
                ("embedding_dtype", self.embedding_dtype),
                ("normalize_embeddings", self.normalize_embeddings),
                ("prompt_variant", self.prompt_variant),
            ],
        )

    def _create_agent(self, player_name, player_id):
        """建立 Twin Agent"""
        db_path = self._database_path(player_name)
        print(f"[Twin] Loading memory for {player_name}: {db_path}")
        database = TwinDatabase.load(db_path)
        embedding_cache = TwinEmbeddingCache.load(
            self.embedding_cache_dir,
            database,
            self.embedding_model_name,
            revision=self.embedding_revision,
            normalize_embeddings=self.normalize_embeddings,
            dtype=self.embedding_dtype,
            provider=self.embedding_provider,
            embedding_dimension=self.embedding_dimension,
        )
        if self.embedding_model is None:
            self.embedding_model = create_embedding_client(self.embedding_settings)
        print(f"[Twin] Memory and embeddings ready for {player_name}: {len(database.memories)} memories")

        twin_config = self.config.get('twin', {})

        agent = TwinAgent(
            player_name=player_name,
            database=database,
            embedding_cache=embedding_cache,
            model_name=twin_config.get('model_name'),
            api_key=twin_config.get('api_key'),
            host=twin_config.get('host'),
            port=twin_config.get('port'),
            base_url=twin_config.get('base_url'),
            host_url=twin_config.get('host_url'),
            top_k=self.top_k,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_mode,
            lambda_weight=self.lambda_weight,
            embedding_model_name=self.embedding_model_name,
            embedding_device=self.embedding_device,
            embedding_batch_size=self.embedding_batch_size,
            embedding_revision=self.embedding_revision,
            normalize_embeddings=self.normalize_embeddings,
            embedding_dtype=self.embedding_dtype,
            embedding_model=self.embedding_model,
            enable_tracing=self.enable_tracing,
            prompt_variant=self.prompt_variant,
            embedding_provider=self.embedding_provider,
            embedding_dimension=self.embedding_dimension,
        )
        print(f"[Twin] Agent created for {player_name}")

        return agent

    def _database_path(self, player_name):
        safe_name = player_name.replace(" ", "_").replace(".", "")
        return os.path.join(self.memory_dir, f"{safe_name}_db.pkl")

    @staticmethod
    def _resolve_project_path(path):
        return path if os.path.isabs(path) else os.path.join(project_path, path)

    def _get_agent(self, player_name):
        """取得指定選手的 Agent；target/opponent 必須使用各自的 player_name。"""
        if player_name not in self.player2index:
            raise ValueError(f"Player '{player_name}' not found in player map")

        if player_name not in self.agents:
            player_id = self.player2index[player_name]
            self.agents[player_name] = self._create_agent(player_name, player_id)

        return self.agents[player_name]

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
            'match_id': row['match_id'],
            'rally': row['rally'],
            'round': [row['ball_round']],
            'score': [row['player_score'], row['opponent_score']],
            'action': [None]
        }

    def simulate(self):
        """執行 Twin aspect-memory 模擬，收集trajectory"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        test_match_ids = self.get_player_test_matches()
        target_max_len = self.get_player_max_sequence_length(self.player_name)
        if target_max_len == 0:
            return []

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
            opponent_agent = self._get_agent(opponent_name)

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

                current_player_name = first_row['player']
                current_player_id = self.player_id if current_player_name == self.player_name else opponent_player_id

                state_values = [first_row[col] for col in state_col]
                state_values[-1] = current_player_id
                current_state = torch.tensor(state_values, dtype=torch.float32)

                simulation_ended_early = False
                target_pred_count = 0
                last_action = None
                simulated_history = []

                while target_pred_count < target_max_len:
                    current_player_id = int(current_state[17].item())
                    is_target_turn = current_player_id == self.player_id
                    current_player_name = self.player_name if is_target_turn else opponent_name
                    current_player_id = self.player_id if is_target_turn else opponent_player_id
                    current_state[17] = current_player_id
                    state_for_pred = current_state
                    info = self.build_rollout_info(match_id, rally_num, state_for_pred, last_action)
                    info['current_player_name'] = current_player_name
                    info['rally_history'] = simulated_history
                    ball_round = int(state_for_pred[1].item())

                    launch = (state_for_pred[1].item() == 1)

                    env_state = self._state_to_env_format(state_for_pred)
                    current_agent = self.agent if is_target_turn else opponent_agent
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
                    if is_target_turn:
                        target_pred_count += 1

                    simulated_history.append({
                        'player': current_player_name,
                        'is_target_turn': is_target_turn,
                        'turn_role': 'target' if is_target_turn else 'opponent',
                        'rally': int(rally_num),
                        'ball_round': ball_round,
                        'player_score': float(state_for_pred[2].item()),
                        'opponent_score': float(state_for_pred[3].item()),
                        'score_status': float(state_for_pred[4].item()),
                        'player_location_x': float(state_for_pred[5].item()),
                        'player_location_y': float(state_for_pred[6].item()),
                        'hit_x': float(state_for_pred[10].item()),
                        'hit_y': float(state_for_pred[11].item()),
                        'opponent_type': int(state_for_pred[12].item()),
                        'opponent_location_x': float(state_for_pred[13].item()),
                        'opponent_location_y': float(state_for_pred[14].item()),
                        'player_type': int(output_shot),
                        'landing_x': land_2d[0],
                        'landing_y': land_2d[1],
                        'moving_x': move_2d[0],
                        'moving_y': move_2d[1],
                    })

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
                    last_action = action

                self.agent.on_rally_end(simulated_history)
                opponent_agent.on_rally_end(simulated_history)

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
