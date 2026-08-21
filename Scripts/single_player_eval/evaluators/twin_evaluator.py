"""Physical-rally evaluator for the Twin aspect-memory agent."""

import json
import hashlib
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_path)

from Agent.twin import (
    TwinAgent,
    TwinDatabase,
    TwinEmbeddingCache,
    embedding_cache_paths,
    embedding_model_key,
)
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
    build_rollout_metadata,
)
from Utils.artifact_names import player_database_filename
from Utils.embeddings import create_embedding_client, resolve_embedding_settings
from Utils.helper import action_col, state_col, terminate, transition

from .base_evaluator import BaseEvaluator


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TwinEvaluator(BaseEvaluator):
    """Evaluate each atomic physical rally once with two player-specific Twins."""

    algorithm_name = "twin"
    agent_class = TwinAgent
    retrieval_mode = "twin_aspect_two_stage_v1"
    uses_conditioned_retrieval = True

    def __init__(self, player_name=None, experiment="rally", device="cuda", skip_first_n=0,
                 train_split_path=None, conditioned_top_k=None, lambda_weight=None,
                 enable_tracing=True, save_trajectories=False, output_experiment=None,
                 all_rallies=False,
                 instruction_mode="no_instruct", case_grounding=True,
                 save_prompt_traces=False):
        if instruction_mode not in {"instruct", "no_instruct"}:
            raise ValueError("instruction_mode must be instruct or no_instruct")
        super().__init__(
            player_name,
            "rally",
            device,
            skip_first_n,
            save_trajectories=save_trajectories,
            style_shift_instruction_mode="off",
            verbose_style_shift=False,
            output_experiment=output_experiment or f"rally_{instruction_mode}",
        )
        self.all_rallies = all_rallies
        self.instruction_mode = instruction_mode
        self.case_grounding = bool(case_grounding)
        self.save_prompt_traces = bool(save_prompt_traces)
        self.train_split_path = train_split_path or self.config["paths"]["rally_split"]

        twin_config = self.config.get("twin", {})
        retrieval = self.config.get("retrieval", {})
        stage1 = retrieval.get("stage1", {})
        stage2 = retrieval.get("stage2", {})
        self.normal_top_k = int(stage1.get("normal_top_k", 10))
        self.miss_top_k = int(stage1.get("miss_top_k", 5))
        self.conditioned_top_k = int(
            conditioned_top_k
            if conditioned_top_k is not None
            else stage2.get("conditioned_top_k", 10)
        )
        self.lambda_weight = float(
            lambda_weight if lambda_weight is not None else twin_config.get("lambda_weight", 0.5)
        )
        self.semantic_start_ball_round = int(
            twin_config.get("semantic_start_ball_round", 3)
        )
        self.online_terminal_aspect_repair = twin_config.get(
            "online_terminal_aspect_repair", True
        )
        required_top_k = [self.normal_top_k, self.miss_top_k]
        if self.uses_conditioned_retrieval:
            required_top_k.append(self.conditioned_top_k)
        if min(required_top_k) < 1:
            raise ValueError("Twin retrieval top-k values must all be positive")
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("Twin lambda_weight must be between 0 and 1")
        if self.semantic_start_ball_round < 1:
            raise ValueError("Twin semantic_start_ball_round must be at least 1")
        if not isinstance(self.online_terminal_aspect_repair, bool):
            raise TypeError("Twin online_terminal_aspect_repair must be a bool")

        self.embedding_settings = resolve_embedding_settings(config=self.config)
        self.embedding_provider = self.embedding_settings["provider"]
        self.embedding_model_name = self.embedding_settings["model_name"]
        self.embedding_revision = self.embedding_settings["revision"]
        self.embedding_dimension = self.embedding_settings["dimension"]
        self.embedding_batch_size = self.embedding_settings["batch_size"]
        self.embedding_dtype = self.embedding_settings["dtype"]
        self.normalize_embeddings = self.embedding_settings["normalize_embeddings"]
        self.enable_tracing = enable_tracing
        self.database_dir = self._resolve_project_path(self.config["paths"]["twin"]["data"])
        self.embedding_cache_dir = self._resolve_project_path(
            self.config["paths"]["twin"]["embeddings"]
        )
        self.embedding_model = None
        manifest_path = os.path.join(self.database_dir, "preparation_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.memory_manifest = json.load(handle)
        embedding_manifest_path = os.path.join(
            self.embedding_cache_dir,
            embedding_model_key(
                self.embedding_model_name,
                self.embedding_revision,
                self.normalize_embeddings,
                self.embedding_dtype,
                self.embedding_provider,
                self.embedding_dimension,
            ),
            "preparation_manifest.json",
        )
        with open(embedding_manifest_path, "r", encoding="utf-8") as handle:
            self.embedding_manifest = json.load(handle)
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        if self.memory_manifest.get("split_sha256") != dataset["split_sha256"]:
            raise ValueError(f"Twin preparation manifest split mismatch: {manifest_path}")

        with open(self.player_map_path, "r", encoding="utf-8") as handle:
            self.player2index = json.load(handle)
        if self.player_name is not None and self.player_name not in self.player2index:
            raise ValueError(f"Player {self.player_name!r} not found in player map")
        self.player_id = self.player2index.get(self.player_name)

        self.df = pd.read_csv(self.merged_csv_path).reset_index(drop=True)
        self.agents = {}
        self.agent = self._get_agent(self.player_name) if self.player_name is not None else None

        self.print_runtime_parameters(
            self.algorithm_name,
            top_k=self.conditioned_top_k if self.uses_conditioned_retrieval else None,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            retrieval_mode=self.retrieval_mode,
            lambda_weight=self.lambda_weight,
            include_llm=True,
            extra_rows=[
                ("database_dir", self.database_dir),
                ("embedding_cache_dir", self.embedding_cache_dir),
                ("embedding_provider", self.embedding_provider),
                ("embedding_model", self.embedding_model_name),
                ("instruction_mode", self.instruction_mode),
                ("case_grounding", self.case_grounding),
                ("semantic_start_ball_round", self.semantic_start_ball_round),
                ("online_terminal_aspect_repair", self.online_terminal_aspect_repair),
            ],
        )

    @staticmethod
    def _resolve_project_path(path):
        return path if os.path.isabs(path) else os.path.join(project_path, path)

    def _database_path(self, player_name):
        return os.path.join(self.database_dir, player_database_filename(player_name))

    def _create_agent(self, player_name, player_id):
        database_path = self._database_path(player_name)
        player_manifest = (self.memory_manifest.get("players") or {}).get(player_name)
        if not isinstance(player_manifest, dict):
            raise ValueError(f"Player {player_name!r} missing from Twin preparation manifest")
        if player_manifest.get("database_file") != os.path.basename(database_path):
            raise ValueError(f"Twin database filename mismatch for {player_name}: {database_path}")
        if player_manifest.get("database_sha256") != _file_sha256(database_path):
            raise ValueError(f"Twin database checksum mismatch: {database_path}")
        database = TwinDatabase.load(database_path)
        if database.player_name != player_name or database.player_id != player_id:
            raise ValueError(f"Twin database identity mismatch: {database_path}")
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        if database.build_metadata.get("split_sha256") != dataset["split_sha256"]:
            raise ValueError(f"Twin database split mismatch: {database_path}")
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
        embedding_player = (self.embedding_manifest.get("players") or {}).get(player_name)
        if not isinstance(embedding_player, dict):
            raise ValueError(f"Player {player_name!r} missing from Twin embedding manifest")
        array_path, metadata_path = embedding_cache_paths(
            self.embedding_cache_dir,
            player_name,
            self.embedding_model_name,
            self.embedding_revision,
            self.normalize_embeddings,
            self.embedding_dtype,
            self.embedding_provider,
            self.embedding_dimension,
        )
        if embedding_player.get("embedding_sha256") != _file_sha256(array_path):
            raise ValueError(f"Twin embedding checksum mismatch: {array_path}")
        if embedding_player.get("metadata_sha256") != _file_sha256(metadata_path):
            raise ValueError(f"Twin embedding metadata checksum mismatch: {metadata_path}")
        if self.embedding_model is None:
            self.embedding_model = create_embedding_client(self.embedding_settings)
        twin_config = self.config.get("twin", {})
        return self.agent_class(
            player_name=player_name,
            database=database,
            embedding_cache=embedding_cache,
            model_name=twin_config.get("model_name"),
            api_key=twin_config.get("api_key"),
            host=twin_config.get("host"),
            port=twin_config.get("port"),
            base_url=twin_config.get("base_url"),
            host_url=twin_config.get("host_url"),
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            conditioned_top_k=self.conditioned_top_k,
            lambda_weight=self.lambda_weight,
            embedding_model_name=self.embedding_model_name,
            embedding_batch_size=self.embedding_batch_size,
            embedding_revision=self.embedding_revision,
            normalize_embeddings=self.normalize_embeddings,
            embedding_dtype=self.embedding_dtype,
            embedding_model=self.embedding_model,
            enable_tracing=self.enable_tracing,
            instruction_mode=self.instruction_mode,
            embedding_provider=self.embedding_provider,
            embedding_dimension=self.embedding_dimension,
            case_grounding=self.case_grounding,
            save_prompt_traces=self.save_prompt_traces,
            semantic_start_ball_round=self.semantic_start_ball_round,
            online_terminal_aspect_repair=self.online_terminal_aspect_repair,
        )

    def _get_agent(self, player_name):
        if player_name not in self.player2index:
            raise ValueError(f"Player {player_name!r} not found in player map")
        if player_name not in self.agents:
            self.agents[player_name] = self._create_agent(
                player_name, self.player2index[player_name]
            )
        return self.agents[player_name]

    def get_algorithm_name(self):
        return self.algorithm_name

    @staticmethod
    def _state_to_env_format(state):
        return [
            int(state[12].item()),
            (state[5].item(), state[6].item()),
            (state[13].item(), state[14].item()),
            (state[10].item(), state[11].item()),
        ]

    @staticmethod
    def _build_history_stroke(state, player_name, action):
        output_shot, _, output_land, output_move, _ = action
        return {
            "player": player_name,
            "rally": int(state[0].item()),
            "ball_round": int(state[1].item()),
            "player_score": int(state[2].item()),
            "opponent_score": int(state[3].item()),
            "score_status": int(state[4].item()),
            "player_location_x": float(state[5].item()),
            "player_location_y": float(state[6].item()),
            "ball_distance": float(state[7].item()),
            "ball_distance_x": float(state[8].item()),
            "ball_distance_y": float(state[9].item()),
            "hit_x": float(state[10].item()),
            "hit_y": float(state[11].item()),
            "opponent_type": int(state[12].item()),
            "opponent_location_x": float(state[13].item()),
            "opponent_location_y": float(state[14].item()),
            "opponent_move_x": float(state[15].item()),
            "opponent_move_y": float(state[16].item()),
            "landing_x": float(output_land[0]),
            "landing_y": float(output_land[1]),
            "player_type": int(output_shot),
            "moving_x": float(output_move[0]),
            "moving_y": float(output_move[1]),
        }

    def simulate_rally(self, record):
        match_id, set_num, rally_num = rally_key(record)
        rally_df = filter_rally_rows(self.df, [record]).sort_values("ball_round").reset_index(drop=True)
        if rally_df.empty:
            raise ValueError(f"No model-ready rows found for rally {(match_id, set_num, rally_num)}")
        if rally_df[state_col + action_col].isna().any(axis=None):
            raise ValueError(
                f"Incomplete model row in physical rally {(match_id, set_num, rally_num)}; "
                "the rally cannot be evaluated atomically"
            )
        players = list(record.get("players") or [])
        if len(players) != 2:
            raise ValueError(f"Expected two players for rally {(match_id, set_num, rally_num)}")
        agents = {player: self._get_agent(player) for player in players}
        for agent in agents.values():
            agent.reset_history()
        player_ids = {player: self.player2index[player] for player in players}
        id_to_player = {player_id: player for player, player_id in player_ids.items()}

        profiles = record.get("player_profiles") or {}
        instructions = {}
        for player in players:
            if self.instruction_mode == "no_instruct":
                instructions[player] = ""
                continue
            profile = profiles.get(player) or {}
            label = profile.get("label")
            instruction = profile.get("instruction")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Missing tactical profile label for {player!r}")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing tactical profile instruction for {player!r}")
            instructions[player] = f"{label.strip()}. {instruction.strip()}"

        gt_sequence = [build_gt_shot_from_row(row) for _, row in rally_df.iterrows()]
        rollout_limit = len(gt_sequence) + int(
            self.config.get("evaluation", {}).get("rollout_extra_steps", 2)
        )
        pred_sequence = []
        rally_history = []
        first_row = rally_df.iloc[0]
        current_player = str(first_row["player"])
        state_values = [first_row[column] for column in state_col]
        state_values[-1] = player_ids[current_player]
        current_state = torch.tensor(state_values, dtype=torch.float32)
        termination_reason = "max_rollout_reached"
        last_action = None
        prompt_traces = []

        while len(pred_sequence) < rollout_limit:
            player_id = int(current_state[17].item())
            if player_id not in id_to_player:
                raise ValueError(f"Unknown player id {player_id} during rally")
            current_player = id_to_player[player_id]
            opponent = next(player for player in players if player != current_player)
            info = self.build_rollout_info(match_id, rally_num, current_state, last_action)
            info.update({
                "set": set_num,
                "current_player_name": current_player,
                "opponent_player_name": opponent,
                "rally_history": list(rally_history),
                "rally_profile_instruction": instructions[current_player],
                "instruction_mode": self.instruction_mode,
            })
            ball_round = int(current_state[1].item())
            action, reason = agents[current_player].predict(
                self._state_to_env_format(current_state),
                info,
                launch=ball_round == 1,
            )
            prompt_trace = None
            if getattr(self, "save_prompt_traces", False):
                prompt_trace = agents[current_player].tracer.consume_local_trace()
                if prompt_trace is None:
                    raise RuntimeError("Missing local prompt trace for generated action")
            if action is None:
                if prompt_trace is not None:
                    prompt_trace["action_source"] = "action_none"
                    prompt_trace["fallback_source"] = None
                    prompt_traces.append(prompt_trace)
                termination_reason = "action_none"
                break
            output_shot, _, output_land, output_move, _ = action
            landing = [float(output_land[0]), float(output_land[1])]
            moving = [float(output_move[0]), float(output_move[1])]
            decision = dict(agents[current_player].last_decision_trace or {})
            if prompt_trace is not None:
                action_source = decision.get("action_source", "agent")
                fallback_source = decision.get("fallback_source")
                if fallback_source is None:
                    fallback_parts = [
                        part for part in action_source.split("+")
                        if part.startswith("fallback")
                    ]
                    fallback_source = "+".join(fallback_parts) or None
                prompt_trace["action_source"] = action_source
                prompt_trace["fallback_source"] = fallback_source
                prompt_traces.append(prompt_trace)
            pred_sequence.append(build_pred_shot(
                current_player,
                int(output_shot),
                landing,
                moving,
                ball_round,
                decision=decision,
            ))
            rally_history.append(self._build_history_stroke(current_state, current_player, action))
            action_tensor = torch.tensor([
                landing[0], landing[1], float(output_shot), moving[0], moving[1]
            ], dtype=torch.float32)
            next_state = transition(action_tensor, current_state.cpu(), player_ids[opponent])
            if terminate(next_state):
                termination_reason = "natural_terminate"
                break
            current_state = next_state
            last_action = action

        for agent in agents.values():
            agent.on_rally_end(rally_history)
        trajectory = {
            "match_id": match_id,
            "set": set_num,
            "rally": rally_num,
            "players": players,
            "player_profiles": profiles,
            "applied_instructions": instructions,
            "instruction_mode": self.instruction_mode,
            "evaluation_unit": "physical_rally",
            "target_player": None,
            "opponent_player": None,
            "gt_sequence": gt_sequence,
            "pred_sequence": pred_sequence,
            "target_player_won": None,
            **build_rollout_metadata(
                gt_sequence,
                pred_sequence,
                termination_reason,
                rollout_limit=rollout_limit,
            ),
        }
        if getattr(self, "save_prompt_traces", False):
            trajectory["_prompt_traces"] = prompt_traces
        return trajectory

    def simulate(self, records=None):
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        records = list(records if records is not None else dataset["test"])
        if not self.all_rallies and self.player_name is not None:
            records = [record for record in records if self.player_name in record.get("players", [])]
        trajectories = [
            self.simulate_rally(record)
            for record in tqdm(records, desc=f"Simulating {self.algorithm_name} physical rallies")
        ]
        self._save_trajectories_if_needed(trajectories)
        return trajectories
