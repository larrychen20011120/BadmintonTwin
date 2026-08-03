"""Canonical physical-rally evaluator for RallyNet."""

import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Agent.rallynet import RallyNetPolicy, RallyNetRollout
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
    build_rollout_metadata,
)
from Utils.helper import action_col, state_col, terminate, transition

from .base_evaluator import BaseEvaluator


def build_state_tensor_from_row(row, player_id):
    values = [row[column] for column in state_col]
    values[-1] = player_id
    return torch.tensor(values, dtype=torch.float32)


def _normalized_rally_keys(values):
    if values is None:
        return None
    normalized = set()
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            normalized.add(tuple(int(item) for item in value))
        else:
            normalized.add(rally_key(value))
    return normalized


class RallyNetEvaluator(BaseEvaluator):
    """Roll RallyNet out once for each selected physical test rally."""

    def __init__(
        self,
        player_name=None,
        experiment="rally",
        device="cuda",
        skip_first_n=0,
        model_path=None,
        seed=None,
        sample=None,
        error_threshold=None,
        save_trajectories=False,
        all_rallies=False,
        records_limit=None,
    ):
        super().__init__(
            player_name,
            experiment,
            device,
            skip_first_n,
            save_trajectories=save_trajectories,
        )
        if records_limit is not None and records_limit < 0:
            raise ValueError("records_limit must be non-negative")
        self.all_rallies = all_rallies
        self.records_limit = records_limit

        algorithm_config = self.config.get("rallynet", {})
        rallynet_paths = self.config["paths"]["rallynet"]
        self.model_path = model_path or rallynet_paths["model"]
        self.seed = int(seed if seed is not None else algorithm_config.get("seed", 42))
        self.sample = bool(
            sample if sample is not None else algorithm_config.get("sample", True)
        )
        self.error_threshold = float(
            error_threshold
            if error_threshold is not None
            else algorithm_config.get("error_threshold", 0.5)
        )
        self.policy = RallyNetPolicy(
            self.model_path,
            device=self.device,
            seed=self.seed,
            sample=self.sample,
            error_threshold=self.error_threshold,
        )
        self.player2index = self.policy.player_map

        with open(self.player_map_path, "r", encoding="utf-8") as handle:
            configured_player_map = json.load(handle)
        if configured_player_map != self.player2index:
            raise ValueError(
                "RallyNet checkpoint player map does not match configured player map"
            )
        if not self.all_rallies and self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in checkpoint player map")
        self.player_id = self.player2index.get(self.player_name)

        data_dir = rallynet_paths["data"]
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(self._project_path(), data_dir)
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data_metadata = json.load(handle)
        if data_metadata.get("player_map") != configured_player_map:
            raise ValueError(
                "Prepared RallyNet player map does not match configured player map"
            )
        self._verify_provenance(data_metadata)

        self.df = pd.read_csv(self.merged_csv_path)
        required_columns = set(
            state_col + action_col + ["match_id", "set", "rally", "ball_round"]
        )
        missing_columns = sorted(required_columns - set(self.df.columns))
        if missing_columns:
            raise ValueError(f"Missing RallyNet evaluation columns: {missing_columns}")

        self.print_runtime_parameters(
            "rallynet",
            seed=self.seed,
            extra_rows=[
                ("rallynet.model_path", self.model_path),
                ("rallynet.sample", self.sample),
                ("rallynet.error_threshold", self.error_threshold),
                ("rallynet.records_limit", self.records_limit),
            ],
        )

    def _verify_provenance(self, data_metadata):
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        provenance = self.policy.checkpoint.get("training", {}).get("provenance", {})

        if data_metadata.get("experiment") != self.experiment:
            raise ValueError("Prepared RallyNet data belongs to a different experiment")
        if provenance.get("experiment") != self.experiment:
            raise ValueError("RallyNet checkpoint belongs to a different experiment")
        if (
            not data_metadata.get("dataset_fingerprint")
            or provenance.get("dataset_fingerprint")
            != data_metadata.get("dataset_fingerprint")
        ):
            raise ValueError(
                "RallyNet checkpoint fingerprint does not match prepared evaluation data"
            )

        split_sha256 = dataset["split_sha256"]
        if data_metadata.get("split_sha256") != split_sha256:
            raise ValueError("Prepared RallyNet data does not match the physical-rally split")
        if provenance.get("split_sha256") != split_sha256:
            raise ValueError("RallyNet checkpoint does not match the physical-rally split")

        expected_train_keys = {rally_key(record) for record in dataset["train"]}
        metadata_train_keys = _normalized_rally_keys(data_metadata.get("train_rally_keys"))
        checkpoint_train_keys = _normalized_rally_keys(provenance.get("train_rally_keys"))
        metadata_requested_keys = _normalized_rally_keys(
            data_metadata.get("requested_train_rally_keys")
        )
        checkpoint_requested_keys = _normalized_rally_keys(
            provenance.get("requested_train_rally_keys")
        )
        metadata_skipped_keys = _normalized_rally_keys(
            data_metadata.get("skipped_train_rallies")
        )
        checkpoint_skipped_keys = _normalized_rally_keys(
            provenance.get("skipped_train_rallies")
        )
        if metadata_requested_keys != expected_train_keys:
            raise ValueError("Prepared RallyNet data does not cover the configured train split")
        if checkpoint_requested_keys != expected_train_keys:
            raise ValueError("RallyNet checkpoint does not cover the configured train split")
        if metadata_train_keys is None or metadata_skipped_keys is None:
            raise ValueError("Prepared RallyNet data is missing physical-rally provenance")
        if metadata_train_keys & metadata_skipped_keys:
            raise ValueError("Prepared and skipped RallyNet training rallies overlap")
        if metadata_train_keys | metadata_skipped_keys != expected_train_keys:
            raise ValueError("Prepared RallyNet training-rally provenance is incomplete")
        if checkpoint_train_keys != metadata_train_keys:
            raise ValueError(
                "RallyNet checkpoint physical training rallies do not match prepared data"
            )
        if checkpoint_skipped_keys != metadata_skipped_keys:
            raise ValueError(
                "RallyNet checkpoint skipped training rallies do not match prepared data"
            )

    def get_algorithm_name(self):
        return "rallynet"

    @staticmethod
    def _validate_rally_frame(rally_df, key):
        ball_rounds = pd.to_numeric(rally_df["ball_round"], errors="coerce")
        round_values = ball_rounds.to_numpy(dtype=np.float64)
        if (
            ball_rounds.isna().any()
            or not np.equal(ball_rounds, np.floor(ball_rounds)).all()
            or len(round_values) == 0
            or int(round_values[0]) != 1
            or np.any(np.diff(round_values) <= 0)
        ):
            raise ValueError(f"Invalid ball-round sequence for RallyNet rally {key}")

        players = rally_df["player"].astype(str).tolist()
        if any(players[index] == players[index - 1] for index in range(1, len(players))):
            raise ValueError(f"Players do not alternate in RallyNet rally {key}")

        numeric_state_columns = [column for column in state_col if column != "player"]
        numeric_states = rally_df[numeric_state_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric_states.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Invalid state values for RallyNet rally {key}")

        shot_types = pd.to_numeric(rally_df["player_type"], errors="coerce")
        if (
            shot_types.isna().any()
            or not np.equal(shot_types, np.floor(shot_types)).all()
            or not shot_types.astype(int).isin(range(1, 12)).all()
        ):
            raise ValueError(f"Invalid shot types for RallyNet rally {key}")
        shot_values = shot_types.astype(int).to_numpy()
        if shot_values[0] not in (1, 10, 11):
            raise ValueError(f"Invalid successful serve for RallyNet rally {key}")
        if len(shot_values) > 1 and not np.isin(
            shot_values[1:][shot_values[1:] != 11], np.arange(2, 11)
        ).all():
            raise ValueError(f"Invalid successful non-serve for RallyNet rally {key}")
        if np.any(shot_values[:-1] == 11):
            raise ValueError(f"Non-terminal Type 11 for RallyNet rally {key}")
        returned = rally_df.loc[shot_types.astype(int) != 11, [
            "landing_x", "landing_y", "moving_x", "moving_y"
        ]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(returned.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Invalid action coordinates for RallyNet rally {key}")

    @staticmethod
    def _coordinates(prediction, key):
        values = prediction.get(key)
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().tolist()
        return list(values)

    def simulate(self):
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        records = dataset["test"]
        if not self.all_rallies:
            records = [record for record in records if self.player_name in record["players"]]
        selected_rows = filter_rally_rows(self.df, records)
        rally_frames = {
            tuple(int(value) for value in key): frame.sort_values("ball_round").reset_index(drop=True)
            for key, frame in selected_rows.groupby(
                ["match_id", "set", "rally"], sort=False
            )
        }
        self.skipped_malformed_rallies = []
        valid_records = []
        for record in records:
            key = rally_key(record)
            if key not in rally_frames:
                self.skipped_malformed_rallies.append({
                    "match_id": key[0],
                    "set": key[1],
                    "rally": key[2],
                    "reason": "no model-ready source rows",
                })
                continue
            try:
                self._validate_rally_frame(rally_frames[key], key)
            except ValueError as exc:
                self.skipped_malformed_rallies.append({
                    "match_id": key[0],
                    "set": key[1],
                    "rally": key[2],
                    "reason": str(exc),
                })
                continue
            valid_records.append(record)
        records = valid_records
        if self.records_limit is not None:
            records = records[: self.records_limit]

        extra_steps = int(
            self.config.get("evaluation", {}).get("rollout_extra_steps", 2)
        )
        if extra_steps < 0:
            raise ValueError("evaluation.rollout_extra_steps must be non-negative")
        required_history = max(
            (len(rally_frames[rally_key(record)]) + extra_steps for record in records),
            default=0,
        )
        model_max_length = int(self.policy.model.model_config["max_sequence_length"])
        if required_history > model_max_length:
            raise ValueError(
                "RallyNet model max_sequence_length "
                f"({model_max_length}) is smaller than the required rollout history "
                f"({required_history})"
            )

        trajectories = []
        seen = set()
        description = (
            "Simulating physical rallies"
            if self.all_rallies
            else f"Simulating {self.player_name}"
        )
        for record in tqdm(records, desc=description):
            key = rally_key(record)
            if key in seen:
                raise ValueError(f"Duplicate physical test rally: {key}")
            seen.add(key)
            match_id, set_number, rally_number = key
            rally_df = rally_frames[key]

            players = list(dict.fromkeys(rally_df["player"].astype(str)))
            if len(players) != 2:
                raise ValueError(f"Expected two players for rally {key}, found {players}")
            missing_players = [player for player in players if player not in self.player2index]
            if missing_players:
                raise ValueError(
                    f"Players missing from RallyNet checkpoint for rally {key}: {missing_players}"
                )
            player_ids = {player: int(self.player2index[player]) for player in players}
            if len(set(player_ids.values())) != 2:
                raise ValueError(f"RallyNet players do not have distinct IDs for rally {key}")
            id_to_player = {player_id: player for player, player_id in player_ids.items()}

            ground_truth = [
                build_gt_shot_from_row(row) for _, row in rally_df.iterrows()
            ]
            max_rollout_steps = len(ground_truth) + extra_steps
            predictions = []
            first_row = rally_df.iloc[0]
            first_player = str(first_row["player"])
            current_state = build_state_tensor_from_row(
                first_row, player_ids[first_player]
            )
            rollout = RallyNetRollout(self.policy)
            termination_reason = "max_rollout_reached"

            for _ in range(max_rollout_steps):
                current_player_id = int(current_state[17].item())
                if current_player_id not in id_to_player:
                    raise ValueError(
                        f"Unknown RallyNet player ID {current_player_id} during rally {key}"
                    )
                current_player = id_to_player[current_player_id]
                ball_round = int(current_state[1].item())
                action, prediction = rollout.step(
                    current_state, launch=ball_round == 1
                )
                if action is None:
                    termination_reason = "action_none"
                    break

                shot_type = int(prediction["shot_type"])
                landing = self._coordinates(prediction, "landing")
                moving = self._coordinates(prediction, "moving")
                predictions.append(
                    build_pred_shot(
                        current_player,
                        shot_type,
                        landing,
                        moving,
                        ball_round,
                    )
                )
                other_player_id = next(
                    player_id
                    for player_id in id_to_player
                    if player_id != current_player_id
                )
                next_state = transition(action.cpu(), current_state.cpu(), other_player_id)
                if terminate(next_state):
                    termination_reason = "natural_terminate"
                    break
                current_state = next_state

            target_player = None if self.all_rallies else self.player_name
            opponent_player = None
            if target_player in players:
                opponent_player = next(
                    player for player in players if player != target_player
                )
            trajectories.append({
                "match_id": match_id,
                "set": set_number,
                "rally": rally_number,
                "evaluation_unit": "physical_rally",
                "players": players,
                "player_profiles": record.get("player_profiles", {}),
                "instruction_mode": "no_instruct",
                "target_player": target_player,
                "opponent_player": opponent_player,
                "gt_sequence": ground_truth,
                "pred_sequence": predictions,
                "target_player_won": None,
                **build_rollout_metadata(
                    ground_truth,
                    predictions,
                    termination_reason,
                    rollout_limit=max_rollout_steps,
                ),
            })

        self._save_trajectories_if_needed(trajectories)
        return trajectories
