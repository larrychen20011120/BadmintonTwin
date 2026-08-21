#!/usr/bin/env python3
"""Run a controlled single-state StateRAG instruction case study.

StateRAG is the project's single-state ``NaiveRAG`` method.  The experiment
holds one physical rally's initial state and the opponent policy fixed, varies
only the target player's rally-profile instruction, and repeats every condition.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
from Preprocessing.rally_feature import CLUSTER_FEATURE_COLUMNS, extract_rally_features
from Scripts.single_player_eval.evaluators.naive_rag_evaluator import NaiveRAGEvaluator
from Scripts.single_player_eval.evaluators.timerag_evaluator import TimeRAGEvaluator
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row
from Utils.artifact_names import player_database_filename
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import action_col, coord2area, state_col, terminate, transition


SHOT_ID_TO_ZH = {
    1: "發短球",
    2: "長球",
    3: "推撲球",
    4: "殺球",
    5: "接殺防守",
    6: "平球",
    7: "網前球",
    8: "挑球",
    9: "切球",
    10: "發長球",
    11: "接不到",
}

RATIO_SPECS = {
    "attack_shot_ratio": ("attack_shot_count", "target_nonserve_shot_count"),
    "defense_shot_ratio": ("defense_shot_count", "target_nonserve_shot_count"),
    "side_ratio": ("side_landing_count", "valid_landing_count"),
    "frontcourt_ratio": ("frontcourt_landing_count", "valid_landing_count"),
    "backcourt_ratio": ("backcourt_landing_count", "valid_landing_count"),
    "front_back_switch_ratio": (
        "front_back_switch_count",
        "front_back_transition_count",
    ),
    "left_right_switch_ratio": (
        "left_right_switch_count",
        "left_right_transition_count",
    ),
}

CONTINUOUS_FEATURES = {
    "log_rally_length",
    "average_movement",
    "landing_opponent_distance",
}

_WORKER_RUNNER = None


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "condition"


def _next_run_directory(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    generations = []
    for path in output_root.iterdir():
        match = re.fullmatch(r"run_(\d+)", path.name) if path.is_dir() else None
        if match:
            generations.append(int(match.group(1)))
    generation = max(generations, default=0) + 1
    while True:
        run_dir = output_root / f"run_{generation:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            generation += 1


def _instruction_vector(cluster):
    center = cluster.get("center_z") or {}
    values = np.asarray([center.get(name, np.nan) for name in CLUSTER_FEATURE_COLUMNS], dtype=float)
    return values if np.isfinite(values).all() else None


def feasible_instructions(player_profile):
    """Return profile clusters that can be used as grounded instructions."""
    feasible = []
    for cluster in player_profile.get("clusters", []):
        if not isinstance(cluster.get("label"), str) or not cluster["label"].strip():
            continue
        if not isinstance(cluster.get("instruction"), str) or not cluster["instruction"].strip():
            continue
        if _instruction_vector(cluster) is None:
            continue
        feasible.append(cluster)
    return sorted(feasible, key=lambda cluster: int(cluster["cluster_id"]))


def select_diverse_instructions(clusters, count=3, cluster_ids=None):
    """Select supported, separated cluster instructions deterministically."""
    by_id = {int(cluster["cluster_id"]): cluster for cluster in clusters}
    if cluster_ids is not None:
        missing = sorted(set(cluster_ids) - set(by_id))
        if missing:
            raise ValueError(f"Unavailable instruction cluster ids: {missing}")
        return [by_id[int(cluster_id)] for cluster_id in cluster_ids]
    if count < 1:
        raise ValueError("num_instructions must be positive")
    if len(clusters) < count:
        raise ValueError(f"Need {count} feasible instructions, found {len(clusters)}")

    first = max(
        clusters,
        key=lambda cluster: (int(cluster.get("train_size", 0)), -int(cluster["cluster_id"])),
    )
    selected = [first]
    remaining = [cluster for cluster in clusters if cluster is not first]
    while len(selected) < count:
        selected_vectors = [_instruction_vector(cluster) for cluster in selected]

        def separation(candidate):
            vector = _instruction_vector(candidate)
            minimum_distance = min(
                float(np.linalg.norm(vector - selected_vector))
                for selected_vector in selected_vectors
            )
            return minimum_distance, int(candidate.get("train_size", 0)), -int(candidate["cluster_id"])

        chosen = max(remaining, key=separation)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _database_exists(config, player_name):
    database_dir = _resolve_path(config["paths"]["naive_rag"]["data"])
    return (database_dir / player_database_filename(player_name)).is_file()


def _select_target_player(config, profiles, records, requested, instruction_count):
    players_in_records = {
        player
        for record in records
        for player in record.get("players", [])
        if len(record.get("players", [])) == 2
    }
    candidates = []
    for player_name, profile in profiles.get("players", {}).items():
        clusters = feasible_instructions(profile)
        if (
            player_name in players_in_records
            and len(clusters) >= instruction_count
            and _database_exists(config, player_name)
        ):
            candidates.append((player_name, profile, clusters))
    if requested is not None:
        matches = [candidate for candidate in candidates if candidate[0] == requested]
        if not matches:
            raise ValueError(
                f"Target player {requested!r} lacks test data, a StateRAG database, "
                f"or {instruction_count} feasible instructions"
            )
        return matches[0]
    if not candidates:
        raise ValueError("No player supports the requested StateRAG case study")
    return max(
        candidates,
        key=lambda item: (
            len(item[2]),
            int(item[1].get("test_rally_count", 0)),
            item[0],
        ),
    )


def _select_record(config, records, target_player, requested_key=None, max_shots=24):
    merged = pd.read_csv(_resolve_path(config["paths"]["merged_csv"]))
    merged = merged.dropna(subset=state_col + action_col).reset_index(drop=True)
    candidates = [
        record
        for record in records
        if target_player in record.get("players", [])
        and len(record.get("players", [])) == 2
        and all(_database_exists(config, player) for player in record["players"])
    ]
    if requested_key is not None:
        candidates = [record for record in candidates if rally_key(record) == requested_key]
        if not candidates:
            raise ValueError(f"Requested rally {requested_key} is unavailable for {target_player}")
        record = candidates[0]
        rows = filter_rally_rows(merged, [record]).sort_values("ball_round").reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"Requested rally {requested_key} has no model-ready rows")
        if int(rows.iloc[0]["ball_round"]) != 1:
            raise ValueError(f"Requested rally {requested_key} has no model-ready initial shot")
        return record, rows

    candidate_rows = filter_rally_rows(merged, candidates)
    rows_by_key = {
        tuple(int(value) for value in key): rows.sort_values("ball_round").reset_index(drop=True)
        for key, rows in candidate_rows.groupby(["match_id", "set", "rally"], sort=False)
    }
    details = []
    for record in candidates:
        rows = rows_by_key.get(rally_key(record), merged.iloc[0:0])
        if rows.empty or int(rows.iloc[0]["ball_round"]) != 1:
            continue
        details.append((record, rows, str(rows.iloc[0]["player"]) == target_player))
    if not details:
        raise ValueError(f"No model-ready test rally is available for {target_player}")

    target_first = [detail for detail in details if detail[2]] or details
    eligible_lengths = [min(len(rows), max_shots) for _, rows, _ in target_first]
    median_length = float(np.median(eligible_lengths))
    record, rows, _ = min(
        target_first,
        key=lambda detail: (
            abs(min(len(detail[1]), max_shots) - median_length),
            rally_key(detail[0]),
        ),
    )
    return record, rows


def _public_cluster(cluster):
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "label": cluster["label"].strip(),
        "description": str(cluster.get("description") or "").strip(),
        "instruction": cluster["instruction"].strip(),
        "train_size": int(cluster.get("train_size", 0)),
        "center_z": {
            name: float(cluster["center_z"][name]) for name in CLUSTER_FEATURE_COLUMNS
        },
    }


def build_case_spec(config, args):
    dataset = load_rally_dataset(
        _resolve_path(config["paths"]["rally_split"]),
        _resolve_path(config["paths"]["rally_profiles"]),
    )
    profiles = _load_json(_resolve_path(config["paths"]["rally_profiles"]))
    target_player, player_profile, clusters = _select_target_player(
        config,
        profiles,
        dataset["test"],
        args.target_player,
        len(args.cluster_ids) if args.cluster_ids else args.num_instructions,
    )
    requested_key = None
    requested_parts = (args.match_id, args.set_number, args.rally)
    if any(value is not None for value in requested_parts):
        if not all(value is not None for value in requested_parts):
            raise ValueError("--match-id, --set, and --rally must be provided together")
        requested_key = tuple(int(value) for value in requested_parts)
    record, rows = _select_record(
        config,
        dataset["test"],
        target_player,
        requested_key,
        max_shots=args.max_shots,
    )
    selected = select_diverse_instructions(
        clusters,
        count=args.num_instructions,
        cluster_ids=args.cluster_ids,
    )
    conditions = [{
        "key": "no_instruction",
        "instruction": None,
        "cluster_id": None,
        "label": "No instruction",
        "center_z": None,
    }]
    for order, cluster in enumerate(selected, start=1):
        public = _public_cluster(cluster)
        conditions.append({
            "key": f"instruction_{order:02d}_cluster_{public['cluster_id']}",
            **public,
            "applied_instruction": f"{public['label']}. {public['instruction']}",
        })

    first_row = rows.iloc[0]
    initial_state = {column: first_row[column] for column in state_col}
    initial_state["player"] = str(first_row["player"])
    initial_state = {
        key: int(value) if key in {"rally", "ball_round", "player_score", "opponent_score", "score_status", "opponent_type"}
        else float(value) if key != "player"
        else value
        for key, value in initial_state.items()
    }
    return {
        "schema_version": 1,
        "algorithm": "staterag (naive_rag)",
        "design": "single_initial_state_instruction_intervention",
        "seed": int(args.seed),
        "rollouts_per_condition": int(args.rollouts),
        "max_shots": int(args.max_shots),
        "cap_rule": "At max_shots, the actor of the final shot loses the point.",
        "natural_end_rule": "The actor of the terminal shot loses the point.",
        "target_player": target_player,
        "opponent_player": next(player for player in record["players"] if player != target_player),
        "record": copy.deepcopy(record),
        "source_data_point": {
            "key": list(rally_key(record)),
            "initial_state": initial_state,
            "ground_truth_sequence": [
                build_gt_shot_from_row(row) for _, row in rows.iterrows()
            ],
        },
        "feature_names": list(CLUSTER_FEATURE_COLUMNS),
        "smoothing_alpha": float(
            profiles.get("clustering_settings", {}).get("smoothing_alpha", 2.0)
        ),
        "target_train_statistics": copy.deepcopy(player_profile["train_statistics"]),
        "all_feasible_instructions": [_public_cluster(cluster) for cluster in clusters],
        "conditions": conditions,
        "case_grounding": bool(args.case_grounding),
        "conditioned_top_k": args.conditioned_top_k,
        "llm_seed_note": "The seed controls local RNGs; the LLM endpoint is stochastic unless configured deterministic.",
    }


class StateRAGCaseRunner:
    """Run one intervention while keeping the opponent on the normal prompt."""

    def __init__(self, spec):
        agent_type = spec.get("agent_type", "naive_rag")
        if agent_type not in {"naive_rag", "timerag"}:
            raise ValueError(f"Unsupported case-study agent type: {agent_type}")
        common = {
            "player_name": None,
            "device": "cpu",
            "conditioned_top_k": spec.get("conditioned_top_k"),
            "enable_tracing": False,
            "save_trajectories": False,
            "case_grounding": bool(spec.get("case_grounding", True)),
            "save_prompt_traces": False,
            "all_rallies": True,
            "action_architecture": spec.get(
                "action_architecture", "two_stage_v1"
            ),
        }
        evaluator_class = TimeRAGEvaluator if agent_type == "timerag" else NaiveRAGEvaluator
        if agent_type == "timerag":
            common["window_size"] = spec.get("window_size")
        self.normal = evaluator_class(instruction_mode="no_instruct", **common)
        self.instructed = evaluator_class(instruction_mode="instruct", **common)
        self.spec = spec

    @staticmethod
    def _state_snapshot(state):
        return {
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
        }

    def simulate(self, condition, rollout_index, rollout_seed):
        record = self.spec["record"]
        match_id, set_number, rally_number = rally_key(record)
        rows = filter_rally_rows(self.normal.df, [record]).sort_values("ball_round").reset_index(drop=True)
        players = list(record["players"])
        target_player = self.spec["target_player"]
        opponent_player = self.spec["opponent_player"]
        instruction = condition.get("applied_instruction")
        target_evaluator = self.instructed if instruction else self.normal
        agents = {
            target_player: target_evaluator._get_agent(target_player),
            opponent_player: self.normal._get_agent(opponent_player),
        }
        for agent in agents.values():
            agent.reset_history()

        player_ids = self.normal.player2index
        id_to_player = {int(player_ids[player]): player for player in players}
        first_row = rows.iloc[0]
        state_values = [first_row[column] for column in state_col]
        state_values[-1] = player_ids[str(first_row["player"])]
        current_state = torch.tensor(state_values, dtype=torch.float32)
        history = []
        sequence = []
        last_action = None
        termination_reason = "max_shots_reached"
        loser = None

        while len(sequence) < int(self.spec["max_shots"]):
            current_player = id_to_player[int(current_state[17].item())]
            opponent = next(player for player in players if player != current_player)
            active_instruction = instruction if current_player == target_player and instruction else ""
            info = self.normal.build_rollout_info(
                match_id,
                rally_number,
                current_state,
                last_action,
            )
            info.update({
                "set": set_number,
                "current_player_name": current_player,
                "opponent_player_name": opponent,
                "rally_history": list(history),
                "style_shift_instruction": active_instruction,
                "rally_profile_instruction": active_instruction,
                "instruction_mode": "instruct" if active_instruction else "no_instruct",
            })
            launch = int(current_state[1].item()) == 1
            action, _ = agents[current_player].predict(
                target_evaluator._state_to_env_format(current_state),
                info,
                launch,
            )
            if action is None:
                termination_reason = "action_none"
                loser = current_player
                break

            shot_type, _, landing, moving, _ = action
            landing = [float(landing[0]), float(landing[1])]
            moving = [float(moving[0]), float(moving[1])]
            shot = {
                "player": current_player,
                "ball_round": int(current_state[1].item()),
                "shot_type": int(shot_type),
                "landing_x": None if int(shot_type) == 11 else landing[0],
                "landing_y": None if int(shot_type) == 11 else landing[1],
                "moving_x": None if int(shot_type) == 11 else moving[0],
                "moving_y": None if int(shot_type) == 11 else moving[1],
                **self._state_snapshot(current_state),
            }
            sequence.append(shot)
            history.append(self.normal._build_history_stroke(current_state, current_player, action))
            action_tensor = torch.tensor(
                [landing[0], landing[1], float(shot_type), moving[0], moving[1]],
                dtype=torch.float32,
            )
            next_state = transition(action_tensor, current_state.cpu(), player_ids[opponent])
            if terminate(next_state.cpu()):
                termination_reason = "natural_terminate"
                loser = current_player
                break
            current_state = next_state
            last_action = action

        if loser is None:
            if not sequence:
                raise RuntimeError("The capped rally contains no generated shot")
            loser = sequence[-1]["player"]
        winner = next(player for player in players if player != loser)
        return {
            "schema_version": 1,
            "algorithm": self.spec.get("algorithm", "staterag (naive_rag)"),
            "action_architecture": self.spec.get(
                "action_architecture", "two_stage_v1"
            ),
            "condition": condition["key"],
            "condition_cluster_id": condition.get("cluster_id"),
            "instruction": instruction,
            "rollout_index": int(rollout_index),
            "rollout_seed": int(rollout_seed),
            "match_id": match_id,
            "set": set_number,
            "rally": rally_number,
            "players": players,
            "target_player": target_player,
            "opponent_player": opponent_player,
            "initial_state": copy.deepcopy(self.spec["source_data_point"]["initial_state"]),
            "pred_sequence": sequence,
            "rally_length": len(sequence),
            "termination_reason": termination_reason,
            "winner": winner,
            "loser": loser,
            "point_score": {winner: 1, loser: 0},
            "target_player_won": winner == target_player,
            "point_rule": (
                self.spec["cap_rule"]
                if termination_reason == "max_shots_reached"
                else "The acting player loses when the agent returns no action."
                if termination_reason == "action_none"
                else self.spec["natural_end_rule"]
            ),
        }


def _rollout_seed(base_seed, condition_key, rollout_index):
    payload = f"{base_seed}:{condition_key}:{rollout_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_worker(spec):
    global _WORKER_RUNNER
    os.chdir(PROJECT_ROOT)
    _WORKER_RUNNER = StateRAGCaseRunner(spec)


def _run_worker_task(task):
    condition, rollout_index, rollout_seed = task
    _seed_everything(rollout_seed)
    return _WORKER_RUNNER.simulate(condition, rollout_index, rollout_seed)


def _trajectory_path(run_dir, condition_key, rollout_index):
    return Path(run_dir) / "trajectories" / _safe_name(condition_key) / f"rollout_{rollout_index:03d}.json"


def _condition_rollout_count(spec, condition):
    return int(condition.get("rollout_count", spec["rollouts_per_condition"]))


def run_rollouts(spec, run_dir, workers=1):
    tasks = []
    for condition in spec["conditions"]:
        for rollout_index in range(1, _condition_rollout_count(spec, condition) + 1):
            path = _trajectory_path(run_dir, condition["key"], rollout_index)
            if path.is_file():
                continue
            seed = _rollout_seed(spec["seed"], condition["key"], rollout_index)
            tasks.append((condition, rollout_index, seed))
    if not tasks:
        return

    if workers == 1:
        runner = StateRAGCaseRunner(spec)

        def sequential_results():
            for condition, rollout_index, seed in tasks:
                _seed_everything(seed)
                yield runner.simulate(condition, rollout_index, seed)

        results = sequential_results()
    else:
        context = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(spec,),
        )
        futures = [executor.submit(_run_worker_task, task) for task in tasks]

        def completed_results():
            try:
                for future in concurrent.futures.as_completed(futures):
                    yield future.result()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        results = completed_results()

    for trajectory in tqdm(results, total=len(tasks), desc="StateRAG case-study rollouts"):
        path = _trajectory_path(
            run_dir,
            trajectory["condition"],
            trajectory["rollout_index"],
        )
        _atomic_write_json(path, trajectory)


def _trajectory_feature_row(trajectory):
    rows = []
    for shot in trajectory["pred_sequence"]:
        rows.append({
            "match_id": int(trajectory["match_id"]),
            "set": int(trajectory["set"]),
            "rally": int(trajectory["rally"]),
            "ball_round": int(shot["ball_round"]),
            "player": shot["player"],
            "type": SHOT_ID_TO_ZH[int(shot["shot_type"])],
            "landing_x": shot["landing_x"],
            "landing_y": shot["landing_y"],
            "moving_x": shot["moving_x"],
            "moving_y": shot["moving_y"],
            "player_location_x": shot["player_location_x"],
            "player_location_y": shot["player_location_y"],
            "ball_distance": shot["ball_distance"],
            "hit_x": shot["hit_x"],
            "hit_y": shot["hit_y"],
            "opponent_location_x": shot["opponent_location_x"],
            "opponent_location_y": shot["opponent_location_y"],
        })
    features = extract_rally_features(pd.DataFrame(rows))
    target = features[features["player"] == trajectory["target_player"]]
    if target.empty:
        raise ValueError(f"Rollout {trajectory['rollout_index']} has no target-player action")
    return target.iloc[0].to_dict()


def _transform_feature_row(raw, statistics, smoothing_alpha=2.0):
    alpha = float(smoothing_alpha)
    baseline = statistics["smoothing_baselines"]
    transformed = {}
    for feature, (numerator, denominator) in RATIO_SPECS.items():
        transformed[feature] = (
            float(raw[numerator]) + alpha * float(baseline[feature])
        ) / (float(raw[denominator]) + alpha)
    transformed["signed_side_bias"] = (
        float(raw["right_landing_count"])
        - float(raw["left_landing_count"])
        + alpha * float(baseline["signed_side_bias"])
    ) / (float(raw["valid_landing_count"]) + alpha)
    transformed["log_rally_length"] = math.log1p(max(float(raw["rally_length"]), 0.0))
    transformed["average_movement"] = float(raw["average_movement"])
    transformed["landing_opponent_distance"] = float(raw["landing_opponent_distance"])

    z = {}
    for feature in CLUSTER_FEATURE_COLUMNS:
        value = transformed[feature]
        if feature in CONTINUOUS_FEATURES:
            bounds = statistics["clipping"][feature]
            if math.isfinite(value):
                value = min(max(value, float(bounds["lower"])), float(bounds["upper"]))
        if not math.isfinite(value):
            value = float(statistics["imputation_medians"][feature])
        transformed[feature] = value
        z[feature] = (
            value - float(statistics["means"][feature])
        ) / float(statistics["stds"][feature])
    return transformed, z


def _distance(z_values, center):
    return float(np.linalg.norm(np.asarray([
        float(z_values[name]) - float(center[name]) for name in CLUSTER_FEATURE_COLUMNS
    ])))


def _distribution(values):
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    return {
        key: {"count": count, "rate": count / total if total else 0.0}
        for key, count in sorted(counts.items())
    }


def _mean_std(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _load_condition_trajectories(spec, run_dir, condition):
    trajectories = []
    for rollout_index in range(1, _condition_rollout_count(spec, condition) + 1):
        path = _trajectory_path(run_dir, condition["key"], rollout_index)
        if not path.is_file():
            raise ValueError(f"Missing trajectory: {path}")
        trajectory = _load_json(path)
        if trajectory.get("condition") != condition["key"]:
            raise ValueError(f"Condition mismatch in {path}")
        trajectories.append(trajectory)
    return trajectories


def summarize_run(spec, run_dir):
    statistics = spec["target_train_statistics"]
    condition_data = {}
    for condition in spec["conditions"]:
        trajectories = _load_condition_trajectories(spec, run_dir, condition)
        analyzed = []
        for trajectory in trajectories:
            raw = _trajectory_feature_row(trajectory)
            transformed, z_values = _transform_feature_row(
                raw,
                statistics,
                spec.get("smoothing_alpha", 2.0),
            )
            analyzed.append((trajectory, raw, transformed, z_values))
        condition_data[condition["key"]] = analyzed

    baseline = condition_data["no_instruction"]
    baseline_matrix = np.asarray([
        [entry[3][name] for name in CLUSTER_FEATURE_COLUMNS] for entry in baseline
    ])
    baseline_center = baseline_matrix.mean(axis=0)
    summaries = {}
    representative_paths = {}

    for condition in spec["conditions"]:
        analyzed = condition_data[condition["key"]]
        trajectories = [entry[0] for entry in analyzed]
        target_shots = [
            shot
            for trajectory in trajectories
            for shot in trajectory["pred_sequence"]
            if shot["player"] == spec["target_player"]
        ]
        all_shots = [shot for trajectory in trajectories for shot in trajectory["pred_sequence"]]
        landing_areas = []
        for shot in target_shots:
            if shot["landing_x"] is None:
                landing_areas.append("Missed shot")
            else:
                area = coord2area(float(shot["landing_x"]), float(shot["landing_y"]))
                landing_areas.append(LANG_AREA_MAPPING.get(area, "Out of Bounds"))

        if condition.get("center_z") is not None:
            distances = [_distance(entry[3], condition["center_z"]) for entry in analyzed]
            representative_index = int(np.argmin(distances))
        else:
            distances = [float(np.linalg.norm(row - baseline_center)) for row in baseline_matrix]
            representative_index = int(np.argmin(distances))
        representative = trajectories[representative_index]
        representative_paths[condition["key"]] = str(
            _trajectory_path(run_dir, condition["key"], representative["rollout_index"])
            .relative_to(run_dir)
        )

        wins = sum(bool(trajectory["target_player_won"]) for trajectory in trajectories)
        feature_summary = {
            name: _mean_std([entry[2][name] for entry in analyzed])
            for name in CLUSTER_FEATURE_COLUMNS
        }
        summary = {
            "label": condition["label"],
            "cluster_id": condition.get("cluster_id"),
            "instruction": condition.get("applied_instruction"),
            "rollouts": len(trajectories),
            "target_wins": wins,
            "target_losses": len(trajectories) - wins,
            "target_win_rate": wins / len(trajectories),
            "target_win_rate_wilson_95": _wilson_interval(wins, len(trajectories)),
            "rally_length": _mean_std([trajectory["rally_length"] for trajectory in trajectories]),
            "rally_length_distribution": _distribution(
                trajectory["rally_length"] for trajectory in trajectories
            ),
            "termination_distribution": _distribution(
                trajectory["termination_reason"] for trajectory in trajectories
            ),
            "winner_distribution": _distribution(trajectory["winner"] for trajectory in trajectories),
            "all_shot_type_distribution": _distribution(
                INDEX2EN_SHOT[int(shot["shot_type"])] for shot in all_shots
            ),
            "target_shot_type_distribution": _distribution(
                INDEX2EN_SHOT[int(shot["shot_type"])] for shot in target_shots
            ),
            "target_landing_area_distribution": _distribution(landing_areas),
            "target_feature_distribution": feature_summary,
            "representative_rollout": int(representative["rollout_index"]),
            "representative_trajectory": representative_paths[condition["key"]],
            "representative_distance": float(distances[representative_index]),
        }
        if condition.get("center_z") is not None:
            baseline_distances = [
                _distance(entry[3], condition["center_z"]) for entry in baseline
            ]
            summary["instruction_adherence"] = {
                "mean_distance_to_instruction_cluster": float(np.mean(distances)),
                "baseline_mean_distance_to_instruction_cluster": float(np.mean(baseline_distances)),
                "distance_improvement_over_no_instruction": float(
                    np.mean(baseline_distances) - np.mean(distances)
                ),
            }
        summaries[condition["key"]] = summary

    baseline_summary = summaries["no_instruction"]
    for condition in spec["conditions"]:
        if condition["key"] == "no_instruction":
            continue
        summary = summaries[condition["key"]]
        summary["effect_vs_no_instruction"] = {
            "target_win_rate_delta": summary["target_win_rate"] - baseline_summary["target_win_rate"],
            "mean_rally_length_delta": summary["rally_length"]["mean"] - baseline_summary["rally_length"]["mean"],
            "target_feature_mean_delta": {
                name: summary["target_feature_distribution"][name]["mean"]
                - baseline_summary["target_feature_distribution"][name]["mean"]
                for name in CLUSTER_FEATURE_COLUMNS
            },
        }
    return {
        "schema_version": 1,
        "target_player": spec["target_player"],
        "opponent_player": spec["opponent_player"],
        "source_key": spec["source_data_point"]["key"],
        "conditions": summaries,
    }


def _latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _rate(distribution, key):
    return float(distribution.get(key, {}).get("rate", 0.0))


def _trajectory_latex(condition, trajectory, representative_distance):
    label = _latex_escape(condition["label"])
    instruction = condition.get("applied_instruction")
    lines = [
        rf"\subsection*{{{label}}}",
        (rf"\textbf{{Instruction:}} {_latex_escape(instruction)}\\" if instruction else r"\textbf{Instruction:} None\\"),
        rf"Representative rollout: {trajectory['rollout_index']}; selection distance: {representative_distance:.3f}; winner: {_latex_escape(trajectory['winner'])}.",
        "",
        r"\begin{center}",
        r"\begin{tikzpicture}[x=2.3cm,y=2.3cm]",
        r"\draw[thick] (-1,-1) rectangle (1,1);",
        r"\draw[thick] (-1,0) -- (1,0);",
        r"\draw[densely dashed] (-1,-0.58) -- (1,-0.58);",
        r"\draw[densely dashed] (-1,0.58) -- (1,0.58);",
        r"\node[anchor=west] at (-1,-1.15) {circle: landing; triangle: recovery; blue: target; red: opponent};",
    ]
    for shot in trajectory["pred_sequence"]:
        if shot["landing_x"] is None:
            continue
        color = "blue" if shot["player"] == trajectory["target_player"] else "red"
        round_number = int(shot["ball_round"])
        lines.append(
            rf"\fill[{color}] ({float(shot['landing_x']):.4f},{float(shot['landing_y']):.4f}) circle (1.3pt) node[above,font=\tiny] {{{round_number}}};"
        )
        lines.append(
            rf"\fill[{color}] ({float(shot['moving_x']):.4f},{float(shot['moving_y']):.4f}) -- ++(-0.025,-0.04) -- ++(0.05,0) -- cycle;"
        )
    lines.extend([
        r"\end{tikzpicture}",
        r"\end{center}",
        r"\begin{longtable}{rlllp{2.1cm}}",
        r"\toprule",
        r"Round & Actor & Shot & Landing area & Landing $(x,y)$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Round & Actor & Shot & Landing area & Landing $(x,y)$ \\",
        r"\midrule",
        r"\endhead",
    ])
    for shot in trajectory["pred_sequence"]:
        actor = "Target" if shot["player"] == trajectory["target_player"] else "Opponent"
        shot_name = INDEX2EN_SHOT[int(shot["shot_type"])]
        if shot["landing_x"] is None:
            area_name = "Missed shot"
            coordinate = "--"
        else:
            area = coord2area(float(shot["landing_x"]), float(shot["landing_y"]))
            area_name = LANG_AREA_MAPPING.get(area, "Out of Bounds")
            coordinate = f"({float(shot['landing_x']):.2f}, {float(shot['landing_y']):.2f})"
        lines.append(
            f"{int(shot['ball_round'])} & {actor} & {_latex_escape(shot_name)} & "
            f"{_latex_escape(area_name)} & {coordinate} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return lines


def write_latex_report(spec, summary, run_dir):
    key = tuple(spec["source_data_point"]["key"])
    lines = [
        "% Requires: \\usepackage{booktabs,longtable,tikz}",
        r"\section{StateRAG Instruction Case Study}",
        (
            f"A fixed initial state (match {key[0]}, set {key[1]}, rally {key[2]}) "
            f"was rolled out {spec['rollouts_per_condition']} times per condition. "
            f"The target player was {_latex_escape(spec['target_player'])}; the opponent was "
            f"{_latex_escape(spec['opponent_player'])}. Rallies were capped at "
            f"{spec['max_shots']} shots, with the final actor losing at the cap."
        ),
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Outcome and target-player behavior under each instruction. Positive $\Delta d$ means the instructed rollouts moved closer to the requested training cluster than no instruction.}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Condition & Win (\%) & Length & Attack (\%) & Defense (\%) & Front (\%) & Back (\%) & $\Delta d$ \\",
        r"\midrule",
    ]
    for condition in spec["conditions"]:
        condition_summary = summary["conditions"][condition["key"]]
        features = condition_summary["target_feature_distribution"]
        adherence = condition_summary.get("instruction_adherence", {})
        lines.append(
            f"{_latex_escape(condition_summary['label'])} & "
            f"{100.0 * condition_summary['target_win_rate']:.1f} & "
            f"{condition_summary['rally_length']['mean']:.2f} & "
            f"{100.0 * features['attack_shot_ratio']['mean']:.1f} & "
            f"{100.0 * features['defense_shot_ratio']['mean']:.1f} & "
            f"{100.0 * features['frontcourt_ratio']['mean']:.1f} & "
            f"{100.0 * features['backcourt_ratio']['mean']:.1f} & "
            f"{adherence.get('distance_improvement_over_no_instruction', 0.0):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    lines.extend([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Target-player shot distribution by condition (percentage of target actions).}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Condition & Clear & Push & Smash & Defense & Drive & Net & Lob & Drop \\",
        r"\midrule",
    ])
    shot_columns = ["Clear", "Push Shot", "Smash", "Smash Defence", "Drive", "Net Shot", "Lob", "Drop"]
    for condition in spec["conditions"]:
        condition_summary = summary["conditions"][condition["key"]]
        distribution = condition_summary["target_shot_type_distribution"]
        values = " & ".join(f"{100.0 * _rate(distribution, shot):.1f}" for shot in shot_columns)
        lines.append(f"{_latex_escape(condition_summary['label'])} & {values} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    for condition in spec["conditions"]:
        condition_summary = summary["conditions"][condition["key"]]
        trajectory = _load_json(Path(run_dir) / condition_summary["representative_trajectory"])
        lines.extend(_trajectory_latex(
            condition,
            trajectory,
            condition_summary["representative_distance"],
        ))
    output_path = Path(run_dir) / "case_study.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-player")
    parser.add_argument("--match-id", type=int)
    parser.add_argument("--set", dest="set_number", type=int)
    parser.add_argument("--rally", type=int)
    parser.add_argument("--cluster-ids", type=int, nargs="+")
    parser.add_argument("--num-instructions", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--max-shots", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--conditioned-top-k", type=int)
    parser.add_argument("--case-grounding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-root",
        default="Experiments/naive_rag/case_study/staterag",
    )
    parser.add_argument("--resume-run")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.rollouts < 1 or args.max_shots < 1 or args.workers < 1:
        raise ValueError("--rollouts, --max-shots, and --workers must be positive")
    os.chdir(PROJECT_ROOT)
    with (PROJECT_ROOT / "Scripts" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.resume_run:
        run_dir = _resolve_path(args.resume_run)
        spec = _load_json(run_dir / "manifest.json")
    else:
        if args.summarize_only:
            raise ValueError("--summarize-only requires --resume-run")
        spec = build_case_spec(config, args)
        if args.dry_run:
            preview = {
                "target_player": spec["target_player"],
                "opponent_player": spec["opponent_player"],
                "source_key": spec["source_data_point"]["key"],
                "feasible_instruction_count": len(spec["all_feasible_instructions"]),
                "selected_conditions": spec["conditions"],
            }
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            return
        run_dir = _next_run_directory(_resolve_path(args.output_root))
        _atomic_write_json(run_dir / "manifest.json", spec)

    if not args.summarize_only:
        run_rollouts(spec, run_dir, workers=args.workers)
    summary = summarize_run(spec, run_dir)
    _atomic_write_json(run_dir / "summary.json", summary)
    latex_path = write_latex_report(spec, summary, run_dir)
    print(f"Case study: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
