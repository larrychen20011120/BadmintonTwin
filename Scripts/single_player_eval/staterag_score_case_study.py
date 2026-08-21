#!/usr/bin/env python3
"""Run four sequential-score StateRAG case-study conditions.

Each condition starts from the same role-relative 0:0 service state and runs 32
consecutive rallies. The rally winner gains one point and serves next. The
target player receives one of three dataset profile instructions or no
instruction; the opponent always uses the normal StateRAG prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Preprocessing.rally_data import load_rally_dataset, rally_key
from Scripts.single_player_eval.staterag_case_study import (
    StateRAGCaseRunner,
    _atomic_write_json,
    _database_exists,
    _distribution,
    _latex_escape,
    _load_json,
    _mean_std,
    _next_run_directory,
    _public_cluster,
    _resolve_path,
    _rollout_seed,
    _safe_name,
    _seed_everything,
    _trajectory_latex,
    _wilson_interval,
    feasible_instructions,
    select_diverse_instructions,
)
from Scripts.single_player_eval.trajectory_store import build_gt_shot_from_row
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import action_col, coord2area, state_col, terminate, transition


def _score_path(run_dir, condition_key, rally_index):
    return (
        Path(run_dir)
        / "conditions"
        / _safe_name(condition_key)
        / f"rally_{rally_index:03d}.json"
    )


def _initial_state_from_row(row):
    integer_fields = {
        "rally",
        "ball_round",
        "player_score",
        "opponent_score",
        "score_status",
        "opponent_type",
    }
    state = {}
    for column in state_col:
        value = row[column]
        if column == "player":
            state[column] = str(value)
        elif column in integer_fields:
            state[column] = int(value)
        else:
            state[column] = float(value)
    return state


def _candidate_sources(config, dataset, profiles, target_player=None):
    merged = pd.read_csv(_resolve_path(config["paths"]["merged_csv"]))
    merged = merged.dropna(subset=state_col + action_col).copy()
    merged = merged.sort_values(["match_id", "set", "rally", "ball_round"])
    first_rows = {
        tuple(int(value) for value in key): rows.iloc[0]
        for key, rows in merged.groupby(["match_id", "set", "rally"], sort=False)
    }
    records_by_key = {rally_key(record): record for record in dataset["test"]}
    candidates = []
    for key, record in records_by_key.items():
        row = first_rows.get(key)
        if row is None or int(row["ball_round"]) != 1:
            continue
        server = str(row["player"])
        players = list(record.get("players") or [])
        if len(players) != 2 or server not in players:
            continue
        if target_player is not None and server != target_player:
            continue
        profile = profiles.get("players", {}).get(server) or {}
        clusters = feasible_instructions(profile)
        if len(clusters) < 3:
            continue
        if not all(_database_exists(config, player) for player in players):
            continue
        candidates.append({
            "record": record,
            "row": row,
            "target_player": server,
            "opponent_player": next(player for player in players if player != server),
            "profile": profile,
            "clusters": clusters,
            "score_total": int(row["player_score"]) + int(row["opponent_score"]),
            "is_zero_zero": int(row["player_score"]) == 0 and int(row["opponent_score"]) == 0,
        })
    return candidates, merged


def _select_source(config, dataset, profiles, selection_seed, target_player=None,
                   requested_key=None):
    candidates, merged = _candidate_sources(config, dataset, profiles, target_player)
    if requested_key is not None:
        candidates = [
            candidate
            for candidate in candidates
            if rally_key(candidate["record"]) == requested_key
        ]
        if not candidates:
            raise ValueError(
                f"Requested rally {requested_key} is not a model-ready service state "
                "whose server has three profile instructions"
            )
    if not candidates:
        raise ValueError("No eligible StateRAG service-state case was found")

    zero_zero = [candidate for candidate in candidates if candidate["is_zero_zero"]]
    if zero_zero:
        pool = zero_zero
    else:
        minimum_score = min(candidate["score_total"] for candidate in candidates)
        pool = [candidate for candidate in candidates if candidate["score_total"] == minimum_score]
    pool = sorted(pool, key=lambda candidate: rally_key(candidate["record"]))
    selected = random.Random(selection_seed).choice(pool)
    key = rally_key(selected["record"])
    selected["rows"] = merged[
        (merged["match_id"].astype(int) == key[0])
        & (merged["set"].astype(int) == key[1])
        & (merged["rally"].astype(int) == key[2])
    ].sort_values("ball_round").reset_index(drop=True)
    return selected


def build_score_case_spec(config, args):
    split_path = _resolve_path(config["paths"]["rally_split"])
    profiles_path = _resolve_path(config["paths"]["rally_profiles"])
    dataset = load_rally_dataset(split_path, profiles_path)
    profiles = _load_json(profiles_path)
    requested_parts = (args.match_id, args.set_number, args.rally)
    requested_key = None
    if any(value is not None for value in requested_parts):
        if not all(value is not None for value in requested_parts):
            raise ValueError("--match-id, --set, and --rally must be provided together")
        requested_key = tuple(int(value) for value in requested_parts)
    selected = _select_source(
        config,
        dataset,
        profiles,
        selection_seed=args.selection_seed,
        target_player=args.target_player,
        requested_key=requested_key,
    )
    chosen_clusters = select_diverse_instructions(
        selected["clusters"],
        count=3,
        cluster_ids=args.cluster_ids,
    )
    conditions = [{
        "key": "no_instruction",
        "label": "No instruction",
        "cluster_id": None,
        "instruction": None,
        "applied_instruction": None,
        "center_z": None,
    }]
    for order, cluster in enumerate(chosen_clusters, start=1):
        public = _public_cluster(cluster)
        conditions.append({
            "key": f"instruction_{order:02d}_cluster_{public['cluster_id']}",
            **public,
            "applied_instruction": f"{public['label']}. {public['instruction']}",
        })

    record = selected["record"]
    rows = selected["rows"]
    source_score = {
        selected["target_player"]: int(selected["row"]["player_score"]),
        selected["opponent_player"]: int(selected["row"]["opponent_score"]),
    }
    return {
        "schema_version": 1,
        "algorithm": "staterag (naive_rag)",
        "design": "sequential_score_winner_serves",
        "seed": int(args.seed),
        "selection_seed": int(args.selection_seed),
        "rallies_per_condition": 32,
        "max_shots": int(args.max_shots),
        "target_player": selected["target_player"],
        "opponent_player": selected["opponent_player"],
        "players": list(record["players"]),
        "initial_server": selected["target_player"],
        "initial_score": {
            selected["target_player"]: 0,
            selected["opponent_player"]: 0,
        },
        "source_score": source_score,
        "source_is_zero_zero": bool(selected["is_zero_zero"]),
        "record": copy.deepcopy(record),
        "source_data_point": {
            "key": list(rally_key(record)),
            "initial_state": _initial_state_from_row(selected["row"]),
            "ground_truth_sequence": [
                build_gt_shot_from_row(row) for _, row in rows.iterrows()
            ],
        },
        "conditions": conditions,
        "all_feasible_instruction_count": len(selected["clusters"]),
        "case_grounding": False,
        "conditioned_top_k": args.conditioned_top_k,
        "score_rule": "The rally winner gains one point and serves the next rally.",
        "cap_rule": "At max_shots, the actor of the final shot loses the rally.",
        "state_rule": (
            "Court geometry is reset to the same role-relative service template; "
            "server identity and score fields are updated before each rally."
        ),
    }


def build_service_state(spec, server, score, rally_index, player_ids):
    """Build the fixed service template from the current server's perspective."""
    players = list(spec["players"])
    if server not in players:
        raise ValueError(f"Unknown server: {server}")
    receiver = next(player for player in players if player != server)
    template = spec["source_data_point"]["initial_state"]
    values = [template[column] for column in state_col]
    values[0] = int(rally_index)
    values[1] = 1
    values[2] = int(score[server])
    values[3] = int(score[receiver])
    values[4] = int(score[server]) - int(score[receiver])
    values[12] = 0
    values[17] = int(player_ids[server])
    return torch.tensor(values, dtype=torch.float32), receiver


class SequentialScoreRunner(StateRAGCaseRunner):
    def simulate_scored_rally(self, condition, rally_index, rollout_seed, server, score):
        players = list(self.spec["players"])
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
        current_state, receiver = build_service_state(
            self.spec,
            server,
            score,
            rally_index,
            player_ids,
        )
        score_before = {player: int(score[player]) for player in players}
        sequence = []
        history = []
        last_action = None
        termination_reason = "max_shots_reached"
        loser = None
        match_id, set_number, _ = self.spec["source_data_point"]["key"]

        while len(sequence) < int(self.spec["max_shots"]):
            current_player = id_to_player[int(current_state[17].item())]
            current_opponent = next(player for player in players if player != current_player)
            active_instruction = (
                instruction if current_player == target_player and instruction else ""
            )
            info = self.normal.build_rollout_info(
                match_id,
                rally_index,
                current_state,
                last_action,
            )
            info.update({
                "set": set_number,
                "current_player_name": current_player,
                "opponent_player_name": current_opponent,
                "rally_history": list(history),
                "style_shift_instruction": active_instruction,
                "rally_profile_instruction": active_instruction,
                "instruction_mode": "instruct" if active_instruction else "no_instruct",
            })
            action, _ = agents[current_player].predict(
                self.normal._state_to_env_format(current_state),
                info,
                launch=len(sequence) == 0,
            )
            if action is None:
                termination_reason = "action_none"
                loser = current_player
                break

            shot_type, _, landing, moving, _ = action
            landing = [float(landing[0]), float(landing[1])]
            moving = [float(moving[0]), float(moving[1])]
            sequence.append({
                "player": current_player,
                "ball_round": int(current_state[1].item()),
                "shot_type": int(shot_type),
                "landing_x": None if int(shot_type) == 11 else landing[0],
                "landing_y": None if int(shot_type) == 11 else landing[1],
                "moving_x": None if int(shot_type) == 11 else moving[0],
                "moving_y": None if int(shot_type) == 11 else moving[1],
                **self._state_snapshot(current_state),
            })
            history.append(
                self.normal._build_history_stroke(current_state, current_player, action)
            )
            action_tensor = torch.tensor(
                [landing[0], landing[1], float(shot_type), moving[0], moving[1]],
                dtype=torch.float32,
            )
            next_state = transition(
                action_tensor,
                current_state.cpu(),
                player_ids[current_opponent],
            )
            if terminate(next_state.cpu()):
                termination_reason = "natural_terminate"
                loser = current_player
                break
            current_state = next_state
            last_action = action

        if loser is None:
            if not sequence:
                raise RuntimeError("A capped rally contains no generated action")
            loser = sequence[-1]["player"]
        winner = next(player for player in players if player != loser)
        score_after = dict(score_before)
        score_after[winner] += 1
        return {
            "schema_version": 1,
            "algorithm": "staterag (naive_rag)",
            "condition": condition["key"],
            "condition_cluster_id": condition.get("cluster_id"),
            "instruction": instruction,
            "rally_index": int(rally_index),
            "rollout_index": int(rally_index),
            "rollout_seed": int(rollout_seed),
            "source_key": list(self.spec["source_data_point"]["key"]),
            "players": players,
            "target_player": target_player,
            "opponent_player": opponent_player,
            "server": server,
            "receiver": receiver,
            "score_before": score_before,
            "pred_sequence": sequence,
            "rally_length": len(sequence),
            "termination_reason": termination_reason,
            "winner": winner,
            "loser": loser,
            "score_after": score_after,
            "next_server": winner,
            "target_player_won": winner == target_player,
        }


def _validate_chain_trajectory(trajectory, spec, condition, rally_index, score, server):
    expected_score = {player: int(score[player]) for player in spec["players"]}
    if trajectory.get("condition") != condition["key"]:
        raise ValueError(f"Condition mismatch at rally {rally_index}")
    if int(trajectory.get("rally_index", -1)) != rally_index:
        raise ValueError(f"Rally index mismatch at rally {rally_index}")
    if trajectory.get("score_before") != expected_score:
        raise ValueError(f"Score continuity mismatch at rally {rally_index}")
    if trajectory.get("server") != server:
        raise ValueError(f"Server continuity mismatch at rally {rally_index}")
    winner = trajectory.get("winner")
    if winner not in spec["players"] or trajectory.get("next_server") != winner:
        raise ValueError(f"Winner/server mismatch at rally {rally_index}")
    expected_after = dict(expected_score)
    expected_after[winner] += 1
    if trajectory.get("score_after") != expected_after:
        raise ValueError(f"Score update mismatch at rally {rally_index}")


def run_condition_chain(spec, condition, run_dir, runner=None):
    runner = runner or SequentialScoreRunner(spec)
    score = {player: int(spec["initial_score"][player]) for player in spec["players"]}
    server = spec["initial_server"]
    missing_seen = False
    for rally_index in range(1, int(spec["rallies_per_condition"]) + 1):
        output_path = _score_path(run_dir, condition["key"], rally_index)
        if output_path.is_file():
            if missing_seen:
                raise ValueError(
                    f"Non-contiguous resume files for {condition['key']} at rally {rally_index}"
                )
            trajectory = _load_json(output_path)
        else:
            missing_seen = True
            seed = _rollout_seed(spec["seed"], condition["key"], rally_index)
            _seed_everything(seed)
            trajectory = runner.simulate_scored_rally(
                condition,
                rally_index,
                seed,
                server,
                score,
            )
            _atomic_write_json(output_path, trajectory)
        _validate_chain_trajectory(
            trajectory,
            spec,
            condition,
            rally_index,
            score,
            server,
        )
        score = {player: int(trajectory["score_after"][player]) for player in spec["players"]}
        server = trajectory["next_server"]
    return {
        "condition": condition["key"],
        "final_score": score,
        "next_server": server,
    }


def _condition_worker(payload):
    spec, condition, run_dir = payload
    os.chdir(PROJECT_ROOT)
    return run_condition_chain(spec, condition, run_dir)


def run_all_conditions(spec, run_dir, workers=4):
    payloads = [(spec, condition, str(run_dir)) for condition in spec["conditions"]]
    if workers == 1:
        return [_condition_worker(payload) for payload in payloads]
    worker_count = min(int(workers), len(payloads))
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = [executor.submit(_condition_worker, payload) for payload in payloads]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def _load_chain(spec, run_dir, condition):
    trajectories = []
    score = {player: int(spec["initial_score"][player]) for player in spec["players"]}
    server = spec["initial_server"]
    for rally_index in range(1, int(spec["rallies_per_condition"]) + 1):
        path = _score_path(run_dir, condition["key"], rally_index)
        if not path.is_file():
            raise ValueError(f"Missing trajectory: {path}")
        trajectory = _load_json(path)
        _validate_chain_trajectory(
            trajectory,
            spec,
            condition,
            rally_index,
            score,
            server,
        )
        trajectories.append(trajectory)
        score = trajectory["score_after"]
        server = trajectory["next_server"]
    return trajectories


def summarize_score_run(spec, run_dir):
    summaries = {}
    target = spec["target_player"]
    for condition in spec["conditions"]:
        trajectories = _load_chain(spec, run_dir, condition)
        target_shots = [
            shot
            for trajectory in trajectories
            for shot in trajectory["pred_sequence"]
            if shot["player"] == target
        ]
        landing_areas = []
        for shot in target_shots:
            if shot["landing_x"] is None:
                landing_areas.append("Missed shot")
            else:
                area = coord2area(float(shot["landing_x"]), float(shot["landing_y"]))
                landing_areas.append(LANG_AREA_MAPPING.get(area, "Out of Bounds"))
        wins = sum(bool(trajectory["target_player_won"]) for trajectory in trajectories)
        majority_won = wins >= len(trajectories) / 2
        representative_pool = [
            trajectory
            for trajectory in trajectories
            if bool(trajectory["target_player_won"]) == majority_won
        ] or trajectories
        median_length = float(np.median([
            trajectory["rally_length"] for trajectory in representative_pool
        ]))
        representative = min(
            representative_pool,
            key=lambda trajectory: (
                abs(trajectory["rally_length"] - median_length),
                trajectory["rally_index"],
            ),
        )
        final_score = trajectories[-1]["score_after"]
        summaries[condition["key"]] = {
            "label": condition["label"],
            "cluster_id": condition.get("cluster_id"),
            "instruction": condition.get("applied_instruction"),
            "rallies": len(trajectories),
            "final_score": final_score,
            "target_wins": wins,
            "target_losses": len(trajectories) - wins,
            "target_win_rate": wins / len(trajectories),
            "target_win_rate_wilson_95": _wilson_interval(wins, len(trajectories)),
            "rally_length": _mean_std([
                trajectory["rally_length"] for trajectory in trajectories
            ]),
            "rally_length_distribution": _distribution(
                trajectory["rally_length"] for trajectory in trajectories
            ),
            "termination_distribution": _distribution(
                trajectory["termination_reason"] for trajectory in trajectories
            ),
            "server_distribution": _distribution(
                trajectory["server"] for trajectory in trajectories
            ),
            "target_shot_type_distribution": _distribution(
                INDEX2EN_SHOT[int(shot["shot_type"])] for shot in target_shots
            ),
            "target_landing_area_distribution": _distribution(landing_areas),
            "score_progression": [{
                "rally_index": trajectory["rally_index"],
                "server": trajectory["server"],
                "winner": trajectory["winner"],
                "score_before": trajectory["score_before"],
                "score_after": trajectory["score_after"],
            } for trajectory in trajectories],
            "representative_rally": representative["rally_index"],
            "representative_trajectory": str(
                _score_path(
                    run_dir,
                    condition["key"],
                    representative["rally_index"],
                ).relative_to(run_dir)
            ),
        }
    return {
        "schema_version": 1,
        "target_player": target,
        "opponent_player": spec["opponent_player"],
        "source_key": spec["source_data_point"]["key"],
        "conditions": summaries,
    }


def _score_progression_latex(spec, condition, condition_summary):
    target = spec["target_player"]
    opponent = spec["opponent_player"]
    target_points = [(0, 0)]
    opponent_points = [(0, 0)]
    for item in condition_summary["score_progression"]:
        target_points.append((item["rally_index"], item["score_after"][target]))
        opponent_points.append((item["rally_index"], item["score_after"][opponent]))
    target_path = " -- ".join(f"({x},{y})" for x, y in target_points)
    opponent_path = " -- ".join(f"({x},{y})" for x, y in opponent_points)
    return [
        rf"\subsection*{{Score progression: {_latex_escape(condition['label'])}}}",
        r"\begin{center}",
        r"\begin{tikzpicture}[x=0.22cm,y=0.16cm]",
        r"\draw[->] (0,0) -- (33,0) node[right] {rally};",
        r"\draw[->] (0,0) -- (0,33) node[above] {points};",
        rf"\draw[blue,thick] {target_path};",
        rf"\draw[red,thick] {opponent_path};",
        rf"\node[blue,anchor=west] at (20,31) {{{_latex_escape(target)}}};",
        rf"\node[red,anchor=west] at (20,28) {{{_latex_escape(opponent)}}};",
        r"\end{tikzpicture}",
        r"\end{center}",
        "",
    ]


def write_score_latex(spec, summary, run_dir):
    source_key = spec["source_data_point"]["key"]
    lines = [
        "% Requires: \\usepackage{booktabs,longtable,tikz}",
        r"\section{Sequential-Score StateRAG Case Study}",
        (
            f"All conditions use the same role-relative service state from match "
            f"{source_key[0]}, set {source_key[1]}, rally {source_key[2]}. Each condition "
            f"starts at 0--0 and runs {spec['rallies_per_condition']} rallies; the winner "
            "scores one point and serves next."
        ),
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sequential-score outcomes and rally lengths.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Target score & Opponent score & Target win (\%) & Mean length \\",
        r"\midrule",
    ]
    target = spec["target_player"]
    opponent = spec["opponent_player"]
    for condition in spec["conditions"]:
        item = summary["conditions"][condition["key"]]
        lines.append(
            f"{_latex_escape(item['label'])} & {item['final_score'][target]} & "
            f"{item['final_score'][opponent]} & {100 * item['target_win_rate']:.1f} & "
            f"{item['rally_length']['mean']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    for condition in spec["conditions"]:
        item = summary["conditions"][condition["key"]]
        lines.extend(_score_progression_latex(spec, condition, item))
        trajectory = _load_json(Path(run_dir) / item["representative_trajectory"])
        lines.extend(_trajectory_latex(condition, trajectory, 0.0))
    output_path = Path(run_dir) / "case_study.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-player")
    parser.add_argument("--match-id", type=int)
    parser.add_argument("--set", dest="set_number", type=int)
    parser.add_argument("--rally", type=int)
    parser.add_argument("--cluster-ids", type=int, nargs=3)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shots", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditioned-top-k", type=int)
    parser.add_argument(
        "--output-root",
        default="Experiments/naive_rag/case_study/staterag_sequential_score",
    )
    parser.add_argument("--resume-run")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.max_shots < 1 or args.workers < 1:
        raise ValueError("--max-shots and --workers must be positive")
    os.chdir(PROJECT_ROOT)
    with (PROJECT_ROOT / "Scripts" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.resume_run:
        run_dir = _resolve_path(args.resume_run)
        spec = _load_json(run_dir / "manifest.json")
    else:
        if args.summarize_only:
            raise ValueError("--summarize-only requires --resume-run")
        spec = build_score_case_spec(config, args)
        if args.dry_run:
            print(json.dumps({
                "source_key": spec["source_data_point"]["key"],
                "source_is_zero_zero": spec["source_is_zero_zero"],
                "target_player": spec["target_player"],
                "opponent_player": spec["opponent_player"],
                "conditions": spec["conditions"],
            }, ensure_ascii=False, indent=2))
            return
        run_dir = _next_run_directory(_resolve_path(args.output_root))
        _atomic_write_json(run_dir / "manifest.json", spec)

    if not args.summarize_only:
        results = run_all_conditions(spec, run_dir, workers=args.workers)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    summary = summarize_score_run(spec, run_dir)
    _atomic_write_json(run_dir / "summary.json", summary)
    latex_path = write_score_latex(spec, summary, run_dir)
    print(f"Case study: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
