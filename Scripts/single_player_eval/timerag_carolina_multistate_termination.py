#!/usr/bin/env python3
"""Run TimeRAG two-stage termination sweeps for five Carolina MARIN states."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.single_player_eval.staterag_case_study import (
    _atomic_write_json,
    _distribution,
    _load_condition_trajectories,
    _load_json,
    _mean_std,
    _next_run_directory,
    _resolve_path,
    build_case_spec,
    run_rollouts,
)


TARGET_PLAYER = "Carolina MARIN"
STATE_KEYS = (
    (35, 1, 1),
    (25, 1, 31),
    (93, 1, 37),
    (98, 1, 14),
    (98, 1, 26),
)
EXCLUDED_STATE_KEY = (98, 2, 1)
CLUSTER_IDS = (0, 1, 2)
ROLLOUTS_PER_CONDITION = 5
WINDOW_SIZE = 5


def _state_label(index):
    return f"state_{index:02d}"


def _build_state_specs(config, args):
    if EXCLUDED_STATE_KEY in STATE_KEYS:
        raise ValueError("The previously used Carolina state must remain excluded")
    specs = []
    for index, (match_id, set_number, rally) in enumerate(STATE_KEYS, start=1):
        namespace = SimpleNamespace(
            target_player=TARGET_PLAYER,
            match_id=match_id,
            set_number=set_number,
            rally=rally,
            cluster_ids=list(CLUSTER_IDS),
            num_instructions=len(CLUSTER_IDS),
            seed=int(args.seed) + index * 1000,
            rollouts=ROLLOUTS_PER_CONDITION,
            max_shots=int(args.max_shots),
            case_grounding=False,
            conditioned_top_k=args.conditioned_top_k,
        )
        spec = build_case_spec(config, namespace)
        spec.update({
            "algorithm": "timerag",
            "agent_type": "timerag",
            "action_architecture": "two_stage_v1",
            "window_size": WINDOW_SIZE,
            "design": "carolina_five_states_four_conditions_five_timerag_two_stage_rollouts",
            "state_index": index,
            "state_label": _state_label(index),
            "independent_rollouts": True,
            "score_carryover": False,
            "history_reset_per_rollout": True,
        })
        specs.append(spec)
    return specs


def _state_worker(payload):
    spec, state_dir = payload
    os.chdir(PROJECT_ROOT)
    run_rollouts(spec, state_dir, workers=1)
    return spec["state_label"]


def run_states(specs, run_dir, workers):
    payloads = [
        (spec, str(Path(run_dir) / "states" / spec["state_label"]))
        for spec in specs
    ]
    if workers == 1:
        return [_state_worker(payload) for payload in payloads]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(int(workers), len(payloads)),
        mp_context=context,
    ) as executor:
        return list(executor.map(_state_worker, payloads))


def _terminal_outcome(trajectory):
    reason = trajectory["termination_reason"]
    if reason == "max_shots_reached":
        return "Maximum shot limit reached"
    if reason == "action_none":
        return "Agent returned no action"
    return (
        "Missed shot"
        if int(trajectory["pred_sequence"][-1]["shot_type"]) == 11
        else "Out of bounds"
    )


def validate_trajectories(trajectories, spec):
    expected_state = spec["source_data_point"]["initial_state"]
    expected_indices = set(range(1, ROLLOUTS_PER_CONDITION + 1))
    indices = set()
    for trajectory in trajectories:
        index = int(trajectory["rollout_index"])
        indices.add(index)
        if trajectory.get("initial_state") != expected_state:
            raise ValueError(f"Rollout {index} did not reset to the source state")
        if "score_before" in trajectory or "score_after" in trajectory:
            raise ValueError("Independent rollouts must not contain score carryover fields")
        if trajectory.get("algorithm") != "timerag":
            raise ValueError(f"Rollout {index} is not labeled as TimeRAG")
        if trajectory.get("action_architecture") != "two_stage_v1":
            raise ValueError(f"Rollout {index} is not two-stage")
    if indices != expected_indices:
        raise ValueError(f"Incomplete rollout indices: {sorted(indices)}")


def _score_context(spec):
    initial = spec["source_data_point"]["initial_state"]
    server = initial["player"]
    if server == spec["target_player"]:
        target_score = initial["player_score"]
        opponent_score = initial["opponent_score"]
    else:
        target_score = initial["opponent_score"]
        opponent_score = initial["player_score"]
    return server, int(target_score), int(opponent_score)


def summarize_states(specs, run_dir):
    overall = []
    by_condition = {}
    state_summaries = []
    for spec in specs:
        state_dir = Path(run_dir) / "states" / spec["state_label"]
        condition_summaries = {}
        state_trajectories = []
        for condition in spec["conditions"]:
            trajectories = _load_condition_trajectories(spec, state_dir, condition)
            validate_trajectories(trajectories, spec)
            state_trajectories.extend(trajectories)
            overall.extend(trajectories)
            by_condition.setdefault(condition["key"], {
                "label": condition["label"],
                "cluster_id": condition.get("cluster_id"),
                "trajectories": [],
            })["trajectories"].extend(trajectories)
            condition_summaries[condition["key"]] = {
                "label": condition["label"],
                "cluster_id": condition.get("cluster_id"),
                "rollouts": len(trajectories),
                "natural_terminations": sum(
                    trajectory["termination_reason"] == "natural_terminate"
                    for trajectory in trajectories
                ),
                "termination_reasons": _distribution(
                    trajectory["termination_reason"] for trajectory in trajectories
                ),
                "terminal_outcomes": _distribution(
                    _terminal_outcome(trajectory) for trajectory in trajectories
                ),
                "length": _mean_std([
                    trajectory["rally_length"] for trajectory in trajectories
                ]),
                "length_distribution": _distribution(
                    trajectory["rally_length"] for trajectory in trajectories
                ),
                "target_wins": sum(
                    bool(trajectory["target_player_won"])
                    for trajectory in trajectories
                ),
            }
        server, target_score, opponent_score = _score_context(spec)
        state_summaries.append({
            "state_index": spec["state_index"],
            "state_label": spec["state_label"],
            "source_key": spec["source_data_point"]["key"],
            "target_player": spec["target_player"],
            "opponent_player": spec["opponent_player"],
            "server": server,
            "target_score": target_score,
            "opponent_score": opponent_score,
            "rollouts": len(state_trajectories),
            "natural_terminations": sum(
                trajectory["termination_reason"] == "natural_terminate"
                for trajectory in state_trajectories
            ),
            "terminal_outcomes": _distribution(
                _terminal_outcome(trajectory) for trajectory in state_trajectories
            ),
            "conditions": condition_summaries,
        })
    condition_totals = {}
    for condition_key, condition in by_condition.items():
        trajectories = condition["trajectories"]
        condition_totals[condition_key] = {
            "label": condition["label"],
            "cluster_id": condition["cluster_id"],
            "rollouts": len(trajectories),
            "natural_terminations": sum(
                trajectory["termination_reason"] == "natural_terminate"
                for trajectory in trajectories
            ),
            "terminal_outcomes": _distribution(
                _terminal_outcome(trajectory) for trajectory in trajectories
            ),
            "length": _mean_std([
                trajectory["rally_length"] for trajectory in trajectories
            ]),
        }
    return {
        "schema_version": 1,
        "design": "five_states_four_conditions_five_timerag_two_stage_rollouts",
        "algorithm": "timerag",
        "action_architecture": "two_stage_v1",
        "window_size": WINDOW_SIZE,
        "target_player": TARGET_PLAYER,
        "excluded_state_key": list(EXCLUDED_STATE_KEY),
        "total_trajectories": len(overall),
        "natural_terminations": sum(
            trajectory["termination_reason"] == "natural_terminate"
            for trajectory in overall
        ),
        "termination_reasons": _distribution(
            trajectory["termination_reason"] for trajectory in overall
        ),
        "terminal_outcomes": _distribution(
            _terminal_outcome(trajectory) for trajectory in overall
        ),
        "condition_totals": condition_totals,
        "states": state_summaries,
    }


def write_report(summary, run_dir):
    lines = [
        "# Carolina MARIN TimeRAG Two-Stage Termination Sweep",
        "",
        f"- Total trajectories: {summary['total_trajectories']}",
        f"- Natural terminations: {summary['natural_terminations']}",
        f"- Window size: {summary['window_size']}",
        "- Every rollout resets score, state, and history.",
        "",
        "## Condition Summary",
        "",
        "| Condition | Natural | Mean length | Missed | Out of bounds | Maximum limit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in summary["condition_totals"].values():
        outcomes = condition["terminal_outcomes"]
        lines.append(
            f"| {condition['label']} | {condition['natural_terminations']}/25 | "
            f"{condition['length']['mean']:.1f} | "
            f"{outcomes.get('Missed shot', {}).get('count', 0)} | "
            f"{outcomes.get('Out of bounds', {}).get('count', 0)} | "
            f"{outcomes.get('Maximum shot limit reached', {}).get('count', 0)} |"
        )
    lines.extend([
        "",
        "## State Summary",
        "",
        "| State | Source key | Opponent | Server | Score (Carolina-opponent) | Natural | Missed | Out of bounds | Maximum limit |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for state in summary["states"]:
        outcomes = state["terminal_outcomes"]
        lines.append(
            f"| {state['state_label']} | {tuple(state['source_key'])} | "
            f"{state['opponent_player']} | {state['server']} | "
            f"{state['target_score']}-{state['opponent_score']} | "
            f"{state['natural_terminations']}/20 | "
            f"{outcomes.get('Missed shot', {}).get('count', 0)} | "
            f"{outcomes.get('Out of bounds', {}).get('count', 0)} | "
            f"{outcomes.get('Maximum shot limit reached', {}).get('count', 0)} |"
        )
    lines.extend(["", "## State and Condition Details", ""])
    for state in summary["states"]:
        lines.extend([
            f"### {state['state_label']}: {tuple(state['source_key'])}",
            "",
            "| Condition | Natural | Mean length | Missed | Out of bounds | Maximum limit |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for condition in state["conditions"].values():
            outcomes = condition["terminal_outcomes"]
            lines.append(
                f"| {condition['label']} | {condition['natural_terminations']}/5 | "
                f"{condition['length']['mean']:.1f} | "
                f"{outcomes.get('Missed shot', {}).get('count', 0)} | "
                f"{outcomes.get('Out of bounds', {}).get('count', 0)} | "
                f"{outcomes.get('Maximum shot limit reached', {}).get('count', 0)} |"
            )
        lines.append("")
    output_path = Path(run_dir) / "termination_report.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _master_manifest(specs, args):
    return {
        "schema_version": 1,
        "design": "five_states_four_conditions_five_timerag_two_stage_rollouts",
        "algorithm": "timerag",
        "action_architecture": "two_stage_v1",
        "window_size": WINDOW_SIZE,
        "target_player": TARGET_PLAYER,
        "state_keys": [list(key) for key in STATE_KEYS],
        "excluded_state_key": list(EXCLUDED_STATE_KEY),
        "cluster_ids": list(CLUSTER_IDS),
        "conditions_per_state": 4,
        "rollouts_per_condition": ROLLOUTS_PER_CONDITION,
        "total_trajectories": len(STATE_KEYS) * 4 * ROLLOUTS_PER_CONDITION,
        "max_shots": int(args.max_shots),
        "case_grounding": False,
        "independent_rollouts": True,
        "score_carryover": False,
        "states": [{
            "state_label": spec["state_label"],
            "source_key": spec["source_data_point"]["key"],
            "manifest": f"states/{spec['state_label']}/manifest.json",
        } for spec in specs],
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shots", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditioned-top-k", type=int)
    parser.add_argument(
        "--output-root",
        default="Experiments/timerag/case_study/carolina_five_state_termination",
    )
    parser.add_argument("--resume-run")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _parser().parse_args()
    if args.max_shots < 1 or args.workers < 1:
        raise ValueError("--max-shots and --workers must be positive")
    os.chdir(PROJECT_ROOT)
    with (PROJECT_ROOT / "Scripts" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.resume_run:
        run_dir = _resolve_path(args.resume_run)
        master = _load_json(run_dir / "manifest.json")
        specs = [_load_json(run_dir / state["manifest"]) for state in master["states"]]
    else:
        if args.summarize_only:
            raise ValueError("--summarize-only requires --resume-run")
        specs = _build_state_specs(config, args)
        if args.dry_run:
            print(yaml.safe_dump(_master_manifest(specs, args), sort_keys=False))
            return
        run_dir = _next_run_directory(_resolve_path(args.output_root))
        for spec in specs:
            state_dir = run_dir / "states" / spec["state_label"]
            state_dir.mkdir(parents=True)
            _atomic_write_json(state_dir / "manifest.json", spec)
        _atomic_write_json(run_dir / "manifest.json", _master_manifest(specs, args))

    if not args.summarize_only:
        completed = run_states(specs, run_dir, args.workers)
        print(f"Completed states: {', '.join(sorted(completed))}")
    summary = summarize_states(specs, run_dir)
    _atomic_write_json(run_dir / "summary.json", summary)
    report_path = write_report(summary, run_dir)
    print(f"Case study: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
