#!/usr/bin/env python3
"""Run and render a balanced StateRAG case study for Carolina MARIN."""

from __future__ import annotations

import argparse
import itertools
import math
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
    _latex_escape,
    _load_condition_trajectories,
    _load_json,
    _mean_std,
    _next_run_directory,
    _resolve_path,
    _trajectory_path,
    build_case_spec,
    run_rollouts,
)
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


TARGET_PLAYER = "Carolina MARIN"
SOURCE_KEY = (98, 2, 1)
CLUSTER_IDS = (0, 1, 2)
ROLLOUTS_PER_CONDITION = 10

CONDITION_KINDS = {
    "no_instruction": "normal",
    "instruction_01_cluster_0": "high_switching_deep",
    "instruction_02_cluster_1": "aggressive_midcourt_deep",
    "instruction_03_cluster_2": "deep_defensive_reset",
}

COLORS = {
    "normal": "normalgray",
    "high_switching_deep": "switchblue",
    "aggressive_midcourt_deep": "attackorange",
    "deep_defensive_reset": "resetgreen",
}

ATTACK_SHOTS = {"Push Shot", "Smash", "Drive", "Net Shot", "Drop"}
RESET_SHOTS = {"Clear", "Lob", "Smash Defence"}


def _area_name(x, y):
    if x is None or y is None:
        return "Missed shot"
    return LANG_AREA_MAPPING.get(coord2area(float(x), float(y)), "Out of Bounds")


def _depth(area):
    if area.startswith("Front"):
        return "front"
    if area.startswith("Rear") or area.startswith("Deep"):
        return "back"
    return "middle"


def _side(area):
    if "Left" in area:
        return "left"
    if "Right" in area:
        return "right"
    return "center"


def _target_actions(trajectory):
    target = trajectory["target_player"]
    final_round = int(trajectory["pred_sequence"][-1]["ball_round"])
    actions = []
    for shot in trajectory["pred_sequence"]:
        shot_type = int(shot["shot_type"])
        if shot["player"] != target or shot_type in {1, 10, 11}:
            continue
        if (
            trajectory["termination_reason"] == "natural_terminate"
            and int(shot["ball_round"]) == final_round
        ):
            continue
        area = _area_name(shot["landing_x"], shot["landing_y"])
        actions.append({
            "round": int(shot["ball_round"]),
            "shot": INDEX2EN_SHOT[shot_type],
            "area": area,
            "depth": _depth(area),
            "side": _side(area),
        })
    return actions


def assess_alignment(trajectory, kind):
    if kind == "normal":
        return {
            "rating": "not applicable",
            "highlight_rounds": [],
            "explanation": "Reference trajectory without a tactical instruction.",
        }
    actions = _target_actions(trajectory)
    if not actions:
        return {
            "rating": "weak",
            "highlight_rounds": [],
            "explanation": "No successful non-serve Carolina MARIN action is available.",
        }

    if kind == "high_switching_deep":
        depth_switches = 0
        side_switches = 0
        highlighted = set()
        for previous, current in zip(actions, actions[1:]):
            if {previous["depth"], current["depth"]} == {"front", "back"}:
                depth_switches += 1
                highlighted.update({previous["round"], current["round"]})
            if {previous["side"], current["side"]} == {"left", "right"}:
                side_switches += 1
                highlighted.update({previous["round"], current["round"]})
        back_landings = sum(action["depth"] == "back" for action in actions)
        rating = (
            "strong"
            if depth_switches >= 2 and side_switches >= 1 and len(actions) >= 5
            else "partial"
            if depth_switches or side_switches or back_landings
            else "weak"
        )
        explanation = (
            f"{rating.title()} alignment: {depth_switches} front-to-back switches, "
            f"{side_switches} left-to-right switches, and {back_landings} back-court "
            f"landings across a {trajectory['rally_length']}-shot rally."
        )
    elif kind == "aggressive_midcourt_deep":
        highlighted = {
            action["round"]
            for action in actions
            if action["shot"] in ATTACK_SHOTS
            and action["depth"] in {"middle", "back"}
        }
        rating = "strong" if len(highlighted) >= 2 else "partial" if highlighted else "weak"
        explanation = (
            f"{rating.title()} alignment: {len(highlighted)} attacking shots land in "
            "the middle or back court to pressure the opponent with depth."
        )
    elif kind == "deep_defensive_reset":
        highlighted = {
            action["round"]
            for action in actions
            if action["shot"] in RESET_SHOTS and action["depth"] == "back"
        }
        back_landings = sum(action["depth"] == "back" for action in actions)
        rating = "strong" if len(highlighted) >= 2 else "partial" if highlighted else "weak"
        explanation = (
            f"{rating.title()} alignment: {len(highlighted)} defensive reset shots and "
            f"{back_landings} total placements reach the back court."
        )
    else:
        raise ValueError(f"Unknown Carolina condition kind: {kind}")
    return {
        "rating": rating,
        "highlight_rounds": sorted(highlighted),
        "explanation": explanation,
    }


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


def validate_independent_rollouts(trajectories, spec):
    expected_state = spec["source_data_point"]["initial_state"]
    expected_architecture = spec.get("action_architecture", "two_stage_v1")
    expected_algorithm = (
        "naive_rag_single_stage"
        if expected_architecture == "single_stage_v1"
        else "staterag (naive_rag)"
    )
    expected_indices = set(range(1, ROLLOUTS_PER_CONDITION + 1))
    indices = set()
    for trajectory in trajectories:
        index = int(trajectory["rollout_index"])
        indices.add(index)
        if trajectory.get("initial_state") != expected_state:
            raise ValueError(f"Rollout {index} did not reset to the source state")
        if "score_before" in trajectory or "score_after" in trajectory:
            raise ValueError("Independent rollouts must not contain score carryover fields")
        if trajectory.get("action_architecture") != expected_architecture:
            raise ValueError(f"Rollout {index} has the wrong action architecture")
        if trajectory.get("algorithm") != expected_algorithm:
            raise ValueError(f"Rollout {index} has the wrong algorithm label")
    if indices != expected_indices:
        raise ValueError(f"Incomplete rollout indices: {sorted(indices)}")


def select_balanced_cases(condition_trajectories):
    keys = list(condition_trajectories)
    pools = [condition_trajectories[key] for key in keys]

    best = None
    best_key = None
    for combination in itertools.product(*pools):
        outcomes = [_terminal_outcome(trajectory) for trajectory in combination]
        lengths = [int(trajectory["rally_length"]) for trajectory in combination]
        winners = [trajectory["winner"] for trajectory in combination]
        alignments = [
            assess_alignment(trajectory, CONDITION_KINDS[key])
            for key, trajectory in zip(keys, combination)
        ]
        mean = sum(lengths) / len(lengths)
        standard_deviation = math.sqrt(
            sum((length - mean) ** 2 for length in lengths) / len(lengths)
        )
        score = (
            -len(set(outcomes)),
            max(lengths) - min(lengths),
            -len(set(zip(outcomes, winners))),
            standard_deviation,
            -sum(len(alignment["highlight_rounds"]) for alignment in alignments),
            tuple(int(trajectory["rollout_index"]) for trajectory in combination),
        )
        if best_key is None or score < best_key:
            best_key = score
            best = list(zip(keys, combination, alignments))
    return best


def _build_spec(config, args):
    namespace = SimpleNamespace(
        target_player=TARGET_PLAYER,
        match_id=SOURCE_KEY[0],
        set_number=SOURCE_KEY[1],
        rally=SOURCE_KEY[2],
        cluster_ids=list(CLUSTER_IDS),
        num_instructions=len(CLUSTER_IDS),
        seed=int(args.seed),
        rollouts=ROLLOUTS_PER_CONDITION,
        max_shots=int(args.max_shots),
        case_grounding=False,
        conditioned_top_k=args.conditioned_top_k,
    )
    spec = build_case_spec(config, namespace)
    spec.update({
        "algorithm": (
            "naive_rag_single_stage"
            if args.action_architecture == "single_stage_v1"
            else "staterag (naive_rag)"
        ),
        "design": (
            "carolina_naive_rag_single_stage_four_conditions_ten_independent_rollouts"
            if args.action_architecture == "single_stage_v1"
            else "carolina_naive_rag_two_stage_four_conditions_ten_independent_rollouts"
        ),
        "action_architecture": args.action_architecture,
        "independent_rollouts": True,
        "score_carryover": False,
        "history_reset_per_rollout": True,
    })
    return spec


def _condition_title(condition):
    cluster_id = condition.get("cluster_id")
    return (
        "Normal play"
        if cluster_id is None
        else f"C{cluster_id}: {condition['label']}"
    )


def summarize_run(spec, run_dir):
    trajectories_by_condition = {}
    condition_summaries = {}
    for condition in spec["conditions"]:
        trajectories = _load_condition_trajectories(spec, run_dir, condition)
        validate_independent_rollouts(trajectories, spec)
        trajectories_by_condition[condition["key"]] = trajectories
        condition_summaries[condition["key"]] = {
            "title": _condition_title(condition),
            "instruction": (
                "No tactical instruction; follow the learned normal policy."
                if condition["key"] == "no_instruction"
                else condition["instruction"]
            ),
            "rollouts": len(trajectories),
            "natural_terminations": sum(
                trajectory["termination_reason"] == "natural_terminate"
                for trajectory in trajectories
            ),
            "length": _mean_std([
                trajectory["rally_length"] for trajectory in trajectories
            ]),
            "length_distribution": _distribution(
                trajectory["rally_length"] for trajectory in trajectories
            ),
            "outcome_distribution": _distribution(
                _terminal_outcome(trajectory) for trajectory in trajectories
            ),
        }

    selected = select_balanced_cases(trajectories_by_condition)
    selected_cases = []
    for condition_key, trajectory, alignment in selected:
        condition = next(item for item in spec["conditions"] if item["key"] == condition_key)
        kind = CONDITION_KINDS[condition_key]
        selected_cases.append({
            "condition": condition_key,
            "kind": kind,
            "title": _condition_title(condition),
            "instruction": (
                "No tactical instruction; follow the learned normal policy."
                if kind == "normal"
                else condition["instruction"]
            ),
            "color": COLORS[kind],
            "rollout_index": int(trajectory["rollout_index"]),
            "trajectory": str(
                _trajectory_path(run_dir, condition_key, trajectory["rollout_index"])
                .relative_to(run_dir)
            ),
            "rally_length": int(trajectory["rally_length"]),
            "winner": trajectory["winner"],
            "terminal_outcome": _terminal_outcome(trajectory),
            "alignment": alignment,
        })

    selected_lengths = [case["rally_length"] for case in selected_cases]
    initial = spec["source_data_point"]["initial_state"]
    match_id, set_number, rally_number = spec["source_data_point"]["key"]
    return {
        "schema_version": 1,
        "design": spec["design"],
        "action_architecture": spec.get("action_architecture", "two_stage_v1"),
        "target_player": spec["target_player"],
        "opponent_player": spec["opponent_player"],
        "source_key": spec["source_data_point"]["key"],
        "total_trajectories": sum(len(value) for value in trajectories_by_condition.values()),
        "conditions": condition_summaries,
        "selection_priority": [
            "maximize distinct terminal outcomes",
            "minimize selected rally-length range",
            "maximize distinct outcome-and-winner pairs",
            "minimize selected rally-length standard deviation",
            "maximize instruction-aligned highlighted actions",
        ],
        "selected_length_range": max(selected_lengths) - min(selected_lengths),
        "distinct_terminal_outcomes": len({case["terminal_outcome"] for case in selected_cases}),
        "starting_situation": (
            f"Source match {match_id}, set {set_number}, rally {rally_number}. "
            f"{spec['target_player']} serves against {spec['opponent_player']} at "
            f"{initial['player_score']}--{initial['opponent_score']}. Their normalized "
            f"court positions are ({initial['player_location_x']:.2f}, "
            f"{initial['player_location_y']:.2f}) and "
            f"({initial['opponent_location_x']:.2f}, "
            f"{initial['opponent_location_y']:.2f}), respectively. Every case starts "
            "from this identical situation with an empty rally history."
        ),
        "selected_cases": selected_cases,
    }


def _action_text(shot, trajectory):
    role = "Player" if shot["player"] == trajectory["target_player"] else "Opponent"
    shot_name = INDEX2EN_SHOT[int(shot["shot_type"])]
    action = (
        shot_name
        if int(shot["shot_type"]) == 11
        else f"{shot_name} to {_area_name(shot['landing_x'], shot['landing_y'])}"
    )
    return int(shot["ball_round"]), role, action


def _action_cell(shot, trajectory, highlighted_rounds, color):
    ball_round, role, action = _action_text(shot, trajectory)
    rendered_action = _latex_escape(action)
    if role == "Player" and ball_round in highlighted_rounds:
        rendered_action = rf"\textcolor{{{color}}}{{\textbf{{{rendered_action}}}}}"
    return f"{ball_round}. {_latex_escape(role)}: {rendered_action}"


def _result_text(trajectory):
    final = trajectory["pred_sequence"][-1]
    if int(final["shot_type"]) == 11:
        return f"{trajectory['winner']} wins after {trajectory['loser']} misses the shuttle."
    if trajectory["termination_reason"] == "max_shots_reached":
        return f"{trajectory['winner']} wins when the rally reaches the maximum shot limit."
    shot_name = INDEX2EN_SHOT[int(final["shot_type"])]
    return (
        f"{trajectory['winner']} wins after {trajectory['loser']}'s "
        f"{shot_name} lands out of bounds."
    )


def write_latex(summary, run_dir):
    columns = [
        (case, _load_json(Path(run_dir) / case["trajectory"]))
        for case in summary["selected_cases"]
    ]
    architecture_label = (
        "single-stage NaiveRAG"
        if summary["action_architecture"] == "single_stage_v1"
        else "two-stage NaiveRAG"
    )
    lines = [
        "% Requires: \\usepackage{xcolor,booktabs,tabularx,array}",
        r"\definecolor{normalgray}{HTML}{4F5965}",
        r"\definecolor{switchblue}{HTML}{2457A6}",
        r"\definecolor{attackorange}{HTML}{D66B20}",
        r"\definecolor{resetgreen}{HTML}{178A62}",
        r"\begin{table*}[t]",
        r"\raggedright",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabularx}{\textwidth}{*{4}{>{\raggedright\arraybackslash}X}}",
        r"\toprule",
    ]
    lines.append(" & ".join(
        rf"\textcolor{{{case['color']}}}{{\textbf{{\large {_latex_escape(case['title'])}}}}}"
        for case, _ in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(" & ".join(_latex_escape(case["instruction"]) for case, _ in columns) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    context = rf"\textit{{\textbf{{Starting situation.}} {_latex_escape(summary['starting_situation'])}}}"
    lines.append(" & ".join(context for _ in columns) + r" \\")
    lines.append(r"\midrule")
    maximum_length = max(len(trajectory["pred_sequence"]) for _, trajectory in columns)
    for row_index in range(maximum_length):
        cells = []
        for case, trajectory in columns:
            if row_index >= len(trajectory["pred_sequence"]):
                cells.append("")
            else:
                cells.append(_action_cell(
                    trajectory["pred_sequence"][row_index],
                    trajectory,
                    set(case["alignment"]["highlight_rounds"]),
                    case["color"],
                ))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(
        rf"\textbf{{Instruction alignment.}} \textcolor{{{case['color']}}}"
        rf"{{{_latex_escape(case['alignment']['explanation'])}}}"
        for case, _ in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(" & ".join(
        rf"\textbf{{Result.}} {_latex_escape(_result_text(trajectory))}"
        for _, trajectory in columns
    ) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        (
            rf"\caption{{Carolina MARIN {architecture_label} trajectories from one "
            r"identical starting "
            r"situation. Every condition has ten independent rollouts. The selected set "
            r"first maximizes terminal-outcome diversity and then minimizes the rally-length "
            r"range. Colored Player actions mark shots or sequences aligned with the "
            r"instruction in the same column. Opponent denotes Akane YAMAGUCHI.}"
        ),
        r"\label{tab:carolina-balanced-case-study}",
        r"\end{table*}",
    ])
    output_path = Path(run_dir) / "explanatory_case_study.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shots", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditioned-top-k", type=int)
    parser.add_argument(
        "--action-architecture",
        choices=("single_stage_v1", "two_stage_v1"),
        default="single_stage_v1",
    )
    parser.add_argument(
        "--output-root",
        default="Experiments/naive_rag/case_study/carolina_naive_rag_single_stage",
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
        spec = _load_json(run_dir / "manifest.json")
        architecture = spec.get("action_architecture", "two_stage_v1")
        spec["algorithm"] = (
            "naive_rag_single_stage"
            if architecture == "single_stage_v1"
            else "staterag (naive_rag)"
        )
        _atomic_write_json(run_dir / "manifest.json", spec)
    else:
        if args.summarize_only:
            raise ValueError("--summarize-only requires --resume-run")
        spec = _build_spec(config, args)
        if args.dry_run:
            print(yaml.safe_dump({
                "target_player": spec["target_player"],
                "opponent_player": spec["opponent_player"],
                "source_key": spec["source_data_point"]["key"],
                "cluster_ids": [condition.get("cluster_id") for condition in spec["conditions"]],
                "rollouts_per_condition": spec["rollouts_per_condition"],
                "total_trajectories": len(spec["conditions"]) * spec["rollouts_per_condition"],
                "case_grounding": spec["case_grounding"],
                "action_architecture": spec["action_architecture"],
            }, sort_keys=False))
            return
        run_dir = _next_run_directory(_resolve_path(args.output_root))
        _atomic_write_json(run_dir / "manifest.json", spec)

    if not args.summarize_only:
        run_rollouts(spec, run_dir, workers=args.workers)
    summary = summarize_run(spec, run_dir)
    _atomic_write_json(run_dir / "summary.json", summary)
    latex_path = write_latex(summary, run_dir)
    print(f"Case study: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
