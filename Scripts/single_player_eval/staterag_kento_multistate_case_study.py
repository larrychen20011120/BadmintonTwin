#!/usr/bin/env python3
"""Run independent StateRAG trajectories for Kento across three states."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
import multiprocessing
import os
import random
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.single_player_eval.staterag_case_study import (
    _atomic_write_json,
    _condition_rollout_count,
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


STATE_KEYS = (
    (1, 2, 14),
    (3, 2, 13),
    (6, 1, 14),
)
CLUSTER_IDS = (0, 4, 1)
TARGET_PLAYER = "Kento MOMOTA"
ROLLOUTS_PER_CONDITION = 5
NORMAL_ROLLOUTS = 10

CONDITION_KINDS = {
    "no_instruction": "normal",
    "instruction_01_cluster_0": "wide_front",
    "instruction_02_cluster_4": "short_net",
    "instruction_03_cluster_1": "deep_reset",
}

COLORS = {
    "normal": "normalgray",
    "wide_front": "wideblue",
    "short_net": "netgreen",
    "deep_reset": "deeporange",
}

EXPLANATORY_STATE_LABEL = "state_02"
EXPLANATORY_ROLLOUTS = {
    "no_instruction": 3,
    "instruction_01_cluster_0": 4,
    "instruction_02_cluster_4": 5,
    "instruction_03_cluster_1": 5,
}
HIGHLIGHT_ROUNDS = {
    "no_instruction": (),
    "instruction_01_cluster_0": (3, 5, 7, 9, 13, 15, 17),
    "instruction_02_cluster_4": (5, 7),
    "instruction_03_cluster_1": (3, 7, 11),
}
ALIGNMENT_EXPLANATIONS = {
    "no_instruction": "Reference trajectory without a tactical instruction.",
    "instruction_01_cluster_0": (
        "Strong alignment: four wide front-court landings and seven depth switches."
    ),
    "instruction_02_cluster_4": (
        "Partial alignment: the fifth and seventh shots stay in the front court, "
        "and the seventh is an attacking Net Shot."
    ),
    "instruction_03_cluster_1": (
        "Partial alignment: the third-shot Clear and seventh-shot Lob reset play "
        "to the back court; the eleventh shot also stays deep."
    ),
}

RESET_SHOTS = {"Clear", "Lob", "Smash Defence"}
ATTACK_SHOTS = {"Push Shot", "Smash", "Drive", "Net Shot", "Drop"}

def _area_name(x, y):
    if x is None or y is None:
        return "Missed shot"
    return LANG_AREA_MAPPING.get(coord2area(float(x), float(y)), "Out of Bounds")


def _is_front(area):
    return area.startswith("Front")


def _is_rear(area):
    return area.startswith("Rear") or area.startswith("Deep")


def _state_label(index):
    return f"state_{index:02d}"


def _build_state_specs(config, args):
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
            "state_index": index,
            "state_label": _state_label(index),
            "independent_rollouts": True,
            "score_carryover": False,
            "history_reset_per_rollout": True,
        })
        _configure_rollout_counts(spec)
        specs.append(spec)
    return specs


def _configure_rollout_counts(spec):
    for condition in spec["conditions"]:
        condition["rollout_count"] = (
            NORMAL_ROLLOUTS
            if condition["key"] == "no_instruction"
            else ROLLOUTS_PER_CONDITION
        )


def _state_worker(payload):
    spec, state_dir = payload
    os.chdir(PROJECT_ROOT)
    run_rollouts(spec, state_dir, workers=1)
    return spec["state_label"]


def run_states(specs, run_dir, workers=3):
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


def validate_independent_rollouts(trajectories, spec, condition=None):
    expected = spec["source_data_point"]["initial_state"]
    seen = set()
    for trajectory in trajectories:
        rollout_index = int(trajectory["rollout_index"])
        if rollout_index in seen:
            raise ValueError(f"Duplicate rollout index: {rollout_index}")
        seen.add(rollout_index)
        if trajectory.get("initial_state") != expected:
            raise ValueError(f"Rollout {rollout_index} did not reset to the source state")
        if "score_before" in trajectory or "score_after" in trajectory:
            raise ValueError("Independent rollouts must not contain score carryover fields")
    expected_count = (
        _condition_rollout_count(spec, condition)
        if condition is not None
        else int(spec["rollouts_per_condition"])
    )
    expected_indices = set(range(1, expected_count + 1))
    if seen != expected_indices:
        raise ValueError(f"Incomplete independent rollout indices: {sorted(seen)}")


def select_shortest_natural(trajectories):
    natural = [
        trajectory
        for trajectory in trajectories
        if trajectory.get("termination_reason") == "natural_terminate"
    ]
    if not natural:
        return None
    return min(
        natural,
        key=lambda trajectory: (
            int(trajectory["rally_length"]),
            int(trajectory["rollout_index"]),
        ),
    )


def select_representative(trajectories, kind, seed):
    selected = select_shortest_natural(trajectories)
    if selected is not None:
        return selected, "shortest_natural_termination"
    if kind == "normal":
        return random.Random(int(seed)).choice(trajectories), "random_fallback_no_natural_termination"
    return None, "no_natural_termination"


def _successful_target_actions(trajectory):
    target = trajectory["target_player"]
    final_round = int(trajectory["pred_sequence"][-1]["ball_round"])
    actions = []
    for shot in trajectory["pred_sequence"]:
        if shot["player"] != target or int(shot["shot_type"]) in {1, 10, 11}:
            continue
        if (
            trajectory["termination_reason"] == "natural_terminate"
            and int(shot["ball_round"]) == final_round
        ):
            continue
        actions.append({
            "shot_type": INDEX2EN_SHOT[int(shot["shot_type"])],
            "landing_area": _area_name(shot["landing_x"], shot["landing_y"]),
        })
    return actions


def assess_adherence(trajectory, kind):
    if kind == "normal":
        return {"rating": "not_applicable", "evidence": "No tactical instruction."}
    actions = _successful_target_actions(trajectory)
    if not actions:
        return {"rating": "weak", "evidence": "No successful non-serve Player action."}
    count = len(actions)
    front = sum(_is_front(action["landing_area"]) for action in actions)
    rear = sum(_is_rear(action["landing_area"]) for action in actions)
    if kind == "wide_front":
        wide_front = sum(
            _is_front(action["landing_area"])
            and "Center" not in action["landing_area"]
            for action in actions
        )
        depth = [
            "front" if _is_front(action["landing_area"])
            else "rear" if _is_rear(action["landing_area"])
            else "mid"
            for action in actions
        ]
        switches = sum(current != previous for previous, current in zip(depth, depth[1:]))
        rating = "strong" if wide_front >= 2 and switches >= 2 else "partial" if wide_front or switches else "weak"
        evidence = (
            f"Wide front-court landings: {wide_front} of {count}; "
            f"depth switches: {switches}; rally length: {trajectory['rally_length']}."
        )
    elif kind == "short_net":
        front_attacks = sum(
            action["shot_type"] in ATTACK_SHOTS and _is_front(action["landing_area"])
            for action in actions
        )
        rating = "strong" if front / count >= 0.6 and front_attacks >= 2 else "partial" if front else "weak"
        evidence = (
            f"Front-court landings: {front} of {count}; "
            f"front-court attacks: {front_attacks}; rally length: {trajectory['rally_length']}."
        )
    elif kind == "deep_reset":
        rear_resets = sum(
            action["shot_type"] in RESET_SHOTS and _is_rear(action["landing_area"])
            for action in actions
        )
        rating = "strong" if rear_resets / count >= 0.5 else "partial" if rear_resets else "weak"
        evidence = (
            f"Deep reset shots: {rear_resets} of {count}; "
            f"back-court landings: {rear} of {count}; rally length: {trajectory['rally_length']}."
        )
    else:
        raise ValueError(f"Unknown condition kind: {kind}")
    return {"rating": rating, "evidence": evidence}


def _terminal_outcome(trajectory):
    if trajectory["termination_reason"] == "max_shots_reached":
        return "Maximum shot limit reached"
    if trajectory["termination_reason"] == "action_none":
        return "Agent returned no action"
    final = trajectory["pred_sequence"][-1]
    return "Missed shot" if int(final["shot_type"]) == 11 else "Out of bounds"


def summarize_states(specs, run_dir):
    state_summaries = []
    for spec in specs:
        state_dir = Path(run_dir) / "states" / spec["state_label"]
        condition_summaries = {}
        for condition in spec["conditions"]:
            trajectories = _load_condition_trajectories(spec, state_dir, condition)
            kind = CONDITION_KINDS[condition["key"]]
            validate_independent_rollouts(trajectories, spec, condition)
            shortest_natural = select_shortest_natural(trajectories)
            selected, selection_reason = select_representative(
                trajectories,
                kind,
                seed=spec["seed"],
            )
            selected_summary = None
            if selected is not None:
                selected_summary = {
                    "rollout_index": int(selected["rollout_index"]),
                    "rally_length": int(selected["rally_length"]),
                    "winner": selected["winner"],
                    "target_player_won": bool(selected["target_player_won"]),
                    "terminal_outcome": _terminal_outcome(selected),
                    "termination_reason": selected["termination_reason"],
                    "selection_reason": selection_reason,
                    "trajectory": str(
                        _trajectory_path(
                            state_dir,
                            condition["key"],
                            selected["rollout_index"],
                        ).relative_to(run_dir)
                    ),
                    "adherence": assess_adherence(selected, kind),
                }
            condition_summaries[condition["key"]] = {
                "label": condition["label"],
                "instruction": condition.get("applied_instruction"),
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
                "shortest_natural_rollout_index": (
                    int(shortest_natural["rollout_index"])
                    if shortest_natural is not None
                    else None
                ),
                "representative": selected_summary,
            }
        state_summaries.append({
            "state_index": spec["state_index"],
            "state_label": spec["state_label"],
            "source_key": spec["source_data_point"]["key"],
            "initial_state": spec["source_data_point"]["initial_state"],
            "target_player": spec["target_player"],
            "opponent_player": spec["opponent_player"],
            "conditions": condition_summaries,
        })
    return {
        "schema_version": 3,
        "design": "three_states_with_ten_normal_and_five_instructed_independent_rollouts",
        "total_trajectories": sum(
            _condition_rollout_count(spec, condition)
            for spec in specs
            for condition in spec["conditions"]
        ),
        "states": state_summaries,
        "explanatory_case": _build_explanatory_case(specs, run_dir),
    }


def _action_text(shot, trajectory):
    role = "Player" if shot["player"] == trajectory["target_player"] else "Opponent"
    shot_name = INDEX2EN_SHOT[int(shot["shot_type"])]
    if int(shot["shot_type"]) == 11:
        action = shot_name
    else:
        area = _area_name(shot["landing_x"], shot["landing_y"])
        action = f"{shot_name} to {area}"
    return int(shot["ball_round"]), role, action


def _trajectory_actions(trajectory, limit=10):
    actions = []
    for shot in trajectory["pred_sequence"]:
        ball_round, role, action = _action_text(shot, trajectory)
        actions.append(f"{ball_round}. {role}: {action}")
    if len(actions) <= limit:
        return actions
    side = limit // 2
    return [*actions[:side], "...", *actions[-side:]]


def _build_explanatory_case(specs, run_dir):
    spec = next(
        (item for item in specs if item["state_label"] == EXPLANATORY_STATE_LABEL),
        None,
    )
    if spec is None:
        raise ValueError(f"Missing explanatory state: {EXPLANATORY_STATE_LABEL}")
    state_dir = Path(run_dir) / "states" / spec["state_label"]
    conditions = []
    for condition in spec["conditions"]:
        condition_key = condition["key"]
        rollout_index = EXPLANATORY_ROLLOUTS[condition_key]
        path = _trajectory_path(state_dir, condition_key, rollout_index)
        trajectory = _load_json(path)
        if trajectory.get("termination_reason") != "natural_terminate":
            raise ValueError(f"Explanatory trajectory must terminate naturally: {path}")
        if trajectory.get("initial_state") != spec["source_data_point"]["initial_state"]:
            raise ValueError(f"Explanatory trajectory did not reset to the source state: {path}")
        kind = CONDITION_KINDS[condition_key]
        conditions.append({
            "key": condition_key,
            "kind": kind,
            "title": "Normal play" if kind == "normal" else condition["label"],
            "instruction": (
                "No tactical instruction; follow the learned normal policy."
                if kind == "normal"
                else condition["instruction"]
            ),
            "color": COLORS[kind],
            "rollout_index": rollout_index,
            "trajectory": str(path.relative_to(run_dir)),
            "highlight_rounds": list(HIGHLIGHT_ROUNDS[condition_key]),
            "alignment": ALIGNMENT_EXPLANATIONS[condition_key],
            "adherence": assess_adherence(trajectory, kind),
            "rally_length": int(trajectory["rally_length"]),
            "winner": trajectory["winner"],
            "terminal_outcome": _terminal_outcome(trajectory),
        })
    initial = spec["source_data_point"]["initial_state"]
    match_id, set_number, rally_number = spec["source_data_point"]["key"]
    return {
        "state_label": spec["state_label"],
        "source_key": spec["source_data_point"]["key"],
        "starting_situation": (
            f"Source match {match_id}, set {set_number}, rally {rally_number}. Kento "
            f"MOMOTA serves against CHOU Tien Chen at {initial['player_score']}--"
            f"{initial['opponent_score']}. Their normalized court positions are "
            f"({initial['player_location_x']:.2f}, {initial['player_location_y']:.2f}) "
            f"and ({initial['opponent_location_x']:.2f}, "
            f"{initial['opponent_location_y']:.2f}), respectively. Every case starts "
            "from this identical situation with an empty rally history."
        ),
        "conditions": conditions,
    }


def _case_action_cell(shot, trajectory, highlighted_rounds, color):
    ball_round, role, action = _action_text(shot, trajectory)
    action = _latex_escape(action)
    if role == "Player" and ball_round in highlighted_rounds:
        action = rf"\textcolor{{{color}}}{{\textbf{{{action}}}}}"
    return f"{ball_round}. {_latex_escape(role)}: {action}"


def _case_result(trajectory):
    final = trajectory["pred_sequence"][-1]
    winner = trajectory["winner"]
    loser = trajectory["loser"]
    if int(final["shot_type"]) == 11:
        return f"{winner} wins after {loser} misses the shuttle."
    shot_name = INDEX2EN_SHOT[int(final["shot_type"])]
    return f"{winner} wins after {loser}'s {shot_name} lands out of bounds."


def write_latex(summary, run_dir):
    explanatory = summary["explanatory_case"]
    columns = []
    for condition in explanatory["conditions"]:
        trajectory = _load_json(Path(run_dir) / condition["trajectory"])
        columns.append((condition, trajectory))
    lines = [
        "% Requires: \\usepackage{xcolor,booktabs,tabularx,array}",
        r"\definecolor{normalgray}{HTML}{4F5965}",
        r"\definecolor{wideblue}{HTML}{2457A6}",
        r"\definecolor{netgreen}{HTML}{178A62}",
        r"\definecolor{deeporange}{HTML}{D66B20}",
        r"\begin{table*}[t]",
        r"\raggedright",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabularx}{\textwidth}{*{4}{>{\raggedright\arraybackslash}X}}",
        r"\toprule",
    ]
    lines.append(" & ".join(
        rf"\textcolor{{{condition['color']}}}{{\textbf{{\large {_latex_escape(condition['title'])}}}}}"
        for condition, _ in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(" & ".join(
        _latex_escape(condition["instruction"])
        for condition, _ in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    context = rf"\textit{{\textbf{{Starting situation.}} {_latex_escape(explanatory['starting_situation'])}}}"
    lines.append(" & ".join(context for _ in columns) + r" \\")
    lines.append(r"\midrule")
    maximum_length = max(len(trajectory["pred_sequence"]) for _, trajectory in columns)
    for row_index in range(maximum_length):
        cells = []
        for condition, trajectory in columns:
            if row_index >= len(trajectory["pred_sequence"]):
                cells.append("")
                continue
            cells.append(_case_action_cell(
                trajectory["pred_sequence"][row_index],
                trajectory,
                set(condition["highlight_rounds"]),
                condition["color"],
            ))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(
        rf"\textbf{{Instruction alignment.}} \textcolor{{{condition['color']}}}{{{_latex_escape(condition['alignment'])}}}"
        for condition, _ in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(" & ".join(
        rf"\textbf{{Result.}} {_latex_escape(_case_result(trajectory))}"
        for _, trajectory in columns
    ) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        (
            r"\caption{Four StateRAG trajectories from one identical starting situation. "
            r"The selected cases provide interpretable instruction-following behavior and "
            r"varied terminal outcomes, including a missed shot. Colored action phrases "
            r"mark Player shots or short sequences that align with the instruction in the "
            r"same column; uncolored actions provide context. Player denotes Kento MOMOTA "
            r"and Opponent denotes CHOU Tien Chen.}"
        ),
        r"\label{tab:kento-explanatory-case-study}",
        r"\end{table*}",
    ])
    output_path = Path(run_dir) / "explanatory_case_study.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _master_manifest(specs, args):
    return {
        "schema_version": 2,
        "design": "three_states_with_ten_normal_and_five_instructed_independent_rollouts",
        "target_player": TARGET_PLAYER,
        "state_keys": [list(key) for key in STATE_KEYS],
        "cluster_ids": list(CLUSTER_IDS),
        "conditions_per_state": 4,
        "normal_rollouts_per_state": NORMAL_ROLLOUTS,
        "instructed_rollouts_per_condition": ROLLOUTS_PER_CONDITION,
        "total_trajectories": sum(
            _condition_rollout_count(spec, condition)
            for spec in specs
            for condition in spec["conditions"]
        ),
        "independent_rollouts": True,
        "score_carryover": False,
        "seed": int(args.seed),
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
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--conditioned-top-k", type=int)
    parser.add_argument(
        "--output-root",
        default="Experiments/naive_rag/case_study/kento_multistate",
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
        specs = [
            _load_json(run_dir / state["manifest"])
            for state in master["states"]
        ]
        for spec in specs:
            _configure_rollout_counts(spec)
            state_dir = run_dir / "states" / spec["state_label"]
            _atomic_write_json(state_dir / "manifest.json", spec)
        _atomic_write_json(run_dir / "manifest.json", _master_manifest(specs, args))
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
        completed = run_states(specs, run_dir, workers=args.workers)
        print(f"Completed states: {', '.join(sorted(completed))}")
    summary = summarize_states(specs, run_dir)
    _atomic_write_json(run_dir / "summary.json", summary)
    latex_path = write_latex(summary, run_dir)
    print(f"Case study: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
