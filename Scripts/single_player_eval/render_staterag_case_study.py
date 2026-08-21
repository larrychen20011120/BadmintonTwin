#!/usr/bin/env python3
"""Render three instruction-following StateRAG trajectories as LaTeX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.single_player_eval.staterag_case_study import (
    _latex_escape,
    _load_json,
)
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


CONDITIONS = (
    {
        "key": "no_instruction",
        "rally": 10,
        "kind": "normal",
        "color": "normalgray",
        "title": "Normal",
    },
    {
        "key": "instruction_01_cluster_0",
        "rally": 3,
        "kind": "defensive",
        "color": "defensiveblue",
        "title": "Deep defensive resets",
    },
    {
        "key": "instruction_02_cluster_1",
        "rally": 12,
        "kind": "front",
        "color": "frontgreen",
        "title": "Front-court net pressure",
    },
    {
        "key": "instruction_03_cluster_2",
        "rally": 2,
        "kind": "drive_sequence",
        "color": "driveorange",
        "title": "Back-to-front drive pressure",
    },
)

RESET_SHOTS = {"Clear", "Lob", "Smash Defence"}
MAX_VISIBLE_ACTIONS = 12

SHORT_SHOTS = {
    "Serve short": "Short serve",
    "Serve long": "Long serve",
    "Push Shot": "Push",
    "Smash Defence": "Smash Def.",
    "Net Shot": "Net",
    "Missed shot": "Miss",
}

SHORT_AREAS = {
    "Front Center": "FC",
    "Front Left Side": "FL",
    "Front Right Side": "FR",
    "Front Left Side Line": "FL-line",
    "Front Right Side Line": "FR-line",
    "Mid Center": "MC",
    "Mid Left Side": "ML",
    "Mid Right Side": "MR",
    "Mid Left Side Line": "ML-line",
    "Mid Right Side Line": "MR-line",
    "Rear Center": "RC",
    "Rear Left Side": "RL",
    "Rear Right Side": "RR",
    "Rear Left Side Line": "RL-line",
    "Rear Right Side Line": "RR-line",
    "Deep Left Corner": "DLC",
    "Deep Right Corner": "DRC",
    "Missed shot": "Miss",
    "Out of Bounds": "OOB",
}


def _area_name(x, y):
    if x is None or y is None:
        return "Missed shot"
    return LANG_AREA_MAPPING.get(coord2area(float(x), float(y)), "Out of Bounds")


def _is_front(area):
    return area.startswith("Front")


def _is_rear(area):
    return area.startswith("Rear") or area.startswith("Deep")


def target_actions(trajectory):
    """Return target-player shots with readable action metadata."""
    target = trajectory["target_player"]
    actions = []
    for shot in trajectory["pred_sequence"]:
        if shot["player"] != target:
            continue
        actions.append({
            "player_action": len(actions) + 1,
            "global_round": int(shot["ball_round"]),
            "shot_type": INDEX2EN_SHOT[int(shot["shot_type"])],
            "landing_area": _area_name(shot["landing_x"], shot["landing_y"]),
            "recovery_area": _area_name(shot["moving_x"], shot["moving_y"]),
            "landing_x": shot["landing_x"],
            "landing_y": shot["landing_y"],
            "moving_x": shot["moving_x"],
            "moving_y": shot["moving_y"],
        })
    if not actions:
        raise ValueError("Selected trajectory has no target-player actions")
    return actions


def rally_actions(trajectory, first_round=None, last_round=None):
    """Return both players' actions in one optionally clipped exchange."""
    actions = []
    terminal_round = int(trajectory["pred_sequence"][-1]["ball_round"])
    for shot in trajectory["pred_sequence"]:
        global_round = int(shot["ball_round"])
        if first_round is not None and global_round < first_round:
            continue
        if last_round is not None and global_round > last_round:
            continue
        landing_area = _area_name(shot["landing_x"], shot["landing_y"])
        terminal_error = (
            global_round == terminal_round
            and trajectory.get("termination_reason") == "natural_terminate"
        )
        actions.append({
            "shot": len(actions) + 1,
            "global_round": global_round,
            "player": shot["player"],
            "is_target": shot["player"] == trajectory["target_player"],
            "shot_type": INDEX2EN_SHOT[int(shot["shot_type"])],
            "landing_area": landing_area,
            "terminal_error": terminal_error,
        })
    if not actions:
        raise ValueError("Selected exchange contains no actions")
    return actions


def adherence_marks(actions, kind):
    """Return aligned action indices and aligned consecutive action pairs."""
    aligned = set()
    pairs = []
    if kind == "defensive":
        aligned = {
            index
            for index, action in enumerate(actions)
            if action["shot_type"] in RESET_SHOTS
            and _is_rear(action["landing_area"])
        }
    elif kind == "front":
        aligned = {
            index
            for index, action in enumerate(actions)
            if _is_front(action["landing_area"])
        }
    elif kind == "drive_sequence":
        for index, (current, following) in enumerate(zip(actions, actions[1:])):
            if (
                current["shot_type"] == "Drive"
                and _is_rear(current["landing_area"])
                and following["shot_type"] == "Drive"
                and _is_front(following["landing_area"])
            ):
                aligned.update({index, index + 1})
                pairs.append((index, index + 1))
    else:
        raise ValueError(f"Unknown instruction kind: {kind}")
    return aligned, pairs


def _colored_phrase(value, color, aligned):
    value = _latex_escape(value)
    return rf"\textcolor{{{color}}}{{\textbf{{{value}}}}}" if aligned else value


def _action_cell(action, condition):
    target_action = action["is_target"]
    shot_aligned = False
    landing_aligned = False
    if target_action and condition["kind"] == "defensive":
        shot_aligned = action["shot_type"] in RESET_SHOTS
        landing_aligned = _is_rear(action["landing_area"])
    elif target_action and condition["kind"] == "front":
        landing_aligned = _is_front(action["landing_area"])
        shot_aligned = landing_aligned and action["shot_type"] in {
            "Drop", "Drive", "Net Shot", "Smash",
        }
    elif target_action and condition["kind"] == "drive_sequence":
        shot_aligned = action["shot_type"] in {"Drive", "Net Shot", "Smash"}
        landing_aligned = _is_rear(action["landing_area"]) or _is_front(
            action["landing_area"]
        )

    if action["terminal_error"]:
        shot_aligned = False
        landing_aligned = False
    player = "Player" if target_action else "Opponent"
    if target_action:
        player = rf"\textbf{{{player}}}"
    else:
        player = rf"\textcolor{{gray}}{{{player}}}"
    short_shot = SHORT_SHOTS.get(action["shot_type"], action["shot_type"])
    short_area = SHORT_AREAS.get(action["landing_area"], action["landing_area"])
    if action["shot_type"] == "Missed shot":
        action_text = rf"{player}: Miss"
    else:
        action_text = (
            rf"{player}: "
            rf"{_colored_phrase(short_shot, condition['color'], shot_aligned)} "
            rf"$\rightarrow$ {_colored_phrase(short_area, condition['color'], landing_aligned)}"
        )
    return rf"\mbox{{\textbf{{{action['shot']}.}} {action_text}}}"


def _compact_actions(actions, limit=MAX_VISIBLE_ACTIONS):
    if len(actions) <= limit:
        return list(actions)
    side = limit // 2
    return [*actions[:side], None, *actions[-side:]]


def _condition_data(condition, profile, trajectory):
    actions = rally_actions(trajectory)
    instruction_text = (
        profile.get("instruction")
        or profile.get("applied_instruction")
        or "No tactical instruction."
    )
    winner_role = (
        "Player" if trajectory["winner"] == trajectory["target_player"] else "Opponent"
    )
    terminal_shot = trajectory["pred_sequence"][-1]
    terminal_reason = "Miss" if int(terminal_shot["shot_type"]) == 11 else "Out of bounds"
    player_score = int(trajectory["score_before"][trajectory["target_player"]])
    opponent_score = int(trajectory["score_before"][trajectory["opponent_player"]])
    return {
        "condition": condition,
        "title": condition["title"],
        "instruction": instruction_text,
        "context": f"Player serves; score {player_score}--{opponent_score}.",
        "actions": _compact_actions(actions),
        "result": f"{winner_role} wins ({terminal_reason}).",
    }


def render_trajectory_comparison(run_dir, output_path=None):
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    profiles = {condition["key"]: condition for condition in manifest["conditions"]}
    columns = []
    for condition in CONDITIONS:
        profile = profiles.get(condition["key"])
        if profile is None:
            raise ValueError(f"Missing condition in manifest: {condition['key']}")
        trajectory_path = (
            run_dir
            / "conditions"
            / condition["key"]
            / f"rally_{condition['rally']:03d}.json"
        )
        trajectory = _load_json(trajectory_path)
        if trajectory.get("condition") != condition["key"]:
            raise ValueError(f"Condition mismatch in {trajectory_path}")
        if trajectory.get("termination_reason") != "natural_terminate":
            raise ValueError(f"Selected trajectory is not an early natural stop: {trajectory_path}")
        if trajectory.get("server") != trajectory.get("target_player"):
            raise ValueError(f"Selected trajectory is not served by Player: {trajectory_path}")
        columns.append(_condition_data(condition, profile, trajectory))

    lines = [
        "% Requires: \\usepackage{xcolor,booktabs,tabularx,array}",
        r"\definecolor{normalgray}{HTML}{4F5965}",
        r"\definecolor{defensiveblue}{HTML}{2457A6}",
        r"\definecolor{frontgreen}{HTML}{178A62}",
        r"\definecolor{driveorange}{HTML}{D66B20}",
        r"\begin{table*}[t]",
        r"\raggedright",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabularx}{\textwidth}{*{4}{>{\raggedright\arraybackslash}X}}",
        r"\toprule",
    ]
    lines.append(" & ".join(
        rf"\textcolor{{{column['condition']['color']}}}{{\textbf{{{_latex_escape(column['title'])}}}}}"
        for column in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(" & ".join(
        _latex_escape(column["instruction"]) for column in columns
    ) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(" & ".join(
        rf"\textit{{{_latex_escape(column['context'])}}}"
        for column in columns
    ) + r" \\")
    lines.append(r"\midrule")
    maximum_rows = max(len(column["actions"]) for column in columns)
    for row_index in range(maximum_rows):
        cells = []
        for column in columns:
            if row_index >= len(column["actions"]):
                cells.append("")
                continue
            action = column["actions"][row_index]
            cells.append(
                r"\mbox{\textcolor{gray}{\ldots}}"
                if action is None
                else _action_cell(action, column["condition"])
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(
        rf"\mbox{{\textbf{{Result.}} {_latex_escape(column['result'])}}}"
        for column in columns
    ) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        (
            r"\caption{Normal play and three instruction-conditioned rallies. Each "
            r"column is top-aligned and numbered independently; Player denotes the "
            r"instructed policy and Opponent the other agent. Long trajectories retain "
            r"their opening and ending actions with omitted middle actions marked by "
            r"ellipsis. Colored phrases indicate instruction-aligned Player behavior, "
            r"and each result reports the winner and terminal outcome. Area abbreviations: "
            r"F/M/R denote front/mid/rear, C/L/R denote center/left/right, and DLC/DRC "
            r"denote deep left/right corners.}"
        ),
        r"\label{tab:staterag-instruction-rallies}",
        r"\end{table*}",
    ])
    output_path = Path(output_path) if output_path else run_dir / "trajectory_comparison.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default=(
            "Experiments/naive_rag/case_study/"
            "staterag_sequential_score/run_001"
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    output_path = Path(args.output) if args.output else None
    if output_path is not None and not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    rendered = render_trajectory_comparison(run_dir, output_path)
    print(f"LaTeX trajectory comparison: {rendered}")


if __name__ == "__main__":
    main()
