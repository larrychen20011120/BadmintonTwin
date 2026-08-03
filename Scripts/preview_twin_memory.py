#!/usr/bin/env python3
"""Preview the Twin memories produced from one rally without building a database."""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml
from jinja2 import Template


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.twin import TwinDatabase


DEFAULT_CONFIG_PATH = os.path.join(project_path, "Scripts", "config.yaml")
DEFAULT_PREVIOUS_RALLY_SUMMARY = "No previous-rally summary is used by the current Twin memory schema."


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Twin memory annotation for one rally and print the resulting memories.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Project config YAML path")
    parser.add_argument("--player", required=True, help="Exact target player name")
    parser.add_argument("--match-id", required=True, type=int, help="Match ID")
    parser.add_argument("--set", dest="set_number", type=int, help="Set number when the dataset has sets")
    parser.add_argument("--rally", required=True, type=int, help="Rally number")
    parser.add_argument(
        "--previous-rally-summary",
        default=DEFAULT_PREVIOUS_RALLY_SUMMARY,
        help="Deprecated compatibility option; current Twin memories do not use it",
    )
    parser.add_argument("--no-llm", action="store_true", help="Use fallback annotation without an LLM request")
    parser.add_argument("--show-prompt", action="store_true", help="Print the full user prompt sent to the LLM")
    parser.add_argument("--show-annotation", action="store_true", help="Print the rally annotation JSON")
    parser.add_argument("--output", help="Optionally write the complete preview as JSON")
    return parser


def resolve_path(path, config_path):
    if os.path.isabs(path):
        return path
    project_candidate = os.path.join(project_path, path)
    if os.path.exists(project_candidate):
        return project_candidate
    return os.path.join(os.path.dirname(config_path), path)


def load_inputs(config_path):
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    paths = config.get("paths", {})
    try:
        merged_csv_path = resolve_path(paths["merged_csv"], config_path)
        player_map_path = resolve_path(paths["player_map"], config_path)
    except (KeyError, TypeError) as exc:
        raise ValueError("Config must define paths.merged_csv and paths.player_map") from exc

    dataframe = pd.read_csv(merged_csv_path)
    with open(player_map_path, "r", encoding="utf-8") as file:
        player_map = json.load(file)
    return config_path, dataframe, player_map


def select_rally(dataframe, match_id, rally_number, set_number=None):
    required_columns = {"match_id", "rally", "ball_round", "player", "player_type"}
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"Merged data is missing required columns: {missing}")

    selected = dataframe[
        (dataframe["match_id"] == match_id)
        & (dataframe["rally"] == rally_number)
    ]
    if "set" in dataframe.columns:
        available_sets = sorted(
            int(value) for value in selected["set"].dropna().unique()
        )
        if set_number is None:
            if len(available_sets) > 1:
                raise ValueError(
                    f"Match {match_id} rally {rally_number} exists in sets {available_sets}; provide --set."
                )
            if len(available_sets) == 1:
                set_number = available_sets[0]
        if set_number is not None:
            selected = selected[selected["set"] == set_number]
    elif set_number is not None:
        raise ValueError("The merged dataset has no set column; omit --set.")

    if selected.empty:
        set_text = f", set {set_number}" if set_number is not None else ""
        raise ValueError(
            f"No rally found for match {match_id}{set_text}, rally {rally_number}."
        )
    if selected["player_type"].isna().any():
        raise ValueError("Selected rally contains rows without player_type.")
    selected = selected.copy()
    selected["player"] = selected["player"].astype(str).str.strip()
    return selected.sort_values("ball_round").reset_index(drop=True), set_number


def render_memory_prompt(database, rally_df, previous_rally_summary):
    del previous_rally_summary
    template_path = os.path.join(project_path, "Prompts", "New", "twin", "memory_user.jinja")
    with open(template_path, "r", encoding="utf-8") as file:
        template = Template(file.read())
    return template.render(
        rally_meta_text=database.build_rally_meta_text(rally_df),
        rally_turn_based_text=database.build_rally_turn_based_text(rally_df),
        first_ball_round=int(rally_df["ball_round"].min()),
        last_ball_round=int(rally_df["ball_round"].max()),
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_preview(rally_df, player_name, player_id, previous_rally_summary,
                  use_llm=True, config_path=None):
    players = {str(value).strip() for value in rally_df["player"].dropna()}
    if player_name not in players:
        raise ValueError(
            f"Player '{player_name}' is not in this rally. Available players: {sorted(players)}"
        )

    database = TwinDatabase(rally_df, player_name, player_id)
    prompt = render_memory_prompt(database, rally_df, previous_rally_summary)
    llm = None
    if use_llm:
        from Utils.llm import LLM

        llm = LLM(config_path=config_path)
    annotation = database.annotate_rally(
        rally_df,
        llm=llm,
        use_llm=use_llm,
    )
    memories = database.build_stroke_annotations(
        rally_df,
        annotation,
    )
    return {
        "rally_meta_text": database.build_rally_meta_text(rally_df),
        "rally_turn_based_text": database.build_rally_turn_based_text(rally_df),
        "llm_user_prompt": prompt,
        "annotation": annotation,
        "memories": memories,
    }


def print_preview(preview, selection, show_prompt=False, show_annotation=False):
    print("Twin single-rally memory preview")
    print(f"Player: {selection['player']}")
    print(f"Match: {selection['match_id']}")
    if selection.get("set") is not None:
        print(f"Set: {selection['set']}")
    print(f"Rally: {selection['rally']}")
    print(f"Annotation mode: {selection['annotation_mode']}")
    print(f"Target-player memories: {len(preview['memories'])}")

    if show_prompt:
        print("\n" + "=" * 80)
        print("LLM user prompt")
        print("=" * 80)
        print(preview["llm_user_prompt"])
    if show_annotation:
        print("\n" + "=" * 80)
        print("Rally annotation")
        print("=" * 80)
        print(json.dumps(to_jsonable(preview["annotation"]), indent=2, ensure_ascii=False))

    for index, memory in enumerate(preview["memories"], start=1):
        print("\n" + "=" * 80)
        print(f"Memory {index}: {memory['memory_id']}")
        print("=" * 80)
        print(memory["semantic_text"])
        print(memory["reachability_text"])
        print(memory["action_verbalization"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config_path, dataframe, player_map = load_inputs(args.config)
        player_name = args.player.strip()
        if player_name not in player_map:
            raise ValueError(f"Player '{player_name}' is not in the player map.")
        rally_df, resolved_set = select_rally(
            dataframe,
            args.match_id,
            args.rally,
            args.set_number,
        )
        preview = build_preview(
            rally_df,
            player_name,
            player_map[player_name],
            args.previous_rally_summary,
            use_llm=not args.no_llm,
            config_path=config_path,
        )
        selection = {
            "player": player_name,
            "match_id": args.match_id,
            "set": resolved_set,
            "rally": args.rally,
            "annotation_mode": "fallback" if args.no_llm else "llm",
        }
        print_preview(
            preview,
            selection,
            show_prompt=args.show_prompt,
            show_annotation=args.show_annotation,
        )
        if args.output:
            output_path = os.path.abspath(args.output)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            payload = {"selection": selection, **preview}
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(to_jsonable(payload), file, indent=2, ensure_ascii=False, allow_nan=False)
            print(f"\nPreview JSON saved to: {output_path}")
        return 0
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
