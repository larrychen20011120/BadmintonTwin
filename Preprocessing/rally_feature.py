import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Utils.helper import coord2area  # noqa: E402


ATTACK_SHOTS = {"殺球", "切球", "推撲球"}
DEFENSIVE_RESET_SHOTS = {"發長球", "長球", "挑球", "接殺防守"}
SERVE_SHOTS = {"發短球", "發長球"}

FRONT_AREA_IDS = {17, 18, 19, 20, 21, 22, 23, 24, 25, 29}
BACK_AREA_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 27, 31}
SIDE_AREA_IDS = {
    1, 4, 5, 8,
    9, 12, 13, 16,
    17, 20, 21, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
}
LEFT_AREA_IDS = {1, 5, 9, 13, 17, 21, 29, 30, 31, 32, 2, 6, 10, 14, 18, 22}
RIGHT_AREA_IDS = {4, 8, 12, 16, 20, 24, 25, 26, 27, 28, 3, 7, 11, 15, 19, 23}
IN_COURT_AREA_IDS = set(range(1, 33))


RALLY_KEY_COLUMNS = ["match_id", "set", "rally"]

CLUSTER_FEATURE_COLUMNS = [
    "attack_shot_ratio",
    "defense_shot_ratio",
    "side_ratio",
    "frontcourt_ratio",
    "backcourt_ratio",
    "signed_side_bias",
    "front_back_switch_ratio",
    "left_right_switch_ratio",
    "log_rally_length",
    "average_movement",
    "landing_opponent_distance",
]


def _area_series(df, area_col, x_col, y_col):
    if area_col in df.columns:
        areas = pd.to_numeric(df[area_col], errors="coerce").copy()
    else:
        areas = pd.Series(np.nan, index=df.index, dtype="float64")

    missing = areas.isna()
    if x_col in df.columns and y_col in df.columns and missing.any():
        coords = df.loc[missing, [x_col, y_col]].dropna()
        for idx, row in coords.iterrows():
            areas.loc[idx] = coord2area(float(row[x_col]), float(row[y_col]))

    return areas


def _stroke_movement_distance(player_df):
    if player_df.empty:
        return pd.Series(dtype="float64")

    if "ball_distance" in player_df.columns:
        approach = pd.to_numeric(player_df["ball_distance"], errors="coerce")
    elif {"hit_x", "hit_y", "player_location_x", "player_location_y"}.issubset(
        player_df.columns
    ):
        approach = np.sqrt(
            (
                pd.to_numeric(player_df["hit_x"], errors="coerce")
                - pd.to_numeric(player_df["player_location_x"], errors="coerce")
            )
            ** 2
            + (
                pd.to_numeric(player_df["hit_y"], errors="coerce")
                - pd.to_numeric(player_df["player_location_y"], errors="coerce")
            )
            ** 2
        )
    else:
        approach = pd.Series(np.nan, index=player_df.index, dtype="float64")

    if {"hit_x", "hit_y", "moving_x", "moving_y"}.issubset(player_df.columns):
        recovery = np.sqrt(
            (
                pd.to_numeric(player_df["moving_x"], errors="coerce")
                - pd.to_numeric(player_df["hit_x"], errors="coerce")
            )
            ** 2
            + (
                pd.to_numeric(player_df["moving_y"], errors="coerce")
                - pd.to_numeric(player_df["hit_y"], errors="coerce")
            )
            ** 2
        )
    elif "moving_distance" in player_df.columns:
        recovery = pd.to_numeric(player_df["moving_distance"], errors="coerce")
    else:
        recovery = pd.Series(np.nan, index=player_df.index, dtype="float64")

    return approach + recovery


def _landing_opponent_distance(player_df):
    if player_df.empty:
        return pd.Series(dtype="float64")
    if "opponent_moving_distance" in player_df.columns:
        return pd.to_numeric(player_df["opponent_moving_distance"], errors="coerce")
    if {"landing_x", "landing_y", "opponent_location_x", "opponent_location_y"}.issubset(
        player_df.columns
    ):
        return np.sqrt(
            (
                pd.to_numeric(player_df["landing_x"], errors="coerce")
                - pd.to_numeric(player_df["opponent_location_x"], errors="coerce")
            )
            ** 2
            + (
                pd.to_numeric(player_df["landing_y"], errors="coerce")
                - pd.to_numeric(player_df["opponent_location_y"], errors="coerce")
            )
            ** 2
        )
    return pd.Series(np.nan, index=player_df.index, dtype="float64")


def _require_columns(df, columns):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required rally columns: {missing}")


def _switch_counts(areas, label_func):
    labels = [label for label in areas.map(label_func).tolist() if label is not None]
    transition_count = max(len(labels) - 1, 0)
    switch_count = sum(current != previous for previous, current in zip(labels, labels[1:]))
    return int(switch_count), int(transition_count)


def _front_back_label(area):
    if area in FRONT_AREA_IDS:
        return "front"
    if area in BACK_AREA_IDS:
        return "back"
    return None


def _left_right_label(area):
    if area in LEFT_AREA_IDS:
        return "left"
    if area in RIGHT_AREA_IDS:
        return "right"
    return None


def _mean_or_nan(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def extract_rally_features(df):
    """Return one tactical feature record per player and physical rally."""
    _require_columns(df, RALLY_KEY_COLUMNS + ["ball_round", "player", "type"])

    records = []
    grouped = df.groupby(RALLY_KEY_COLUMNS, sort=True, dropna=False)
    for key, rally_df in grouped:
        if any(pd.isna(value) for value in key):
            continue

        rally_df = rally_df.sort_values("ball_round", kind="stable")
        rally_rounds = pd.to_numeric(rally_df["ball_round"], errors="coerce")
        if not rally_rounds.notna().any():
            continue
        rally_length = float(rally_rounds.max())

        players = rally_df["player"].dropna().astype(str).str.strip().unique().tolist()
        for player_name in players:
            player_df = rally_df[rally_df["player"].astype("string").str.strip() == player_name]
            shot_types = player_df["type"].astype("string")
            valid_actions = player_df[shot_types.notna() & (shot_types != "接不到")]
            valid_types = valid_actions["type"].astype("string")
            tactical_rows = valid_actions[~valid_types.isin(SERVE_SHOTS)]
            tactical_types = tactical_rows["type"].astype("string")

            landing_areas = _area_series(tactical_rows, "landing_area", "landing_x", "landing_y")
            landing_areas = landing_areas[landing_areas.isin(IN_COURT_AREA_IDS)]

            left_count = int(landing_areas.isin(LEFT_AREA_IDS).sum())
            right_count = int(landing_areas.isin(RIGHT_AREA_IDS).sum())
            front_back_switches, front_back_transitions = _switch_counts(
                landing_areas, _front_back_label
            )
            left_right_switches, left_right_transitions = _switch_counts(
                landing_areas, _left_right_label
            )

            movement = _stroke_movement_distance(tactical_rows)
            opponent_distance = _landing_opponent_distance(tactical_rows)

            records.append({
                "player": player_name,
                "match_id": int(key[0]),
                "set": int(key[1]),
                "rally": int(key[2]),
                "rally_length": rally_length,
                "valid_action_count": int(len(valid_actions)),
                "target_nonserve_shot_count": int(len(tactical_rows)),
                "attack_shot_count": int(tactical_types.isin(ATTACK_SHOTS).sum()),
                "defense_shot_count": int(
                    tactical_types.isin(DEFENSIVE_RESET_SHOTS - SERVE_SHOTS).sum()
                ),
                "valid_landing_count": int(len(landing_areas)),
                "side_landing_count": int(landing_areas.isin(SIDE_AREA_IDS).sum()),
                "frontcourt_landing_count": int(landing_areas.isin(FRONT_AREA_IDS).sum()),
                "backcourt_landing_count": int(landing_areas.isin(BACK_AREA_IDS).sum()),
                "left_landing_count": left_count,
                "right_landing_count": right_count,
                "front_back_switch_count": front_back_switches,
                "front_back_transition_count": front_back_transitions,
                "left_right_switch_count": left_right_switches,
                "left_right_transition_count": left_right_transitions,
                "average_movement": _mean_or_nan(movement),
                "landing_opponent_distance": _mean_or_nan(opponent_distance),
            })

    columns = [
        "player",
        *RALLY_KEY_COLUMNS,
        "rally_length",
        "valid_action_count",
        "target_nonserve_shot_count",
        "attack_shot_count",
        "defense_shot_count",
        "valid_landing_count",
        "side_landing_count",
        "frontcourt_landing_count",
        "backcourt_landing_count",
        "left_landing_count",
        "right_landing_count",
        "front_back_switch_count",
        "front_back_transition_count",
        "left_right_switch_count",
        "left_right_transition_count",
        "average_movement",
        "landing_opponent_distance",
    ]
    return pd.DataFrame(records, columns=columns)
