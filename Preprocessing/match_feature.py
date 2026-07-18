import numpy as np
import pandas as pd

from Utils.helper import coord2area


POWER_ATTACK_SHOTS = {'殺球', '推撲球'}
ATTACK_SHOTS = {'殺球', '切球', '推撲球'}
BALANCE_SHOTS = {'發短球', '平球', '網前球'}
DEFENSIVE_RESET_SHOTS = {'發長球', '長球', '挑球', '接殺防守'}
DEFENSE_SHOTS = DEFENSIVE_RESET_SHOTS
SERVE_SHOTS = {'發短球', '發長球'}
LOSE_REASONS = {'掛網', '未過網', '犯規', '出界'}

FRONT_AREA_IDS = {17, 18, 19, 20, 21, 22, 23, 24, 25, 29}
MID_AREA_IDS = {9, 10, 11, 12, 13, 14, 15, 16, 26, 30}
BACK_AREA_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 31, 32}
LEFT_AREA_IDS = {1, 5, 9, 13, 17, 21, 29, 30, 31, 32, 2, 6, 10, 14, 18, 22}
RIGHT_AREA_IDS = {4, 8, 12, 16, 20, 24, 25, 26, 27, 28, 3, 7, 11, 15, 19, 23}
SIDELINE_ONLY_AREA_IDS = {25, 26, 27, 29, 30, 31}
DEEP_CORNER_AREA_IDS = {28, 32}
IN_COURT_AREA_IDS = set(range(1, 33))


def _safe_ratio(count, total):
    if total == 0:
        return np.nan
    return count / total


def _area_series(df, area_col, x_col, y_col):
    if area_col in df.columns:
        areas = pd.to_numeric(df[area_col], errors='coerce').copy()
    else:
        areas = pd.Series(np.nan, index=df.index, dtype='float64')

    missing = areas.isna()
    if x_col in df.columns and y_col in df.columns and missing.any():
        coords = df.loc[missing, [x_col, y_col]].dropna()
        for idx, row in coords.iterrows():
            areas.loc[idx] = coord2area(float(row[x_col]), float(row[y_col]))

    return areas


def _valid_shot_rows(player_df):
    if 'type' not in player_df.columns:
        return player_df.iloc[0:0]

    shot_type = player_df['type'].astype('string')
    return player_df[shot_type.notna() & (shot_type != '接不到')]


def _valid_nonserve_rows(player_df):
    valid_rows = _valid_shot_rows(player_df)
    if valid_rows.empty or 'type' not in valid_rows.columns:
        return valid_rows.iloc[0:0]
    shot_type = valid_rows['type'].astype('string')
    return valid_rows[~shot_type.isin(SERVE_SHOTS)]


def _dominant_ratio(series):
    if len(series) == 0:
        return np.nan
    counts = series.dropna().value_counts()
    if counts.empty:
        return np.nan
    return counts.max() / len(series)


def _front_back_label(area):
    if area in FRONT_AREA_IDS:
        return 'front'
    if area in BACK_AREA_IDS:
        return 'back'
    return None


def _side_label(area):
    if area in LEFT_AREA_IDS:
        return 'left'
    if area in RIGHT_AREA_IDS:
        return 'right'
    return None


def _switch_ratio(areas, label_func):
    labels = [label for label in areas.map(label_func).tolist() if label is not None]
    if len(labels) < 2:
        return np.nan
    switches = sum(curr != prev for prev, curr in zip(labels, labels[1:]))
    return switches / (len(labels) - 1)


def _error_ratio(player_df):
    if player_df.empty or 'rally_id' not in player_df.columns or 'lose_reason' not in player_df.columns:
        return np.nan
    participated_rallies = player_df['rally_id'].dropna().nunique()
    if participated_rallies == 0:
        return np.nan
    error_rallies = player_df.loc[
        player_df['lose_reason'].notna() & player_df['lose_reason'].isin(LOSE_REASONS),
        'rally_id',
    ].dropna().nunique()
    return error_rallies / participated_rallies


def _loss_rows(player_df):
    if player_df.empty:
        return player_df.iloc[0:0]
    masks = []
    if 'lose_reason' in player_df.columns:
        masks.append(player_df['lose_reason'].notna() & player_df['lose_reason'].isin(LOSE_REASONS))
    if 'type' in player_df.columns:
        masks.append(player_df['type'].astype('string') == '接不到')
    if not masks:
        return player_df.iloc[0:0]
    mask = masks[0]
    for extra_mask in masks[1:]:
        mask = mask | extra_mask
    return player_df[mask]


def _lost_rally_ids(player_df):
    if player_df.empty or 'rally_id' not in player_df.columns:
        return set()
    return set(_loss_rows(player_df)['rally_id'].dropna().unique().tolist())


def _stroke_movement_distance(player_df):
    if player_df.empty:
        return pd.Series(dtype='float64')

    if 'ball_distance' in player_df.columns:
        approach = pd.to_numeric(player_df['ball_distance'], errors='coerce')
    elif {'hit_x', 'hit_y', 'player_location_x', 'player_location_y'}.issubset(player_df.columns):
        approach = np.sqrt(
            (pd.to_numeric(player_df['hit_x'], errors='coerce') - pd.to_numeric(player_df['player_location_x'], errors='coerce')) ** 2
            + (pd.to_numeric(player_df['hit_y'], errors='coerce') - pd.to_numeric(player_df['player_location_y'], errors='coerce')) ** 2
        )
    else:
        approach = pd.Series(np.nan, index=player_df.index, dtype='float64')

    if {'hit_x', 'hit_y', 'moving_x', 'moving_y'}.issubset(player_df.columns):
        recovery = np.sqrt(
            (pd.to_numeric(player_df['moving_x'], errors='coerce') - pd.to_numeric(player_df['hit_x'], errors='coerce')) ** 2
            + (pd.to_numeric(player_df['moving_y'], errors='coerce') - pd.to_numeric(player_df['hit_y'], errors='coerce')) ** 2
        )
    elif 'moving_distance' in player_df.columns:
        recovery = pd.to_numeric(player_df['moving_distance'], errors='coerce')
    else:
        recovery = pd.Series(np.nan, index=player_df.index, dtype='float64')

    return approach + recovery


def _landing_opponent_distance(player_df):
    if player_df.empty:
        return pd.Series(dtype='float64')
    if 'opponent_moving_distance' in player_df.columns:
        return pd.to_numeric(player_df['opponent_moving_distance'], errors='coerce')
    if {'landing_x', 'landing_y', 'opponent_location_x', 'opponent_location_y'}.issubset(player_df.columns):
        return np.sqrt(
            (pd.to_numeric(player_df['landing_x'], errors='coerce') - pd.to_numeric(player_df['opponent_location_x'], errors='coerce')) ** 2
            + (pd.to_numeric(player_df['landing_y'], errors='coerce') - pd.to_numeric(player_df['opponent_location_y'], errors='coerce')) ** 2
        )
    return pd.Series(np.nan, index=player_df.index, dtype='float64')


def _long_rally_loss_ratio(match_df, player_df, long_rally_threshold):
    if pd.isna(long_rally_threshold):
        return np.nan
    rally_lengths = _rally_lengths(match_df)
    if rally_lengths.empty:
        return np.nan
    long_rally_ids = set(rally_lengths[rally_lengths >= long_rally_threshold].index.tolist())
    if not long_rally_ids:
        return np.nan
    lost_long_rallies = long_rally_ids & _lost_rally_ids(player_df)
    return len(lost_long_rallies) / len(long_rally_ids)


def _rally_lengths(match_df):
    if match_df.empty or 'rally_id' not in match_df.columns or 'ball_round' not in match_df.columns:
        return pd.Series(dtype='float64')
    return match_df.groupby('rally_id')['ball_round'].max().dropna()


def _player_rally_lengths(df, match_ids):
    match_df = df[df['match_id'].isin(match_ids)]
    if match_df.empty or 'rally_id' not in match_df.columns or 'ball_round' not in match_df.columns:
        return pd.Series(dtype='float64')
    return match_df.groupby(['match_id', 'rally_id'])['ball_round'].max().dropna()


def _player_rally_movement(player_df):
    if player_df.empty or 'rally_id' not in player_df.columns or 'moving_distance' not in player_df.columns:
        return pd.Series(dtype='float64')
    group_cols = ['match_id', 'rally_id'] if 'match_id' in player_df.columns else ['rally_id']
    return player_df.groupby(group_cols)['moving_distance'].sum().dropna()


def _style_profile_for_match(df, match_id, player_name, long_rally_threshold):
    match_df = df[df['match_id'] == match_id]
    player_df = match_df[match_df['player'] == player_name]
    valid_shots = _valid_nonserve_rows(player_df)

    shot_count = len(valid_shots)
    shot_types = valid_shots['type'] if shot_count else pd.Series(dtype='object')

    landing_areas = _area_series(valid_shots, 'landing_area', 'landing_x', 'landing_y')
    valid_landing_areas = landing_areas[landing_areas.isin(IN_COURT_AREA_IDS)]
    landing_count = len(valid_landing_areas)

    left_landing_count = valid_landing_areas.isin(LEFT_AREA_IDS).sum()
    right_landing_count = valid_landing_areas.isin(RIGHT_AREA_IDS).sum()

    rally_lengths = _rally_lengths(match_df)
    rally_count = len(rally_lengths)

    if rally_count == 0:
        average_rally_length = np.nan
    else:
        average_rally_length = rally_lengths.mean()

    average_movement = _stroke_movement_distance(valid_shots).mean()
    landing_opponent_distance = _landing_opponent_distance(valid_shots).mean()
    return {
        'match_id': match_id,
        'attack_shot_ratio': _safe_ratio(shot_types.isin(ATTACK_SHOTS).sum(), shot_count),
        'defense_shot_ratio': _safe_ratio(shot_types.isin(DEFENSIVE_RESET_SHOTS - SERVE_SHOTS).sum(), shot_count),
        'sideline_ratio': _safe_ratio(
            valid_landing_areas.isin(SIDELINE_ONLY_AREA_IDS).sum(),
            landing_count,
        ),
        'frontcourt_ratio': _safe_ratio(
            valid_landing_areas.isin(FRONT_AREA_IDS).sum(),
            landing_count,
        ),
        'deepcourt_ratio': _safe_ratio(
            valid_landing_areas.isin(DEEP_CORNER_AREA_IDS).sum(),
            landing_count,
        ),
        'signed_side_bias': _safe_ratio(right_landing_count - left_landing_count, landing_count),
        'average_rally_length': average_rally_length,
        'average_movement': average_movement,
        'long_rally_loss_ratio': _long_rally_loss_ratio(match_df, player_df, long_rally_threshold),
        'landing_opponent_distance': landing_opponent_distance,
    }


def extract_style_profile_features(df, match_ids, player_name):
    player_rally_lengths = _player_rally_lengths(df, match_ids)
    long_rally_threshold = player_rally_lengths.quantile(0.75) if len(player_rally_lengths) else np.nan
    features_list = [
        _style_profile_for_match(df, mid, player_name, long_rally_threshold)
        for mid in match_ids
    ]

    return pd.DataFrame(features_list)
