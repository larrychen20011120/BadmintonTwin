import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from jinja2 import Template

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Preprocessing.match_feature import extract_style_profile_features
from Utils.llm import LLM
from Utils.runtime_report import llm_rows, print_runtime_box


FEATURE_COLS = [
    'attack_shot_ratio',
    'defense_shot_ratio',
    'sideline_ratio',
    'frontcourt_ratio',
    'deepcourt_ratio',
    'signed_side_bias',
    'average_rally_length',
    'average_movement',
    'long_rally_loss_ratio',
    'landing_opponent_distance',
]

FEATURE_DIMENSIONS = {
    'style_profile': FEATURE_COLS,
}

FEATURE_LABELS = {
    'attack_shot_ratio': 'attack shot ratio',
    'defense_shot_ratio': 'defense shot ratio',
    'sideline_ratio': 'sideline landing ratio',
    'frontcourt_ratio': 'frontcourt landing ratio',
    'deepcourt_ratio': 'deepcourt landing ratio',
    'signed_side_bias': 'signed left-right landing bias',
    'average_rally_length': 'average rally length',
    'average_movement': 'average movement',
    'long_rally_loss_ratio': 'long-rally loss ratio',
    'landing_opponent_distance': 'landing-opponent distance',
}

def _json_dumps(data):
    return json.dumps(data, ensure_ascii=False, indent=2)


def _clean_number(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value


def _clean_dict(data):
    return {key: _clean_number(value) for key, value in data.items()}


def _to_nested_profile(flat_profile):
    return {key: _clean_number(value) for key, value in flat_profile.items()}


def _profile_from_row(row, feature_cols):
    return _to_nested_profile({col: row[col] for col in feature_cols})


def _mean_profile(df, feature_cols):
    if df.empty:
        return _to_nested_profile({col: np.nan for col in feature_cols})
    return _to_nested_profile(dict(df[feature_cols].mean(skipna=True)))


def _load_config():
    config_path = os.path.join(project_path, 'Scripts', 'config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_path, path)


def _build_match_player_df(merged_df):
    records = []
    for match_id, match_df in merged_df.groupby('match_id'):
        player_counts = match_df['player'].dropna().value_counts()
        if len(player_counts) < 2:
            continue
        player, opponent = player_counts.index[:2].tolist()
        records.append({
            'match_id': int(match_id),
            'player': player,
            'opponent': opponent,
        })
    return pd.DataFrame(records)


def _load_match_player_df(merged_df, match_path):
    if match_path and os.path.exists(match_path):
        return pd.read_csv(match_path)
    return _build_match_player_df(merged_df)


def _iqr_value(value, median, iqr, cap):
    if pd.isna(value) or pd.isna(median):
        return 0.0
    if pd.isna(iqr) or iqr == 0:
        return 0.0 if value == median else cap
    return (value - median) / iqr


def _deviation_strength(abs_iqr):
    if abs_iqr >= 3.0:
        return 'extreme'
    if abs_iqr >= 2.0:
        return 'strong'
    if abs_iqr >= 1.5:
        return 'moderate'
    return 'normal'


def _side_bias_label(value, neutral_threshold=0.05):
    if pd.isna(value):
        return 'unknown'
    if value > neutral_threshold:
        return 'right-biased'
    if value < -neutral_threshold:
        return 'left-biased'
    return 'balanced'


def _supporting_signal(feature, signed_iqr, value, baseline_median, baseline_iqr):
    direction = 'higher' if signed_iqr > 0 else 'lower'
    signal = {
        'feature': feature,
        'label': FEATURE_LABELS.get(feature, feature),
        'direction': direction,
        'signed_iqr': _clean_number(signed_iqr),
        'abs_iqr': _clean_number(abs(float(signed_iqr))),
        'deviation_strength': _deviation_strength(abs(float(signed_iqr))),
        'current_value': _clean_number(value),
        'baseline_median': _clean_number(baseline_median),
        'baseline_iqr': _clean_number(baseline_iqr),
    }
    if feature == 'signed_side_bias':
        signal.update({
            'current_side_bias': _side_bias_label(value),
            'baseline_side_bias': _side_bias_label(baseline_median),
            'relative_side_shift': 'more rightward than usual' if signed_iqr > 0 else 'more leftward than usual',
        })
    return signal


def _score_player_matches(feature_df, feature_threshold, feature_count_threshold, total_threshold, cap, max_shifted_ratio):
    rows = []
    signals_by_idx = {}
    baselines_by_idx = {}
    values = feature_df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')

    for idx, row in values.iterrows():
        baseline = values.drop(index=idx)
        median = baseline.median(skipna=True)
        iqr = baseline.quantile(0.75) - baseline.quantile(0.25)
        baselines_by_idx[idx] = _clean_dict(dict(median))

        supporting_signals = []
        capped_abs = []
        for feature in FEATURE_COLS:
            signed_iqr = _iqr_value(row[feature], median[feature], iqr[feature], cap)
            capped_abs.append(min(abs(signed_iqr), cap))
            if abs(signed_iqr) >= feature_threshold:
                supporting_signals.append(_supporting_signal(feature, signed_iqr, row[feature], median[feature], iqr[feature]))

        supporting_signals = sorted(supporting_signals, key=lambda item: abs(item['signed_iqr']), reverse=True)
        total_score = float(np.mean(capped_abs))
        shifted_candidate = len(supporting_signals) >= feature_count_threshold and total_score >= total_threshold
        rows.append({
            'row_index': idx,
            'match_id': int(feature_df.loc[idx, 'match_id']),
            'total_score': total_score,
            'shifted_feature_count': len(supporting_signals),
            'shifted_candidate': bool(shifted_candidate),
            'shifted': False,
        })
        signals_by_idx[idx] = supporting_signals

    scores = pd.DataFrame(rows).set_index('row_index') if rows else pd.DataFrame()
    if scores.empty:
        return scores, signals_by_idx, baselines_by_idx

    if max_shifted_ratio is None or max_shifted_ratio <= 0:
        max_shifted_count = len(scores)
    else:
        max_shifted_count = int(np.floor(len(scores) * max_shifted_ratio))

    candidates = scores[scores['shifted_candidate']].sort_values(
        ['total_score', 'shifted_feature_count', 'match_id'],
        ascending=[False, False, True],
    )
    keep_ids = set(candidates.head(max_shifted_count)['match_id'].astype(int).tolist())
    scores['shifted'] = scores['match_id'].astype(int).isin(keep_ids)

    return scores, signals_by_idx, baselines_by_idx


def construct_user_prompt(
    user_prompt_template,
    normal_profile,
    abnormal_profile,
    supporting_signals,
    total_score,
    shifted_feature_count,
):
    return user_prompt_template.render(
        normal_profile=_json_dumps(normal_profile),
        abnormal_profile=_json_dumps(abnormal_profile),
        supporting_signals=_json_dumps(supporting_signals),
        total_score=_clean_number(total_score),
        shifted_feature_count=int(shifted_feature_count),
    )


def get_difference(llm, schema, system_prompt='', user_prompt=''):
    result = llm.get_verified_response(schema, system_prompt, user_prompt)
    response = result[0] if isinstance(result, tuple) else result
    print(response)
    if response is None:
        raise RuntimeError('LLM failed to generate a valid style-shift JSON response.')
    return response['instruction']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enough_thres', type=int, default=8, help='Minimum player match count needed for anomaly detection.')
    parser.add_argument('--feature_threshold', type=float, default=1.5, help='Minimum absolute leave-one-out IQR deviation for a supporting signal.')
    parser.add_argument('--feature_count_threshold', type=int, default=3, help='Minimum supporting signals required for a shifted candidate.')
    parser.add_argument('--total_threshold', type=float, default=1.1, help='Minimum average capped IQR deviation required for a shifted candidate.')
    parser.add_argument('--cap', type=float, default=3.0, help='Cap applied to each absolute IQR deviation when computing total_score.')
    parser.add_argument('--max_shifted_ratio', type=float, default=0.25, help='Per-player cap for shifted matches using floor(match_count * ratio). Set 0 to disable.')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)

    args = parser.parse_args()

    config = _load_config()
    paths = config.get('paths', {}) if isinstance(config.get('paths', {}), dict) else {}
    merged_path = _project_path(paths.get('merged_csv', os.path.join('Data', 'intermediate_data', 'merged.csv')))
    match_path = _project_path(paths.get('match_player_csv', os.path.join('Data', 'intermediate_data', 'match_player.csv')))
    save_path = _project_path(paths.get('style_shift_json', os.path.join('Data', 'intermediate_data', 'style_shift.json')))
    user_template_path = _project_path(os.path.join('Prompts', 'Preprocessing', 'style_shift.jinja'))
    schema_path = _project_path(os.path.join('Schemas', 'Preprocessing', 'style_shift_schema.json'))

    merged_df = pd.read_csv(merged_path)
    match_df = _load_match_player_df(merged_df, match_path)

    with open(user_template_path, 'r', encoding='utf-8') as f:
        user_prompt_template = Template(f.read())
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    llm_settings = LLM.resolve_effective_settings(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    print_runtime_box(
        'Runtime Parameters: style_shift',
        [
            ('enough_thres', args.enough_thres),
            ('feature_threshold', args.feature_threshold),
            ('feature_count_threshold', args.feature_count_threshold),
            ('total_threshold', args.total_threshold),
            ('cap', args.cap),
            ('max_shifted_ratio', args.max_shifted_ratio if args.max_shifted_ratio > 0 else 'disabled'),
            ('max_shifted_count_rule', 'floor(match_count * max_shifted_ratio)'),
            ('merged_path', merged_path),
            ('match_path', match_path if os.path.exists(match_path) else 'derived from merged.csv'),
            ('save_path', save_path),
        ] + llm_rows(llm_settings),
    )

    llm = LLM(model_name=args.model_name, api_key=args.api_key, base_url=args.base_url)

    player_counts = match_df['player'].value_counts()
    opponent_counts = match_df['opponent'].value_counts()
    total_counts = player_counts.add(opponent_counts, fill_value=0).astype(int)

    enough_players = total_counts[total_counts >= args.enough_thres].index
    not_enough_players = total_counts[total_counts < args.enough_thres].index

    data = {
        'normal': [],
        'shifted': [],
    }

    for player_name in not_enough_players:
        match_ids = match_df[(match_df['player'] == player_name) | (match_df['opponent'] == player_name)]['match_id'].tolist()
        feature_df = extract_style_profile_features(
            merged_df,
            match_ids,
            player_name,
        )
        feature_cols = [col for col in feature_df.columns if col != 'match_id']

        data['normal'].append({
            'name': player_name,
            'match_ids': [int(mid) for mid in match_ids],
            'match_profile': _mean_profile(feature_df, feature_cols),
        })

    for player_name in enough_players:
        match_ids = match_df[(match_df['player'] == player_name) | (match_df['opponent'] == player_name)]['match_id'].tolist()
        feature_df = extract_style_profile_features(
            merged_df,
            match_ids,
            player_name,
        )
        feature_cols = [col for col in feature_df.columns if col != 'match_id']

        scores_df, supporting_signals_by_idx, baselines_by_idx = _score_player_matches(
            feature_df,
            args.feature_threshold,
            args.feature_count_threshold,
            args.total_threshold,
            args.cap,
            args.max_shifted_ratio,
        )

        shifted_indices = scores_df.index[scores_df['shifted']].tolist()
        normal_indices = scores_df.index[~scores_df['shifted']].tolist()
        normal_df = feature_df.loc[normal_indices]
        baseline_df = normal_df if not normal_df.empty else feature_df
        baseline_flat = _clean_dict(dict(baseline_df[feature_cols].mean(skipna=True)))
        normal_profile = _to_nested_profile(baseline_flat)

        data['normal'].append({
            'name': player_name,
            'match_ids': [int(mid) for mid in normal_df['match_id'].tolist()],
            'match_profile': normal_profile,
        })

        match_profiles = []
        ids = []
        iqr_scores = []
        total_scores = []
        shifted_feature_counts = []
        supporting_signals = []
        inferred_reasons = []
        instructions = []

        abnormal_df = feature_df.loc[shifted_indices]
        for idx, row in abnormal_df.iterrows():
            abnormal_profile = _profile_from_row(row, feature_cols)
            row_score = scores_df.loc[idx]
            row_supporting_signals = supporting_signals_by_idx.get(idx, [])
            row_baseline_profile = _to_nested_profile(baselines_by_idx.get(idx, baseline_flat))

            user_prompt = construct_user_prompt(
                user_prompt_template,
                row_baseline_profile,
                abnormal_profile,
                row_supporting_signals,
                row_score['total_score'],
                row_score['shifted_feature_count'],
            )
            inferred_reason = get_difference(llm, schema, user_prompt=user_prompt)

            ids.append(int(row['match_id']))
            match_profiles.append(abnormal_profile)
            iqr_scores.append({
                'total_score': _clean_number(row_score['total_score']),
                'shifted_feature_count': int(row_score['shifted_feature_count']),
            })
            total_scores.append(_clean_number(row_score['total_score']))
            shifted_feature_counts.append(int(row_score['shifted_feature_count']))
            supporting_signals.append(row_supporting_signals)
            inferred_reasons.append(inferred_reason)
            instructions.append(inferred_reason)

        data['shifted'].append({
            'name': player_name,
            'match_ids': ids,
            'match_profiles': match_profiles,
            'iqr_scores': iqr_scores,
            'total_scores': total_scores,
            'shifted_feature_counts': shifted_feature_counts,
            'supporting_signals': supporting_signals,
            'inferred_reasons': inferred_reasons,
            'instructions': instructions,
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print('JSON File Dumped!!')
