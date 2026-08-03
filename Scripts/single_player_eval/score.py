#!/usr/bin/env python3
"""Canonical scoring for the latest run or all completed runs under a root."""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from dtaidistance import dtw_ndim
except ImportError:
    dtw_ndim = None


METRIC_NAMES = (
    'land_dtw',
    'move_dtw',
    'shot_lcs_ratio',
    'shotland_jsd',
    'shotdist_jsd',
    'rallylen_jsd',
)
SEQUENCE_METRIC_NAMES = METRIC_NAMES[:3]
COORDINATE_KEYS = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
N_AREA_GROUPS = 18
OUT_OF_BOUNDS_GROUP = 17

# Raw areas use the same canonical grouping as badminton_config.area_groups.
AREA_TO_GROUP = {
    32: 0, 28: 1, 1: 2, 5: 2, 4: 3, 8: 3,
    2: 4, 3: 4, 6: 4, 7: 4, 9: 5, 13: 5,
    12: 6, 16: 6, 10: 7, 11: 7, 14: 7, 15: 7,
    20: 8, 24: 8, 17: 9, 21: 9, 18: 10, 19: 10,
    22: 10, 23: 10, 29: 11, 25: 12, 30: 13, 26: 14,
    31: 15, 27: 16, 33: 17,
}
RAW_AREA_BOUNDS = (
    (50, 150, 104, 204, 1), (104, 150, 177.5, 204, 2),
    (177.5, 150, 251, 204, 3), (251, 150, 305, 204, 4),
    (50, 204, 104, 258, 5), (104, 204, 177.5, 258, 6),
    (177.5, 204, 251, 258, 7), (251, 204, 305, 258, 8),
    (50, 258, 104, 312, 9), (104, 258, 177.5, 312, 10),
    (177.5, 258, 251, 312, 11), (251, 258, 305, 312, 12),
    (50, 312, 104, 366, 13), (104, 312, 177.5, 366, 14),
    (177.5, 312, 251, 366, 15), (251, 312, 305, 366, 16),
    (50, 366, 104, 423, 17), (104, 366, 177.5, 423, 18),
    (177.5, 366, 251, 423, 19), (251, 366, 305, 423, 20),
    (50, 423, 104, 480, 21), (104, 423, 177.5, 480, 22),
    (177.5, 423, 251, 480, 23), (251, 423, 305, 480, 24),
    (305, 204, 355, 366, 26), (305, 366, 355, 480, 25),
    (305, 0, 355, 204, 27), (177.5, 0, 305, 150, 28),
    (0, 204, 50, 366, 30), (0, 366, 50, 480, 29),
    (0, 0, 50, 204, 31), (50, 0, 177.5, 150, 32),
)


def load_trajectories(input_path):
    """Load rally JSON documents without depending on trajectory_store.py."""
    input_path = Path(input_path)
    if input_path.is_dir():
        paths = sorted(input_path.rglob('*.json'))
        paths = [
            path for path in paths
            if path.name not in {'manifest.json', 'summary.json', 'experiment_metrics.json'}
        ]
    elif input_path.suffix.lower() == '.json':
        paths = [input_path]
    else:
        raise ValueError(f'Unsupported trajectory path: {input_path}')

    trajectories = []
    for path in paths:
        with path.open('r', encoding='utf-8') as handle:
            document = json.load(handle)
        trajectories.extend(document if isinstance(document, list) else [document])
    print(f'Trajectories loaded from {input_path} ({len(trajectories)} rallies)')
    return trajectories


def trajectory_players(trajectory):
    target_player = trajectory.get('target_player')
    if target_player:
        return [str(target_player)]
    players = trajectory.get('players')
    if isinstance(players, list):
        return list(dict.fromkeys(str(player) for player in players if player))

    inferred = []
    for sequence_name in ('gt_sequence', 'pred_sequence'):
        for step in trajectory.get(sequence_name, []):
            player = step.get('player')
            if player and player not in inferred:
                inferred.append(player)
    return inferred


def as_player_view(trajectory, player):
    view = dict(trajectory)
    view['target_player'] = player
    view['opponent_player'] = next(
        (candidate for candidate in trajectory_players(trajectory) if candidate != player),
        trajectory.get('opponent_player'),
    )
    return view


def extract_sequences(trajectories, target_player_only=False, skip_n=0):
    data = {
        'all_land_pred': [], 'all_land_gt': [],
        'all_move_pred': [], 'all_move_gt': [],
        'all_shot_pred': [], 'all_shot_gt': [],
        'all_ball_rounds_pred': [], 'all_ball_rounds_gt': [],
        'all_rally_lengths_pred': [], 'all_rally_lengths_gt': [],
        'land_pred_seqs': [], 'land_gt_seqs': [],
        'move_pred_seqs': [], 'move_gt_seqs': [],
        'shot_pred_seqs': [], 'shot_gt_seqs': [],
        'rally_player_won': [], 'target_player': None,
    }

    def finite_positions(step):
        try:
            if int(step.get('shot_type')) == 11:
                return None
            values = [float(step[name]) for name in COORDINATE_KEYS]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return values[:2], values[2:]

    for trajectory in trajectories:
        target_player = trajectory.get('target_player', '')
        data['target_player'] = target_player
        for sequence_name in ('gt', 'pred'):
            sequence = trajectory.get(f'{sequence_name}_sequence', [])
            landing_sequence = []
            movement_sequence = []
            shot_sequence = []
            skipped = 0
            for step in sequence:
                if target_player_only and step.get('player', '') != target_player:
                    continue
                if skipped < skip_n:
                    skipped += 1
                    continue
                shot_type = step['shot_type']
                data[f'all_shot_{sequence_name}'].append(shot_type)
                data[f'all_ball_rounds_{sequence_name}'].append(step.get('ball_round', 0))
                shot_sequence.append(shot_type)
                positions = finite_positions(step)
                if positions is not None:
                    landing, movement = positions
                    data[f'all_land_{sequence_name}'].append(landing)
                    data[f'all_move_{sequence_name}'].append(movement)
                    landing_sequence.append(landing)
                    movement_sequence.append(movement)
            data[f'land_{sequence_name}_seqs'].append(
                np.asarray(landing_sequence, dtype=float).reshape(-1, 2)
            )
            data[f'move_{sequence_name}_seqs'].append(
                np.asarray(movement_sequence, dtype=float).reshape(-1, 2)
            )
            data[f'shot_{sequence_name}_seqs'].append(shot_sequence)

        data['all_rally_lengths_pred'].append(len(trajectory.get('pred_sequence', [])))
        data['all_rally_lengths_gt'].append(len(trajectory.get('gt_sequence', [])))
        won = trajectory.get('target_player_won')
        data['rally_player_won'].append(won if isinstance(won, bool) else False)

    for key in ('all_land_pred', 'all_land_gt', 'all_move_pred', 'all_move_gt'):
        data[key] = np.asarray(data[key], dtype=float).reshape(-1, 2)
    for key in ('all_shot_pred', 'all_shot_gt', 'all_ball_rounds_pred', 'all_ball_rounds_gt'):
        data[key] = np.asarray(data[key], dtype=int)
    return data


def normalized_dtw_distance(predicted, ground_truth):
    if dtw_ndim is None:
        return float('nan')
    distance = dtw_ndim.distance(predicted, ground_truth)
    warping_path = dtw_ndim.warping_path(predicted, ground_truth)
    return float(distance / np.sqrt(len(warping_path))) if warping_path else float('inf')


class SequenceMetrics:
    @staticmethod
    def compute_dtw(land_pred_seqs, land_gt_seqs, move_pred_seqs, move_gt_seqs):
        if dtw_ndim is None:
            return {'land_dtw': float('nan'), 'move_dtw': float('nan')}
        land_values = []
        move_values = []
        for predicted, ground_truth in zip(land_pred_seqs, land_gt_seqs):
            if len(predicted) >= 2 and len(ground_truth) >= 2:
                try:
                    land_values.append(normalized_dtw_distance(predicted, ground_truth))
                except Exception:
                    pass
        for predicted, ground_truth in zip(move_pred_seqs, move_gt_seqs):
            if len(predicted) >= 2 and len(ground_truth) >= 2:
                try:
                    move_values.append(normalized_dtw_distance(predicted, ground_truth))
                except Exception:
                    pass
        return {
            'land_dtw': float(np.mean(land_values)) if land_values else float('inf'),
            'move_dtw': float(np.mean(move_values)) if move_values else float('inf'),
        }

    @staticmethod
    def _lcs_length(predicted, ground_truth):
        if len(predicted) < len(ground_truth):
            shorter, longer = predicted, ground_truth
        else:
            shorter, longer = ground_truth, predicted
        previous = [0] * (len(shorter) + 1)
        for token in longer:
            current = [0] * (len(shorter) + 1)
            for index, other_token in enumerate(shorter, start=1):
                if token == other_token:
                    current[index] = previous[index - 1] + 1
                else:
                    current[index] = max(previous[index], current[index - 1])
            previous = current
        return previous[-1]

    @classmethod
    def compute_lcs_ratio(cls, shot_pred_seqs, shot_gt_seqs):
        values = []
        for predicted, ground_truth in zip(shot_pred_seqs, shot_gt_seqs):
            maximum_length = max(len(predicted), len(ground_truth))
            if maximum_length:
                values.append(cls._lcs_length(predicted, ground_truth) / maximum_length)
        return {'lcs_ratio': float(np.mean(values)) if values else float('inf')}


def coordinate_area_group(normalized_x, normalized_y):
    x = normalized_x * 177.5 + 177.5
    y = normalized_y * 240.0 + 240.0
    raw_area = 33
    for min_x, min_y, max_x, max_y, area_id in RAW_AREA_BOUNDS:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            raw_area = area_id
    return AREA_TO_GROUP.get(raw_area, OUT_OF_BOUNDS_GROUP)


def js_divergence(predicted, ground_truth):
    predicted = np.asarray(predicted, dtype=np.float64) + 1e-10
    ground_truth = np.asarray(ground_truth, dtype=np.float64) + 1e-10
    predicted /= predicted.sum()
    ground_truth /= ground_truth.sum()
    midpoint = (predicted + ground_truth) / 2.0
    return float(
        0.5 * np.sum(predicted * np.log2(predicted / midpoint))
        + 0.5 * np.sum(ground_truth * np.log2(ground_truth / midpoint))
    )


class DistributionMetrics:
    VALID_SHOTS = tuple(range(1, 11))

    @staticmethod
    def _safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _area_from_step(step):
        try:
            return coordinate_area_group(float(step['landing_x']), float(step['landing_y']))
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def _sequence_actions(cls, trajectories, sequence_name, target_player_only):
        actions = []
        for trajectory in trajectories:
            target_player = trajectory.get('target_player', '')
            for step in trajectory.get(sequence_name, []):
                if target_player_only and step.get('player', '') != target_player:
                    continue
                actions.append(step)
        return actions

    @classmethod
    def _count_js(cls, predicted_counts, ground_truth_counts, bins):
        if not predicted_counts and not ground_truth_counts:
            return float('nan')
        predicted = np.array([predicted_counts.get(index, 0) for index in range(bins)])
        ground_truth = np.array([ground_truth_counts.get(index, 0) for index in range(bins)])
        return js_divergence(predicted, ground_truth)

    @classmethod
    def shot_distribution_js(cls, trajectories, target_player_only):
        predicted_counts = defaultdict(int)
        ground_truth_counts = defaultdict(int)
        for step in cls._sequence_actions(trajectories, 'pred_sequence', target_player_only):
            shot_type = cls._safe_int(step.get('shot_type'))
            if shot_type in cls.VALID_SHOTS:
                predicted_counts[shot_type - 1] += 1
        for step in cls._sequence_actions(trajectories, 'gt_sequence', target_player_only):
            shot_type = cls._safe_int(step.get('shot_type'))
            if shot_type in cls.VALID_SHOTS:
                ground_truth_counts[shot_type - 1] += 1
        return cls._count_js(predicted_counts, ground_truth_counts, len(cls.VALID_SHOTS))

    @classmethod
    def shot_conditioned_landing_js(cls, trajectories, target_player_only):
        predicted = defaultdict(lambda: defaultdict(int))
        ground_truth = defaultdict(lambda: defaultdict(int))
        for step in cls._sequence_actions(trajectories, 'pred_sequence', target_player_only):
            shot_type = cls._safe_int(step.get('shot_type'))
            area = cls._area_from_step(step)
            if shot_type in cls.VALID_SHOTS and area is not None:
                predicted[shot_type][area] += 1
        for step in cls._sequence_actions(trajectories, 'gt_sequence', target_player_only):
            shot_type = cls._safe_int(step.get('shot_type'))
            area = cls._area_from_step(step)
            if shot_type in cls.VALID_SHOTS and area is not None:
                ground_truth[shot_type][area] += 1
        return cls._weighted_conditioned_js(predicted, ground_truth, N_AREA_GROUPS)

    @classmethod
    def rally_length_js(cls, trajectories):
        predicted_counts = defaultdict(int)
        ground_truth_counts = defaultdict(int)
        for trajectory in trajectories:
            predicted = cls._rally_round_length(
                trajectory.get('pred_sequence', []), trajectory.get('rally_length_pred')
            )
            ground_truth = cls._rally_round_length(
                trajectory.get('gt_sequence', []), trajectory.get('rally_length_gt')
            )
            predicted_counts[predicted] += 1
            ground_truth_counts[ground_truth] += 1
        maximum = max([*predicted_counts, *ground_truth_counts], default=0)
        return cls._count_js(predicted_counts, ground_truth_counts, maximum + 1)

    @classmethod
    def _rally_round_length(cls, sequence, fallback):
        rounds = [cls._safe_int(step.get('ball_round')) for step in sequence]
        rounds = [round_id for round_id in rounds if round_id is not None]
        if rounds:
            return max(rounds)
        fallback_length = cls._safe_int(fallback)
        return max(fallback_length or 0, 0)

    @classmethod
    def _weighted_conditioned_js(cls, predicted, ground_truth, bins):
        weighted_js = 0.0
        total_weight = 0
        for key, ground_truth_counts in ground_truth.items():
            support = sum(ground_truth_counts.values())
            divergence = cls._count_js(predicted.get(key, {}), ground_truth_counts, bins)
            if support and math.isfinite(divergence):
                weighted_js += divergence * support
                total_weight += support
        return float(weighted_js / total_weight) if total_weight else float('nan')


def finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def physical_key(trajectory):
    try:
        return (
            int(trajectory['match_id']),
            int(trajectory['set']),
            int(trajectory['rally']),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Trajectory has an invalid physical rally identity') from exc


def _validate_sequence(sequence, label, key):
    if not isinstance(sequence, list):
        raise ValueError(f'{label} must be a list for rally {key}')
    previous_round = 0
    for step in sequence:
        if not isinstance(step, dict):
            raise ValueError(f'{label} contains a non-object step for rally {key}')
        try:
            shot_type = int(step['shot_type'])
            ball_round = int(step['ball_round'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{label} has invalid shot_type/ball_round for rally {key}') from exc
        if not 1 <= shot_type <= 11:
            raise ValueError(f'{label} has shot type {shot_type} outside 1-11 for rally {key}')
        if ball_round <= previous_round:
            raise ValueError(f'{label} ball_round must increase for rally {key}')
        previous_round = ball_round

        coordinates = [step.get(name) for name in COORDINATE_KEYS]
        if shot_type == 11:
            if any(value is not None for value in coordinates):
                raise ValueError(f'Missed shot coordinates must be null for rally {key}')
            continue
        try:
            numeric = [float(value) for value in coordinates]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Non-missed shot coordinates must be numeric for rally {key}') from exc
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f'Non-finite coordinates found for rally {key}')


def validate_trajectories(trajectories):
    if not trajectories:
        raise ValueError('Trajectory run is empty')
    seen = set()
    for trajectory in trajectories:
        key = physical_key(trajectory)
        if key in seen:
            raise ValueError(f'Duplicate physical rally in run: {key}')
        seen.add(key)
        if trajectory.get('target_player') not in (None, ''):
            raise ValueError('score.py expects one-pass physical trajectories with target_player=null')
        _validate_sequence(trajectory.get('gt_sequence'), 'gt_sequence', key)
        _validate_sequence(trajectory.get('pred_sequence'), 'pred_sequence', key)
        for sequence_name, length_name in (
            ('gt_sequence', 'rally_length_gt'),
            ('pred_sequence', 'rally_length_pred'),
        ):
            stored = trajectory.get(length_name)
            if stored is not None and int(stored) != len(trajectory[sequence_name]):
                raise ValueError(f'{length_name} disagrees with {sequence_name} for rally {key}')
    return seen


def resolve_run(path):
    path = Path(path).resolve()
    if path.name == 'rallies' and path.is_dir():
        run_dir = path.parent
        rally_dir = path
    elif path.is_dir() and (path / 'rallies').is_dir():
        run_dir = path
        rally_dir = path / 'rallies'
    else:
        raise ValueError(f'Expected a run_NNN directory or its rallies directory: {path}')

    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing manifest.json in {run_dir}')
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('status') != 'completed':
        raise ValueError(f'Run is not completed: {run_dir}')

    trajectories = load_trajectories(rally_dir)
    keys = validate_trajectories(trajectories)
    expected = manifest.get('num_rallies')
    if expected is not None and int(expected) != len(trajectories):
        raise ValueError(
            f"Manifest expects {expected} rallies but {len(trajectories)} were loaded from {run_dir}"
        )
    return {
        'path': str(run_dir),
        'name': run_dir.name,
        'manifest': manifest,
        'trajectories': trajectories,
        'keys': keys,
        'by_key': {physical_key(trajectory): trajectory for trajectory in trajectories},
    }


def discover_completed_runs(run_root):
    """Return completed direct-child run_NNN directories in generation order."""
    run_root = Path(run_root).resolve()
    if not run_root.is_dir():
        raise ValueError(f'Run root is not a directory: {run_root}')

    completed = []
    for path in run_root.iterdir():
        match = re.fullmatch(r'run_(\d+)', path.name) if path.is_dir() else None
        if not match or not (path / 'rallies').is_dir():
            continue
        manifest_path = path / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(f'Missing manifest.json in {path}')
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('status') == 'completed':
            completed.append((int(match.group(1)), str(path)))

    if not completed:
        raise ValueError(f'No completed run_NNN directories found in {run_root}')
    return [path for _, path in sorted(completed)]


def validate_runs(runs):
    reference = runs[0]
    comparable_fields = (
        'algorithm', 'experiment', 'instruction_mode', 'instruction_variant',
    )
    for run in runs[1:]:
        for field in comparable_fields:
            default = 'original' if field == 'instruction_variant' else None
            if run['manifest'].get(field, default) != reference['manifest'].get(field, default):
                raise ValueError(f'Runs disagree on manifest field {field!r}')
        if run['keys'] != reference['keys']:
            raise ValueError('Runs do not contain the same physical rally keys')
        for key in reference['keys']:
            reference_gt = json.dumps(
                reference['by_key'][key]['gt_sequence'], sort_keys=True, separators=(',', ':')
            )
            candidate_gt = json.dumps(
                run['by_key'][key]['gt_sequence'], sort_keys=True, separators=(',', ':')
            )
            if candidate_gt != reference_gt:
                raise ValueError(f'Ground truth differs across runs for rally {key}')


def compute_sequence_metric_samples(trajectories, target_player_only=False):
    samples = {metric: [] for metric in SEQUENCE_METRIC_NAMES}
    sequence = SequenceMetrics()
    for trajectory in trajectories:
        data = extract_sequences(
            [trajectory],
            target_player_only=target_player_only,
            skip_n=0,
        )
        dtw = sequence.compute_dtw(
            data['land_pred_seqs'],
            data['land_gt_seqs'],
            data['move_pred_seqs'],
            data['move_gt_seqs'],
        )
        lcs = sequence.compute_lcs_ratio(data['shot_pred_seqs'], data['shot_gt_seqs'])
        samples['land_dtw'].append(finite_or_none(dtw.get('land_dtw')))
        samples['move_dtw'].append(finite_or_none(dtw.get('move_dtw')))
        samples['shot_lcs_ratio'].append(finite_or_none(lcs.get('lcs_ratio')))
    return samples


def compute_metric_set(
    trajectories,
    target_player_only=False,
    sequence_samples=None,
):
    if sequence_samples is None:
        sequence_samples = compute_sequence_metric_samples(
            trajectories,
            target_player_only=target_player_only,
        )
    distribution = DistributionMetrics()
    return {
        metric: _mean_metric(sequence_samples[metric])['mean']
        for metric in SEQUENCE_METRIC_NAMES
    } | {
        'shotland_jsd': finite_or_none(
            distribution.shot_conditioned_landing_js(trajectories, target_player_only)
        ),
        'shotdist_jsd': finite_or_none(
            distribution.shot_distribution_js(trajectories, target_player_only)
        ),
        'rallylen_jsd': finite_or_none(distribution.rally_length_js(trajectories)),
    }


def build_scope(
    trajectories,
    target_player_only=False,
    label=None,
    instruction=None,
    sequence_trajectories=None,
    sequence_target_player_only=None,
):
    data = extract_sequences(trajectories, target_player_only=target_player_only)
    sequence_trajectories = sequence_trajectories or trajectories
    if sequence_target_player_only is None:
        sequence_target_player_only = target_player_only
    sequence_samples = compute_sequence_metric_samples(
        sequence_trajectories,
        target_player_only=sequence_target_player_only,
    )
    scope = {
        'support_rallies': len(trajectories),
        'num_gt_shots': int(len(data['all_shot_gt'])),
        'num_pred_shots': int(len(data['all_shot_pred'])),
        'scores': compute_metric_set(
            trajectories,
            target_player_only,
            sequence_samples=sequence_samples,
        ),
    }
    if label is not None:
        scope['label'] = label
    if instruction is not None:
        scope['instruction'] = instruction
    return scope


def compute_hierarchy(trajectories):
    players = sorted({player for trajectory in trajectories for player in trajectory_players(trajectory)})
    rally_player_views = [
        as_player_view(trajectory, player)
        for trajectory in trajectories
        for player in trajectory_players(trajectory)
    ]
    result = {
        'overall': build_scope(
            trajectories,
            target_player_only=False,
            sequence_trajectories=rally_player_views,
            sequence_target_player_only=True,
        ),
        'players': {},
    }
    for player in players:
        player_trajectories = [
            as_player_view(trajectory, player)
            for trajectory in trajectories
            if player in trajectory_players(trajectory)
        ]
        player_result = build_scope(player_trajectories, target_player_only=True)
        clusters = {}
        cluster_ids = sorted({
            int(profile['cluster_id'])
            for trajectory in trajectories
            for name, profile in (trajectory.get('player_profiles') or {}).items()
            if name == player and profile and profile.get('cluster_id') is not None
        })
        for cluster_id in cluster_ids:
            cluster_trajectories = []
            profile_metadata = None
            for trajectory in trajectories:
                profile = (trajectory.get('player_profiles') or {}).get(player)
                if not profile or int(profile.get('cluster_id', -1)) != cluster_id:
                    continue
                cluster_trajectories.append(as_player_view(trajectory, player))
                profile_metadata = profile
            clusters[str(cluster_id)] = build_scope(
                cluster_trajectories,
                target_player_only=True,
                label=(profile_metadata or {}).get('label'),
                instruction=(profile_metadata or {}).get('instruction'),
            )
        player_result['clusters'] = clusters
        result['players'][player] = player_result
    return result


def _mean_metric(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {'mean': None, 'std': None, 'valid_values': 0}
    return {
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
        'valid_values': len(finite),
    }


def aggregate_scope(scopes):
    first = scopes[0]
    scores = {}
    for metric in METRIC_NAMES:
        values = [scope['scores'][metric] for scope in scopes]
        scores[metric] = _mean_metric(values)
    result = {
        'support_rallies': first['support_rallies'],
        'num_gt_shots': first['num_gt_shots'],
        'num_pred_shots': {
            'mean': float(np.mean([scope['num_pred_shots'] for scope in scopes])),
            'std': float(np.std([scope['num_pred_shots'] for scope in scopes])),
        },
        'scores': scores,
    }
    for field in ('label', 'instruction'):
        if field in first:
            result[field] = first[field]
    return result


def aggregate_hierarchies(hierarchies):
    first = hierarchies[0]
    result = {
        'overall': aggregate_scope([hierarchy['overall'] for hierarchy in hierarchies]),
        'players': {},
    }
    for player in first['players']:
        player_scopes = [hierarchy['players'][player] for hierarchy in hierarchies]
        player_result = aggregate_scope(player_scopes)
        player_result['clusters'] = {
            cluster_id: aggregate_scope([
                hierarchy['players'][player]['clusters'][cluster_id]
                for hierarchy in hierarchies
            ])
            for cluster_id in first['players'][player]['clusters']
        }
        result['players'][player] = player_result
    return result


def _format_number(value):
    return 'N/A' if value is None else f'{float(value):.4f}'


def _score_cell(score, aggregation):
    if aggregation == 'average_of_n':
        mean = _format_number(score.get('mean'))
        std = _format_number(score.get('std'))
        return f'{mean} ({std})'
    return _format_number(score)


def _table(headers, rows, right_aligned=None):
    right_aligned = set(right_aligned or [])
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    border = '+-' + '-+-'.join('-' * width for width in widths) + '-+'

    def format_row(row):
        cells = []
        for index, value in enumerate(row):
            cells.append(value.rjust(widths[index]) if index in right_aligned else value.ljust(widths[index]))
        return '| ' + ' | '.join(cells) + ' |'

    lines = [border, format_row(headers), border]
    lines.extend(format_row(row) for row in rendered)
    lines.append(border)
    return '\n'.join(lines)


def render_table(result, title='Trajectory Scores'):
    aggregation = result['aggregation']
    runs = result['runs']
    lines = [
        '=' * 100,
        title,
        '=' * 100,
        f"Aggregation: {aggregation}",
        f"Algorithm: {result['algorithm']}",
        f"Runs: {len(runs)}",
        *[f"  {index + 1}. {run}" for index, run in enumerate(runs)],
        '',
        'Overall Scores',
    ]
    overall = result['scores']['overall']
    overall_rows = []
    for metric in METRIC_NAMES:
        score = overall['scores'][metric]
        valid_values = score.get('valid_values', '') if aggregation == 'average_of_n' else ''
        overall_rows.append((metric, _score_cell(score, aggregation), valid_values))
    lines.append(_table(['Metric', 'Score' if aggregation != 'average_of_n' else 'Mean (Std)', 'Valid Values'], overall_rows, {1, 2}))

    player_rows = []
    for player, scope in result['scores']['players'].items():
        player_rows.append([
            player,
            scope['support_rallies'],
            *[_score_cell(scope['scores'][metric], aggregation) for metric in METRIC_NAMES],
        ])
    lines.extend(['', 'Per-Player Scores', _table(
        ['Player', 'Rallies', 'LandDTW', 'MoveDTW', 'ShotLCS', 'ShotLandJSD', 'ShotDistJSD', 'RallyLenJSD'],
        player_rows,
        set(range(1, 8)),
    )])

    cluster_rows = []
    for player, player_scope in result['scores']['players'].items():
        for cluster_id, scope in player_scope['clusters'].items():
            label = str(scope.get('label') or '')[:32]
            cluster_rows.append([
                player,
                cluster_id,
                label,
                scope['support_rallies'],
                *[_score_cell(scope['scores'][metric], aggregation) for metric in METRIC_NAMES],
            ])
    lines.extend(['', 'Per-Player Cluster Scores', _table(
        ['Player', 'CID', 'Label', 'Rallies', 'LandDTW', 'MoveDTW', 'ShotLCS', 'ShotLandJSD', 'ShotDistJSD', 'RallyLenJSD'],
        cluster_rows,
        set(range(1, 10)),
    )])

    lines.extend([
        '',
        'Lower is better: LandDTW, MoveDTW, ShotLandJSD, ShotDistJSD, RallyLenJSD',
        'Higher is better: ShotLCS',
        '=' * 100,
    ])
    return '\n'.join(lines) + '\n'


def _iter_scopes(scores):
    yield 'overall', '', '', scores['overall']
    for player, player_scope in scores['players'].items():
        yield 'player', player, '', player_scope
        for cluster_id, cluster_scope in player_scope['clusters'].items():
            yield 'cluster', player, cluster_id, cluster_scope


def write_scores_csv(path, results):
    fields = [
        'aggregation', 'scope', 'player', 'cluster_id', 'cluster_label', 'instruction',
        'support_rallies', 'metric', 'value', 'mean', 'std', 'valid_values',
    ]
    if isinstance(results, dict):
        results = [results]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for scope_name, player, cluster_id, scope in _iter_scopes(result['scores']):
                for metric in METRIC_NAMES:
                    score = scope['scores'][metric]
                    row = {
                        'aggregation': result['aggregation'],
                        'scope': scope_name,
                        'player': player,
                        'cluster_id': cluster_id,
                        'cluster_label': scope.get('label', ''),
                        'instruction': scope.get('instruction', ''),
                        'support_rallies': scope['support_rallies'],
                        'metric': metric,
                    }
                    if result['aggregation'] == 'average_of_n':
                        row.update(score)
                    else:
                        row['value'] = score
                    writer.writerow(row)


def build_result(aggregation, runs, scores):
    def without_internal_fields(value):
        if isinstance(value, dict):
            return {
                key: without_internal_fields(item)
                for key, item in value.items()
                if not key.startswith('_')
            }
        if isinstance(value, list):
            return [without_internal_fields(item) for item in value]
        return value

    return {
        'schema_version': 2,
        'aggregation': aggregation,
        'algorithm': runs[0]['manifest'].get('algorithm'),
        'experiment': runs[0]['manifest'].get('experiment'),
        'instruction_mode': runs[0]['manifest'].get('instruction_mode'),
        'runs': [run['path'] for run in runs],
        'scores': without_internal_fields(scores),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Score physical trajectory runs with six canonical metrics.')
    parser.add_argument(
        '--run-root',
        required=True,
        help='Directory containing run_NNN trajectory directories.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Score all completed runs with rally-player sample mean and standard deviation.',
    )
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    discovered_paths = discover_completed_runs(args.run_root)
    selected_paths = discovered_paths if args.all else discovered_paths[-1:]
    runs = [resolve_run(path) for path in selected_paths]
    validate_runs(runs)

    if args.all:
        average_scores = aggregate_hierarchies([
            compute_hierarchy(run['trajectories']) for run in runs
        ])
        average_result = build_result('average_of_n', runs, average_scores)
        result = {
            'schema_version': 3,
            'mode': 'all',
            'algorithm': average_result['algorithm'],
            'experiment': average_result['experiment'],
            'instruction_mode': average_result['instruction_mode'],
            'runs': average_result['runs'],
            'results': {
                'average_of_n': average_result,
            },
        }
        csv_results = [average_result]
        table = render_table(average_result, title='Average-of-N Scores')
    else:
        result = build_result(
            'single',
            runs,
            compute_hierarchy(runs[0]['trajectories']),
        )
        result['mode'] = 'latest'
        csv_results = [result]
        table = render_table(result, title='Latest Run Scores')

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'scores.json')
    csv_path = os.path.join(args.output_dir, 'scores.csv')
    table_path = os.path.join(args.output_dir, 'table.txt')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    write_scores_csv(csv_path, csv_results)

    with open(table_path, 'w', encoding='utf-8') as handle:
        handle.write(table)
    print(table, end='')
    print(f'Scores JSON: {json_path}')
    print(f'Scores CSV: {csv_path}')
    print(f'Table: {table_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
