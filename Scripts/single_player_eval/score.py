#!/usr/bin/env python3
"""Canonical and optional train-reference scoring for physical rally runs."""

import argparse
import copy
import csv
import hashlib
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
    'land_dtw_maxlen',
    'move_dtw_maxlen',
    'shot_lcs_ratio',
    'shotland_jsd',
    'shotdist_jsd',
    'rallylen_jsd',
)
SEQUENCE_METRIC_NAMES = METRIC_NAMES[:5]
TRAIN_REFERENCE_METRIC_NAMES = (
    'shotdist_jsd',
    'shotland_jsd',
    'overall_landing_jsd',
    'rallylen_jsd',
)
COORDINATE_KEYS = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
N_AREA_GROUPS = 18
OUT_OF_BOUNDS_GROUP = 17
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_CSV = PROJECT_ROOT / 'Data' / 'intermediate_data' / 'merged.csv'
DEFAULT_RALLY_PROFILES = PROJECT_ROOT / 'Data' / 'intermediate_data' / 'rally_profiles.json'

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


def dtw_distance_normalizations(predicted, ground_truth):
    if dtw_ndim is None:
        return float('nan'), float('nan')
    distance = dtw_ndim.distance(predicted, ground_truth)
    warping_path = dtw_ndim.warping_path(predicted, ground_truth)
    if not warping_path:
        return float('inf'), float('inf')
    return (
        float(distance / np.sqrt(len(warping_path))),
        float(distance / max(len(predicted), len(ground_truth))),
    )


def normalized_dtw_distance(predicted, ground_truth):
    return dtw_distance_normalizations(predicted, ground_truth)[0]


def max_length_normalized_dtw_distance(predicted, ground_truth):
    return dtw_distance_normalizations(predicted, ground_truth)[1]


class SequenceMetrics:
    @staticmethod
    def compute_dtw(land_pred_seqs, land_gt_seqs, move_pred_seqs, move_gt_seqs):
        if dtw_ndim is None:
            return {
                'land_dtw': float('nan'),
                'move_dtw': float('nan'),
                'land_dtw_maxlen': float('nan'),
                'move_dtw_maxlen': float('nan'),
            }
        land_values = []
        land_maxlen_values = []
        move_values = []
        move_maxlen_values = []
        for predicted, ground_truth in zip(land_pred_seqs, land_gt_seqs):
            if len(predicted) >= 2 and len(ground_truth) >= 2:
                try:
                    path_value, maxlen_value = dtw_distance_normalizations(
                        predicted, ground_truth
                    )
                    land_values.append(path_value)
                    land_maxlen_values.append(maxlen_value)
                except Exception:
                    pass
        for predicted, ground_truth in zip(move_pred_seqs, move_gt_seqs):
            if len(predicted) >= 2 and len(ground_truth) >= 2:
                try:
                    path_value, maxlen_value = dtw_distance_normalizations(
                        predicted, ground_truth
                    )
                    move_values.append(path_value)
                    move_maxlen_values.append(maxlen_value)
                except Exception:
                    pass
        return {
            'land_dtw': float(np.mean(land_values)) if land_values else float('inf'),
            'move_dtw': float(np.mean(move_values)) if move_values else float('inf'),
            'land_dtw_maxlen': (
                float(np.mean(land_maxlen_values)) if land_maxlen_values else float('inf')
            ),
            'move_dtw_maxlen': (
                float(np.mean(move_maxlen_values)) if move_maxlen_values else float('inf')
            ),
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


def coordinate_raw_area(normalized_x, normalized_y):
    x = normalized_x * 177.5 + 177.5
    y = normalized_y * 240.0 + 240.0
    raw_area = 33
    for min_x, min_y, max_x, max_y, area_id in RAW_AREA_BOUNDS:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            raw_area = area_id
    return raw_area


def coordinate_area_group(normalized_x, normalized_y):
    return AREA_TO_GROUP.get(
        coordinate_raw_area(normalized_x, normalized_y),
        OUT_OF_BOUNDS_GROUP,
    )


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
            numeric = float(value)
            return int(numeric) if math.isfinite(numeric) and numeric.is_integer() else None
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _area_from_step(step):
        try:
            x = float(step['landing_x'])
            y = float(step['landing_y'])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return coordinate_area_group(x, y)

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
    def shot_counts(cls, actions):
        counts = defaultdict(int)
        for step in actions:
            shot_type = cls._safe_int(step.get('shot_type'))
            if shot_type in cls.VALID_SHOTS:
                counts[shot_type - 1] += 1
        return counts

    @classmethod
    def conditioned_landing_counts(cls, actions):
        counts = defaultdict(lambda: defaultdict(int))
        for step in actions:
            shot_type = cls._safe_int(step.get('shot_type'))
            area = cls._area_from_step(step)
            if shot_type in cls.VALID_SHOTS and area is not None:
                counts[shot_type][area] += 1
        return counts

    @classmethod
    def landing_counts(cls, actions):
        counts = defaultdict(int)
        for step in actions:
            shot_type = cls._safe_int(step.get('shot_type'))
            area = cls._area_from_step(step)
            if shot_type in cls.VALID_SHOTS and area is not None:
                counts[area] += 1
        return counts

    @staticmethod
    def length_counts(lengths):
        counts = defaultdict(int)
        for length in lengths:
            counts[int(length)] += 1
        return counts

    @staticmethod
    def conditioned_support(counts):
        return sum(sum(values.values()) for values in counts.values())

    @classmethod
    def _count_js(cls, predicted_counts, ground_truth_counts, bins):
        if not predicted_counts and not ground_truth_counts:
            return float('nan')
        predicted = np.array([predicted_counts.get(index, 0) for index in range(bins)])
        ground_truth = np.array([ground_truth_counts.get(index, 0) for index in range(bins)])
        return js_divergence(predicted, ground_truth)

    @classmethod
    def shot_distribution_js(cls, trajectories, target_player_only):
        predicted_counts = cls.shot_counts(
            cls._sequence_actions(trajectories, 'pred_sequence', target_player_only)
        )
        ground_truth_counts = cls.shot_counts(
            cls._sequence_actions(trajectories, 'gt_sequence', target_player_only)
        )
        return cls._count_js(predicted_counts, ground_truth_counts, len(cls.VALID_SHOTS))

    @classmethod
    def shot_conditioned_landing_js(cls, trajectories, target_player_only):
        predicted = cls.conditioned_landing_counts(
            cls._sequence_actions(trajectories, 'pred_sequence', target_player_only)
        )
        ground_truth = cls.conditioned_landing_counts(
            cls._sequence_actions(trajectories, 'gt_sequence', target_player_only)
        )
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

    @classmethod
    def compare_actions_to_reference(cls, actions, lengths, reference):
        shot_counts = cls.shot_counts(actions)
        landing_counts = cls.conditioned_landing_counts(actions)
        overall_landing_counts = cls.landing_counts(actions)
        length_counts = cls.length_counts(lengths)
        reference_shots = reference['shot_counts']
        reference_landings = reference['landing_counts']
        reference_overall_landings = reference['overall_landing_counts']
        reference_lengths = reference['length_counts']
        maximum = max([*length_counts, *reference_lengths], default=0)
        return {
            'scores': {
                'shotdist_jsd': (
                    finite_or_none(cls._count_js(
                        shot_counts, reference_shots, len(cls.VALID_SHOTS)
                    )) if shot_counts else None
                ),
                'shotland_jsd': (
                    finite_or_none(cls._weighted_conditioned_js(
                        landing_counts, reference_landings, N_AREA_GROUPS
                    )) if cls.conditioned_support(landing_counts) else None
                ),
                'overall_landing_jsd': (
                    finite_or_none(cls._count_js(
                        overall_landing_counts,
                        reference_overall_landings,
                        N_AREA_GROUPS,
                    )) if overall_landing_counts else None
                ),
                'rallylen_jsd': (
                    finite_or_none(cls._count_js(
                        length_counts, reference_lengths, maximum + 1
                    )) if length_counts else None
                ),
            },
            'valid_shots': int(sum(shot_counts.values())),
            'valid_landings': int(cls.conditioned_support(landing_counts)),
            'rallies': int(sum(length_counts.values())),
        }


def finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _row_key(row):
    try:
        return tuple(int(float(row[field])) for field in ('match_id', 'set', 'rally'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid physical rally key in merged CSV row: {row}') from exc


def load_train_reference(merged_csv_path, profiles_path, test_keys=None):
    merged_csv_path = Path(merged_csv_path).resolve()
    profiles_path = Path(profiles_path).resolve()
    with profiles_path.open('r', encoding='utf-8') as handle:
        profiles = json.load(handle)

    clusters = {}
    train_assignment_lookup = {}
    test_assignment_lookup = {}
    for player, profile in (profiles.get('players') or {}).items():
        player = str(player)
        cluster_metadata = {
            int(cluster['cluster_id']): cluster
            for cluster in profile.get('clusters', [])
        }
        for cluster_id, metadata in cluster_metadata.items():
            clusters[(player, cluster_id)] = {
                'player': player,
                'cluster_id': cluster_id,
                'label': metadata.get('label'),
                'instruction': metadata.get('instruction'),
                'expected_train_size': metadata.get('train_size'),
                'train_keys': [],
                'actions': [],
                'lengths': [],
                'wins': [],
            }
        for assignment in profile.get('train_assignments', []):
            key = tuple(int(assignment[field]) for field in ('match_id', 'set', 'rally'))
            cluster_id = int(assignment['cluster_id'])
            cluster_key = (player, cluster_id)
            if cluster_key not in clusters:
                raise ValueError(f'Unknown training cluster {cluster_key}')
            lookup_key = (player, key)
            if lookup_key in train_assignment_lookup:
                raise ValueError(f'Duplicate training cluster assignment: {lookup_key}')
            train_assignment_lookup[lookup_key] = cluster_id
            clusters[cluster_key]['train_keys'].append(key)
        for assignment in profile.get('test_assignments', []):
            key = tuple(int(assignment[field]) for field in ('match_id', 'set', 'rally'))
            lookup_key = (player, key)
            if lookup_key in test_assignment_lookup:
                raise ValueError(f'Duplicate test cluster assignment: {lookup_key}')
            test_assignment_lookup[lookup_key] = int(assignment['cluster_id'])

    for cluster_key, cluster in clusters.items():
        expected = cluster['expected_train_size']
        if expected is not None and int(expected) != len(cluster['train_keys']):
            raise ValueError(
                f'Training assignment count mismatch for {cluster_key}: '
                f"expected {expected}, found {len(cluster['train_keys'])}"
            )

    needed_keys = {key for _, key in train_assignment_lookup}
    needed_keys.update(test_keys or [])
    rally_lengths = defaultdict(int)
    rally_winner_names = defaultdict(set)
    rally_players = defaultdict(set)
    rally_final_actions = {}
    found_player_keys = set()
    required_columns = {
        'match_id', 'set', 'rally', 'ball_round', 'player', 'player_type',
        'landing_x', 'landing_y', 'getpoint_player',
    }
    with merged_csv_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'Merged CSV is missing required fields: {missing}')
        for row in reader:
            key = _row_key(row)
            if key not in needed_keys:
                continue
            try:
                ball_round = int(float(row['ball_round']))
            except (TypeError, ValueError):
                continue
            rally_lengths[key] = max(rally_lengths[key], ball_round)
            player = str(row.get('player') or '').strip()
            if player:
                rally_players[key].add(player)
            winner = str(row.get('getpoint_player') or '').strip()
            if winner and winner not in {'player', 'opponent'}:
                rally_winner_names[key].add(winner)

            final_action = {
                'player': player,
                'shot_type': DistributionMetrics._safe_int(row.get('player_type')),
                'landing_x': None,
                'landing_y': None,
                'lose_reason': str(row.get('lose_reason') or '').strip() or None,
            }
            try:
                final_action['landing_x'] = float(row['landing_x'])
                final_action['landing_y'] = float(row['landing_y'])
            except (TypeError, ValueError):
                pass
            previous = rally_final_actions.get(key)
            if previous is None or ball_round >= previous[0]:
                rally_final_actions[key] = (ball_round, final_action)

            lookup_key = (player, key)
            cluster_id = train_assignment_lookup.get(lookup_key)
            if cluster_id is None:
                continue
            found_player_keys.add(lookup_key)
            shot_type = DistributionMetrics._safe_int(row.get('player_type'))
            if shot_type is None:
                continue
            action = {'shot_type': shot_type}
            try:
                action['landing_x'] = float(row['landing_x'])
                action['landing_y'] = float(row['landing_y'])
            except (TypeError, ValueError):
                action['landing_x'] = None
                action['landing_y'] = None
            clusters[(player, cluster_id)]['actions'].append(action)

    missing_assignments = sorted(set(train_assignment_lookup) - found_player_keys)
    if missing_assignments:
        raise ValueError(f'Training assignments missing player rows: {missing_assignments[:5]}')

    winners = {}
    for key in needed_keys:
        candidates = rally_winner_names.get(key, set())
        if len(candidates) > 1:
            raise ValueError(f'Conflicting rally winners for {key}: {sorted(candidates)}')
        explicit = next(iter(candidates), None)
        if explicit is not None and explicit not in rally_players.get(key, set()):
            raise ValueError(f'Rally winner {explicit!r} is not a participant in {key}')
        if explicit is not None:
            winners[key] = explicit
            continue
        final = rally_final_actions.get(key)
        winners[key] = (
            terminal_action_winner(sorted(rally_players.get(key, set())), final[1])
            if final is not None else None
        )

    for (player, key), cluster_id in train_assignment_lookup.items():
        if key not in rally_lengths:
            raise ValueError(f'Training rally not found in merged CSV: {key}')
        cluster = clusters[(player, cluster_id)]
        cluster['lengths'].append(rally_lengths[key])
        winner = winners.get(key)
        if winner is not None:
            cluster['wins'].append(winner == player)

    for cluster in clusters.values():
        actions = cluster.pop('actions')
        cluster['shot_counts'] = DistributionMetrics.shot_counts(actions)
        cluster['landing_counts'] = DistributionMetrics.conditioned_landing_counts(actions)
        cluster['overall_landing_counts'] = DistributionMetrics.landing_counts(actions)
        cluster['length_counts'] = DistributionMetrics.length_counts(cluster['lengths'])

    return {
        'merged_csv_path': str(merged_csv_path),
        'profiles_path': str(profiles_path),
        'merged_csv_sha256': _file_sha256(merged_csv_path),
        'profiles_sha256': _file_sha256(profiles_path),
        'split_sha256': profiles.get('split_sha256'),
        'clusters': clusters,
        'test_assignments': test_assignment_lookup,
        'rally_winners': winners,
    }


def validate_train_reference_run(run, reference, allow_mismatch=False):
    manifest = run['manifest']
    variant = manifest.get('instruction_variant', 'original')
    if variant != 'original':
        raise ValueError('Train-reference scoring currently supports original instructions only')
    mismatches = []
    for field in ('merged_csv_sha256', 'profiles_sha256', 'split_sha256'):
        expected = manifest.get(field)
        actual = reference.get(field)
        if expected != actual:
            mismatches.append({'field': field, 'run': expected, 'reference': actual})
    if mismatches and not allow_mismatch:
        fields = ', '.join(item['field'] for item in mismatches)
        raise ValueError(f'Train-reference provenance mismatch: {fields}')

    for trajectory in run['trajectories']:
        key = physical_key(trajectory)
        for player in trajectory_players(trajectory):
            canonical = reference['test_assignments'].get((player, key))
            profile = (trajectory.get('player_profiles') or {}).get(player)
            if canonical is None:
                if profile and profile.get('cluster_id') is not None:
                    raise ValueError(
                        f'Unexpected embedded cluster assignment for {player} rally {key}'
                    )
                continue
            if not profile or profile.get('cluster_id') is None:
                raise ValueError(f'Missing embedded cluster assignment for {player} rally {key}')
            embedded = int(profile['cluster_id'])
            if embedded != canonical:
                raise ValueError(
                    f'Test cluster assignment mismatch for {player} rally {key}: '
                    f'{embedded} != {canonical}'
                )
    return mismatches


def terminal_action_winner(players, final):
    players = list(dict.fromkeys(str(player) for player in players if player))
    if len(players) != 2 or not isinstance(final, dict):
        return None
    loser = str(final.get('player') or '')
    if loser not in players:
        return None
    shot_type = DistributionMetrics._safe_int(final.get('shot_type'))
    terminal = shot_type == 11 or bool(final.get('lose_reason'))
    if not terminal:
        try:
            # Evaluators pass actions through float32 tensors before termination.
            x = float(np.float32(final['landing_x']))
            y = float(np.float32(final['landing_y']))
            terminal = (
                math.isfinite(x)
                and math.isfinite(y)
                and (coordinate_raw_area(x, y) > 24 or (x == 0 and y == 1))
            )
        except (KeyError, TypeError, ValueError):
            terminal = False
    return next((player for player in players if player != loser), None) if terminal else None


def predicted_winner(trajectory):
    reason = trajectory.get('termination_reason')
    if reason != 'natural_terminate':
        return None, str(reason or 'unknown')
    sequence = trajectory.get('pred_sequence') or []
    if not sequence:
        return None, 'empty_prediction'
    final = sequence[-1]
    winner = terminal_action_winner(trajectory_players(trajectory), final)
    if winner is None:
        return None, 'invalid_natural_termination'
    return winner, 'resolved'


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
        players = trajectory_players(trajectory)
        if len(players) != 2:
            raise ValueError(f'Expected exactly two players for rally {key}')
        player_set = set(players)
        for sequence_name in ('gt_sequence', 'pred_sequence'):
            invalid_actors = {
                str(step.get('player') or '')
                for step in trajectory.get(sequence_name, [])
                if str(step.get('player') or '') not in player_set
            }
            if invalid_actors:
                raise ValueError(
                    f'{sequence_name} contains unknown players for rally {key}: '
                    f'{sorted(invalid_actors)}'
                )
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
    metadata_fields = (
        'algorithm', 'experiment', 'instruction_mode', 'instruction_variant',
        'config_fingerprint', 'split_sha256', 'profiles_sha256', 'merged_csv_sha256',
    )
    for trajectory in trajectories:
        key = physical_key(trajectory)
        for field in metadata_fields:
            default = 'original' if field == 'instruction_variant' else None
            if trajectory.get(field, default) != manifest.get(field, default):
                raise ValueError(
                    f'Trajectory {key} disagrees with manifest field {field!r}'
                )
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
        samples['land_dtw_maxlen'].append(finite_or_none(dtw.get('land_dtw_maxlen')))
        samples['move_dtw_maxlen'].append(finite_or_none(dtw.get('move_dtw_maxlen')))
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


def _mean_value(values):
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _player_actions(trajectories, player, sequence_name):
    return [
        step
        for trajectory in trajectories
        for step in trajectory.get(sequence_name, [])
        if str(step.get('player') or '') == player
    ]


def _sequence_lengths(trajectories, sequence_name):
    fallback_name = 'rally_length_pred' if sequence_name == 'pred_sequence' else 'rally_length_gt'
    return [
        DistributionMetrics._rally_round_length(
            trajectory.get(sequence_name, []), trajectory.get(fallback_name)
        )
        for trajectory in trajectories
    ]


def _comparison_hierarchy(cluster_results, comparison):
    players = {}
    for player in sorted({key[0] for key in cluster_results}):
        player_clusters = {
            str(cluster_id): {
                'label': result.get('label'),
                'instruction': result.get('instruction'),
                'support': result['support'],
                'eligible_for_macro': result['eligible_for_macro'],
                'metric_eligible_for_macro': {
                    metric: (
                        result['eligible_for_macro']
                        and result[comparison]['scores'][metric] is not None
                    )
                    for metric in TRAIN_REFERENCE_METRIC_NAMES
                },
                'scores': result[comparison]['scores'],
                'valid_shots': result[comparison]['valid_shots'],
                'valid_landings': result[comparison]['valid_landings'],
            }
            for (name, cluster_id), result in sorted(cluster_results.items())
            if name == player
        }
        scores = {
            metric: _mean_value([
                cluster['scores'][metric]
                for cluster in player_clusters.values()
                if cluster['metric_eligible_for_macro'][metric]
            ])
            for metric in TRAIN_REFERENCE_METRIC_NAMES
        }
        players[player] = {
            'scores': scores,
            'num_clusters_total': len(player_clusters),
            'num_clusters_eligible': sum(
                cluster['eligible_for_macro'] for cluster in player_clusters.values()
            ),
            'num_clusters_scored': {
                metric: sum(
                    cluster['metric_eligible_for_macro'][metric]
                    for cluster in player_clusters.values()
                )
                for metric in TRAIN_REFERENCE_METRIC_NAMES
            },
            'clusters': player_clusters,
        }
    overall_scores = {
        metric: _mean_value([
            player['scores'][metric]
            for player in players.values()
            if player['scores'][metric] is not None
        ])
        for metric in TRAIN_REFERENCE_METRIC_NAMES
    }
    return {
        'overall': {
            'scores': overall_scores,
            'num_players_total': len(players),
            'num_players_scored': {
                metric: sum(
                    player['scores'][metric] is not None for player in players.values()
                )
                for metric in TRAIN_REFERENCE_METRIC_NAMES
            },
            'num_clusters_total': sum(
                player['num_clusters_total'] for player in players.values()
            ),
            'num_clusters_eligible': sum(
                player['num_clusters_eligible'] for player in players.values()
            ),
        },
        'players': players,
    }


def _win_rate_hierarchy(cluster_results, min_cluster_test_rallies):
    metric_names = (
        'train_win_rate',
        'generated_win_rate',
        'win_rate_difference',
        'win_rate_absolute_difference',
        'gt_test_win_rate',
        'gt_test_vs_train_win_rate_difference',
    )
    players = {}
    for player in sorted({key[0] for key in cluster_results}):
        clusters = {
            str(cluster_id): {
                'label': result.get('label'),
                'instruction': result.get('instruction'),
                'support': result['support'],
                **result['win_rate'],
            }
            for (name, cluster_id), result in sorted(cluster_results.items())
            if name == player
        }
        for cluster in clusters.values():
            cluster['eligible_for_macro'] = (
                cluster['support']['test_rallies'] >= min_cluster_test_rallies
                and cluster['support']['resolved_generated_rallies']
                >= min_cluster_test_rallies
            )
        eligible = [cluster for cluster in clusters.values() if cluster['eligible_for_macro']]
        players[player] = {
            metric: _mean_value([cluster[metric] for cluster in eligible])
            for metric in metric_names
        } | {
            'generated_win_rate_coverage': (
                sum(cluster['support']['resolved_generated_rallies'] for cluster in clusters.values())
                / sum(cluster['support']['test_rallies'] for cluster in clusters.values())
                if sum(cluster['support']['test_rallies'] for cluster in clusters.values()) else None
            ),
            'num_clusters_total': len(clusters),
            'num_clusters_scored': len(eligible),
            'clusters': clusters,
        }
    overall = {
        metric: _mean_value([
            player[metric] for player in players.values() if player[metric] is not None
        ])
        for metric in metric_names
    }
    overall.update({
        'generated_win_rate_coverage': _mean_value([
            player['generated_win_rate_coverage'] for player in players.values()
        ]),
        'num_players_total': len(players),
        'num_players_scored': sum(
            player['win_rate_absolute_difference'] is not None for player in players.values()
        ),
        'num_clusters_total': sum(player['num_clusters_total'] for player in players.values()),
        'num_clusters_scored': sum(player['num_clusters_scored'] for player in players.values()),
    })
    return {'overall': overall, 'players': players}


def compute_train_reference_hierarchy(trajectories, reference, min_cluster_test_rallies=5):
    grouped = defaultdict(list)
    unassigned_player_views = 0
    for trajectory in trajectories:
        key = physical_key(trajectory)
        for player in trajectory_players(trajectory):
            canonical = reference['test_assignments'].get((player, key))
            profile = (trajectory.get('player_profiles') or {}).get(player)
            if canonical is None:
                if profile and profile.get('cluster_id') is not None:
                    raise ValueError(
                        f'Unexpected embedded cluster assignment for {player} rally {key}'
                    )
                unassigned_player_views += 1
                continue
            if not profile or profile.get('cluster_id') is None:
                raise ValueError(f'Missing embedded cluster assignment for {player} rally {key}')
            cluster_id = int(profile['cluster_id'])
            cluster_key = (player, cluster_id)
            if cluster_key not in reference['clusters']:
                raise ValueError(f'Missing training reference for cluster {cluster_key}')
            if canonical != cluster_id:
                raise ValueError(f'Test cluster assignment mismatch for {player} rally {key}')
            grouped[cluster_key].append(trajectory)

    predicted_outcomes = {
        physical_key(trajectory): predicted_winner(trajectory)
        for trajectory in trajectories
    }
    cluster_results = {}
    for cluster_key, cluster_trajectories in sorted(grouped.items()):
        player, cluster_id = cluster_key
        train = reference['clusters'][cluster_key]
        generated = DistributionMetrics.compare_actions_to_reference(
            _player_actions(cluster_trajectories, player, 'pred_sequence'),
            _sequence_lengths(cluster_trajectories, 'pred_sequence'),
            train,
        )
        gt_test = DistributionMetrics.compare_actions_to_reference(
            _player_actions(cluster_trajectories, player, 'gt_sequence'),
            _sequence_lengths(cluster_trajectories, 'gt_sequence'),
            train,
        )

        generated_wins = []
        gt_test_wins = []
        unresolved_reasons = defaultdict(int)
        for trajectory in cluster_trajectories:
            key = physical_key(trajectory)
            winner, status = predicted_outcomes[key]
            if winner is None:
                unresolved_reasons[status] += 1
            else:
                generated_wins.append(winner == player)
            gt_winner = reference['rally_winners'].get(key)
            if gt_winner is not None:
                gt_test_wins.append(gt_winner == player)

        train_win_rate = _mean_value(train['wins'])
        generated_win_rate = _mean_value(generated_wins)
        gt_test_win_rate = _mean_value(gt_test_wins)
        win_difference = (
            generated_win_rate - train_win_rate
            if generated_win_rate is not None and train_win_rate is not None else None
        )
        gt_difference = (
            gt_test_win_rate - train_win_rate
            if gt_test_win_rate is not None and train_win_rate is not None else None
        )
        test_rallies = len(cluster_trajectories)
        support = {
            'train_rallies': len(train['train_keys']),
            'test_rallies': test_rallies,
            'resolved_train_rallies': len(train['wins']),
            'unresolved_train_rallies': len(train['train_keys']) - len(train['wins']),
            'resolved_gt_test_rallies': len(gt_test_wins),
            'unresolved_gt_test_rallies': test_rallies - len(gt_test_wins),
            'train_valid_shots': int(sum(train['shot_counts'].values())),
            'train_valid_landings': int(
                DistributionMetrics.conditioned_support(train['landing_counts'])
            ),
            'generated_valid_shots': generated['valid_shots'],
            'generated_valid_landings': generated['valid_landings'],
            'gt_test_valid_shots': gt_test['valid_shots'],
            'gt_test_valid_landings': gt_test['valid_landings'],
            'resolved_generated_rallies': len(generated_wins),
            'unresolved_generated_rallies': test_rallies - len(generated_wins),
            'action_none_count': unresolved_reasons.get('action_none', 0),
            'max_rollout_count': unresolved_reasons.get('max_rollout_reached', 0),
            'unresolved_reasons': dict(sorted(unresolved_reasons.items())),
        }
        cluster_results[cluster_key] = {
            'cluster_id': cluster_id,
            'label': train.get('label'),
            'instruction': train.get('instruction'),
            'support': support,
            'eligible_for_macro': test_rallies >= min_cluster_test_rallies,
            'generated_test_vs_train': generated,
            'gt_test_vs_train_diagnostic': gt_test,
            'win_rate': {
                'train_win_rate': train_win_rate,
                'generated_win_rate': generated_win_rate,
                'win_rate_difference': win_difference,
                'win_rate_absolute_difference': (
                    abs(win_difference) if win_difference is not None else None
                ),
                'gt_test_win_rate': gt_test_win_rate,
                'gt_test_vs_train_win_rate_difference': gt_difference,
                'generated_win_rate_coverage': (
                    len(generated_wins) / test_rallies if test_rallies else None
                ),
            },
        }

    return {
        'schema_version': 1,
        'provenance': {
            key: reference[key]
            for key in (
                'merged_csv_path', 'profiles_path', 'merged_csv_sha256',
                'profiles_sha256', 'split_sha256',
            )
        },
        'aggregation': {
            'min_cluster_test_rallies': int(min_cluster_test_rallies),
            'cluster_to_player': 'equal',
            'player_to_overall': 'equal',
            'empty_generated_distribution': 'null_excluded',
            'win_rate_min_resolved_rallies': int(min_cluster_test_rallies),
        },
        'generated_test_vs_train': _comparison_hierarchy(
            cluster_results, 'generated_test_vs_train'
        ),
        'gt_test_vs_train_diagnostic': _comparison_hierarchy(
            cluster_results, 'gt_test_vs_train_diagnostic'
        ),
        'win_rate': _win_rate_hierarchy(
            cluster_results, min_cluster_test_rallies
        ),
        'coverage': {
            'assigned_player_cluster_views': int(sum(len(values) for values in grouped.values())),
            'unassigned_player_views': int(unassigned_player_views),
            'player_clusters_observed': len(cluster_results),
            'player_clusters_eligible': sum(
                result['eligible_for_macro'] for result in cluster_results.values()
            ),
        },
    }


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


def aggregate_train_reference_hierarchies(hierarchies):
    first = hierarchies[0]

    def scope_signature(hierarchy, name):
        return {
            player: set(scope['clusters'])
            for player, scope in hierarchy[name]['players'].items()
        }

    for hierarchy in hierarchies[1:]:
        for name in ('generated_test_vs_train', 'gt_test_vs_train_diagnostic'):
            if scope_signature(hierarchy, name) != scope_signature(first, name):
                raise ValueError('Train-reference runs do not contain identical player-cluster scopes')

    def aggregate_comparison(name):
        first_comparison = first[name]
        result = copy.deepcopy(first_comparison)
        result['overall']['scores'] = {
            metric: _mean_metric([
                hierarchy[name]['overall']['scores'][metric]
                for hierarchy in hierarchies
            ])
            for metric in TRAIN_REFERENCE_METRIC_NAMES
        }
        result['overall']['num_players_scored'] = {
            metric: _mean_metric([
                hierarchy[name]['overall']['num_players_scored'][metric]
                for hierarchy in hierarchies
            ])
            for metric in TRAIN_REFERENCE_METRIC_NAMES
        }
        for player, player_scope in result['players'].items():
            player_scope['scores'] = {
                metric: _mean_metric([
                    hierarchy[name]['players'][player]['scores'][metric]
                    for hierarchy in hierarchies
                ])
                for metric in TRAIN_REFERENCE_METRIC_NAMES
            }
            player_scope['num_clusters_scored'] = {
                metric: _mean_metric([
                    hierarchy[name]['players'][player]['num_clusters_scored'][metric]
                    for hierarchy in hierarchies
                ])
                for metric in TRAIN_REFERENCE_METRIC_NAMES
            }
            for cluster_id, cluster_scope in player_scope['clusters'].items():
                cluster_scope['scores'] = {
                    metric: _mean_metric([
                        hierarchy[name]['players'][player]['clusters'][cluster_id]['scores'][metric]
                        for hierarchy in hierarchies
                    ])
                    for metric in TRAIN_REFERENCE_METRIC_NAMES
                }
                cluster_scope['valid_shots'] = _mean_metric([
                    hierarchy[name]['players'][player]['clusters'][cluster_id]['valid_shots']
                    for hierarchy in hierarchies
                ])
                cluster_scope['valid_landings'] = _mean_metric([
                    hierarchy[name]['players'][player]['clusters'][cluster_id]['valid_landings']
                    for hierarchy in hierarchies
                ])
                cluster_scope['metric_eligible_for_macro'] = {
                    metric: _mean_metric([
                        hierarchy[name]['players'][player]['clusters'][cluster_id][
                            'metric_eligible_for_macro'
                        ][metric]
                        for hierarchy in hierarchies
                    ])
                    for metric in TRAIN_REFERENCE_METRIC_NAMES
                }
                for field in (
                    'generated_valid_shots', 'generated_valid_landings',
                    'resolved_generated_rallies', 'unresolved_generated_rallies',
                    'action_none_count', 'max_rollout_count',
                ):
                    cluster_scope['support'][field] = _mean_metric([
                        hierarchy[name]['players'][player]['clusters'][cluster_id][
                            'support'
                        ][field]
                        for hierarchy in hierarchies
                    ])
                reasons = set().union(*(
                    hierarchy[name]['players'][player]['clusters'][cluster_id][
                        'support'
                    ]['unresolved_reasons']
                    for hierarchy in hierarchies
                ))
                cluster_scope['support']['unresolved_reasons'] = {
                    reason: _mean_metric([
                        hierarchy[name]['players'][player]['clusters'][cluster_id][
                            'support'
                        ]['unresolved_reasons'].get(reason, 0)
                        for hierarchy in hierarchies
                    ])
                    for reason in sorted(reasons)
                }
        return result

    win_fields = (
        'train_win_rate', 'generated_win_rate', 'win_rate_difference',
        'win_rate_absolute_difference', 'gt_test_win_rate',
        'gt_test_vs_train_win_rate_difference', 'generated_win_rate_coverage',
    )
    win_rate = copy.deepcopy(first['win_rate'])
    for field in win_fields:
        win_rate['overall'][field] = _mean_metric([
            hierarchy['win_rate']['overall'][field] for hierarchy in hierarchies
        ])
    for field in ('num_players_scored', 'num_clusters_scored'):
        win_rate['overall'][field] = _mean_metric([
            hierarchy['win_rate']['overall'][field] for hierarchy in hierarchies
        ])
    for player, player_scope in win_rate['players'].items():
        for field in win_fields:
            player_scope[field] = _mean_metric([
                hierarchy['win_rate']['players'][player][field]
                for hierarchy in hierarchies
            ])
        player_scope['num_clusters_scored'] = _mean_metric([
            hierarchy['win_rate']['players'][player]['num_clusters_scored']
            for hierarchy in hierarchies
        ])
        for cluster_id, cluster_scope in player_scope['clusters'].items():
            for field in win_fields:
                cluster_scope[field] = _mean_metric([
                    hierarchy['win_rate']['players'][player]['clusters'][cluster_id][field]
                    for hierarchy in hierarchies
                ])
            cluster_scope['eligible_for_macro'] = _mean_metric([
                hierarchy['win_rate']['players'][player]['clusters'][cluster_id][
                    'eligible_for_macro'
                ]
                for hierarchy in hierarchies
            ])
            reasons = set().union(*(
                hierarchy['win_rate']['players'][player]['clusters'][cluster_id][
                    'support'
                ]['unresolved_reasons']
                for hierarchy in hierarchies
            ))
            cluster_scope['support']['unresolved_reasons'] = {
                reason: _mean_metric([
                    hierarchy['win_rate']['players'][player]['clusters'][cluster_id][
                        'support'
                    ]['unresolved_reasons'].get(reason, 0)
                    for hierarchy in hierarchies
                ])
                for reason in sorted(reasons)
            }
            for field in (
                'generated_valid_shots', 'generated_valid_landings',
                'resolved_generated_rallies', 'unresolved_generated_rallies',
                'action_none_count', 'max_rollout_count',
            ):
                cluster_scope['support'][field] = _mean_metric([
                    hierarchy['win_rate']['players'][player]['clusters'][cluster_id][
                        'support'
                    ][field]
                    for hierarchy in hierarchies
                ])

    return {
        'schema_version': first['schema_version'],
        'provenance': {
            **{
                key: value for key, value in first['provenance'].items()
                if key not in {'mismatches', 'run'}
            },
            'mismatches_by_run': [
                {
                    'run': hierarchy['provenance'].get('run'),
                    'mismatches': hierarchy['provenance'].get('mismatches', []),
                }
                for hierarchy in hierarchies
            ],
        },
        'aggregation': {**first['aggregation'], 'runs': 'mean_and_std_of_run_level_values'},
        'generated_test_vs_train': aggregate_comparison('generated_test_vs_train'),
        'gt_test_vs_train_diagnostic': aggregate_comparison('gt_test_vs_train_diagnostic'),
        'win_rate': win_rate,
        'coverage': first['coverage'],
    }


def _format_number(value):
    return 'N/A' if value is None else f'{float(value):.4f}'


def _score_cell(score, aggregation):
    if aggregation == 'average_of_n':
        mean = _format_number(score.get('mean'))
        std = _format_number(score.get('std'))
        return f'{mean} ({std})'
    return _format_number(score)


def _eligibility_cell(value):
    if isinstance(value, dict):
        valid = int(value.get('valid_values', 0))
        mean = value.get('mean')
        eligible = round(float(mean) * valid) if mean is not None else 0
        return f'{eligible}/{valid}'
    return 'yes' if value else 'no'


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
        [
            'Player', 'Rallies', 'LandDTW', 'MoveDTW', 'LandDTWMaxLen',
            'MoveDTWMaxLen', 'ShotLCS', 'ShotLandJSD', 'ShotDistJSD',
            'RallyLenJSD',
        ],
        player_rows,
        set(range(1, 10)),
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
        [
            'Player', 'CID', 'Label', 'Rallies', 'LandDTW', 'MoveDTW',
            'LandDTWMaxLen', 'MoveDTWMaxLen', 'ShotLCS', 'ShotLandJSD',
            'ShotDistJSD', 'RallyLenJSD',
        ],
        cluster_rows,
        set(range(1, 12)),
    )])

    lines.extend([
        '',
        'Lower is better: both DTW variants, ShotLandJSD, ShotDistJSD, RallyLenJSD',
        'Higher is better: ShotLCS',
        '=' * 100,
    ])
    return '\n'.join(lines) + '\n'


def render_train_reference_table(train_reference, aggregation):
    generated = train_reference['generated_test_vs_train']
    diagnostic = train_reference['gt_test_vs_train_diagnostic']
    win_rate = train_reference['win_rate']
    rows = []
    for metric in TRAIN_REFERENCE_METRIC_NAMES:
        rows.append([
            metric,
            _score_cell(generated['overall']['scores'][metric], aggregation),
            _score_cell(diagnostic['overall']['scores'][metric], aggregation),
        ])
    lines = [
        '',
        'Train-Reference Two-Level Macro',
        _table(['Metric', 'Generated vs Train', 'GT Test vs Train'], rows, {1, 2}),
        '',
        'Train-Reference Win Rate',
        _table(
            ['Metric', 'Value'],
            [
                [field, _score_cell(win_rate['overall'][field], aggregation)]
                for field in (
                    'train_win_rate', 'generated_win_rate', 'win_rate_difference',
                    'win_rate_absolute_difference', 'gt_test_win_rate',
                    'gt_test_vs_train_win_rate_difference',
                    'generated_win_rate_coverage',
                )
            ],
            {1},
        ),
        '',
        'Train-Reference Coverage',
        _table(
            ['Field', 'Value'],
            [[field, value] for field, value in train_reference['coverage'].items()],
            {1},
        ),
    ]
    cluster_rows = []
    for player, player_scope in generated['players'].items():
        diagnostic_player = diagnostic['players'][player]
        outcome_player = win_rate['players'][player]
        for cluster_id, scope in player_scope['clusters'].items():
            diagnostic_scope = diagnostic_player['clusters'][cluster_id]
            outcome_scope = outcome_player['clusters'][cluster_id]
            cluster_rows.append([
                player,
                cluster_id,
                scope['support']['test_rallies'],
                _eligibility_cell(outcome_scope['eligible_for_macro']),
                *[
                    _eligibility_cell(scope['metric_eligible_for_macro'][metric])
                    for metric in TRAIN_REFERENCE_METRIC_NAMES
                ],
                *[
                    _score_cell(scope['scores'][metric], aggregation)
                    for metric in TRAIN_REFERENCE_METRIC_NAMES
                ],
                _score_cell(outcome_scope['win_rate_difference'], aggregation),
                _score_cell(outcome_scope['win_rate_absolute_difference'], aggregation),
                _score_cell(diagnostic_scope['scores']['shotdist_jsd'], aggregation),
            ])
    lines.extend([
        '',
        'Per-Player-Cluster Train-Reference Values',
        _table(
            [
                'Player', 'CID', 'Test', 'WinMacro', 'ShotDistIn', 'ShotLandIn',
                'OverallLandIn', 'RallyLenIn', 'ShotDist', 'ShotLand',
                'OverallLand', 'RallyLen', 'WinDiff', 'AbsWinDiff',
                'GTTrainShotDist',
            ],
            cluster_rows,
            set(range(1, 15)),
        ),
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


def _iter_train_reference_scopes(hierarchy):
    yield 'overall', '', '', hierarchy['overall']
    for player, player_scope in hierarchy['players'].items():
        yield 'player', player, '', player_scope
        for cluster_id, cluster_scope in player_scope['clusters'].items():
            yield 'cluster', player, cluster_id, cluster_scope


def _csv_stat_value(value):
    return value.get('mean') if isinstance(value, dict) else value


def write_train_reference_scores_csv(path, train_reference, aggregation):
    fields = [
        'aggregation', 'comparison', 'scope', 'player', 'cluster_id', 'cluster_label',
        'metric', 'value', 'mean', 'std', 'valid_values', 'train_rallies',
        'test_rallies', 'valid_shots', 'valid_landings', 'eligible_for_macro',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in ('generated_test_vs_train', 'gt_test_vs_train_diagnostic'):
            hierarchy = train_reference[comparison]
            for scope_name, player, cluster_id, scope in _iter_train_reference_scopes(hierarchy):
                for metric in TRAIN_REFERENCE_METRIC_NAMES:
                    score = scope['scores'][metric]
                    support = scope.get('support', {})
                    row = {
                        'aggregation': aggregation,
                        'comparison': comparison,
                        'scope': scope_name,
                        'player': player,
                        'cluster_id': cluster_id,
                        'cluster_label': scope.get('label', ''),
                        'metric': metric,
                        'train_rallies': _csv_stat_value(support.get('train_rallies', '')),
                        'test_rallies': _csv_stat_value(support.get('test_rallies', '')),
                        'valid_shots': _csv_stat_value(scope.get('valid_shots', '')),
                        'valid_landings': _csv_stat_value(scope.get('valid_landings', '')),
                        'eligible_for_macro': _csv_stat_value(
                            scope.get('metric_eligible_for_macro', {}).get(metric, '')
                        ),
                    }
                    if aggregation == 'average_of_n':
                        row.update(score)
                    else:
                        row['value'] = score
                    writer.writerow(row)


def write_train_reference_outcomes_csv(path, train_reference, aggregation):
    fields = [
        'aggregation', 'scope', 'player', 'cluster_id', 'cluster_label',
        'metric', 'value', 'mean', 'std', 'valid_values',
        'train_rallies', 'test_rallies', 'resolved_generated_rallies',
        'unresolved_generated_rallies', 'action_none_count', 'max_rollout_count',
        'resolved_train_rallies', 'unresolved_train_rallies',
        'resolved_gt_test_rallies', 'unresolved_gt_test_rallies',
        'eligible_for_macro',
    ]
    win_rate = train_reference['win_rate']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        scopes = [('overall', '', '', win_rate['overall'])]
        for player, player_scope in win_rate['players'].items():
            scopes.append(('player', player, '', player_scope))
            scopes.extend(
                ('cluster', player, cluster_id, cluster_scope)
                for cluster_id, cluster_scope in player_scope['clusters'].items()
            )
        rate_fields = (
            'train_win_rate', 'generated_win_rate', 'win_rate_difference',
            'win_rate_absolute_difference', 'gt_test_win_rate',
            'gt_test_vs_train_win_rate_difference', 'generated_win_rate_coverage',
        )
        for scope_name, player, cluster_id, scope in scopes:
            support = scope.get('support', {})
            for metric in rate_fields:
                score = scope.get(metric)
                row = {
                    'aggregation': aggregation,
                    'scope': scope_name,
                    'player': player,
                    'cluster_id': cluster_id,
                    'cluster_label': scope.get('label', ''),
                    'metric': metric,
                    'eligible_for_macro': _csv_stat_value(
                        scope.get('eligible_for_macro', '')
                    ),
                    **{
                        field: _csv_stat_value(support.get(field, ''))
                        for field in (
                            'train_rallies', 'test_rallies',
                            'resolved_generated_rallies',
                            'unresolved_generated_rallies', 'action_none_count',
                            'max_rollout_count', 'resolved_train_rallies',
                            'unresolved_train_rallies', 'resolved_gt_test_rallies',
                            'unresolved_gt_test_rallies',
                        )
                    },
                }
                if aggregation == 'average_of_n':
                    row.update(score)
                else:
                    row['value'] = score
                writer.writerow(row)


def build_result(aggregation, runs, scores, train_reference=None):
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

    result = {
        'schema_version': 2,
        'aggregation': aggregation,
        'algorithm': runs[0]['manifest'].get('algorithm'),
        'experiment': runs[0]['manifest'].get('experiment'),
        'instruction_mode': runs[0]['manifest'].get('instruction_mode'),
        'runs': [run['path'] for run in runs],
        'scores': without_internal_fields(scores),
    }
    if train_reference is not None:
        result['train_reference'] = without_internal_fields(train_reference)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Score physical trajectory runs with canonical and alternate DTW metrics.'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--run-root',
        help='Directory containing run_NNN trajectory directories.',
    )
    source.add_argument(
        '--run',
        help='One completed run_NNN directory or its rallies directory.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Score all completed runs with rally-player sample mean and standard deviation.',
    )
    parser.add_argument(
        '--train-reference', action='store_true',
        help='Compare original-condition generated clusters with their GT training distributions.',
    )
    parser.add_argument(
        '--merged-csv', default=str(DEFAULT_MERGED_CSV),
        help='Canonical merged CSV used to build training-cluster references.',
    )
    parser.add_argument(
        '--rally-profiles', default=str(DEFAULT_RALLY_PROFILES),
        help='Train-fitted rally profile artifact containing cluster assignments.',
    )
    parser.add_argument(
        '--min-cluster-test-rallies', type=int, default=5,
        help='Minimum test support for a player-cluster to enter headline macros.',
    )
    parser.add_argument(
        '--allow-reference-mismatch', action='store_true',
        help='Allow hash mismatches and record them in output provenance.',
    )
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.run and args.all:
        parser.error('--all can only be used with --run-root')
    if args.min_cluster_test_rallies < 1:
        parser.error('--min-cluster-test-rallies must be at least 1')
    return args


def main():
    args = parse_args()
    if args.run:
        selected_paths = [args.run]
    else:
        discovered_paths = discover_completed_runs(args.run_root)
        selected_paths = discovered_paths if args.all else discovered_paths[-1:]
    runs = [resolve_run(path) for path in selected_paths]
    validate_runs(runs)

    train_references = None
    if args.train_reference:
        test_keys = set().union(*(run['keys'] for run in runs))
        reference = load_train_reference(
            args.merged_csv,
            args.rally_profiles,
            test_keys=test_keys,
        )
        mismatches = [
            validate_train_reference_run(run, reference, args.allow_reference_mismatch)
            for run in runs
        ]
        train_references = [
            compute_train_reference_hierarchy(
                run['trajectories'], reference, args.min_cluster_test_rallies
            )
            for run in runs
        ]
        for hierarchy, mismatch, run in zip(train_references, mismatches, runs):
            hierarchy['provenance']['run'] = run['path']
            hierarchy['provenance']['mismatches'] = mismatch

    if args.all:
        average_scores = aggregate_hierarchies([
            compute_hierarchy(run['trajectories']) for run in runs
        ])
        average_train_reference = (
            aggregate_train_reference_hierarchies(train_references)
            if train_references is not None else None
        )
        average_result = build_result(
            'average_of_n', runs, average_scores, average_train_reference
        )
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
        if average_train_reference is not None:
            table += render_train_reference_table(
                average_train_reference, 'average_of_n'
            )
    else:
        result = build_result(
            'single',
            runs,
            compute_hierarchy(runs[0]['trajectories']),
            train_references[0] if train_references is not None else None,
        )
        result['mode'] = 'selected' if args.run else 'latest'
        csv_results = [result]
        table = render_table(
            result,
            title='Selected Run Scores' if args.run else 'Latest Run Scores',
        )
        if train_references is not None:
            table += render_train_reference_table(train_references[0], 'single')

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'scores.json')
    csv_path = os.path.join(args.output_dir, 'scores.csv')
    table_path = os.path.join(args.output_dir, 'table.txt')
    train_scores_path = os.path.join(args.output_dir, 'train_reference_scores.csv')
    train_outcomes_path = os.path.join(args.output_dir, 'train_reference_outcomes.csv')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    write_scores_csv(csv_path, csv_results)
    if args.train_reference:
        train_reference = average_result['train_reference'] if args.all else result['train_reference']
        aggregation = 'average_of_n' if args.all else 'single'
        write_train_reference_scores_csv(
            train_scores_path, train_reference, aggregation
        )
        write_train_reference_outcomes_csv(
            train_outcomes_path, train_reference, aggregation
        )
    else:
        for stale_path in (train_scores_path, train_outcomes_path):
            if os.path.isfile(stale_path):
                os.remove(stale_path)

    with open(table_path, 'w', encoding='utf-8') as handle:
        handle.write(table)
    print(table, end='')
    print(f'Scores JSON: {json_path}')
    print(f'Scores CSV: {csv_path}')
    if args.train_reference:
        print(f'Train-reference scores CSV: {train_scores_path}')
        print(f'Train-reference outcomes CSV: {train_outcomes_path}')
    print(f'Table: {table_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
