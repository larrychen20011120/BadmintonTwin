#!/usr/bin/env python3
"""
Standalone Trajectory Evaluation Script

Evaluates saved trajectories without re-running agent simulation.
Computes comprehensive metrics including:
- Sequence-level: DTW, Edit Distance, LCS Ratio, shot n-gram similarity, shot transition JS
- Distribution: selected shot, rally-length, and state-conditioned JS metrics

Usage:
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/twin/trajectories/Kento_MOMOTA_norm.pkl \\
        --output-dir Experiments/twin/results/single_player/ \\
        --metrics all \\
        --visualize

    # Compute specific metrics only
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/bc/trajectories/Viktor_AXELSEN_norm.pkl \\
        --metrics dtw edit_distance shot_similarity \\
        --output-dir results/

Output Files (with --visualize):
    evaluation_results.json              - Complete metrics in JSON
    evaluation_summary.csv               - Summary table in CSV
    rally_length_distribution.png        - Pred vs GT rally length KDE, excluding missed shots
    landing_kde.png                      - Landing position KDE heatmap, excluding missed shots
    shot_distribution.png                - Normalized shot type distribution, excluding missed shots
    shot_conditioned_landing_kde.png     - Per-shot Pred/GT landing KDE court panels
    shot_transition_matrices.png         - P(next shot | previous shot), excluding missed shots
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

# Setup paths
_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _project_path)

from Utils.runtime_report import print_runtime_box

from Utils.helper import edit_distance, normalized_dtw_ndim_distance
from Utils.badminton_config import EN_SHOT2INDEX, INDEX2EN_SHOT, area_groups

# Try to import optional dependencies
try:
    from dtaidistance import dtw_ndim
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. DTW metrics will be skipped.")

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance, gaussian_kde
    from scipy.spatial.distance import cdist
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except Exception as e:
    SCIPY_AVAILABLE = False
    print(f"Warning: SciPy not available. Distribution metrics will be skipped. ({e})")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"Warning: Matplotlib not available. Visualizations will be skipped. ({e})")

# Canonical area grouping comes from Utils.badminton_config.area_groups.
_GROUP_NAMES = list(area_groups.keys())
N_GROUPS = len(_GROUP_NAMES)
OUT_OF_BOUNDS_GROUP = _GROUP_NAMES.index('Out of Bounds') if 'Out of Bounds' in _GROUP_NAMES else N_GROUPS - 1
N_VALID_GROUPS = N_GROUPS - 1  # Exclude Out of Bounds.


def short_area_group_label(name: str) -> str:
    """Compact display label for canonical badminton_config area groups."""
    if name == 'Out of Bounds':
        return 'Out'

    label = name
    replacements = [
        ('Deep Left Corner', 'Deep L'),
        ('Deep Right Corner', 'Deep R'),
        ('Rear Left Side', 'Rear L'),
        ('Rear Right Side', 'Rear R'),
        ('Rear Center', 'Rear C'),
        ('Mid Left Side', 'Mid L'),
        ('Mid Right Side', 'Mid R'),
        ('Mid Center', 'Mid C'),
        ('Front Left Side', 'Front L'),
        ('Front Right Side', 'Front R'),
        ('Front Center', 'Front C'),
        ('Front Left Side Line', 'FL Line'),
        ('Front Right Side Line', 'FR Line'),
        ('Mid Left Side Line', 'ML Line'),
        ('Mid Right Side Line', 'MR Line'),
        ('Rear Left Side Line', 'RL Line'),
        ('Rear Right Side Line', 'RR Line'),
    ]
    for full_name, short_name in replacements:
        if label == full_name:
            return short_name
    return label


GROUP_LABELS = [short_area_group_label(name) for name in _GROUP_NAMES]
MISSED_SHOT_TYPE = 11
VALID_SHOT_TYPES = tuple(range(1, MISSED_SHOT_TYPE))
DEPTH_NAMES = ('rear', 'mid', 'front')
REAR_GROUPS = {
    'Deep Left Corner', 'Deep Right Corner',
    'Rear Left Side', 'Rear Right Side', 'Rear Center',
    'Rear Left Side Line', 'Rear Right Side Line',
}
MID_GROUPS = {
    'Mid Left Side', 'Mid Right Side', 'Mid Center',
    'Mid Left Side Line', 'Mid Right Side Line',
}
FRONT_GROUPS = {
    'Front Left Side', 'Front Right Side', 'Front Center',
    'Front Left Side Line', 'Front Right Side Line',
}
GROUP_TO_DEPTH = {}
for group_name in REAR_GROUPS:
    GROUP_TO_DEPTH[group_name] = 0
for group_name in MID_GROUPS:
    GROUP_TO_DEPTH[group_name] = 1
for group_name in FRONT_GROUPS:
    GROUP_TO_DEPTH[group_name] = 2

# Build raw area id (1-33) -> canonical group index (0-N_GROUPS-1).
_AREA_TO_GROUP = {}
for gid, name in enumerate(_GROUP_NAMES):
    for area_id in area_groups.get(name, [33]):
        _AREA_TO_GROUP[area_id] = gid
for a in range(34):
    _AREA_TO_GROUP.setdefault(a, OUT_OF_BOUNDS_GROUP)

RAW_AREA_BOUNDS = {
    1: (50, 150, 104, 204),
    2: (104, 150, 177.5, 204),
    3: (177.5, 150, 251, 204),
    4: (251, 150, 305, 204),
    5: (50, 204, 104, 258),
    6: (104, 204, 177.5, 258),
    7: (177.5, 204, 251, 258),
    8: (251, 204, 305, 258),
    9: (50, 258, 104, 312),
    10: (104, 258, 177.5, 312),
    11: (177.5, 258, 251, 312),
    12: (251, 258, 305, 312),
    13: (50, 312, 104, 366),
    14: (104, 312, 177.5, 366),
    15: (177.5, 312, 251, 366),
    16: (251, 312, 305, 366),
    17: (50, 366, 104, 423),
    18: (104, 366, 177.5, 423),
    19: (177.5, 366, 251, 423),
    20: (251, 366, 305, 423),
    21: (50, 423, 104, 480),
    22: (104, 423, 177.5, 480),
    23: (177.5, 423, 251, 480),
    24: (251, 423, 305, 480),
    25: (305, 366, 355, 480),
    26: (305, 204, 355, 366),
    27: (305, 0, 355, 204),
    28: (177.5, 0, 305, 150),
    29: (0, 366, 50, 480),
    30: (0, 204, 50, 366),
    31: (0, 0, 50, 204),
    32: (50, 0, 177.5, 150),
}

# =============================================================================
# Utility Functions
# =============================================================================

def load_trajectories(trajectory_file: str) -> List[Dict]:
    """Load trajectory pickle file."""
    with open(trajectory_file, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def coord2area(norm_x: float, norm_y: float) -> int:
    """Convert normalized coordinates to raw court area id (1-33)."""
    from Utils.helper import coord2area as _coord2area
    return _coord2area(norm_x, norm_y)


def coord2group(norm_x: float, norm_y: float) -> int:
    """Convert normalized coordinates to badminton_config area group index."""
    area_id = coord2area(norm_x, norm_y)
    return _AREA_TO_GROUP.get(area_id, OUT_OF_BOUNDS_GROUP)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two discrete distributions."""
    if not SCIPY_AVAILABLE:
        return float('nan')
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Add smoothing
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def histogram(data: np.ndarray, bins: int, range_min: float, range_max: float) -> np.ndarray:
    """Normalized histogram with Laplace (+1) smoothing."""
    hist, _ = np.histogram(data, bins=bins, range=(range_min, range_max))
    hist = hist.astype(np.float64) + 1.0
    return hist / hist.sum()


def extract_sequences(trajectories: List[Dict], target_player_only: bool = False, skip_n: int = 0) -> Dict:
    """
    Extract flat arrays and per-rally sequences from trajectories.
    
    Args:
        trajectories: List of trajectory dicts
        target_player_only: If True, only extract target player's shots
        skip_n: Number of initial steps to skip per trajectory (for both GT and pred)
    
    Returns:
        Dict with extracted arrays
    """
    data = {
        'all_land_pred': [],
        'all_land_gt': [],
        'all_move_pred': [],
        'all_move_gt': [],
        'all_shot_pred': [],
        'all_shot_gt': [],
        'all_ball_rounds_pred': [],
        'all_ball_rounds_gt': [],
        'all_rally_lengths_pred': [],
        'all_rally_lengths_gt': [],
        'land_pred_seqs': [],
        'land_gt_seqs': [],
        'move_pred_seqs': [],
        'move_gt_seqs': [],
        'shot_pred_seqs': [],
        'shot_gt_seqs': [],
        'rally_player_won': [],
        'target_player': None,
    }

    def finite_positions(step):
        try:
            if int(step.get('shot_type')) == 11:
                return None
            values = [
                float(step['landing_x']),
                float(step['landing_y']),
                float(step['moving_x']),
                float(step['moving_y']),
            ]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return values[:2], values[2:]

    for traj in trajectories:
        gt_seq = traj.get('gt_sequence', [])
        pred_seq = traj.get('pred_sequence', [])
        target_player = traj.get('target_player', '')
        data['target_player'] = target_player
        
        # Extract per-rally data
        land_pred_rally = []
        land_gt_rally = []
        move_pred_rally = []
        move_gt_rally = []
        shot_pred_rally = []
        shot_gt_rally = []

        skipped_gt = 0
        for s in gt_seq:
            player = s.get('player', '')
            if target_player_only and player != target_player:
                continue
            if skipped_gt < skip_n:
                skipped_gt += 1
                continue
            
            data['all_shot_gt'].append(s['shot_type'])
            data['all_ball_rounds_gt'].append(s['ball_round'])
            shot_gt_rally.append(s['shot_type'])
            positions = finite_positions(s)
            if positions is not None:
                landing, movement = positions
                data['all_land_gt'].append(landing)
                data['all_move_gt'].append(movement)
                land_gt_rally.append(landing)
                move_gt_rally.append(movement)

        skipped_pred = 0
        for s in pred_seq:
            player = s.get('player', '')
            if target_player_only and player != target_player:
                continue
            if skipped_pred < skip_n:
                skipped_pred += 1
                continue
            
            data['all_shot_pred'].append(s['shot_type'])
            data['all_ball_rounds_pred'].append(s.get('ball_round', 0))
            shot_pred_rally.append(s['shot_type'])
            positions = finite_positions(s)
            if positions is not None:
                landing, movement = positions
                data['all_land_pred'].append(landing)
                data['all_move_pred'].append(movement)
                land_pred_rally.append(landing)
                move_pred_rally.append(movement)

        data['all_rally_lengths_pred'].append(len(pred_seq))
        data['all_rally_lengths_gt'].append(len(gt_seq))

        data['land_pred_seqs'].append(np.asarray(land_pred_rally, dtype=float).reshape(-1, 2))
        data['move_pred_seqs'].append(np.asarray(move_pred_rally, dtype=float).reshape(-1, 2))
        data['shot_pred_seqs'].append(shot_pred_rally)
        data['land_gt_seqs'].append(np.asarray(land_gt_rally, dtype=float).reshape(-1, 2))
        data['move_gt_seqs'].append(np.asarray(move_gt_rally, dtype=float).reshape(-1, 2))
        data['shot_gt_seqs'].append(shot_gt_rally)

        won = traj.get('target_player_won')
        data['rally_player_won'].append(won if isinstance(won, bool) else False)

    # Convert to numpy arrays
    for key in ['all_land_pred', 'all_land_gt', 'all_move_pred', 'all_move_gt']:
        if data[key]:
            data[key] = np.array(data[key])
        else:
            data[key] = np.array([]).reshape(0, 2)
    
    for key in ['all_shot_pred', 'all_shot_gt', 'all_ball_rounds_pred', 'all_ball_rounds_gt']:
        if data[key]:
            data[key] = np.array(data[key], dtype=int)
        else:
            data[key] = np.array([], dtype=int)

    return data


def non_missed_rally_lengths(trajectories: List[Dict], seq_key: str) -> List[int]:
    """Count modeled shots per rally, excluding Missed (11)."""
    lengths = []
    for traj in trajectories:
        count = 0
        for step in traj.get(seq_key, []):
            try:
                shot = int(step.get('shot_type'))
            except (TypeError, ValueError):
                continue
            if shot in VALID_SHOT_TYPES:
                count += 1
        lengths.append(count)
    return lengths


# =============================================================================
# Sequence Metrics (DTW, Edit Distance, LCS Ratio, Shot Similarity)
# =============================================================================

class SequenceMetrics:
    """Computes sequence-level metrics for positions and shot-type sequences."""
    
    @staticmethod
    def compute_dtw(land_pred_seqs: List[np.ndarray], land_gt_seqs: List[np.ndarray],
                    move_pred_seqs: List[np.ndarray], move_gt_seqs: List[np.ndarray]) -> Dict:
        """
        Compute DTW (Dynamic Time Warping) distance for landing and move positions.
        
        Returns average DTW normalized by sqrt(warping path length).
        """
        if not DTW_AVAILABLE:
            return {'land_dtw': float('nan'), 'move_dtw': float('nan')}
        
        land_dtw_list = []
        move_dtw_list = []
        
        for land_pred, land_gt in zip(land_pred_seqs, land_gt_seqs):
            if len(land_pred) >= 2 and len(land_gt) >= 2:
                try:
                    land_dtw_list.append(normalized_dtw_ndim_distance(land_pred, land_gt))
                except Exception as e:
                    pass
        
        for move_pred, move_gt in zip(move_pred_seqs, move_gt_seqs):
            if len(move_pred) >= 2 and len(move_gt) >= 2:
                try:
                    move_dtw_list.append(normalized_dtw_ndim_distance(move_pred, move_gt))
                except Exception as e:
                    pass
        
        return {
            'land_dtw': float(np.mean(land_dtw_list)) if land_dtw_list else float('inf'),
            'move_dtw': float(np.mean(move_dtw_list)) if move_dtw_list else float('inf'),
            'num_valid_rallies': len(land_dtw_list),
        }
    
    @staticmethod
    def compute_edit_distance(shot_pred_seqs: List[List[int]], shot_gt_seqs: List[List[int]]) -> Dict:
        """
        Compute normalized edit distance (Levenshtein) for shot sequences.
        """
        edit_dist_list = []
        
        for shot_pred, shot_gt in zip(shot_pred_seqs, shot_gt_seqs):
            if len(shot_pred) == 0 and len(shot_gt) == 0:
                continue
            
            dist = edit_distance(shot_pred, shot_gt)
            edit_dist_list.append(dist)
        
        return {
            'edit_distance': float(np.mean(edit_dist_list)) if edit_dist_list else float('inf'),
            'edit_distance_std': float(np.std(edit_dist_list)) if edit_dist_list else 0.0,
            'num_valid_rallies': len(edit_dist_list),
        }

    @staticmethod
    def _lcs_length(seq_pred: List[int], seq_gt: List[int]) -> int:
        """Compute longest common subsequence length for two shot sequences."""
        if len(seq_pred) < len(seq_gt):
            shorter, longer = seq_pred, seq_gt
        else:
            shorter, longer = seq_gt, seq_pred

        prev = [0] * (len(shorter) + 1)
        for token in longer:
            curr = [0] * (len(shorter) + 1)
            for j, other_token in enumerate(shorter, start=1):
                if token == other_token:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[-1]

    @classmethod
    def compute_lcs_ratio(cls, shot_pred_seqs: List[List[int]], shot_gt_seqs: List[List[int]]) -> Dict:
        """Compute symmetric LCS ratio for shot sequences, normalized by max length."""
        lcs_ratio_list = []

        for shot_pred, shot_gt in zip(shot_pred_seqs, shot_gt_seqs):
            max_len = max(len(shot_pred), len(shot_gt))
            if max_len == 0:
                continue

            lcs_len = cls._lcs_length(shot_pred, shot_gt)
            lcs_ratio_list.append(lcs_len / max_len)

        return {
            'lcs_ratio': float(np.mean(lcs_ratio_list)) if lcs_ratio_list else float('inf'),
            'lcs_ratio_std': float(np.std(lcs_ratio_list)) if lcs_ratio_list else 0.0,
            'num_valid_rallies': len(lcs_ratio_list),
        }

    @staticmethod
    def _valid_shots(seq: List[int]) -> List[int]:
        """Keep only valid modeled shot types."""
        valid = []
        for shot in seq:
            try:
                shot_id = int(shot)
            except (TypeError, ValueError):
                continue
            if 1 <= shot_id <= 11:
                valid.append(shot_id)
        return valid

    @classmethod
    def _ngram_counts(cls, seq: List[int], n: int) -> Dict[Tuple[int, ...], int]:
        valid_seq = cls._valid_shots(seq)
        counts = defaultdict(int)
        for i in range(len(valid_seq) - n + 1):
            counts[tuple(valid_seq[i:i + n])] += 1
        return counts

    @staticmethod
    def _overlap_f1(pred_counts: Dict[Tuple[int, ...], int], gt_counts: Dict[Tuple[int, ...], int]) -> float:
        pred_total = sum(pred_counts.values())
        gt_total = sum(gt_counts.values())
        if pred_total == 0 and gt_total == 0:
            return float('nan')
        if pred_total == 0 or gt_total == 0:
            return 0.0

        overlap = sum(min(pred_counts.get(key, 0), gt_counts.get(key, 0)) for key in set(pred_counts) | set(gt_counts))
        precision = overlap / pred_total
        recall = overlap / gt_total
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    @classmethod
    def compute_ngram_similarity(cls, shot_pred_seqs: List[List[int]], shot_gt_seqs: List[List[int]], n: int) -> Dict:
        """Compare shot n-gram distributions per rally using JS divergence and overlap F1."""
        js_list = []
        f1_list = []

        for shot_pred, shot_gt in zip(shot_pred_seqs, shot_gt_seqs):
            pred_counts = cls._ngram_counts(shot_pred, n)
            gt_counts = cls._ngram_counts(shot_gt, n)
            if not pred_counts and not gt_counts:
                continue

            keys = sorted(set(pred_counts) | set(gt_counts))
            if keys:
                p = np.array([pred_counts.get(key, 0) for key in keys], dtype=np.float64)
                q = np.array([gt_counts.get(key, 0) for key in keys], dtype=np.float64)
                js_list.append(js_divergence(p, q))

            f1_list.append(cls._overlap_f1(pred_counts, gt_counts))

        prefix = 'shot_bigram' if n == 2 else 'shot_trigram' if n == 3 else f'shot_{n}gram'
        return {
            f'{prefix}_js': float(np.mean(js_list)) if js_list else float('nan'),
            f'{prefix}_js_std': float(np.std(js_list)) if js_list else 0.0,
            f'{prefix}_f1': float(np.mean(f1_list)) if f1_list else float('nan'),
            f'{prefix}_f1_std': float(np.std(f1_list)) if f1_list else 0.0,
            'num_valid_rallies': len(f1_list),
        }

    @classmethod
    def compute_transition_js(cls, shot_pred_seqs: List[List[int]], shot_gt_seqs: List[List[int]]) -> Dict:
        """Compute rally-level weighted JS divergence between Markov transition rows."""
        transition_js_list = []

        for shot_pred, shot_gt in zip(shot_pred_seqs, shot_gt_seqs):
            pred_transitions = defaultdict(lambda: defaultdict(int))
            gt_transitions = defaultdict(lambda: defaultdict(int))

            pred_seq = cls._valid_shots(shot_pred)
            gt_seq = cls._valid_shots(shot_gt)
            for seq, transitions in [(pred_seq, pred_transitions), (gt_seq, gt_transitions)]:
                for i in range(len(seq) - 1):
                    transitions[seq[i]][seq[i + 1]] += 1

            weighted_js = 0.0
            total_weight = 0
            for prev_shot in range(1, 12):
                gt_support = sum(gt_transitions[prev_shot].values())
                if gt_support == 0:
                    continue

                p = np.array([pred_transitions[prev_shot].get(next_shot, 0) for next_shot in range(1, 12)], dtype=np.float64) + 1.0
                q = np.array([gt_transitions[prev_shot].get(next_shot, 0) for next_shot in range(1, 12)], dtype=np.float64) + 1.0
                weighted_js += js_divergence(p, q) * gt_support
                total_weight += gt_support

            if total_weight > 0:
                transition_js_list.append(weighted_js / total_weight)

        return {
            'shot_transition_js': float(np.mean(transition_js_list)) if transition_js_list else float('nan'),
            'shot_transition_js_std': float(np.std(transition_js_list)) if transition_js_list else 0.0,
            'num_valid_rallies': len(transition_js_list),
        }


# =============================================================================
# Experiment-Level Table Helpers
# =============================================================================

def parse_trajectory_filename(path: str) -> Dict[str, Any]:
    """Parse generation metadata from a trajectory pickle filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    generation_match = re.match(r'^(?P<base>.+)_gen(?P<generation>\d+)$', stem)
    if generation_match:
        return {
            'trajectory_base': generation_match.group('base'),
            'generation': int(generation_match.group('generation')),
        }

    indexed_match = re.match(r'^\d{4}_(?P<base>.+)$', stem)
    if indexed_match:
        return {
            'trajectory_base': indexed_match.group('base'),
            'generation': 1,
        }

    return {
        'trajectory_base': stem,
        'generation': 1,
    }


def discover_trajectory_files(trajectory_dir: str, latest_only: bool = False) -> List[str]:
    """Discover trajectory pickle files under an experiment trajectory directory."""
    pattern = os.path.join(trajectory_dir, '**', '*.pkl')
    files = sorted(glob.glob(pattern, recursive=True))

    if not latest_only:
        return files

    latest_by_base = {}
    for path in files:
        parsed = parse_trajectory_filename(path)
        base = parsed['trajectory_base']
        generation = parsed['generation']
        current = latest_by_base.get(base)
        if current is None or generation > current[0]:
            latest_by_base[base] = (generation, path)

    return sorted(path for _, path in latest_by_base.values())


def finite_mean(values: List[Any], weights: Optional[List[float]] = None) -> float:
    """Mean over finite numeric values only, optionally weighted."""
    finite_values = []
    finite_weights = []
    for idx, value in enumerate(values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite_values.append(numeric)
            if weights is not None:
                finite_weights.append(float(weights[idx]))

    if not finite_values:
        return float('nan')

    if weights is None:
        return float(np.mean(finite_values))

    weight_sum = float(np.sum(finite_weights))
    if weight_sum <= 0:
        return float('nan')
    return float(np.average(finite_values, weights=finite_weights))


def format_table_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return 'nan'
        if math.isinf(value):
            return 'inf'
        return f'{value:.6f}'
    return str(value)


def print_table(title: str, rows: List[Dict], columns: List[str], max_rows: int = 50):
    """Print a compact text table. Set max_rows <= 0 to print all rows."""
    print()
    print(title)
    print('=' * len(title))
    if not rows:
        print('(no rows)')
        return

    visible_rows = rows if max_rows <= 0 else rows[:max_rows]
    formatted_rows = [
        {column: format_table_value(row.get(column, '')) for column in columns}
        for row in visible_rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }

    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    separator = '-+-'.join('-' * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in formatted_rows:
        print(' | '.join(row[column].ljust(widths[column]) for column in columns))

    if max_rows > 0 and len(rows) > max_rows:
        print(f'... showing {max_rows} of {len(rows)} rows')


def format_fixed_table_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return 'nan'
        if math.isinf(value):
            return 'inf'
        return f'{value:.4f}'
    if isinstance(value, np.integer):
        return str(int(value))
    return str(value)


def build_player_metric_table(title: str,
                              player_rows: List[Dict],
                              metric_specs: List[Tuple[str, str, str]],
                              summary_rows: List[Dict],
                              max_rows: int = 50) -> str:
    """Build a fixed-width player metric table with averages below a rule.

    metric_specs entries are (display_column, player_row_key, summary_row_key).
    """
    base_specs = [
        ('Player', 'target_player', 24),
        ('Gen', 'generation', 4),
        ('M', 'num_matches', 4),
        ('R', 'num_rallies', 6),
        ('GT', 'num_gt_shots', 6),
        ('Pred', 'num_pred_shots', 6),
    ]
    metric_display_specs = [
        (display, display, max(7, len(display)))
        for display, _, _ in metric_specs
    ]
    display_specs = base_specs + metric_display_specs

    rows_to_print = player_rows if max_rows <= 0 else player_rows[:max_rows]
    display_rows = []

    for row in rows_to_print:
        display_row = {
            'target_player': row.get('target_player', ''),
            'generation': row.get('generation', ''),
            'num_matches': row.get('num_matches', ''),
            'num_rallies': row.get('num_rallies', ''),
            'num_gt_shots': row.get('num_gt_shots', ''),
            'num_pred_shots': row.get('num_pred_shots', ''),
        }
        for display, player_key, _ in metric_specs:
            display_row[display] = row.get(player_key, '')
        display_rows.append(display_row)

    summary_display_rows = []
    for row in summary_rows:
        display_row = {
            'target_player': row.get('average_type', ''),
            'generation': '',
            'num_matches': row.get('num_matches', ''),
            'num_rallies': row.get('num_rallies', ''),
            'num_gt_shots': row.get('num_gt_shots', ''),
            'num_pred_shots': row.get('num_pred_shots', ''),
        }
        for display, _, summary_key in metric_specs:
            display_row[display] = row.get(summary_key, '')
        summary_display_rows.append(display_row)

    def format_row(row: Dict) -> str:
        cells = []
        for label, key, width in display_specs:
            value = format_fixed_table_value(row.get(key, ''))
            if key == 'target_player':
                value = value[:width]
            cells.append(value.ljust(width))
        return ' '.join(cells)

    header = ' '.join(label.ljust(width) for label, _, width in display_specs)
    table_width = len(header)
    lines = [
        '',
        '=' * table_width,
        title,
        '=' * table_width,
        header,
        '-' * table_width,
    ]

    if not display_rows and not summary_display_rows:
        lines.append('(no rows)')
        lines.append('=' * table_width)
        return '\n'.join(lines)

    for row in display_rows:
        lines.append(format_row(row))

    if summary_display_rows:
        lines.append('-' * table_width)
        for row in summary_display_rows:
            lines.append(format_row(row))

    if max_rows > 0 and len(player_rows) > max_rows:
        lines.append(f'... showing {max_rows} of {len(player_rows)} player rows')

    lines.append('=' * table_width)
    return '\n'.join(lines)


def print_player_metric_table(title: str,
                              player_rows: List[Dict],
                              metric_specs: List[Tuple[str, str, str]],
                              summary_rows: List[Dict],
                              max_rows: int = 50):
    print(build_player_metric_table(title, player_rows, metric_specs, summary_rows, max_rows))


def write_csv_rows(path: str, rows: List[Dict], columns: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in columns})


def infer_agent_label_from_path(path: str) -> str:
    """Infer a readable agent label from common Experiments/<agent>/... paths."""
    parts = os.path.normpath(path).split(os.sep)
    label_map = {
        'bc': 'BC',
        'naive_rag': 'NaiveRAG',
        'pure_llm': 'PureLLM',
        'timerag': 'TimeRAG',
        'twin': 'Twin',
        'rallynetv2': 'RallyNetV2',
        'random': 'Random',
    }

    if 'Experiments' in parts:
        idx = parts.index('Experiments')
        if idx + 1 < len(parts):
            return label_map.get(parts[idx + 1], parts[idx + 1])

    return 'Agent'


def sequence_metrics_for_trajectories(trajectories: List[Dict], target_player_only: bool, skip_n: int = 0) -> Dict[str, float]:
    data = extract_sequences(trajectories, target_player_only=target_player_only, skip_n=skip_n)
    seq_metrics = SequenceMetrics()
    dtw = seq_metrics.compute_dtw(
        data['land_pred_seqs'], data['land_gt_seqs'],
        data['move_pred_seqs'], data['move_gt_seqs']
    )
    lcs = seq_metrics.compute_lcs_ratio(data['shot_pred_seqs'], data['shot_gt_seqs'])

    return {
        'land_dtw': dtw.get('land_dtw', float('nan')),
        'move_dtw': dtw.get('move_dtw', float('nan')),
        'lcs_ratio': lcs.get('lcs_ratio', float('nan')),
    }


def distribution_metrics_for_trajectories(trajectories: List[Dict], target_player_only: bool) -> Dict[str, float]:
    dist_metrics = DistributionMetrics()

    return {
        'shot_conditioned_landing_js': dist_metrics.shot_conditioned_landing_js(
            trajectories, target_player_only=target_player_only
        ),
        'shot_distribution_js': dist_metrics.shot_distribution_js(
            trajectories, target_player_only=target_player_only
        ),
        'rally_length_js': dist_metrics.rally_length_js(trajectories),
        'state_conditioned_move_js': dist_metrics.state_conditioned_action_js(
            trajectories, action='move', target_player_only=target_player_only
        ),
        'state_conditioned_shot_js': dist_metrics.state_conditioned_action_js(
            trajectories, action='shot', target_player_only=target_player_only
        ),
        'state_conditioned_landing_js': dist_metrics.state_conditioned_action_js(
            trajectories, action='landing', target_player_only=target_player_only
        ),
    }


# =============================================================================
# Distribution Metrics
# =============================================================================

class DistributionMetrics:
    """Computes the selected distribution similarity metrics."""

    VALID_SHOTS = tuple(range(1, 11))
    MIN_STATE_SUPPORT = 5

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _area_from_step(step: Dict, x_key: str, y_key: str) -> Optional[int]:
        if x_key not in step or y_key not in step:
            return None
        try:
            return coord2group(float(step[x_key]), float(step[y_key]))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _depth_from_step(cls, step: Dict, x_key: str, y_key: str) -> Optional[int]:
        area_group = cls._area_from_step(step, x_key, y_key)
        if area_group is None or area_group == OUT_OF_BOUNDS_GROUP:
            return None
        group_name = _GROUP_NAMES[area_group]
        return GROUP_TO_DEPTH.get(group_name)

    @classmethod
    def _include_current_player(cls, step: Dict, target_player: str, target_player_only: bool) -> bool:
        return (not target_player_only) or step.get('player', '') == target_player

    @classmethod
    def _sequence_actions(cls, trajectories: List[Dict], seq_key: str, target_player_only: bool) -> List[Dict]:
        actions = []
        for traj in trajectories:
            target_player = traj.get('target_player', '')
            for step in traj.get(seq_key, []):
                if not cls._include_current_player(step, target_player, target_player_only):
                    continue
                actions.append(step)
        return actions

    @classmethod
    def _count_js(cls, pred_counts: Dict[int, int], gt_counts: Dict[int, int], bins: int) -> float:
        if not pred_counts and not gt_counts:
            return float('nan')
        p = np.array([pred_counts.get(i, 0) for i in range(bins)], dtype=np.float64)
        q = np.array([gt_counts.get(i, 0) for i in range(bins)], dtype=np.float64)
        if p.sum() == 0 and q.sum() == 0:
            return float('nan')
        return js_divergence(p, q)

    @staticmethod
    def _increment(counts: Dict[int, int], key: Optional[int]):
        if key is not None:
            counts[key] += 1

    @classmethod
    def shot_distribution_js(cls, trajectories: List[Dict], target_player_only: bool) -> float:
        """JS divergence of overall shot-type distributions, excluding Missed (11)."""
        pred_counts = defaultdict(int)
        gt_counts = defaultdict(int)

        for step in cls._sequence_actions(trajectories, 'pred_sequence', target_player_only):
            shot = cls._safe_int(step.get('shot_type'))
            if shot in cls.VALID_SHOTS:
                pred_counts[shot - 1] += 1

        for step in cls._sequence_actions(trajectories, 'gt_sequence', target_player_only):
            shot = cls._safe_int(step.get('shot_type'))
            if shot in cls.VALID_SHOTS:
                gt_counts[shot - 1] += 1

        return cls._count_js(pred_counts, gt_counts, bins=len(cls.VALID_SHOTS))

    @classmethod
    def shot_conditioned_landing_js(cls, trajectories: List[Dict], target_player_only: bool) -> float:
        """Weighted JS of P(landing area | shot type), excluding Missed (11)."""
        pred_by_shot = defaultdict(lambda: defaultdict(int))
        gt_by_shot = defaultdict(lambda: defaultdict(int))

        for step in cls._sequence_actions(trajectories, 'pred_sequence', target_player_only):
            shot = cls._safe_int(step.get('shot_type'))
            if shot not in cls.VALID_SHOTS:
                continue
            cls._increment(pred_by_shot[shot], cls._area_from_step(step, 'landing_x', 'landing_y'))

        for step in cls._sequence_actions(trajectories, 'gt_sequence', target_player_only):
            shot = cls._safe_int(step.get('shot_type'))
            if shot not in cls.VALID_SHOTS:
                continue
            cls._increment(gt_by_shot[shot], cls._area_from_step(step, 'landing_x', 'landing_y'))

        return cls._weighted_conditioned_js(pred_by_shot, gt_by_shot, bins=N_GROUPS)

    @classmethod
    def rally_length_js(cls, trajectories: List[Dict]) -> float:
        """JS divergence of complete rally round-length distributions."""
        pred_counts = defaultdict(int)
        gt_counts = defaultdict(int)

        for traj in trajectories:
            pred_len = cls._rally_round_length(traj.get('pred_sequence', []), traj.get('rally_length_pred'))
            gt_len = cls._rally_round_length(traj.get('gt_sequence', []), traj.get('rally_length_gt'))
            pred_counts[pred_len] += 1
            gt_counts[gt_len] += 1

        max_len = max(list(pred_counts.keys()) + list(gt_counts.keys()), default=0)
        return cls._count_js(pred_counts, gt_counts, bins=max_len + 1)

    @classmethod
    def _rally_round_length(cls, seq: List[Dict], fallback: Any) -> int:
        rounds = [cls._safe_int(step.get('ball_round')) for step in seq]
        rounds = [round_id for round_id in rounds if round_id is not None]
        if rounds:
            return max(rounds)
        fallback_len = cls._safe_int(fallback)
        return max(fallback_len or 0, 0)

    @classmethod
    def state_conditioned_action_js(cls, trajectories: List[Dict], action: str, target_player_only: bool) -> float:
        """Weighted JS conditioned on (player previous shot, player previous landing depth)."""
        if action == 'shot':
            bins = len(cls.VALID_SHOTS)
        elif action in ('landing', 'move'):
            bins = N_VALID_GROUPS
        else:
            raise ValueError(f'Unsupported state-conditioned action: {action}')

        pred_by_state = cls._state_action_counts(trajectories, 'pred_sequence', action, target_player_only)
        gt_by_state = cls._state_action_counts(trajectories, 'gt_sequence', action, target_player_only)
        return cls._weighted_conditioned_js(
            pred_by_state, gt_by_state, bins=bins, min_support=cls.MIN_STATE_SUPPORT
        )

    @classmethod
    def _state_action_counts(cls, trajectories: List[Dict], seq_key: str, action: str,
                             target_player_only: bool) -> Dict[Tuple[int, int, int, int], Dict[int, int]]:
        counts = defaultdict(lambda: defaultdict(int))

        for traj in trajectories:
            target_player = traj.get('target_player', '')
            seq = traj.get(seq_key, [])
            for idx in range(2, len(seq)):
                current = seq[idx]
                prev = seq[idx - 1]
                prev_prev = seq[idx - 2]

                if not cls._include_current_player(current, target_player, target_player_only):
                    continue
                if current.get('player') != prev_prev.get('player') or current.get('player') == prev.get('player'):
                    continue

                state_bucket = cls._state_bucket(prev_prev, prev)
                if state_bucket is None:
                    continue

                action_value = cls._action_value(current, action)
                if action_value is None:
                    continue
                counts[state_bucket][action_value] += 1

        return counts

    @classmethod
    def _state_bucket(cls, prev_prev: Dict, prev: Dict) -> Optional[Tuple[int, int]]:
        player_prev_shot = cls._safe_int(prev_prev.get('shot_type'))
        if player_prev_shot not in cls.VALID_SHOTS:
            return None

        player_prev_landing_depth = cls._depth_from_step(prev_prev, 'landing_x', 'landing_y')
        if player_prev_landing_depth is None:
            return None

        return (player_prev_shot, player_prev_landing_depth)

    @classmethod
    def _action_value(cls, step: Dict, action: str) -> Optional[int]:
        if action == 'shot':
            shot = cls._safe_int(step.get('shot_type'))
            return shot - 1 if shot in cls.VALID_SHOTS else None
        if action == 'landing':
            area = cls._area_from_step(step, 'landing_x', 'landing_y')
            return None if area == OUT_OF_BOUNDS_GROUP else area
        if action == 'move':
            area = cls._area_from_step(step, 'moving_x', 'moving_y')
            return None if area == OUT_OF_BOUNDS_GROUP else area
        return None

    @classmethod
    def _weighted_conditioned_js(cls, pred_by_key: Dict[Any, Dict[int, int]],
                                 gt_by_key: Dict[Any, Dict[int, int]], bins: int,
                                 min_support: int = 1) -> float:
        weighted_js = 0.0
        total_weight = 0

        for key, gt_counts in gt_by_key.items():
            gt_support = sum(gt_counts.values())
            if gt_support < min_support:
                continue
            js = cls._count_js(pred_by_key.get(key, {}), gt_counts, bins=bins)
            if not math.isfinite(js):
                continue
            weighted_js += js * gt_support
            total_weight += gt_support

        return float(weighted_js / total_weight) if total_weight > 0 else float('nan')


# =============================================================================
# Visualization Functions
# =============================================================================

class Visualizer:
    """Generates visualization charts for distribution analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _denorm_landing(self, norm_coords: np.ndarray) -> np.ndarray:
        """Convert [-1,1] landing coords to actual court coords [0,355]x[0,480]."""
        if len(norm_coords) == 0:
            return norm_coords
        x = (norm_coords[:, 0] + 1.0) * 177.5
        y = (norm_coords[:, 1] + 1.0) * 240.0
        return np.column_stack([x, y])

    def _denorm_landing_for_court_plot(self, norm_coords: np.ndarray) -> np.ndarray:
        """Convert landing coords to the mirrored court view used by KDE panels."""
        coords = self._denorm_landing(norm_coords)
        if len(coords) == 0:
            return coords
        return np.column_stack([355.0 - coords[:, 0], 480.0 - coords[:, 1]])
    
    def _denorm_move(self, norm_coords: np.ndarray) -> np.ndarray:
        """Convert [-1,1] move coords to [0,355]x[-480,0]."""
        if len(norm_coords) == 0:
            return norm_coords
        x = (norm_coords[:, 0] + 1.0) * 177.5
        y = norm_coords[:, 1] * 240.0 - 240.0
        return np.column_stack([x, y])
    
    def _draw_landing_court(self, ax, color: str = 'k'):
        """Draw badminton court lines for landing half."""
        ax.plot([0, 355], [0, 0], color=color, linestyle='-', linewidth=1.2)
        ax.plot([0, 355], [480, 480], color=color, linestyle='-', linewidth=1.2)
        ax.plot([0, 0], [0, 480], color=color, linestyle='-', linewidth=1.2)
        ax.plot([355, 355], [0, 480], color=color, linestyle='-', linewidth=1.2)
        ax.plot([177.5, 177.5], [0, 480], color=color, linestyle='--', linewidth=0.8, alpha=0.55)
        ax.plot([0, 355], [141.85, 141.85], color=color, linestyle='-', linewidth=0.8, alpha=0.65)
        ax.plot([0, 355], [425.78, 425.78], color=color, linestyle=':', linewidth=0.7, alpha=0.45)

    def _draw_landing_area_groups(self, ax, label: bool = False):
        """Overlay canonical area-group partitions on the landing half court."""
        def edge_key(p1, p2):
            p1 = (round(p1[0], 3), round(p1[1], 3))
            p2 = (round(p2[0], 3), round(p2[1], 3))
            return tuple(sorted([p1, p2]))

        for name in _GROUP_NAMES:
            if name == 'Out of Bounds':
                continue
            bounds = [RAW_AREA_BOUNDS[area_id] for area_id in area_groups.get(name, []) if area_id in RAW_AREA_BOUNDS]
            if not bounds:
                continue

            group_edges = {}
            for x0, y0, x1, y1 in bounds:
                edges = [
                    ((x0, y0), (x1, y0)),
                    ((x1, y0), (x1, y1)),
                    ((x1, y1), (x0, y1)),
                    ((x0, y1), (x0, y0)),
                ]
                for p1, p2 in edges:
                    key = edge_key(p1, p2)
                    if key in group_edges:
                        del group_edges[key]
                    else:
                        group_edges[key] = (p1, p2)

            for p1, p2 in group_edges.values():
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color='white', linewidth=0.8, alpha=0.72, zorder=5,
                )

            if label:
                x0 = min(bound[0] for bound in bounds)
                y0 = min(bound[1] for bound in bounds)
                x1 = max(bound[2] for bound in bounds)
                y1 = max(bound[3] for bound in bounds)
                ax.text(
                    (x0 + x1) / 2, (y0 + y1) / 2,
                    short_area_group_label(name),
                    ha='center', va='center', fontsize=5.5,
                    color='dimgray', alpha=0.72,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.35, pad=0.4),
                )

    def _draw_landing_background(self, ax):
        """Draw court-colored in-bounds and out-of-bounds regions."""
        from matplotlib.patches import Rectangle

        ax.set_facecolor('#1d5f3d')
        ax.add_patch(Rectangle(
            (-35, -35), 425, 540,
            facecolor='#1d5f3d', edgecolor='none', zorder=-10,
        ))
        ax.add_patch(Rectangle(
            (0, 0), 355, 480,
            facecolor='#2f8f5b', edgecolor='none', zorder=-9,
        ))
        ax.add_patch(Rectangle(
            (-35, -35), 425, 540,
            fill=False, edgecolor='white', linewidth=1.1, alpha=0.6, zorder=4,
        ))

    def _draw_net_marker(self, ax):
        """Draw a clear net marker above the landing half."""
        ax.axhspan(480, 500, facecolor='#e8f4f8', alpha=0.9, zorder=0)
        ax.plot([0, 355], [480, 480], color='#f7f7f7', linewidth=3.0, zorder=5)
        ax.plot([0, 355], [483, 483], color='#1f2a33', linewidth=1.0, alpha=0.75, zorder=6)
        ax.text(
            177.5, 490, 'NET', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#1f2a33', clip_on=False,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#1f2a33', alpha=0.88),
            zorder=7,
        )

    def _draw_half_court_reference(self, ax, color: str = 'white'):
        """Draw a clean half-court reference layout for landing KDE panels."""
        from matplotlib.patches import Rectangle

        linewidth = 1.25
        ax.plot([50, 50], [0, 330], color=color, linewidth=linewidth, zorder=5)
        ax.plot([305, 305], [0, 330], color=color, linewidth=linewidth, zorder=5)
        ax.plot([27.4, 327.6], [330, 330], color=color, linewidth=linewidth, zorder=5)
        ax.plot([27.4, 327.6], [0, 0], color=color, linewidth=linewidth, zorder=5)
        ax.plot([27.4, 27.4], [0, 330], color=color, linewidth=linewidth, zorder=5)
        ax.plot([327.6, 327.6], [0, 330], color=color, linewidth=linewidth, zorder=5)
        ax.plot([177.5, 177.5], [0, 330], color=color, linewidth=linewidth, zorder=5)
        ax.plot([27.4, 327.6], [54, 54], color=color, linewidth=linewidth, zorder=5)
        ax.plot([27.4, 327.6], [216, 216], color=color, linewidth=linewidth, zorder=5)

        ax.plot([27.4, 327.6], [0, 0], color='#f7f7f7', linewidth=3.0, alpha=0.85, zorder=6)
        ax.add_patch(Rectangle(
            (0, 0), 355, 480,
            fill=False, edgecolor='black', linewidth=4.0, alpha=1.0,
            zorder=7,
        ))
        ax.text(
            177.5, 0.0, 'NET', ha='center', va='center', fontsize=8,
            fontweight='bold', color='#1f2a33', zorder=7,
            transform=ax.get_xaxis_transform(), clip_on=False,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#1f2a33', alpha=0.9),
        )
    
    def _draw_move_court(self, ax):
        """Draw badminton court lines for move half."""
        ax.plot([0, 355], [0, 0], 'k-', linewidth=1.2)
        ax.plot([0, 355], [-480, -480], 'k-', linewidth=1.2)
        ax.plot([0, 0], [-480, 0], 'k-', linewidth=1.2)
        ax.plot([355, 355], [-480, 0], 'k-', linewidth=1.2)
        ax.plot([177.5, 177.5], [-480, 0], 'k--', linewidth=0.8, alpha=0.5)
        ax.plot([0, 355], [-141.85, -141.85], 'k-', linewidth=0.8, alpha=0.6)
        ax.plot([0, 355], [-425.78, -425.78], 'k:', linewidth=0.7, alpha=0.4)
    
    def plot_landing_kde(self, land_pred: np.ndarray, land_gt: np.ndarray, 
                        title: str, filename: str):
        """Plot overlaid Pred/GT landing KDE on the mirrored half-court view."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        dp = self._denorm_landing_for_court_plot(land_pred)
        dg = self._denorm_landing_for_court_plot(land_gt)

        fig, ax = plt.subplots(figsize=(7, 9))
        self._plot_landing_kde_overlay_panel(ax, dp, dg, title)

        from matplotlib.patches import Patch
        fig.legend(
            handles=[
                Patch(facecolor='#1f4ed8', edgecolor='#0b1f8f', linewidth=1.5, alpha=0.42, label='Pred'),
                Patch(facecolor='#d95f02', edgecolor='#8c2d04', linewidth=1.5, alpha=0.42, label='GT'),
            ],
            loc='upper right', bbox_to_anchor=(0.98, 0.90), fontsize=11,
            frameon=True, framealpha=0.9,
        )
        plt.tight_layout(rect=[0, 0, 0.94, 0.96])
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_landing_kde_panel(self, ax, coords: np.ndarray, cmap: str, title: str,
                                grid_size: int = 90):
        """Draw one landing-court KDE panel, falling back to scatter for sparse data."""
        if len(coords) >= 3 and SCIPY_AVAILABLE:
            try:
                kde_coords = coords.astype(np.float64, copy=True)
                if len(np.unique(kde_coords, axis=0)) < 3:
                    offsets = np.linspace(-0.4, 0.4, len(kde_coords))
                    kde_coords[:, 0] += offsets
                    kde_coords[:, 1] -= offsets

                kde = gaussian_kde(kde_coords.T)
                x = np.linspace(0, 355, grid_size)
                y = np.linspace(0, 480, grid_size)
                X, Y = np.meshgrid(x, y)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                if Z.max() > 0:
                    ax.contourf(X, Y, Z, levels=12, cmap=cmap, alpha=0.86)
            except Exception:
                ax.scatter(coords[:, 0], coords[:, 1], c='black', alpha=0.25, s=12)
        elif len(coords) > 0:
            ax.scatter(coords[:, 0], coords[:, 1], c='black', alpha=0.25, s=12)

        self._draw_landing_court(ax)
        ax.set_xlim(0, 355)
        ax.set_ylim(0, 480)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_landing_kde_overlay_panel(self, ax, pred_coords: np.ndarray, gt_coords: np.ndarray,
                                        title: str, grid_size: int = 80, min_kde_points: int = 10):
        """Draw Pred and GT landing KDEs on one court panel with light overlays."""
        ax.set_facecolor('#4f9a63')
        use_points = (
            0 < len(pred_coords) < min_kde_points
            or 0 < len(gt_coords) < min_kde_points
        )

        def draw_kde(coords: np.ndarray, fill_color: str, contour_color: str, point_color: str):
            if not use_points and len(coords) >= min_kde_points and SCIPY_AVAILABLE:
                try:
                    kde_coords = coords.astype(np.float64, copy=True)
                    if len(np.unique(kde_coords, axis=0)) < 3:
                        offsets = np.linspace(-0.4, 0.4, len(kde_coords))
                        kde_coords[:, 0] += offsets
                        kde_coords[:, 1] -= offsets

                    kde = gaussian_kde(kde_coords.T)
                    x = np.linspace(0, 355, grid_size)
                    y = np.linspace(0, 480, grid_size)
                    X, Y = np.meshgrid(x, y)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    if Z.max() > 0:
                        levels = np.linspace(Z.max() * 0.10, Z.max(), 9)
                        ax.contourf(X, Y, Z, levels=levels, colors=[fill_color], alpha=0.42, zorder=1)
                        ax.contour(X, Y, Z, levels=levels[::2], colors=contour_color,
                                   linewidths=1.45, alpha=0.92, zorder=2)
                        return
                except Exception:
                    pass

            if len(coords) > 0:
                ax.scatter(
                    coords[:, 0], coords[:, 1], c=point_color, alpha=0.82, s=22,
                    edgecolors='white', linewidths=0.45, zorder=3,
                )

        draw_kde(gt_coords, '#d95f02', '#8c2d04', '#8c2d04')
        draw_kde(pred_coords, '#1f4ed8', '#0b1f8f', '#0b1f8f')
        self._draw_half_court_reference(ax, color='white')
        ax.set_xlim(0, 355)
        ax.set_ylim(0, 480)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold', color='#111111', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_shot_conditioned_landing_kde(self, land_pred: np.ndarray, land_gt: np.ndarray,
                                          shot_pred: np.ndarray, shot_gt: np.ndarray,
                                          title: str, filename: str):
        """Plot per-shot landing KDE overlays on court, excluding Missed (11)."""
        if not MATPLOTLIB_AVAILABLE:
            return

        n_panels = len(VALID_SHOT_TYPES)
        ncols = 5
        nrows = int(math.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.2 * nrows))
        axes = axes.flatten()

        for idx, shot in enumerate(VALID_SHOT_TYPES):
            ax = axes[idx]
            pred_mask = shot_pred == shot
            gt_mask = shot_gt == shot
            pred_coords = self._denorm_landing_for_court_plot(land_pred[pred_mask]) if np.sum(pred_mask) > 0 else np.zeros((0, 2))
            gt_coords = self._denorm_landing_for_court_plot(land_gt[gt_mask]) if np.sum(gt_mask) > 0 else np.zeros((0, 2))
            shot_label = INDEX2EN_SHOT.get(shot, f'Shot {shot}')

            self._plot_landing_kde_overlay_panel(
                ax, pred_coords, gt_coords, shot_label
            )

        for ax in axes[n_panels:]:
            ax.axis('off')

        from matplotlib.patches import Patch
        fig.legend(
            handles=[
                Patch(facecolor='#1f4ed8', edgecolor='#0b1f8f', linewidth=1.5, alpha=0.42, label='Pred'),
                Patch(facecolor='#d95f02', edgecolor='#8c2d04', linewidth=1.5, alpha=0.42, label='GT'),
            ],
            loc='upper right', bbox_to_anchor=(0.985, 0.975), fontsize=11,
            frameon=True, framealpha=0.9,
        )
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.96, 0.94])
        plt.savefig(os.path.join(self.output_dir, filename), dpi=130, bbox_inches='tight')
        plt.close()

    def plot_rally_length_distribution(self, rally_lengths_pred: List[int], rally_lengths_gt: List[int],
                                       title: str, filename: str, normalize: bool = True):
        """Plot Pred vs GT rally-length distributions as filled KDE curves."""
        if not MATPLOTLIB_AVAILABLE:
            return

        pred = np.asarray(rally_lengths_pred, dtype=int)
        gt = np.asarray(rally_lengths_gt, dtype=int)
        pred = pred[pred >= 0]
        gt = gt[gt >= 0]
        if len(pred) == 0 and len(gt) == 0:
            return

        all_lengths = np.concatenate([arr for arr in [pred, gt] if len(arr) > 0])
        min_len = int(min(pred.min() if len(pred) else 1, gt.min() if len(gt) else 1))
        max_len = int(max(pred.max() if len(pred) else 0, gt.max() if len(gt) else 0))
        if max_len <= min_len:
            return

        display_max = int(math.ceil(np.percentile(all_lengths, 95)))
        display_max = max(display_max, min_len + 1)
        display_max = min(display_max, max_len)
        pred_display = pred[pred <= display_max]
        gt_display = gt[gt <= display_max]

        lengths = np.arange(min_len, display_max + 1)
        counts_pred = np.bincount(pred_display, minlength=display_max + 1)[min_len:display_max + 1].astype(np.float64)
        counts_gt = np.bincount(gt_display, minlength=display_max + 1)[min_len:display_max + 1].astype(np.float64)

        if normalize:
            if counts_pred.sum() > 0:
                counts_pred = counts_pred / counts_pred.sum()
            if counts_gt.sum() > 0:
                counts_gt = counts_gt / counts_gt.sum()

        dense_x = np.linspace(min_len, display_max, max(400, (display_max - min_len + 1) * 25))

        def fallback_smooth_curve(counts: np.ndarray) -> np.ndarray:
            dense_y = np.interp(dense_x, lengths, counts)
            if len(counts) <= 2:
                return dense_y
            spacing = (dense_x[-1] - dense_x[0]) / max(len(dense_x) - 1, 1)
            sigma_points = max(2.0, 1.25 / max(spacing, 1e-8))
            kernel_radius = int(math.ceil(sigma_points * 3))
            kernel_x = np.arange(-kernel_radius, kernel_radius + 1)
            kernel = np.exp(-(kernel_x ** 2) / (2.0 * sigma_points ** 2))
            kernel = kernel / kernel.sum()
            padded = np.pad(dense_y, kernel_radius, mode='edge')
            return np.convolve(padded, kernel, mode='valid')

        def kde_curve(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
            if not SCIPY_AVAILABLE or len(values) < 2 or len(np.unique(values)) < 2:
                return fallback_smooth_curve(counts)
            try:
                kde = gaussian_kde(values.astype(np.float64), bw_method=0.45)
                curve = kde(dense_x)
                area = np.trapz(curve, dense_x)
                return curve / area if normalize and area > 0 else curve
            except Exception:
                return fallback_smooth_curve(counts)

        curve_pred = kde_curve(pred_display, counts_pred)
        curve_gt = kde_curve(gt_display, counts_gt)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(dense_x, curve_pred, color='steelblue', alpha=0.24)
        ax.plot(dense_x, curve_pred, color='steelblue', linewidth=2.5, label='Pred')
        ax.fill_between(dense_x, curve_gt, color='firebrick', alpha=0.22)
        ax.plot(dense_x, curve_gt, color='firebrick', linewidth=2.5, label='GT')

        marker_y = -0.018 * max(curve_pred.max(), curve_gt.max(), 1e-8)
        ax.plot(pred_display, np.full(len(pred_display), marker_y), '|', color='steelblue', markersize=8, alpha=0.22)
        ax.plot(gt_display, np.full(len(gt_display), marker_y * 1.8), '|', color='firebrick', markersize=8, alpha=0.22)

        ax.set_xlabel('Rally Length')
        ax.set_ylabel('Density' if normalize else 'Smoothed Count')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(min_len, display_max)
        ax.set_ylim(marker_y * 2.6, max(curve_pred.max(), curve_gt.max()) * 1.08)
        if display_max <= 30:
            ax.set_xticks(lengths)
        else:
            step = max(1, math.ceil((display_max - min_len + 1) / 20))
            ax.set_xticks(np.arange(min_len, display_max + 1, step))

        clipped = int(np.sum(pred > display_max) + np.sum(gt > display_max))
        if clipped > 0:
            ax.text(0.99, 0.96, f'x-axis clipped at {display_max} ({clipped} long rallies)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9, color='dimgray')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_shot_distribution(self, shot_pred: np.ndarray, shot_gt: np.ndarray,
                               title: str, filename: str, normalize: bool = True):
        """Plot grouped bar chart of shot type distributions, excluding Missed (11)."""
        if not MATPLOTLIB_AVAILABLE:
            return

        valid_pred = shot_pred[np.isin(shot_pred, VALID_SHOT_TYPES)]
        valid_gt = shot_gt[np.isin(shot_gt, VALID_SHOT_TYPES)]

        shot_labels = [INDEX2EN_SHOT.get(i, f'Type {i}') for i in VALID_SHOT_TYPES]
        counts_pred = np.array([int(np.sum(valid_pred == s)) for s in VALID_SHOT_TYPES], dtype=np.float64)
        counts_gt = np.array([int(np.sum(valid_gt == s)) for s in VALID_SHOT_TYPES], dtype=np.float64)

        if normalize:
            if counts_pred.sum() > 0:
                counts_pred = counts_pred / counts_pred.sum()
            if counts_gt.sum() > 0:
                counts_gt = counts_gt / counts_gt.sum()

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(shot_labels))
        width = 0.42
        ax.bar(x - width / 2, counts_pred, width, label='Pred',
               color='steelblue', alpha=0.85, edgecolor='white')
        ax.bar(x + width / 2, counts_gt, width, label='GT',
               color='firebrick', alpha=0.85, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax.set_ylabel('Probability' if normalize else 'Count')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_per_shot_landing_kde(self, land_pred: np.ndarray, land_gt: np.ndarray,
                                   shot_pred: np.ndarray, shot_gt: np.ndarray,
                                   title: str, filename: str, use_contour: bool = True,
                                   alpha: float = 0.6, grid_size: int = 100):
        """
        Plot per-shot-type landing distribution using Gaussian KDE (Kernel Density Estimation).
        Creates smooth density heatmaps for each shot type, overlaid on the court.
        
        Args:
            land_pred: Predicted landing positions
            land_gt: Ground truth landing positions  
            shot_pred: Predicted shot types
            shot_gt: Ground truth shot types
            title: Plot title
            filename: Output filename
            use_contour: If True, show contour lines instead of filled heatmap
            alpha: Transparency level
            grid_size: Resolution of the KDE grid (default 100)
        """
        if not MATPLOTLIB_AVAILABLE or not SCIPY_AVAILABLE:
            return
        
        from scipy.stats import gaussian_kde
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in VALID_SHOT_TYPES}
        shot_labels['ALL'] = 'All Shots'
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        shot_order = list(VALID_SHOT_TYPES) + ['ALL']
        
        # Create grid for KDE evaluation
        x_grid = np.linspace(0, 355, grid_size)
        y_grid = np.linspace(0, 480, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        for idx, shot in enumerate(shot_order):
            ax = axes[idx]
            
            if shot == 'ALL':
                mask_pred = np.isin(shot_pred, VALID_SHOT_TYPES)
                mask_gt = np.isin(shot_gt, VALID_SHOT_TYPES)
            else:
                mask_pred = shot_pred == shot
                mask_gt = shot_gt == shot
            
            dp = self._denorm_landing(land_pred[mask_pred]) if np.sum(mask_pred) > 0 else np.zeros((0, 2))
            dg = self._denorm_landing(land_gt[mask_gt]) if np.sum(mask_gt) > 0 else np.zeros((0, 2))
            
            # Plot GT KDE
            if len(dg) >= 10:
                try:
                    # Add small noise to avoid singular matrix
                    noise = np.random.normal(0, 0.5, dg.shape)
                    dg_noisy = dg + noise
                    
                    kde_gt = gaussian_kde(dg_noisy.T)
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z_gt = np.reshape(kde_gt(positions).T, X.shape)
                    
                    if use_contour:
                        levels = np.linspace(0, Z_gt.max(), 8)[1:] if Z_gt.max() > 0 else 5
                        ax.contour(X, Y, Z_gt, levels=levels, colors='red', 
                                  linewidths=2, alpha=0.8, linestyles='-')
                    else:
                        ax.contourf(X, Y, Z_gt, levels=15, cmap='Reds', alpha=alpha)
                        # Add contour lines on top
                        ax.contour(X, Y, Z_gt, levels=8, colors='darkred', 
                                  linewidths=1, alpha=0.5)
                except Exception as e:
                    # Fallback to scatter if KDE fails
                    if len(dg) > 0:
                        ax.scatter(dg[:, 0], dg[:, 1], c='red', alpha=0.3, s=20, label='GT')
            elif len(dg) > 0:
                # Not enough points for KDE, use scatter
                ax.scatter(dg[:, 0], dg[:, 1], c='red', alpha=0.3, s=20, label='GT')
            
            # Plot Pred KDE
            if len(dp) >= 10:
                try:
                    noise = np.random.normal(0, 0.5, dp.shape)
                    dp_noisy = dp + noise
                    
                    kde_pred = gaussian_kde(dp_noisy.T)
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z_pred = np.reshape(kde_pred(positions).T, X.shape)
                    
                    if use_contour:
                        levels = np.linspace(0, Z_pred.max(), 8)[1:] if Z_pred.max() > 0 else 5
                        ax.contour(X, Y, Z_pred, levels=levels, colors='blue',
                                  linewidths=2, alpha=0.8, linestyles='--')
                    else:
                        ax.contourf(X, Y, Z_pred, levels=15, cmap='Blues', alpha=alpha*0.8)
                        ax.contour(X, Y, Z_pred, levels=8, colors='darkblue',
                                  linewidths=1, alpha=0.5)
                except Exception as e:
                    if len(dp) > 0:
                        ax.scatter(dp[:, 0], dp[:, 1], c='blue', alpha=0.3, s=20, label='Pred')
            elif len(dp) > 0:
                ax.scatter(dp[:, 0], dp[:, 1], c='blue', alpha=0.3, s=20, label='Pred')
            
            self._draw_landing_court(ax)
            ax.set_xlim(0, 355)
            ax.set_ylim(0, 480)
            ax.set_aspect('equal')
            
            label = shot_labels[shot]
            n_pred, n_gt = len(dp), len(dg)
            ax.set_title(f'{label}\nGT:{n_gt} Pred:{n_pred}', fontsize=10)
            
            if idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='red', lw=2, label='GT KDE' if len(dg) >= 10 else 'GT Points'),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--' if use_contour else '-', 
                          label='Pred KDE' if len(dp) >= 10 else 'Pred Points')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

        for ax in axes[len(shot_order):]:
            ax.axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=120, bbox_inches='tight')
        plt.close()
    
    def plot_per_shot_landing_distribution(self, land_pred: np.ndarray, land_gt: np.ndarray,
                                           shot_pred: np.ndarray, shot_gt: np.ndarray,
                                           title: str, filename: str, use_heatmap: bool = True,
                                           use_contour: bool = False, alpha: float = 0.6,
                                           bins: int = 30, scatter_overlay: bool = False):
        """
        Plot per-shot-type landing distribution comparison.
        Creates a grid of subplots for each shot type.
        
        Args:
            land_pred: Predicted landing positions
            land_gt: Ground truth landing positions
            shot_pred: Predicted shot types
            shot_gt: Ground truth shot types
            title: Plot title
            filename: Output filename
            use_heatmap: If True, use 2D histogram heatmap instead of scatter points
            use_contour: If True, overlay contour lines on the heatmap
            alpha: Transparency for the heatmap
            bins: Number of bins for 2D histogram (default 30)
            scatter_overlay: If True, also show scatter points on top of heatmap
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in VALID_SHOT_TYPES}
        shot_labels['ALL'] = 'All Shots'
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        shot_order = list(VALID_SHOT_TYPES) + ['ALL']
        
        for idx, shot in enumerate(shot_order):
            ax = axes[idx]
            
            if shot == 'ALL':
                mask_pred = np.isin(shot_pred, VALID_SHOT_TYPES)
                mask_gt = np.isin(shot_gt, VALID_SHOT_TYPES)
            else:
                mask_pred = shot_pred == shot
                mask_gt = shot_gt == shot
            
            dp = self._denorm_landing(land_pred[mask_pred]) if np.sum(mask_pred) > 0 else np.zeros((0, 2))
            dg = self._denorm_landing(land_gt[mask_gt]) if np.sum(mask_gt) > 0 else np.zeros((0, 2))
            
            if use_heatmap and len(dg) >= 5:
                # Plot GT as 2D histogram heatmap
                try:
                    hist_gt, xedges, yedges = np.histogram2d(
                        dg[:, 0], dg[:, 1], 
                        bins=bins, 
                        range=[[0, 355], [0, 480]]
                    )
                    if hist_gt.sum() > 0:
                        hist_gt = hist_gt / hist_gt.sum()
                    # Smooth the histogram slightly for better visualization
                    from scipy.ndimage import gaussian_filter
                    hist_gt_smooth = gaussian_filter(hist_gt, sigma=1.0)
                    
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    vmax_gt = hist_gt_smooth.max() if hist_gt_smooth.max() > 0 else 1
                    
                    im_gt = ax.imshow(
                        hist_gt_smooth.T, 
                        extent=extent, 
                        origin='lower',
                        cmap='Reds', 
                        alpha=alpha,
                        aspect='equal',
                        vmin=0,
                        vmax=vmax_gt
                    )
                    
                    if use_contour and hist_gt_smooth.max() > 0:
                        ax.contour(
                            hist_gt_smooth.T,
                            extent=extent,
                            origin='lower',
                            colors='darkred',
                            linewidths=1.5,
                            alpha=0.7,
                            levels=5
                        )
                except Exception as e:
                    print(f"  Warning: Could not create GT heatmap for shot {shot}: {e}")
            
            if use_heatmap and len(dp) >= 5:
                # Plot Pred as 2D histogram heatmap with different colormap
                try:
                    hist_pred, xedges, yedges = np.histogram2d(
                        dp[:, 0], dp[:, 1], 
                        bins=bins, 
                        range=[[0, 355], [0, 480]]
                    )
                    if hist_pred.sum() > 0:
                        hist_pred = hist_pred / hist_pred.sum()
                    from scipy.ndimage import gaussian_filter
                    hist_pred_smooth = gaussian_filter(hist_pred, sigma=1.0)
                    
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    vmax_pred = hist_pred_smooth.max() if hist_pred_smooth.max() > 0 else 1
                    
                    # Use a blended approach: overlay with different alpha
                    im_pred = ax.imshow(
                        hist_pred_smooth.T, 
                        extent=extent, 
                        origin='lower',
                        cmap='Blues', 
                        alpha=alpha * 0.8,
                        aspect='equal',
                        vmin=0,
                        vmax=vmax_pred
                    )
                    
                    if use_contour and hist_pred_smooth.max() > 0:
                        ax.contour(
                            hist_pred_smooth.T,
                            extent=extent,
                            origin='lower',
                            colors='darkblue',
                            linewidths=1.5,
                            alpha=0.7,
                            levels=5
                        )
                except Exception as e:
                    print(f"  Warning: Could not create Pred heatmap for shot {shot}: {e}")
            
            # Optional: overlay scatter points
            if scatter_overlay or not use_heatmap:
                if len(dg) > 0:
                    ax.scatter(dg[:, 0], dg[:, 1], c='red', alpha=0.2, s=10, label='GT')
                if len(dp) > 0:
                    ax.scatter(dp[:, 0], dp[:, 1], c='blue', alpha=0.2, s=10, label='Pred')
            
            self._draw_landing_court(ax)
            ax.set_xlim(0, 355)
            ax.set_ylim(0, 480)
            ax.set_aspect('equal')
            
            label = shot_labels[shot]
            n_pred, n_gt = len(dp), len(dg)
            ax.set_title(f'{label}\nGT:{n_gt} Pred:{n_pred}', fontsize=10)
            
            # Add legend
            if idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', alpha=alpha, label='GT Distribution'),
                    Patch(facecolor='blue', alpha=alpha*0.8, label='Pred Distribution')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

        for ax in axes[len(shot_order):]:
            ax.axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=120, bbox_inches='tight')
        plt.close()
    
    def plot_conditional_heatmap(self, conditional_results: Dict, metric_name: str,
                                 title: str, filename: str):
        """
        Plot heatmap of conditional distribution similarities.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        per_shot = conditional_results.get('per_shot', {})
        
        # Extract JS values
        shot_types = list(range(1, 12))
        js_values = [per_shot.get(str(s), {}).get('js_divergence', 0) for s in shot_types]
        shot_labels = [INDEX2EN_SHOT.get(i, f'S{i}')[:10] for i in shot_types]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        colors = plt.cm.RdYlGn_r(np.array(js_values))
        bars = ax.bar(range(len(shot_types)), js_values, color=colors, edgecolor='black')
        
        ax.set_xticks(range(len(shot_types)))
        ax.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax.set_ylabel('JS Divergence')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=conditional_results.get('weighted_average', 0), 
                  color='red', linestyle='--', linewidth=2, label='Weighted Avg')
        ax.legend()
        ax.set_ylim(0, max(js_values + [0.5]) * 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_transition_heatmap(self, transition_results: Dict, title: str, filename: str):
        """Plot transition probability heatmap."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        per_prev = transition_results.get('per_prev_shot', {})
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shot_labels = [INDEX2EN_SHOT.get(i, f'S{i}')[:10] for i in range(1, 12)]
        js_values = [per_prev.get(str(s), {}).get('js_divergence', 0) for s in range(1, 12)]
        
        colors = plt.cm.RdYlGn_r(np.array(js_values))
        bars = ax.bar(range(11), js_values, color=colors, edgecolor='black')
        
        ax.set_xticks(range(11))
        ax.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax.set_ylabel('JS Divergence')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=transition_results.get('weighted_average', 0),
                  color='red', linestyle='--', linewidth=2, label='Weighted Avg')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # NEW: Actual Conditional Distribution Visualizations
    # =========================================================================
    
    def plot_conditional_landing_distributions(self, land_pred: np.ndarray, land_gt: np.ndarray,
                                                shot_pred: np.ndarray, shot_gt: np.ndarray,
                                                title: str, filename: str):
        """
        Plot actual conditional landing area distributions for each shot type.
        Shows P(Area | Shot Type = X) for both Pred and GT side-by-side.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Convert to area groups
        area_pred = np.array([coord2group(x, y) for x, y in land_pred])
        area_gt = np.array([coord2group(x, y) for x, y in land_gt])
        
        group_labels = GROUP_LABELS
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in VALID_SHOT_TYPES}
        shot_order = list(VALID_SHOT_TYPES)
        
        # Create 3x4 grid
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        for idx, shot in enumerate(shot_order):
            ax = axes[idx]
            
            # Filter by shot type
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            # Get area distributions for this shot type
            areas_pred = area_pred[mask_pred]
            areas_gt = area_gt[mask_gt]
            
            # Build histograms
            hist_pred, _ = np.histogram(areas_pred, bins=N_GROUPS, range=(0, N_GROUPS))
            hist_gt, _ = np.histogram(areas_gt, bins=N_GROUPS, range=(0, N_GROUPS))
            
            # Normalize to get probabilities
            if hist_pred.sum() > 0:
                hist_pred = hist_pred / hist_pred.sum()
            if hist_gt.sum() > 0:
                hist_gt = hist_gt / hist_gt.sum()
            
            # Plot side-by-side bars
            x = np.arange(N_GROUPS)
            width = 0.35
            
            bars1 = ax.bar(x - width/2, hist_pred, width, label='Pred', color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, hist_gt, width, label='GT', color='firebrick', alpha=0.8)
            
            ax.set_xlabel('Area Group')
            ax.set_ylabel('Probability')
            ax.set_title(f'{shot_labels[shot]}\nPred: {len(areas_pred)} shots, GT: {len(areas_gt)} shots', 
                        fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=6)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(max(hist_pred), max(hist_gt)) * 1.2 if len(hist_pred) > 0 else 1)
        
        for ax in axes[len(shot_order):]:
            ax.axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_conditional_move_distributions(self, move_pred: np.ndarray, move_gt: np.ndarray,
                                             shot_pred: np.ndarray, shot_gt: np.ndarray,
                                             title: str, filename: str):
        """
        Plot actual conditional move area distributions for each shot type.
        Shows P(Move Area | Shot Type = X) for both Pred and GT side-by-side.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Convert to area groups
        area_pred = np.array([coord2group(x, y) for x, y in move_pred])
        area_gt = np.array([coord2group(x, y) for x, y in move_gt])
        
        group_labels = GROUP_LABELS
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in VALID_SHOT_TYPES}
        shot_order = list(VALID_SHOT_TYPES)
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        for idx, shot in enumerate(shot_order):
            ax = axes[idx]
            
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            areas_pred = area_pred[mask_pred]
            areas_gt = area_gt[mask_gt]
            
            hist_pred, _ = np.histogram(areas_pred, bins=N_GROUPS, range=(0, N_GROUPS))
            hist_gt, _ = np.histogram(areas_gt, bins=N_GROUPS, range=(0, N_GROUPS))
            
            if hist_pred.sum() > 0:
                hist_pred = hist_pred / hist_pred.sum()
            if hist_gt.sum() > 0:
                hist_gt = hist_gt / hist_gt.sum()
            
            x = np.arange(N_GROUPS)
            width = 0.35
            
            ax.bar(x - width/2, hist_pred, width, label='Pred', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, hist_gt, width, label='GT', color='firebrick', alpha=0.8)
            
            ax.set_xlabel('Move Area Group')
            ax.set_ylabel('Probability')
            ax.set_title(f'{shot_labels[shot]}\nPred: {len(areas_pred)} shots, GT: {len(areas_gt)} shots',
                        fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=6)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(max(hist_pred), max(hist_gt)) * 1.2 if len(hist_pred) > 0 else 1)
        
        for ax in axes[len(shot_order):]:
            ax.axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_conditional_landing_heatmap(self, land_pred: np.ndarray, land_gt: np.ndarray,
                                          shot_pred: np.ndarray, shot_gt: np.ndarray,
                                          title: str, filename: str):
        """
        Plot conditional landing distributions as heatmap.
        Rows = Shot Types, Columns = Area Groups
        Shows the probability distribution P(Area | Shot Type).
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        area_pred = np.array([coord2group(x, y) for x, y in land_pred])
        area_gt = np.array([coord2group(x, y) for x, y in land_gt])
        
        # Build conditional distributions: P(Area | Shot Type), excluding Missed (11).
        shot_labels_short = [INDEX2EN_SHOT.get(i, f'S{i}')[:12] for i in VALID_SHOT_TYPES]
        group_labels = GROUP_LABELS
        
        # Create matrices: valid shot types x canonical area groups.
        mat_pred = np.zeros((len(VALID_SHOT_TYPES), N_GROUPS))
        mat_gt = np.zeros((len(VALID_SHOT_TYPES), N_GROUPS))
        
        for row_idx, shot in enumerate(VALID_SHOT_TYPES):
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            areas_pred = area_pred[mask_pred]
            areas_gt = area_gt[mask_gt]
            
            if len(areas_pred) > 0:
                hist, _ = np.histogram(areas_pred, bins=N_GROUPS, range=(0, N_GROUPS))
                mat_pred[row_idx] = hist / hist.sum() if hist.sum() > 0 else 0
            
            if len(areas_gt) > 0:
                hist, _ = np.histogram(areas_gt, bins=N_GROUPS, range=(0, N_GROUPS))
                mat_gt[row_idx] = hist / hist.sum() if hist.sum() > 0 else 0
        
        # Create side-by-side heatmaps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10))
        
        vmax = max(mat_pred.max(), mat_gt.max())
        if vmax <= 0:
            vmax = 1.0
        
        # Predicted
        im1 = ax1.imshow(mat_pred, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
        ax1.set_yticks(range(len(VALID_SHOT_TYPES)))
        ax1.set_yticklabels(shot_labels_short)
        ax1.set_xticks(range(N_GROUPS))
        ax1.set_xticklabels(group_labels, rotation=45, ha='right')
        ax1.set_xlabel('Landing Area Group')
        ax1.set_ylabel('Shot Type')
        ax1.set_title('Predicted P(Area | Shot Type)')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Ground Truth
        im2 = ax2.imshow(mat_gt, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)
        ax2.set_yticks(range(len(VALID_SHOT_TYPES)))
        ax2.set_yticklabels(shot_labels_short)
        ax2.set_xticks(range(N_GROUPS))
        ax2.set_xticklabels(group_labels, rotation=45, ha='right')
        ax2.set_xlabel('Landing Area Group')
        ax2.set_ylabel('Shot Type')
        ax2.set_title('Ground Truth P(Area | Shot Type)')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference (Pred - GT)
        diff = mat_pred - mat_gt
        im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        ax3.set_yticks(range(len(VALID_SHOT_TYPES)))
        ax3.set_yticklabels(shot_labels_short)
        ax3.set_xticks(range(N_GROUPS))
        ax3.set_xticklabels(group_labels, rotation=45, ha='right')
        ax3.set_xlabel('Landing Area Group')
        ax3.set_ylabel('Shot Type')
        ax3.set_title('Difference (Pred - GT)')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_conditional_mdc_bars(self, mdc_results: Dict, title: str, filename: str):
        """
        Plot Mean Distance from Center ratios for each shot type.
        Shows whether model predicts closer or farther from center than GT.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        per_shot = mdc_results.get('per_shot', {})
        
        shot_labels = [INDEX2EN_SHOT.get(i, f'S{i}')[:12] for i in range(1, 12)]
        ratios = [per_shot.get(str(s), {}).get('ratio', 0) for s in range(1, 12)]
        supports = [per_shot.get(str(s), {}).get('support', 0) for s in range(1, 12)]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Color by ratio: blue (<1 = closer), red (>1 = farther)
        colors = ['steelblue' if r < 1 else 'firebrick' for r in ratios]
        
        bars = ax.bar(range(11), ratios, color=colors, alpha=0.7, edgecolor='black')
        
        # Add support numbers on top of bars
        for i, (bar, support) in enumerate(zip(bars, supports)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={support}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Match')
        ax.axhline(y=mdc_results.get('weighted_average', 0), 
                  color='orange', linestyle='--', linewidth=2, label='Weighted Avg')
        
        ax.set_xticks(range(11))
        ax.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax.set_ylabel('MDC Ratio (Pred / GT)')
        ax.set_xlabel('Shot Type')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add annotation
        ax.text(0.02, 0.98, '< 1: Closer to center\n> 1: Farther from center',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_transition_matrices(self, shot_pred_seqs: List[List[int]], 
                                  shot_gt_seqs: List[List[int]],
                                  title: str, filename: str):
        """
        Plot actual transition matrices P(Next Shot | Previous Shot).
        Shows side-by-side heatmaps of transition probabilities.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Build transition matrices for modeled shot types, excluding Missed (11).
        shot_to_idx = {shot: idx for idx, shot in enumerate(VALID_SHOT_TYPES)}
        n_shots = len(VALID_SHOT_TYPES)
        T_pred = np.zeros((n_shots, n_shots))
        T_gt = np.zeros((n_shots, n_shots))
        
        for seq in shot_pred_seqs:
            for i in range(len(seq) - 1):
                prev, next_s = int(seq[i]), int(seq[i+1])
                if prev in shot_to_idx and next_s in shot_to_idx:
                    T_pred[shot_to_idx[prev], shot_to_idx[next_s]] += 1
        
        for seq in shot_gt_seqs:
            for i in range(len(seq) - 1):
                prev, next_s = int(seq[i]), int(seq[i+1])
                if prev in shot_to_idx and next_s in shot_to_idx:
                    T_gt[shot_to_idx[prev], shot_to_idx[next_s]] += 1
        
        # Normalize rows to get probabilities
        for i in range(n_shots):
            row_sum = T_pred[i].sum()
            if row_sum > 0:
                T_pred[i] = T_pred[i] / row_sum
            row_sum = T_gt[i].sum()
            if row_sum > 0:
                T_gt[i] = T_gt[i] / row_sum
        
        shot_labels = [INDEX2EN_SHOT.get(i, f'S{i}')[:10] for i in VALID_SHOT_TYPES]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        
        vmax = max(T_pred.max(), T_gt.max())
        if vmax <= 0:
            vmax = 1.0
        
        # Predicted transitions
        im1 = ax1.imshow(T_pred, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
        ax1.set_xticks(range(n_shots))
        ax1.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax1.set_yticks(range(n_shots))
        ax1.set_yticklabels(shot_labels)
        ax1.set_xlabel('Next Shot')
        ax1.set_ylabel('Previous Shot')
        ax1.set_title('Predicted P(Next | Previous)')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Ground Truth transitions
        im2 = ax2.imshow(T_gt, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)
        ax2.set_xticks(range(n_shots))
        ax2.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax2.set_yticks(range(n_shots))
        ax2.set_yticklabels(shot_labels)
        ax2.set_xlabel('Next Shot')
        ax2.set_ylabel('Previous Shot')
        ax2.set_title('Ground Truth P(Next | Previous)')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference
        diff = T_pred - T_gt
        im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        ax3.set_xticks(range(n_shots))
        ax3.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax3.set_yticks(range(n_shots))
        ax3.set_yticklabels(shot_labels)
        ax3.set_xlabel('Next Shot')
        ax3.set_ylabel('Previous Shot')
        ax3.set_title('Difference (Pred - GT)')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _state_conditioned_details(pred_by_state: Dict, gt_by_state: Dict, bins: int,
                                   min_support: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Build support and JS matrices for (prev shot, prev landing depth) states."""
        support = np.zeros((len(VALID_SHOT_TYPES), len(DEPTH_NAMES)), dtype=np.float64)
        js_matrix = np.full((len(VALID_SHOT_TYPES), len(DEPTH_NAMES)), np.nan, dtype=np.float64)
        details = []

        for state, gt_counts in gt_by_state.items():
            if len(state) != 2:
                continue
            prev_shot, prev_depth = state
            if prev_shot not in VALID_SHOT_TYPES or not (0 <= prev_depth < len(DEPTH_NAMES)):
                continue

            gt_support = sum(gt_counts.values())
            support[prev_shot - 1, prev_depth] = gt_support
            if gt_support < min_support:
                continue

            pred_counts = pred_by_state.get(state, {})
            p = np.array([pred_counts.get(i, 0) for i in range(bins)], dtype=np.float64)
            q = np.array([gt_counts.get(i, 0) for i in range(bins)], dtype=np.float64)
            js = js_divergence(p, q)
            if not math.isfinite(js):
                continue

            js_matrix[prev_shot - 1, prev_depth] = js
            details.append({
                'state': state,
                'js': js,
                'gt_support': gt_support,
                'pred_support': sum(pred_counts.values()),
                'pred_counts': pred_counts,
                'gt_counts': gt_counts,
            })

        return support, js_matrix, details

    def plot_state_conditioned_coverage(self, pred_by_state: Dict, gt_by_state: Dict,
                                        title: str, filename: str,
                                        min_support: int):
        """Plot GT support coverage for state-conditioned metrics."""
        if not MATPLOTLIB_AVAILABLE:
            return

        support, _, _ = self._state_conditioned_details(
            pred_by_state, gt_by_state, bins=1, min_support=min_support
        )
        pred_states = set(pred_by_state.keys())
        kept_missing = []
        for row_idx, shot in enumerate(VALID_SHOT_TYPES):
            for depth_idx in range(len(DEPTH_NAMES)):
                if support[row_idx, depth_idx] >= min_support and (shot, depth_idx) not in pred_states:
                    kept_missing.append((depth_idx, row_idx))

        fig, ax = plt.subplots(figsize=(16, 7))
        masked = np.ma.masked_where(support <= 0, support)
        im = ax.imshow(masked, cmap='YlGnBu', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Player Previous Landing Depth')
        ax.set_ylabel('Player Previous Shot')
        ax.set_xticks(range(len(DEPTH_NAMES)))
        ax.set_xticklabels([name.title() for name in DEPTH_NAMES])
        ax.set_yticks(range(len(VALID_SHOT_TYPES)))
        ax.set_yticklabels([INDEX2EN_SHOT.get(i, f'S{i}') for i in VALID_SHOT_TYPES])
        plt.colorbar(im, ax=ax, fraction=0.025, label='GT Support')

        if kept_missing:
            xs, ys = zip(*kept_missing)
            ax.scatter(xs, ys, marker='x', s=65, color='crimson', linewidths=1.6,
                       label='GT support kept, missing in Pred')
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_state_conditioned_js_heatmap(self, pred_by_state: Dict, gt_by_state: Dict,
                                          bins: int, title: str, filename: str,
                                          min_support: int):
        """Plot per-state JS for state-conditioned action distributions."""
        if not MATPLOTLIB_AVAILABLE:
            return

        _, js_matrix, _ = self._state_conditioned_details(
            pred_by_state, gt_by_state, bins=bins, min_support=min_support
        )
        fig, ax = plt.subplots(figsize=(16, 7))
        masked = np.ma.masked_invalid(js_matrix)
        im = ax.imshow(masked, cmap='magma_r', aspect='auto', vmin=0, vmax=max(0.5, np.nanmax(js_matrix) if np.isfinite(js_matrix).any() else 0.5))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Player Previous Landing Depth')
        ax.set_ylabel('Player Previous Shot')
        ax.set_xticks(range(len(DEPTH_NAMES)))
        ax.set_xticklabels([name.title() for name in DEPTH_NAMES])
        ax.set_yticks(range(len(VALID_SHOT_TYPES)))
        ax.set_yticklabels([INDEX2EN_SHOT.get(i, f'S{i}') for i in VALID_SHOT_TYPES])
        plt.colorbar(im, ax=ax, fraction=0.025, label='JS Divergence')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_state_conditioned_top_distributions(self, pred_by_state: Dict, gt_by_state: Dict,
                                                 bins: int, action_labels: List[str],
                                                 title: str, filename: str,
                                                 min_support: int, top_k: int = 6):
        """Plot Pred vs GT action distributions for top-JS state-conditioned cases."""
        if not MATPLOTLIB_AVAILABLE:
            return

        _, _, details = self._state_conditioned_details(
            pred_by_state, gt_by_state, bins=bins, min_support=min_support
        )
        details = sorted(details, key=lambda item: item['js'], reverse=True)[:top_k]
        if not details:
            return

        ncols = 2
        nrows = int(math.ceil(len(details) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        x = np.arange(bins)
        width = 0.42
        for idx, item in enumerate(details):
            ax = axes[idx]
            pred_counts = np.array([item['pred_counts'].get(i, 0) for i in range(bins)], dtype=np.float64)
            gt_counts = np.array([item['gt_counts'].get(i, 0) for i in range(bins)], dtype=np.float64)
            pred_probs = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else pred_counts
            gt_probs = gt_counts / gt_counts.sum() if gt_counts.sum() > 0 else gt_counts

            prev_shot, prev_depth = item['state']
            state_title = (
                f"Prev {INDEX2EN_SHOT.get(prev_shot, f'S{prev_shot}')} | "
                f"Land {DEPTH_NAMES[prev_depth].title()} | JS={item['js']:.3f} | "
                f"GT n={item['gt_support']} Pred n={item['pred_support']}"
            )
            ax.bar(x - width / 2, pred_probs, width, label='Pred', color='steelblue', alpha=0.85)
            ax.bar(x + width / 2, gt_probs, width, label='GT', color='firebrick', alpha=0.85)
            ax.set_title(state_title, fontsize=10, fontweight='bold')
            ax.set_ylabel('Probability')
            ax.set_xticks(x)
            ax.set_xticklabels(action_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.25)
            if idx == 0:
                ax.legend()

        for ax in axes[len(details):]:
            ax.axis('off')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_trajectories(trajectory_file: str, 
                          output_dir: str,
                          metrics: List[str] = None,
                          visualize: bool = True,
                          target_player_only: bool = True,
                          skip_n: int = 0) -> Dict:
    """
    Main evaluation function.
    
    Args:
        trajectory_file: Path to trajectory pickle file
        output_dir: Directory for output files
        metrics: List of metrics to compute (None = all)
        visualize: Whether to generate visualizations
        target_player_only: If True, only evaluate target player's shots
        skip_n: Number of initial steps to skip per trajectory
    
    Returns:
        Dict with all computed metrics
    """
    print("="*80)
    print("Trajectory Evaluation")
    print("="*80)
    print(f"Input:  {trajectory_file}")
    print(f"Output: {output_dir}")
    print(f"Target Player Only: {target_player_only}")
    print()
    
    # Load trajectories
    print("Loading trajectories...")
    trajectories = load_trajectories(trajectory_file)
    print(f"  Loaded {len(trajectories)} rallies")
    
    # Extract sequences
    print("Extracting sequences...")
    data = extract_sequences(trajectories, target_player_only=target_player_only, skip_n=skip_n)
    target_player = data.get('target_player', 'Unknown')
    print(f"  Target Player: {target_player}")
    print(f"  Total GT shots: {len(data['all_shot_gt'])}")
    print(f"  Total Pred shots: {len(data['all_shot_pred'])}")
    print()
    
    # Initialize results
    results = {
        'trajectory_file': trajectory_file,
        'target_player': target_player,
        'target_player_only': target_player_only,
        'num_rallies': len(trajectories),
        'num_shots_gt': len(data['all_shot_gt']),
        'num_shots_pred': len(data['all_shot_pred']),
    }
    
    metrics_to_compute = metrics or ['all']
    compute_all = 'all' in metrics_to_compute
    
    # =========================================================================
    # Sequence Metrics
    # =========================================================================
    if compute_all or any(m in ['dtw', 'edit_distance', 'lcs_ratio', 'shot_similarity', 'sequence'] for m in metrics_to_compute):
        print("Computing Sequence Metrics...")
        seq_metrics = SequenceMetrics()
        
        results['sequence_metrics'] = {}
        
        # DTW
        if compute_all or 'dtw' in metrics_to_compute:
            print("  - DTW...")
            dtw_results = seq_metrics.compute_dtw(
                data['land_pred_seqs'], data['land_gt_seqs'],
                data['move_pred_seqs'], data['move_gt_seqs']
            )
            results['sequence_metrics']['dtw'] = dtw_results
            print(f"    Land DTW: {dtw_results['land_dtw']:.4f}")
            print(f"    Move DTW: {dtw_results['move_dtw']:.4f}")
        
        # Edit Distance
        if compute_all or 'edit_distance' in metrics_to_compute:
            print("  - Edit Distance...")
            edit_results = seq_metrics.compute_edit_distance(
                data['shot_pred_seqs'], data['shot_gt_seqs']
            )
            results['sequence_metrics']['edit_distance'] = edit_results
            print(f"    Edit Distance: {edit_results['edit_distance']:.4f}")

        # LCS Ratio
        if compute_all or 'lcs_ratio' in metrics_to_compute:
            print("  - LCS Ratio...")
            lcs_results = seq_metrics.compute_lcs_ratio(
                data['shot_pred_seqs'], data['shot_gt_seqs']
            )
            results['sequence_metrics']['lcs_ratio'] = lcs_results
            print(f"    LCS Ratio: {lcs_results['lcs_ratio']:.4f}")

        # Shot n-gram and Markov transition similarity
        if compute_all or 'shot_similarity' in metrics_to_compute:
            print("  - Shot n-gram similarity...")
            bigram_results = seq_metrics.compute_ngram_similarity(
                data['shot_pred_seqs'], data['shot_gt_seqs'], n=2
            )
            trigram_results = seq_metrics.compute_ngram_similarity(
                data['shot_pred_seqs'], data['shot_gt_seqs'], n=3
            )
            transition_results = seq_metrics.compute_transition_js(
                data['shot_pred_seqs'], data['shot_gt_seqs']
            )
            results['sequence_metrics']['shot_bigram'] = bigram_results
            results['sequence_metrics']['shot_trigram'] = trigram_results
            results['sequence_metrics']['shot_transition'] = transition_results
            print(f"    Bigram JS: {bigram_results['shot_bigram_js']:.4f}")
            print(f"    Bigram F1: {bigram_results['shot_bigram_f1']:.4f}")
            print(f"    Trigram JS: {trigram_results['shot_trigram_js']:.4f}")
            print(f"    Trigram F1: {trigram_results['shot_trigram_f1']:.4f}")
            print(f"    Transition JS: {transition_results['shot_transition_js']:.4f}")
        
        print()
    
    # =========================================================================
    # Distribution Metrics
    # =========================================================================
    if compute_all or 'distribution' in metrics_to_compute:
        print("Computing Distribution Metrics...")
        results['distribution_metrics'] = distribution_metrics_for_trajectories(
            trajectories, target_player_only=target_player_only
        )

        print(f"  Shot-conditioned Landing JS: {results['distribution_metrics']['shot_conditioned_landing_js']:.4f}")
        print(f"  Shot Dist JS: {results['distribution_metrics']['shot_distribution_js']:.4f}")
        print(f"  Rally Length JS: {results['distribution_metrics']['rally_length_js']:.4f}")
        print(f"  State-conditioned Move JS: {results['distribution_metrics']['state_conditioned_move_js']:.4f}")
        print(f"  State-conditioned Shot JS: {results['distribution_metrics']['state_conditioned_shot_js']:.4f}")
        print(f"  State-conditioned Landing JS: {results['distribution_metrics']['state_conditioned_landing_js']:.4f}")
        print()
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    if visualize and MATPLOTLIB_AVAILABLE:
        print("Generating visualizations...")
        visualizer = Visualizer(output_dir)

        valid_land_pred = data['all_land_pred'][np.isin(data['all_shot_pred'], VALID_SHOT_TYPES)]
        valid_land_gt = data['all_land_gt'][np.isin(data['all_shot_gt'], VALID_SHOT_TYPES)]
        valid_shot_pred_count = int(np.sum(np.isin(data['all_shot_pred'], VALID_SHOT_TYPES)))
        valid_shot_gt_count = int(np.sum(np.isin(data['all_shot_gt'], VALID_SHOT_TYPES)))
        non_missed_lengths_pred = non_missed_rally_lengths(trajectories, 'pred_sequence')
        non_missed_lengths_gt = non_missed_rally_lengths(trajectories, 'gt_sequence')

        # Rally length distribution, counting non-missed shots only.
        print("  - Rally length distribution (non-missed shots)...")
        visualizer.plot_rally_length_distribution(
            non_missed_lengths_pred, non_missed_lengths_gt,
            f'{target_player} — Rally Length Distribution (Non-Missed Shots)',
            'rally_length_distribution.png',
            normalize=True
        )
        
        # Landing KDE, excluding missed-shot landing coordinates.
        print("  - Landing distribution KDE (non-missed shots)...")
        visualizer.plot_landing_kde(
            valid_land_pred, valid_land_gt,
            f'{target_player} — Landing Position Distribution (Non-Missed Shots)',
            'landing_kde.png'
        )
        
        # Shot distribution, normalized and excluding Missed (11).
        print("  - Shot type distribution (normalized, non-missed shots)...")
        visualizer.plot_shot_distribution(
            data['all_shot_pred'], data['all_shot_gt'],
            f'{target_player} — Shot Type Distribution (Non-Missed, Normalized)',
            'shot_distribution.png',
            normalize=True
        )

        # Shot-conditioned landing KDE panels on court, matching the overall landing KDE style.
        print("  - Shot-conditioned landing KDE panels...")
        visualizer.plot_shot_conditioned_landing_kde(
            data['all_land_pred'], data['all_land_gt'],
            data['all_shot_pred'], data['all_shot_gt'],
            f'{target_player} — Shot-Conditioned Landing KDE (Non-Missed)',
            'shot_conditioned_landing_kde.png'
        )

        # Shot transition matrix is useful for checking tactical sequence behavior.
        print("  - Shot transition matrices...")
        visualizer.plot_transition_matrices(
            data['shot_pred_seqs'], data['shot_gt_seqs'],
            f'{target_player} — Shot Transition Matrix (Non-Missed)',
            'shot_transition_matrices.png'
        )

        # State-conditioned diagnostics for player previous shot + landing area states.
        print("  - State-conditioned shot/landing distributions...")
        min_state_support = DistributionMetrics.MIN_STATE_SUPPORT
        state_shot_pred = DistributionMetrics._state_action_counts(
            trajectories, 'pred_sequence', 'shot', target_player_only
        )
        state_shot_gt = DistributionMetrics._state_action_counts(
            trajectories, 'gt_sequence', 'shot', target_player_only
        )
        state_land_pred = DistributionMetrics._state_action_counts(
            trajectories, 'pred_sequence', 'landing', target_player_only
        )
        state_land_gt = DistributionMetrics._state_action_counts(
            trajectories, 'gt_sequence', 'landing', target_player_only
        )
        shot_action_labels = [INDEX2EN_SHOT.get(i, f'S{i}') for i in VALID_SHOT_TYPES]
        area_action_labels = GROUP_LABELS[:N_VALID_GROUPS]

        visualizer.plot_state_conditioned_coverage(
            state_land_pred, state_land_gt,
            f'{target_player} — State Coverage: Previous Shot + Previous Landing Depth',
            'state_conditioned_coverage.png',
            min_support=min_state_support,
        )
        visualizer.plot_state_conditioned_js_heatmap(
            state_shot_pred, state_shot_gt,
            bins=len(VALID_SHOT_TYPES),
            title=f'{target_player} — State-Conditioned Shot JS',
            filename='state_conditioned_shot_js.png',
            min_support=min_state_support,
        )
        visualizer.plot_state_conditioned_js_heatmap(
            state_land_pred, state_land_gt,
            bins=N_VALID_GROUPS,
            title=f'{target_player} — State-Conditioned Landing JS',
            filename='state_conditioned_landing_js.png',
            min_support=min_state_support,
        )
        visualizer.plot_state_conditioned_top_distributions(
            state_shot_pred, state_shot_gt,
            bins=len(VALID_SHOT_TYPES),
            action_labels=shot_action_labels,
            title=f'{target_player} — Top State-Conditioned Shot Differences',
            filename='state_conditioned_shot_top_states.png',
            min_support=min_state_support,
        )
        visualizer.plot_state_conditioned_top_distributions(
            state_land_pred, state_land_gt,
            bins=N_VALID_GROUPS,
            action_labels=area_action_labels,
            title=f'{target_player} — Top State-Conditioned Landing Differences',
            filename='state_conditioned_landing_top_states.png',
            min_support=min_state_support,
        )

        print(f"  Non-missed shots used: Pred={valid_shot_pred_count}, GT={valid_shot_gt_count}")
        
        print()
    
    # =========================================================================
    # Save Results
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")
    
    # Summary CSV
    summary_data = {
        'metric': [],
        'value': [],
    }
    
    if 'sequence_metrics' in results:
        if 'dtw' in results['sequence_metrics']:
            summary_data['metric'].append('land_dtw')
            summary_data['value'].append(results['sequence_metrics']['dtw']['land_dtw'])
            summary_data['metric'].append('move_dtw')
            summary_data['value'].append(results['sequence_metrics']['dtw']['move_dtw'])
        if 'edit_distance' in results['sequence_metrics']:
            summary_data['metric'].append('edit_distance')
            summary_data['value'].append(results['sequence_metrics']['edit_distance']['edit_distance'])
        if 'lcs_ratio' in results['sequence_metrics']:
            summary_data['metric'].append('lcs_ratio')
            summary_data['value'].append(results['sequence_metrics']['lcs_ratio']['lcs_ratio'])
        if 'shot_bigram' in results['sequence_metrics']:
            summary_data['metric'].append('shot_bigram_js')
            summary_data['value'].append(results['sequence_metrics']['shot_bigram']['shot_bigram_js'])
            summary_data['metric'].append('shot_bigram_f1')
            summary_data['value'].append(results['sequence_metrics']['shot_bigram']['shot_bigram_f1'])
        if 'shot_trigram' in results['sequence_metrics']:
            summary_data['metric'].append('shot_trigram_js')
            summary_data['value'].append(results['sequence_metrics']['shot_trigram']['shot_trigram_js'])
            summary_data['metric'].append('shot_trigram_f1')
            summary_data['value'].append(results['sequence_metrics']['shot_trigram']['shot_trigram_f1'])
        if 'shot_transition' in results['sequence_metrics']:
            summary_data['metric'].append('shot_transition_js')
            summary_data['value'].append(results['sequence_metrics']['shot_transition']['shot_transition_js'])
    
    if 'distribution_metrics' in results:
        for metric_name in DISTRIBUTION_METRIC_COLUMNS:
            summary_data['metric'].append(metric_name)
            summary_data['value'].append(results['distribution_metrics'].get(metric_name, float('nan')))
    
    import csv
    csv_file = os.path.join(output_dir, 'evaluation_summary.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for m, v in zip(summary_data['metric'], summary_data['value']):
            writer.writerow([m, f'{v:.6f}' if isinstance(v, float) else v])
    print(f"Summary saved to: {csv_file}")
    
    print()
    print("="*80)
    print("Evaluation Complete!")
    print("="*80)
    
    return results


SEQUENCE_METRIC_COLUMNS = [
    'land_dtw', 'move_dtw', 'lcs_ratio',
]

RALLY_LEVEL_COLUMNS = [
    'trajectory_file', 'target_player', 'generation', 'num_matches', 'num_rallies',
    'num_gt_shots', 'num_pred_shots', 'early_end_rate',
] + [f'mean_{column}' for column in SEQUENCE_METRIC_COLUMNS]

RALLY_SKIP_LEVEL_COLUMNS = RALLY_LEVEL_COLUMNS

DISTRIBUTION_METRIC_COLUMNS = [
    'shot_conditioned_landing_js', 'shot_distribution_js', 'rally_length_js',
    'state_conditioned_move_js', 'state_conditioned_shot_js', 'state_conditioned_landing_js',
]

MATCH_LEVEL_COLUMNS = [
    'trajectory_file', 'target_player', 'generation', 'num_matches', 'num_rallies',
    'num_gt_shots', 'num_pred_shots', 'early_end_rate',
] + [f'match_avg_{column}' for column in DISTRIBUTION_METRIC_COLUMNS] + [
    f'rally_weighted_{column}' for column in DISTRIBUTION_METRIC_COLUMNS
]

SUMMARY_METRIC_COLUMNS = [
    *SEQUENCE_METRIC_COLUMNS,
    *DISTRIBUTION_METRIC_COLUMNS,
]

EXPERIMENT_SUMMARY_COLUMNS = [
    'average_type', 'num_trajectory_files', 'num_matches', 'num_rallies',
    'num_gt_shots', 'num_pred_shots',
] + [f'mean_{column}' for column in SUMMARY_METRIC_COLUMNS]


def evaluate_trajectory_directory(trajectory_dir: str,
                                  output_dir: str,
                                  latest_only: bool = False,
                                  target_player_only: bool = True,
                                  print_max_rows: int = 50) -> Dict:
    """Evaluate every trajectory pickle in a directory as one experiment."""
    print('=' * 80)
    print('Experiment Trajectory Evaluation')
    print('=' * 80)
    print(f'Input dir:  {trajectory_dir}')
    print(f'Output dir: {output_dir}')
    print(f'Latest only: {latest_only}')
    print(f'Target Player Only: {target_player_only}')
    print()

    trajectory_files = discover_trajectory_files(trajectory_dir, latest_only=latest_only)
    if not trajectory_files:
        raise ValueError(f'No trajectory pickle files found under: {trajectory_dir}')

    print(f'Found {len(trajectory_files)} trajectory files')

    # Determine skip_n: 1 for target_player_only, 2 for both players
    skip_n = 1 if target_player_only else 2
    print(f'Skip first {skip_n} shot(s) per trajectory for sequence metrics (skip mode)')
    print()

    rally_rows = []
    rally_skip_rows = []
    match_groups = defaultdict(list)
    match_sequence_values = defaultdict(lambda: defaultdict(list))
    match_sequence_values_skip = defaultdict(lambda: defaultdict(list))

    for trajectory_file in trajectory_files:
        trajectories = load_trajectories(trajectory_file)
        parsed = parse_trajectory_filename(trajectory_file)
        trajectory_name = os.path.basename(trajectory_file)
        generation = parsed['generation']

        for traj in trajectories:
            target_player = traj.get('target_player', '')
            match_id = traj.get('match_id', '')
            key = (trajectory_name, target_player, generation, match_id)
            match_groups[key].append(traj)

            sequence_values = sequence_metrics_for_trajectories([traj], target_player_only)
            row = {
                'trajectory_file': trajectory_name,
                'target_player': target_player,
                'generation': generation,
                'match_id': match_id,
                'set': traj.get('set', ''),
                'rally': traj.get('rally', ''),
                'opponent_player': traj.get('opponent_player', ''),
                'num_gt_shots': len(traj.get('gt_sequence', [])),
                'num_pred_shots': len(traj.get('pred_sequence', [])),
                'simulation_ended_early': bool(traj.get('simulation_ended_early', False)),
                **sequence_values,
            }
            rally_rows.append(row)
            for metric_name in SEQUENCE_METRIC_COLUMNS:
                match_sequence_values[key][metric_name].append(row[metric_name])

            # Skip-first-shot version
            sequence_values_skip = sequence_metrics_for_trajectories([traj], target_player_only, skip_n=skip_n)
            # Get shot counts after skipping
            data_skip = extract_sequences([traj], target_player_only=target_player_only, skip_n=skip_n)
            row_skip = {
                'trajectory_file': trajectory_name,
                'target_player': target_player,
                'generation': generation,
                'match_id': match_id,
                'set': traj.get('set', ''),
                'rally': traj.get('rally', ''),
                'opponent_player': traj.get('opponent_player', ''),
                'num_gt_shots': len(data_skip['all_shot_gt']),
                'num_pred_shots': len(data_skip['all_shot_pred']),
                'simulation_ended_early': bool(traj.get('simulation_ended_early', False)),
                **sequence_values_skip,
            }
            rally_skip_rows.append(row_skip)
            for metric_name in SEQUENCE_METRIC_COLUMNS:
                match_sequence_values_skip[key][metric_name].append(row_skip[metric_name])

    match_rows = []
    for key, trajectories in sorted(match_groups.items(), key=lambda item: item[0]):
        trajectory_name, target_player, generation, match_id = key
        data = extract_sequences(trajectories, target_player_only=target_player_only)
        distribution_values = distribution_metrics_for_trajectories(trajectories, target_player_only)
        early_end_count = sum(1 for traj in trajectories if traj.get('simulation_ended_early', False))

        match_rows.append({
            'trajectory_file': trajectory_name,
            'target_player': target_player,
            'generation': generation,
            'match_id': match_id,
            'num_rallies': len(trajectories),
            'num_gt_shots': int(len(data['all_shot_gt'])),
            'num_pred_shots': int(len(data['all_shot_pred'])),
            'early_end_rate': early_end_count / max(len(trajectories), 1),
            **distribution_values,
        })

    match_rows_by_key = defaultdict(list)
    for match_row in match_rows:
        key = (
            match_row['trajectory_file'],
            match_row['target_player'],
            match_row['generation'],
        )
        match_rows_by_key[key].append(match_row)

    rally_rows_by_key = defaultdict(list)
    for rally_row in rally_rows:
        key = (
            rally_row['trajectory_file'],
            rally_row['target_player'],
            rally_row['generation'],
        )
        rally_rows_by_key[key].append(rally_row)

    rally_skip_rows_by_key = defaultdict(list)
    for rally_skip_row in rally_skip_rows:
        key = (
            rally_skip_row['trajectory_file'],
            rally_skip_row['target_player'],
            rally_skip_row['generation'],
        )
        rally_skip_rows_by_key[key].append(rally_skip_row)

    player_rally_rows = []
    for key in sorted(rally_rows_by_key.keys()):
        trajectory_name, target_player, generation = key
        rows = rally_rows_by_key[key]
        related_match_rows = match_rows_by_key.get(key, [])
        num_rallies = len(rows)
        early_end_count = sum(1 for row in rows if row.get('simulation_ended_early'))

        player_rally_rows.append({
            'trajectory_file': trajectory_name,
            'target_player': target_player,
            'generation': generation,
            'num_matches': len({row.get('match_id') for row in rows}),
            'num_rallies': num_rallies,
            'num_gt_shots': sum(int(row.get('num_gt_shots', 0)) for row in related_match_rows),
            'num_pred_shots': sum(int(row.get('num_pred_shots', 0)) for row in related_match_rows),
            'early_end_rate': early_end_count / max(num_rallies, 1),
            **{
                f'mean_{metric_name}': finite_mean([row.get(metric_name) for row in rows])
                for metric_name in SEQUENCE_METRIC_COLUMNS
            },
        })

    player_rally_skip_rows = []
    for key in sorted(rally_skip_rows_by_key.keys()):
        trajectory_name, target_player, generation = key
        rows = rally_skip_rows_by_key[key]
        related_match_rows = match_rows_by_key.get(key, [])
        num_rallies = len(rows)
        early_end_count = sum(1 for row in rows if row.get('simulation_ended_early'))

        player_rally_skip_rows.append({
            'trajectory_file': trajectory_name,
            'target_player': target_player,
            'generation': generation,
            'num_matches': len({row.get('match_id') for row in rows}),
            'num_rallies': num_rallies,
            'num_gt_shots': sum(int(row.get('num_gt_shots', 0)) for row in related_match_rows),
            'num_pred_shots': sum(int(row.get('num_pred_shots', 0)) for row in related_match_rows),
            'early_end_rate': early_end_count / max(num_rallies, 1),
            **{
                f'mean_{metric_name}': finite_mean([row.get(metric_name) for row in rows])
                for metric_name in SEQUENCE_METRIC_COLUMNS
            },
        })

    player_match_rows = []
    for key in sorted(match_rows_by_key.keys()):
        trajectory_name, target_player, generation = key
        rows = match_rows_by_key[key]
        num_rallies = sum(int(row.get('num_rallies', 0)) for row in rows)
        player_row = {
            'trajectory_file': trajectory_name,
            'target_player': target_player,
            'generation': generation,
            'num_matches': len(rows),
            'num_rallies': num_rallies,
            'num_gt_shots': sum(int(row.get('num_gt_shots', 0)) for row in rows),
            'num_pred_shots': sum(int(row.get('num_pred_shots', 0)) for row in rows),
            'early_end_rate': finite_mean(
                [row.get('early_end_rate') for row in rows],
                weights=[row.get('num_rallies', 0) for row in rows],
            ),
        }

        weights = [row.get('num_rallies', 0) for row in rows]
        for metric_name in DISTRIBUTION_METRIC_COLUMNS:
            player_row[f'match_avg_{metric_name}'] = finite_mean(
                [row.get(metric_name) for row in rows]
            )
            player_row[f'rally_weighted_{metric_name}'] = finite_mean(
                [row.get(metric_name) for row in rows],
                weights=weights,
            )

        player_match_rows.append(player_row)

    sequence_match_means = {
        metric_name: [
            finite_mean(match_sequence_values[key][metric_name])
            for key in sorted(match_groups.keys())
        ]
        for metric_name in SEQUENCE_METRIC_COLUMNS
    }
    sequence_match_means_skip = {
        metric_name: [
            finite_mean(match_sequence_values_skip[key][metric_name])
            for key in sorted(match_groups.keys())
        ]
        for metric_name in SEQUENCE_METRIC_COLUMNS
    }
    match_weights = [row['num_rallies'] for row in match_rows]

    summary_rows = []
    summary_rows_skip = []
    for average_type in ['match_average', 'rally_weighted_average']:
        row = {
            'average_type': average_type,
            'num_trajectory_files': len(trajectory_files),
            'num_matches': len(match_rows),
            'num_rallies': len(rally_rows),
            'num_gt_shots': sum(int(match_row['num_gt_shots']) for match_row in match_rows),
            'num_pred_shots': sum(int(match_row['num_pred_shots']) for match_row in match_rows),
        }
        row_skip = {
            'average_type': average_type,
            'num_trajectory_files': len(trajectory_files),
            'num_matches': len(match_rows),
            'num_rallies': len(rally_skip_rows),
            'num_gt_shots': sum(int(row.get('num_gt_shots', 0)) for row in rally_skip_rows),
            'num_pred_shots': sum(int(row.get('num_pred_shots', 0)) for row in rally_skip_rows),
        }

        if average_type == 'match_average':
            for metric_name in SEQUENCE_METRIC_COLUMNS:
                row[f'mean_{metric_name}'] = finite_mean(sequence_match_means[metric_name])
                row_skip[f'mean_{metric_name}'] = finite_mean(sequence_match_means_skip[metric_name])
            for metric_name in SUMMARY_METRIC_COLUMNS[len(SEQUENCE_METRIC_COLUMNS):]:
                row[f'mean_{metric_name}'] = finite_mean([match_row.get(metric_name) for match_row in match_rows])
                row_skip[f'mean_{metric_name}'] = finite_mean([match_row.get(metric_name) for match_row in match_rows])
        else:
            for metric_name in SEQUENCE_METRIC_COLUMNS:
                row[f'mean_{metric_name}'] = finite_mean([rally_row.get(metric_name) for rally_row in rally_rows])
                row_skip[f'mean_{metric_name}'] = finite_mean([rally_row.get(metric_name) for rally_row in rally_skip_rows])
            for metric_name in SUMMARY_METRIC_COLUMNS[len(SEQUENCE_METRIC_COLUMNS):]:
                row[f'mean_{metric_name}'] = finite_mean(
                    [match_row.get(metric_name) for match_row in match_rows],
                    weights=match_weights,
                )
                row_skip[f'mean_{metric_name}'] = finite_mean(
                    [match_row.get(metric_name) for match_row in match_rows],
                    weights=match_weights,
                )

        summary_rows.append(row)
        summary_rows_skip.append(row_skip)

    os.makedirs(output_dir, exist_ok=True)
    rally_csv = os.path.join(output_dir, 'rally_level_metrics.csv')
    match_csv = os.path.join(output_dir, 'match_level_metrics.csv')
    rally_skip_csv = os.path.join(output_dir, 'rally_skip_level_metrics.csv')
    summary_csv = os.path.join(output_dir, 'experiment_summary.csv')
    rally_table_path = os.path.join(output_dir, 'rally_level_metrics_table.txt')
    match_table_path = os.path.join(output_dir, 'match_level_metrics_table.txt')
    rally_skip_table_path = os.path.join(output_dir, 'rally_skip_level_metrics_table.txt')
    json_path = os.path.join(output_dir, 'experiment_metrics.json')

    write_csv_rows(rally_csv, player_rally_rows, RALLY_LEVEL_COLUMNS)
    write_csv_rows(match_csv, player_match_rows, MATCH_LEVEL_COLUMNS)
    write_csv_rows(rally_skip_csv, player_rally_skip_rows, RALLY_SKIP_LEVEL_COLUMNS)
    write_csv_rows(summary_csv, summary_rows, EXPERIMENT_SUMMARY_COLUMNS)

    rally_metric_specs = [
        ('LandDTW', 'mean_land_dtw', 'mean_land_dtw'),
        ('MoveDTW', 'mean_move_dtw', 'mean_move_dtw'),
        ('Shot LCS Ratio', 'mean_lcs_ratio', 'mean_lcs_ratio'),
    ]
    match_metric_specs = [
        ('ShotLand', 'match_avg_shot_conditioned_landing_js', 'mean_shot_conditioned_landing_js'),
        ('ShotJS', 'match_avg_shot_distribution_js', 'mean_shot_distribution_js'),
        ('LenJS', 'match_avg_rally_length_js', 'mean_rally_length_js'),
        ('StateMove', 'match_avg_state_conditioned_move_js', 'mean_state_conditioned_move_js'),
        ('StateShot', 'match_avg_state_conditioned_shot_js', 'mean_state_conditioned_shot_js'),
        ('StateLand', 'match_avg_state_conditioned_landing_js', 'mean_state_conditioned_landing_js'),
    ]

    agent_label = infer_agent_label_from_path(trajectory_dir)
    rally_table = build_player_metric_table(
        f'{agent_label} Agent Rally-Level Metrics By Player (sequence)',
        player_rally_rows,
        rally_metric_specs,
        summary_rows,
        max_rows=print_max_rows,
    )
    match_table = build_player_metric_table(
        f'{agent_label} Agent Match-Level Metrics By Player (distribution)',
        player_match_rows,
        match_metric_specs,
        summary_rows,
        max_rows=print_max_rows,
    )
    rally_skip_table = build_player_metric_table(
        f'{agent_label} Agent Rally-Skip-Level Metrics By Player (sequence, skip first {skip_n})',
        player_rally_skip_rows,
        rally_metric_specs,
        summary_rows_skip,
        max_rows=print_max_rows,
    )

    with open(rally_table_path, 'w') as f:
        f.write(rally_table + '\n')
    with open(match_table_path, 'w') as f:
        f.write(match_table + '\n')
    with open(rally_skip_table_path, 'w') as f:
        f.write(rally_skip_table + '\n')

    results = {
        'trajectory_dir': trajectory_dir,
        'latest_only': latest_only,
        'target_player_only': target_player_only,
        'trajectory_files': trajectory_files,
        'rally_level_metrics': player_rally_rows,
        'match_level_metrics': player_match_rows,
        'rally_skip_level_metrics': player_rally_skip_rows,
        'experiment_summary': summary_rows,
        'rally_level_table': rally_table_path,
        'match_level_table': match_table_path,
        'rally_skip_level_table': rally_skip_table_path,
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(rally_table)
    print(match_table)
    print(rally_skip_table)

    print()
    print(f'Rally-level CSV: {rally_csv}')
    print(f'Match-level CSV: {match_csv}')
    print(f'Rally-skip-level CSV: {rally_skip_csv}')
    print(f'Experiment summary CSV: {summary_csv}')
    print(f'Rally-level table: {rally_table_path}')
    print(f'Match-level table: {match_table_path}')
    print(f'Rally-skip-level table: {rally_skip_table_path}')
    print(f'Experiment metrics JSON: {json_path}')
    print('=' * 80)
    print('Experiment Evaluation Complete!')
    print('=' * 80)

    return results


BEST_OF_RALLY_DETAIL_COLUMNS = [
    'target_player', 'match_id', 'set', 'rally', 'opponent_player',
    'num_generations', 'best_generation', 'best_trajectory_file',
    'land_dtw', 'move_dtw', 'lcs_ratio', 'best_score',
    'land_dtw_skip', 'move_dtw_skip', 'lcs_ratio_skip',
    'num_gt_shots', 'num_pred_shots', 'simulation_ended_early',
]

BEST_OF_MATCH_DETAIL_COLUMNS = [
    'target_player', 'match_id', 'num_rallies', 'num_gt_shots', 'num_pred_shots',
    'early_end_rate', 'mean_land_dtw', 'mean_move_dtw', 'mean_lcs_ratio',
] + DISTRIBUTION_METRIC_COLUMNS

BEST_OF_SUMMARY_COLUMNS = [
    'level', 'num_players', 'num_matches', 'num_rallies',
    'num_gt_shots', 'num_pred_shots', 'early_end_rate',
    'land_dtw', 'move_dtw', 'lcs_ratio', 'best_score',
    'land_dtw_skip', 'move_dtw_skip', 'lcs_ratio_skip',
] + DISTRIBUTION_METRIC_COLUMNS


def best_of_score(sequence_values: Dict[str, Any]) -> float:
    """Best-of selection score: mean of move_dtw, land_dtw, (1 - lcs_ratio).

    Lower is better. Non-finite components are excluded; returns NaN when
    every component is non-finite.
    """
    lcs_ratio = sequence_values.get('lcs_ratio')
    try:
        lcs_component = 1.0 - float(lcs_ratio)
    except (TypeError, ValueError):
        lcs_component = float('nan')

    return finite_mean([
        sequence_values.get('move_dtw'),
        sequence_values.get('land_dtw'),
        lcs_component,
    ])


def build_best_of_table(title: str, row: Dict, metric_specs: List[Tuple[str, str]]) -> str:
    """Build a single-row fixed-width table for best-of summary results."""
    labels = [label for label, _ in metric_specs]
    values = [format_fixed_table_value(row.get(key, '')) for _, key in metric_specs]
    widths = [max(len(label), len(value)) for label, value in zip(labels, values)]
    header = ' '.join(label.ljust(width) for label, width in zip(labels, widths))
    value_line = ' '.join(value.ljust(width) for value, width in zip(values, widths))
    table_width = max(len(header), len(title))
    lines = [
        '',
        '=' * table_width,
        title,
        '=' * table_width,
        header,
        '-' * table_width,
        value_line,
        '=' * table_width,
    ]
    return '\n'.join(lines)


def evaluate_trajectory_directory_best_of(trajectory_dir: str,
                                          output_dir: str,
                                          target_player_only: bool = True) -> Dict:
    """Evaluate a directory by selecting the best generation per ground-truth rally.

    For every ground-truth rally (per player), the generation minimizing
    (move_dtw + land_dtw + (1 - lcs_ratio)) / 3 is selected. Rally-level
    scores are pooled over all selected rallies; match-level metrics are
    computed per match from the selected rallies and averaged over matches.
    """
    print('=' * 80)
    print('Best-of-Generations Trajectory Evaluation')
    print('=' * 80)
    print(f'Input dir:  {trajectory_dir}')
    print(f'Output dir: {output_dir}')
    print(f'Target Player Only: {target_player_only}')
    print()

    trajectory_files = discover_trajectory_files(trajectory_dir, latest_only=False)
    if not trajectory_files:
        raise ValueError(f'No trajectory pickle files found under: {trajectory_dir}')

    print(f'Found {len(trajectory_files)} trajectory files')

    skip_n = 1 if target_player_only else 2
    print(f'Skip first {skip_n} shot(s) per trajectory for sequence metrics (skip mode)')
    print()

    # Group generation candidates by ground-truth rally identity
    rally_groups = defaultdict(list)
    for trajectory_file in trajectory_files:
        trajectories = load_trajectories(trajectory_file)
        parsed = parse_trajectory_filename(trajectory_file)
        trajectory_name = os.path.basename(trajectory_file)
        generation = parsed['generation']

        for traj in trajectories:
            rally_key = (
                traj.get('target_player', ''),
                traj.get('match_id', ''),
                traj.get('set', ''),
                traj.get('rally', ''),
            )
            rally_groups[rally_key].append({
                'generation': generation,
                'trajectory_file': trajectory_name,
                'trajectory': traj,
            })

    print(f'Found {len(rally_groups)} unique ground-truth rallies')
    print('Selecting best generation per rally ...')

    # Select the best generation for each ground-truth rally
    best_rally_rows = []
    for rally_key in sorted(rally_groups.keys(), key=lambda key: tuple(str(part) for part in key)):
        target_player, match_id, set_id, rally_id = rally_key
        candidates = sorted(rally_groups[rally_key],
                            key=lambda item: (item['generation'], item['trajectory_file']))

        best_candidate = None
        best_sequence_values = None
        best_score = float('inf')
        for candidate in candidates:
            sequence_values = sequence_metrics_for_trajectories(
                [candidate['trajectory']], target_player_only)
            score = best_of_score(sequence_values)
            score_cmp = score if math.isfinite(score) else float('inf')
            if score_cmp < best_score:
                best_score = score_cmp
                best_candidate = candidate
                best_sequence_values = sequence_values

        if best_candidate is None:
            # All candidates had non-finite scores; fall back to lowest generation
            best_candidate = candidates[0]
            best_sequence_values = sequence_metrics_for_trajectories(
                [best_candidate['trajectory']], target_player_only)
            best_score = best_of_score(best_sequence_values)

        winner_traj = best_candidate['trajectory']
        sequence_values_skip = sequence_metrics_for_trajectories(
            [winner_traj], target_player_only, skip_n=skip_n)

        best_rally_rows.append({
            'target_player': target_player,
            'match_id': match_id,
            'set': set_id,
            'rally': rally_id,
            'opponent_player': winner_traj.get('opponent_player', ''),
            'num_generations': len(candidates),
            'best_generation': best_candidate['generation'],
            'best_trajectory_file': best_candidate['trajectory_file'],
            'land_dtw': best_sequence_values['land_dtw'],
            'move_dtw': best_sequence_values['move_dtw'],
            'lcs_ratio': best_sequence_values['lcs_ratio'],
            'best_score': best_score,
            'land_dtw_skip': sequence_values_skip['land_dtw'],
            'move_dtw_skip': sequence_values_skip['move_dtw'],
            'lcs_ratio_skip': sequence_values_skip['lcs_ratio'],
            'num_gt_shots': len(winner_traj.get('gt_sequence', [])),
            'num_pred_shots': len(winner_traj.get('pred_sequence', [])),
            'simulation_ended_early': bool(winner_traj.get('simulation_ended_early', False)),
            '_trajectory': winner_traj,
        })

    # Match-level: pool the selected rallies per match and compute metrics
    match_groups = defaultdict(list)
    for row in best_rally_rows:
        match_groups[(row['target_player'], row['match_id'])].append(row['_trajectory'])

    best_match_rows = []
    for match_key in sorted(match_groups.keys(), key=lambda key: tuple(str(part) for part in key)):
        target_player, match_id = match_key
        trajs = match_groups[match_key]
        rows_in_match = [row for row in best_rally_rows
                         if row['target_player'] == target_player and row['match_id'] == match_id]
        data = extract_sequences(trajs, target_player_only=target_player_only)
        distribution_values = distribution_metrics_for_trajectories(trajs, target_player_only)
        early_end_count = sum(1 for traj in trajs if traj.get('simulation_ended_early', False))

        best_match_rows.append({
            'target_player': target_player,
            'match_id': match_id,
            'num_rallies': len(trajs),
            'num_gt_shots': int(len(data['all_shot_gt'])),
            'num_pred_shots': int(len(data['all_shot_pred'])),
            'early_end_rate': early_end_count / max(len(trajs), 1),
            'mean_land_dtw': finite_mean([row['land_dtw'] for row in rows_in_match]),
            'mean_move_dtw': finite_mean([row['move_dtw'] for row in rows_in_match]),
            'mean_lcs_ratio': finite_mean([row['lcs_ratio'] for row in rows_in_match]),
            **distribution_values,
        })

    # Final summaries
    num_players = len({row['target_player'] for row in best_rally_rows})
    num_matches = len(best_match_rows)
    early_end_total = sum(1 for row in best_rally_rows if row['simulation_ended_early'])

    rally_summary = {
        'level': 'rally_level_pooled',
        'num_players': num_players,
        'num_matches': num_matches,
        'num_rallies': len(best_rally_rows),
        'num_gt_shots': sum(int(row['num_gt_shots']) for row in best_rally_rows),
        'num_pred_shots': sum(int(row['num_pred_shots']) for row in best_rally_rows),
        'early_end_rate': early_end_total / max(len(best_rally_rows), 1),
        'land_dtw': finite_mean([row['land_dtw'] for row in best_rally_rows]),
        'move_dtw': finite_mean([row['move_dtw'] for row in best_rally_rows]),
        'lcs_ratio': finite_mean([row['lcs_ratio'] for row in best_rally_rows]),
        'best_score': finite_mean([row['best_score'] for row in best_rally_rows]),
        'land_dtw_skip': finite_mean([row['land_dtw_skip'] for row in best_rally_rows]),
        'move_dtw_skip': finite_mean([row['move_dtw_skip'] for row in best_rally_rows]),
        'lcs_ratio_skip': finite_mean([row['lcs_ratio_skip'] for row in best_rally_rows]),
    }

    match_summary = {
        'level': 'match_level_average',
        'num_players': num_players,
        'num_matches': num_matches,
        'num_rallies': sum(int(row['num_rallies']) for row in best_match_rows),
        'num_gt_shots': sum(int(row['num_gt_shots']) for row in best_match_rows),
        'num_pred_shots': sum(int(row['num_pred_shots']) for row in best_match_rows),
        'early_end_rate': finite_mean([row['early_end_rate'] for row in best_match_rows]),
        'land_dtw': finite_mean([row['mean_land_dtw'] for row in best_match_rows]),
        'move_dtw': finite_mean([row['mean_move_dtw'] for row in best_match_rows]),
        'lcs_ratio': finite_mean([row['mean_lcs_ratio'] for row in best_match_rows]),
    }
    for metric_name in DISTRIBUTION_METRIC_COLUMNS:
        match_summary[metric_name] = finite_mean(
            [row.get(metric_name) for row in best_match_rows])

    summary_rows = [rally_summary, match_summary]

    # Output files
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, 'best_of_summary.csv')
    rally_details_csv = os.path.join(output_dir, 'best_of_rally_details.csv')
    match_details_csv = os.path.join(output_dir, 'best_of_match_details.csv')
    rally_table_path = os.path.join(output_dir, 'best_of_rally_table.txt')
    match_table_path = os.path.join(output_dir, 'best_of_match_table.txt')
    json_path = os.path.join(output_dir, 'best_of_metrics.json')

    write_csv_rows(summary_csv, summary_rows, BEST_OF_SUMMARY_COLUMNS)
    write_csv_rows(rally_details_csv, best_rally_rows, BEST_OF_RALLY_DETAIL_COLUMNS)
    write_csv_rows(match_details_csv, best_match_rows, BEST_OF_MATCH_DETAIL_COLUMNS)

    agent_label = infer_agent_label_from_path(trajectory_dir)
    rally_metric_specs = [
        ('Players', 'num_players'),
        ('Matches', 'num_matches'),
        ('Rallies', 'num_rallies'),
        ('EarlyEnd', 'early_end_rate'),
        ('LandDTW', 'land_dtw'),
        ('MoveDTW', 'move_dtw'),
        ('LCS', 'lcs_ratio'),
        ('BestScore', 'best_score'),
        ('LandDTW_s', 'land_dtw_skip'),
        ('MoveDTW_s', 'move_dtw_skip'),
        ('LCS_s', 'lcs_ratio_skip'),
    ]
    match_metric_specs = [
        ('Players', 'num_players'),
        ('Matches', 'num_matches'),
        ('Rallies', 'num_rallies'),
        ('EarlyEnd', 'early_end_rate'),
        ('LandDTW', 'land_dtw'),
        ('MoveDTW', 'move_dtw'),
        ('LCS', 'lcs_ratio'),
        ('ShotLand', 'shot_conditioned_landing_js'),
        ('ShotJS', 'shot_distribution_js'),
        ('LenJS', 'rally_length_js'),
        ('StateMove', 'state_conditioned_move_js'),
        ('StateShot', 'state_conditioned_shot_js'),
        ('StateLand', 'state_conditioned_landing_js'),
    ]

    rally_table = build_best_of_table(
        f'{agent_label} Best-of-Generations Rally-Level (pooled over all rallies)',
        rally_summary, rally_metric_specs)
    match_table = build_best_of_table(
        f'{agent_label} Best-of-Generations Match-Level (average over matches)',
        match_summary, match_metric_specs)

    with open(rally_table_path, 'w') as f:
        f.write(rally_table + '\n')
    with open(match_table_path, 'w') as f:
        f.write(match_table + '\n')

    results = {
        'trajectory_dir': trajectory_dir,
        'mode': 'best_of_generations',
        'target_player_only': target_player_only,
        'trajectory_files': trajectory_files,
        'summary': summary_rows,
        'rally_details': [
            {key: value for key, value in row.items() if key != '_trajectory'}
            for row in best_rally_rows
        ],
        'match_details': best_match_rows,
        'rally_level_table': rally_table_path,
        'match_level_table': match_table_path,
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(rally_table)
    print(match_table)

    print()
    print(f'Summary CSV: {summary_csv}')
    print(f'Rally details CSV: {rally_details_csv}')
    print(f'Match details CSV: {match_details_csv}')
    print(f'Rally-level table: {rally_table_path}')
    print(f'Match-level table: {match_table_path}')
    print(f'Best-of metrics JSON: {json_path}')
    print('=' * 80)
    print('Best-of-Generations Evaluation Complete!')
    print('=' * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate saved trajectories without re-running simulation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all metrics with visualizations
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/twin/trajectories/Kento_MOMOTA_norm.pkl \\
        --output-dir results/ \\
        --metrics all \\
        --visualize

    # Compute only sequence metrics
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/bc/trajectories/Viktor_AXELSEN_norm.pkl \\
        --metrics dtw edit_distance lcs_ratio shot_similarity

    # Evaluate only selected distribution metrics
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/twin/trajectories/An_Se_Young_norm.pkl \\
        --metrics distribution \\
        --output-dir results/distribution/

    # Include both players (not just target player)
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/pure_llm/trajectories/norm/Akane_YAMAGUCHI_norm_gen001.pkl \\
        --all-players

    # Evaluate all PureLLM trajectories for all players
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-dir Experiments/pure_llm/trajectories/norm \\
        --output-dir Experiments/pure_llm/results/trajectory_analysis/norm \\
        --latest-only \\
        --all-players \\
        --no-visualize

    # Evaluate every trajectory in an experiment directory
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-dir Experiments/twin/trajectories/norm \\
        --output-dir Experiments/twin/results/trajectory_analysis/norm \\
        --latest-only \\
        --no-visualize

    # Best-of-generations: pick the best gen per ground-truth rally
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-dir Experiments/bc/trajectories/norm \\
        --output-dir Experiments/bc/results/trajectory_analysis/norm_best_of \\
        --best-of-generations \\
        --no-visualize

Supported Metrics:
    all              - Compute all metrics
    dtw              - Dynamic Time Warping for landing/move positions
    edit_distance    - Levenshtein edit distance for shot sequences
    lcs_ratio        - Longest common subsequence ratio for shot sequences
    shot_similarity  - Shot bigram/trigram similarity and Markov transition JS
    distribution     - Selected shot, rally-length, and state-conditioned JS metrics
        """
    )
    
    parser.add_argument('--trajectory-file', type=str,
                        help='Path to one trajectory pickle file')
    parser.add_argument('--trajectory-dir', type=str,
                        help='Directory containing trajectory pickle files for experiment-level analysis')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results and visualizations')
    parser.add_argument('--metrics', nargs='+', 
                        choices=['all', 'dtw', 'edit_distance', 'lcs_ratio', 'shot_similarity', 'distribution'],
                        default=['all'],
                        help='Metrics to compute (default: all)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization charts')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualizations')
    parser.add_argument('--all-players', action='store_true',
                        help='Evaluate all players in trajectories (not just target player)')
    parser.add_argument('--latest-only', action='store_true',
                        help='In --trajectory-dir mode, only evaluate the latest generation for each player trajectory')
    parser.add_argument('--best-of-generations', action='store_true',
                        help='In --trajectory-dir mode, select the best generation per ground-truth rally '
                             '(minimizing (move_dtw + land_dtw + (1 - lcs_ratio)) / 3), then report '
                             'rally-level pooled averages and match-level match averages')
    parser.add_argument('--print-max-rows', type=int, default=50,
                        help='Maximum rows to print per table in --trajectory-dir mode; use 0 to print all rows')
    
    args = parser.parse_args()

    print_runtime_box(
        "Runtime Parameters: evaluate_trajectories",
        sorted(vars(args).items()),
    )

    if bool(args.trajectory_file) == bool(args.trajectory_dir):
        print("Error: specify exactly one of --trajectory-file or --trajectory-dir")
        return 1

    if args.trajectory_dir:
        if not os.path.isdir(args.trajectory_dir):
            print(f"Error: Trajectory directory not found: {args.trajectory_dir}")
            return 1
        if args.best_of_generations:
            if args.latest_only:
                print("Error: --best-of-generations cannot be combined with --latest-only")
                return 1
            evaluate_trajectory_directory_best_of(
                trajectory_dir=args.trajectory_dir,
                output_dir=args.output_dir,
                target_player_only=(not args.all_players),
            )
            return 0
        evaluate_trajectory_directory(
            trajectory_dir=args.trajectory_dir,
            output_dir=args.output_dir,
            latest_only=args.latest_only,
            target_player_only=(not args.all_players),
            print_max_rows=args.print_max_rows,
        )
        return 0

    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        return 1
    
    evaluate_trajectories(
        trajectory_file=args.trajectory_file,
        output_dir=args.output_dir,
        metrics=args.metrics,
        visualize=(args.visualize and not args.no_visualize),
        target_player_only=(not args.all_players)
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
