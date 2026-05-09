#!/usr/bin/env python3
"""
Standalone Trajectory Evaluation Script

Evaluates saved trajectories without re-running agent simulation.
Computes comprehensive metrics including:
- Sequence-level: DTW, CTC, Edit Distance
- Distribution: JS divergence, Wasserstein distance
- Conditional distributions: per-shot-type analysis

Usage:
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/twin/trajectories/Kento_MOMOTA_norm.pkl \\
        --output-dir Experiments/twin/results/single_player/ \\
        --metrics all \\
        --visualize

    # Compute specific metrics only
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/bc/trajectories/Viktor_AXELSEN_norm.pkl \\
        --metrics dtw ctc edit_distance \\
        --output-dir results/

Output Files (with --visualize):
    evaluation_results.json              - Complete metrics in JSON
    evaluation_summary.csv               - Summary table in CSV
    landing_kde.png                      - Landing position KDE heatmap
    shot_distribution.png                - Shot type distribution bar chart
    per_shot_landing.png                 - Per-shot landing scatter plots
    conditional_landing_distributions.png - P(Area|Shot Type) distributions
    conditional_landing_heatmap.png      - P(Area|Shot Type) heatmap
    conditional_move_distributions.png   - P(Move Area|Shot Type) distributions
    conditional_mdc_ratios.png           - Mean Distance from Center by shot
    per_shot_landing_js.png              - JS divergence per shot type
    per_prev_shot_transition.png         - Transition JS divergence
    transition_matrices.png              - P(Next|Previous) transition matrices
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

from Utils.helper import edit_distance
from Utils.badminton_config import EN_SHOT2INDEX, INDEX2EN_SHOT

# Try to import optional dependencies
try:
    from dtaidistance import dtw_ndim
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. DTW metrics will be skipped.")

try:
    import torch
    import torch.nn.functional as F
    from torch.nn import CTCLoss
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CTC metrics will be skipped.")

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance, gaussian_kde
    from scipy.spatial.distance import cdist
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Distribution metrics will be skipped.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualizations will be skipped.")


# =============================================================================
# Area Group Mapping (from metrics.py)
# =============================================================================

area_groups = {
    "Deep Left Corner": [32],
    "Deep Right Corner": [28],
    "Rear Left Side": [1, 5],
    "Rear Right Side": [4, 8],
    "Rear Center": [2, 3, 6, 7],
    "Mid Left Side": [9, 13],
    "Mid Right Side": [12, 16],
    "Mid Center": [10, 11, 14, 15],
    "Front Left Side": [20, 24],
    "Front Right Side": [17, 21],
    "Front Center": [18, 19, 22, 23],
    "Front Left Side Line": [29],
    "Front Right Side Line": [25],
    "Mid Left Side Line": [30],
    "Mid Right Side Line": [26],
    "Rear Left Side Line": [31],
    "Rear Right Side Line": [27],
    "Out of Bounds": [33]
}

_GROUP_NAMES = list(area_groups.keys())
N_GROUPS = len(_GROUP_NAMES)  # 18
N_VALID_GROUPS = N_GROUPS - 1  # 17 (exclude Out of Bounds)

# Build area_id (0-33) → group_id (0-17)
_AREA_TO_GROUP = {}
for gid, name in enumerate(_GROUP_NAMES):
    for area_id in area_groups.get(name, [33]):
        _AREA_TO_GROUP[area_id] = gid
for a in range(34):
    _AREA_TO_GROUP.setdefault(a, 17)

# =============================================================================
# Utility Functions
# =============================================================================

def load_trajectories(trajectory_file: str) -> List[Dict]:
    """Load trajectory pickle file."""
    with open(trajectory_file, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def coord2area(norm_x: float, norm_y: float) -> int:
    """Convert normalized coordinates to area id (0-33)."""
    from Utils.helper import coord2area as _coord2area
    return _coord2area(norm_x, norm_y)


def coord2group(norm_x: float, norm_y: float) -> int:
    """Convert normalized coordinates to area group id (0-17)."""
    area_id = coord2area(norm_x, norm_y)
    return _AREA_TO_GROUP.get(area_id, 17)


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


def extract_sequences(trajectories: List[Dict], target_player_only: bool = False) -> Dict:
    """
    Extract flat arrays and per-rally sequences from trajectories.
    
    Args:
        trajectories: List of trajectory dicts
        target_player_only: If True, only extract target player's shots
    
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

        for s in gt_seq:
            player = s.get('player', '')
            if target_player_only and player != target_player:
                continue
            
            data['all_land_gt'].append([s['landing_x'], s['landing_y']])
            data['all_move_gt'].append([s['moving_x'], s['moving_y']])
            data['all_shot_gt'].append(s['shot_type'])
            data['all_ball_rounds_gt'].append(s['ball_round'])
            land_gt_rally.append([s['landing_x'], s['landing_y']])
            move_gt_rally.append([s['moving_x'], s['moving_y']])
            shot_gt_rally.append(s['shot_type'])

        for s in pred_seq:
            player = s.get('player', '')
            if target_player_only and player != target_player:
                continue
            
            data['all_land_pred'].append([s['landing_x'], s['landing_y']])
            data['all_move_pred'].append([s['moving_x'], s['moving_y']])
            data['all_shot_pred'].append(s['shot_type'])
            data['all_ball_rounds_pred'].append(s.get('ball_round', 0))
            land_pred_rally.append([s['landing_x'], s['landing_y']])
            move_pred_rally.append([s['moving_x'], s['moving_y']])
            shot_pred_rally.append(s['shot_type'])

        data['all_rally_lengths_pred'].append(len(pred_seq))
        data['all_rally_lengths_gt'].append(len(gt_seq))

        if len(land_pred_rally) > 0:
            data['land_pred_seqs'].append(np.array(land_pred_rally))
            data['move_pred_seqs'].append(np.array(move_pred_rally))
            data['shot_pred_seqs'].append(shot_pred_rally)
        if len(land_gt_rally) > 0:
            data['land_gt_seqs'].append(np.array(land_gt_rally))
            data['move_gt_seqs'].append(np.array(move_gt_rally))
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


# =============================================================================
# Sequence Metrics (DTW, CTC, Edit Distance)
# =============================================================================

class SequenceMetrics:
    """Computes sequence-level metrics: DTW, CTC, Edit Distance."""
    
    @staticmethod
    def compute_dtw(land_pred_seqs: List[np.ndarray], land_gt_seqs: List[np.ndarray],
                    move_pred_seqs: List[np.ndarray], move_gt_seqs: List[np.ndarray]) -> Dict:
        """
        Compute DTW (Dynamic Time Warping) distance for landing and move positions.
        
        Returns average DTW normalized by sequence length.
        """
        if not DTW_AVAILABLE:
            return {'land_dtw': float('nan'), 'move_dtw': float('nan')}
        
        land_dtw_list = []
        move_dtw_list = []
        
        for land_pred, land_gt in zip(land_pred_seqs, land_gt_seqs):
            if len(land_pred) >= 2 and len(land_gt) >= 2:
                try:
                    dtw_dist = dtw_ndim.distance(land_pred, land_gt)
                    normalized_dtw = dtw_dist / len(land_gt)
                    land_dtw_list.append(normalized_dtw)
                except Exception as e:
                    pass
        
        for move_pred, move_gt in zip(move_pred_seqs, move_gt_seqs):
            if len(move_pred) >= 2 and len(move_gt) >= 2:
                try:
                    dtw_dist = dtw_ndim.distance(move_pred, move_gt)
                    normalized_dtw = dtw_dist / len(move_gt)
                    move_dtw_list.append(normalized_dtw)
                except Exception as e:
                    pass
        
        return {
            'land_dtw': float(np.mean(land_dtw_list)) if land_dtw_list else float('inf'),
            'move_dtw': float(np.mean(move_dtw_list)) if move_dtw_list else float('inf'),
            'land_dtw_std': float(np.std(land_dtw_list)) if land_dtw_list else 0.0,
            'move_dtw_std': float(np.std(move_dtw_list)) if move_dtw_list else 0.0,
            'num_valid_rallies': len(land_dtw_list),
        }
    
    @staticmethod
    def compute_ctc(shot_pred_seqs: List[List[int]], shot_gt_seqs: List[List[int]]) -> Dict:
        """
        Compute CTC (Connectionist Temporal Classification) loss for shot sequences.
        
        Returns average CTC loss across all rallies.
        """
        if not TORCH_AVAILABLE:
            return {'shot_ctc': float('nan')}
        
        ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        ctc_list = []
        
        for shot_pred, shot_gt in zip(shot_pred_seqs, shot_gt_seqs):
            if len(shot_pred) == 0 or len(shot_gt) == 0:
                continue
            
            try:
                # Convert to tensors
                pred_tensor = torch.tensor(shot_pred, dtype=torch.long)
                gt_tensor = torch.tensor(shot_gt, dtype=torch.long)
                
                # Create log probabilities (one-hot with smoothing)
                log_probs = F.one_hot(pred_tensor, num_classes=12).float()
                log_probs = F.log_softmax(log_probs + 1e-10, dim=-1)
                log_probs = log_probs.unsqueeze(1)  # Add batch dimension
                
                # Target: remove blanks (0) from GT
                target = gt_tensor[gt_tensor != 0]
                if len(target) == 0:
                    continue
                
                input_length = torch.tensor([len(pred_tensor)], dtype=torch.long)
                target_length = torch.tensor([len(target)], dtype=torch.long)
                
                loss = ctc_loss_fn(log_probs, target.unsqueeze(0), input_length, target_length)
                ctc_list.append(loss.item())
            except Exception as e:
                pass
        
        return {
            'shot_ctc': float(np.mean(ctc_list)) if ctc_list else float('inf'),
            'shot_ctc_std': float(np.std(ctc_list)) if ctc_list else 0.0,
            'num_valid_rallies': len(ctc_list),
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


def print_player_metric_table(title: str,
                              player_rows: List[Dict],
                              metric_specs: List[Tuple[str, str, str]],
                              summary_rows: List[Dict],
                              max_rows: int = 50):
    """Print player rows with experiment averages appended below a rule.

    metric_specs entries are (display_column, player_row_key, summary_row_key).
    """
    base_columns = [
        'target_player', 'generation', 'num_matches', 'num_rallies',
        'num_gt_shots', 'num_pred_shots', 'early_end_rate',
    ]
    metric_columns = [display for display, _, _ in metric_specs]
    columns = base_columns + metric_columns

    rows_to_print = player_rows if max_rows <= 0 else player_rows[:max_rows]
    display_rows = []

    for row in rows_to_print:
        display_row = {column: row.get(column, '') for column in base_columns}
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
            'early_end_rate': '',
        }
        for display, _, summary_key in metric_specs:
            display_row[display] = row.get(summary_key, '')
        summary_display_rows.append(display_row)

    all_rows = display_rows + summary_display_rows
    print()
    print(title)
    print('=' * len(title))
    if not all_rows:
        print('(no rows)')
        return

    formatted_rows = [
        {column: format_table_value(row.get(column, '')) for column in columns}
        for row in all_rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }
    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    separator = '-+-'.join('-' * widths[column] for column in columns)

    print(header)
    print(separator)
    for idx, row in enumerate(formatted_rows):
        if idx == len(display_rows) and summary_display_rows:
            print(separator)
        print(' | '.join(row[column].ljust(widths[column]) for column in columns))

    if max_rows > 0 and len(player_rows) > max_rows:
        print(f'... showing {max_rows} of {len(player_rows)} player rows')


def write_csv_rows(path: str, rows: List[Dict], columns: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in columns})


def sequence_metrics_for_trajectories(trajectories: List[Dict], target_player_only: bool) -> Dict[str, float]:
    data = extract_sequences(trajectories, target_player_only=target_player_only)
    seq_metrics = SequenceMetrics()
    dtw = seq_metrics.compute_dtw(
        data['land_pred_seqs'], data['land_gt_seqs'],
        data['move_pred_seqs'], data['move_gt_seqs']
    )
    ctc = seq_metrics.compute_ctc(data['shot_pred_seqs'], data['shot_gt_seqs'])
    edit = seq_metrics.compute_edit_distance(data['shot_pred_seqs'], data['shot_gt_seqs'])

    return {
        'land_dtw': dtw.get('land_dtw', float('nan')),
        'move_dtw': dtw.get('move_dtw', float('nan')),
        'shot_ctc': ctc.get('shot_ctc', float('nan')),
        'edit_distance': edit.get('edit_distance', float('nan')),
    }


def distribution_metrics_for_trajectories(trajectories: List[Dict], target_player_only: bool) -> Dict[str, float]:
    data = extract_sequences(trajectories, target_player_only=target_player_only)
    dist_metrics = DistributionMetrics()
    cond_metrics = ConditionalMetrics()

    landing_wasserstein = dist_metrics.wasserstein_1d(data['all_land_pred'], data['all_land_gt'])
    move_wasserstein = dist_metrics.wasserstein_1d(data['all_move_pred'], data['all_move_gt'])
    per_shot_landing = cond_metrics.per_shot_landing_js(
        data['all_land_pred'], data['all_land_gt'],
        data['all_shot_pred'], data['all_shot_gt']
    )
    per_shot_move = cond_metrics.per_shot_move_js(
        data['all_move_pred'], data['all_move_gt'],
        data['all_shot_pred'], data['all_shot_gt']
    )
    per_prev_transition = cond_metrics.per_prev_shot_transition_js(
        data['shot_pred_seqs'], data['shot_gt_seqs']
    )
    per_shot_mdc = cond_metrics.per_shot_landing_mdc(
        data['all_land_pred'], data['all_land_gt'],
        data['all_shot_pred'], data['all_shot_gt']
    )

    return {
        'landing_area_js': dist_metrics.area_distribution_js(data['all_land_pred'], data['all_land_gt']),
        'move_area_js': dist_metrics.area_distribution_js(data['all_move_pred'], data['all_move_gt']),
        'shot_distribution_js': dist_metrics.shot_distribution_js(data['all_shot_pred'], data['all_shot_gt']),
        'landing_x_wasserstein': landing_wasserstein.get('x_wasserstein', float('nan')),
        'landing_y_wasserstein': landing_wasserstein.get('y_wasserstein', float('nan')),
        'landing_mean_wasserstein': landing_wasserstein.get('mean_wasserstein', float('nan')),
        'move_x_wasserstein': move_wasserstein.get('x_wasserstein', float('nan')),
        'move_y_wasserstein': move_wasserstein.get('y_wasserstein', float('nan')),
        'move_mean_wasserstein': move_wasserstein.get('mean_wasserstein', float('nan')),
        'per_shot_landing_js': per_shot_landing.get('weighted_average', float('nan')),
        'per_shot_move_js': per_shot_move.get('weighted_average', float('nan')),
        'per_prev_shot_transition_js': per_prev_transition.get('weighted_average', float('nan')),
        'joint_area_shot_js': cond_metrics.joint_area_shot_js(
            data['all_land_pred'], data['all_shot_pred'],
            data['all_land_gt'], data['all_shot_gt']
        ),
        'mdc_ratio': per_shot_mdc.get('weighted_average', float('nan')),
    }


# =============================================================================
# Distribution Metrics
# =============================================================================

class DistributionMetrics:
    """Computes distribution similarity metrics."""
    
    @staticmethod
    def area_distribution_js(land_pred: np.ndarray, land_gt: np.ndarray) -> float:
        """JS divergence of landing area group distributions (18 groups)."""
        if len(land_pred) == 0 or len(land_gt) == 0:
            return float('nan')
        
        area_pred = np.array([coord2group(x, y) for x, y in land_pred])
        area_gt = np.array([coord2group(x, y) for x, y in land_gt])
        
        p = histogram(area_pred, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        q = histogram(area_gt, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        
        return js_divergence(p, q)
    
    @staticmethod
    def shot_distribution_js(shot_pred: np.ndarray, shot_gt: np.ndarray) -> float:
        """JS divergence of shot type distributions (12 bins, types 0-11)."""
        if len(shot_pred) == 0 or len(shot_gt) == 0:
            return float('nan')
        
        p = histogram(shot_pred, bins=12, range_min=0, range_max=12)
        q = histogram(shot_gt, bins=12, range_min=0, range_max=12)
        
        return js_divergence(p, q)
    
    @staticmethod
    def wasserstein_1d(land_pred: np.ndarray, land_gt: np.ndarray) -> Dict:
        """1D Wasserstein (Earth Mover) distance on x and y marginals."""
        if not SCIPY_AVAILABLE or len(land_pred) == 0 or len(land_gt) == 0:
            return {'x_wasserstein': float('nan'), 'y_wasserstein': float('nan')}
        
        x_dist = float(wasserstein_distance(land_pred[:, 0], land_gt[:, 0]))
        y_dist = float(wasserstein_distance(land_pred[:, 1], land_gt[:, 1]))
        
        return {
            'x_wasserstein': x_dist,
            'y_wasserstein': y_dist,
            'mean_wasserstein': (x_dist + y_dist) / 2.0,
        }
    
    @staticmethod
    def mean_distance_from_center(coords: np.ndarray) -> float:
        """Mean Euclidean distance from court center (normalized space)."""
        if len(coords) == 0:
            return 0.0
        center = np.array([0.0, 0.0])
        return float(np.mean(np.linalg.norm(coords - center, axis=1)))


# =============================================================================
# Conditional Distribution Analysis (KEY FEATURE)
# =============================================================================

class ConditionalMetrics:
    """
    Computes conditional distribution similarities.
    Analyzes how well the model captures shot-type-specific behaviors.
    """
    
    @staticmethod
    def per_shot_landing_js(land_pred: np.ndarray, land_gt: np.ndarray,
                           shot_pred: np.ndarray, shot_gt: np.ndarray) -> Dict:
        """
        For each shot type (1-11), compute JS divergence of landing area distributions.
        
        Returns:
            Dict with per-shot JS and weighted average.
        """
        if len(land_pred) == 0 or len(land_gt) == 0:
            return {'weighted_average': float('nan'), 'per_shot': {}}
        
        area_pred = np.array([coord2group(x, y) for x, y in land_pred])
        area_gt = np.array([coord2group(x, y) for x, y in land_gt])
        
        weighted_js = 0.0
        total_weight = 0
        per_shot = {}
        
        for shot in range(1, 12):
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            if np.sum(mask_gt) == 0:
                continue
            
            # Build histograms for this shot type
            p = histogram(area_pred[mask_pred], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
            q = histogram(area_gt[mask_gt], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
            
            js = js_divergence(p, q)
            weight = float(np.sum(mask_gt))
            
            weighted_js += js * weight
            total_weight += weight
            per_shot[str(shot)] = {
                'js_divergence': js,
                'support_gt': int(weight),
                'support_pred': int(np.sum(mask_pred)),
            }
        
        return {
            'weighted_average': float(weighted_js / max(total_weight, 1e-8)),
            'per_shot': per_shot,
        }
    
    @staticmethod
    def per_shot_move_js(move_pred: np.ndarray, move_gt: np.ndarray,
                        shot_pred: np.ndarray, shot_gt: np.ndarray) -> Dict:
        """
        For each shot type (1-11), compute JS divergence of move area distributions.
        """
        if len(move_pred) == 0 or len(move_gt) == 0:
            return {'weighted_average': float('nan'), 'per_shot': {}}
        
        area_pred = np.array([coord2group(x, y) for x, y in move_pred])
        area_gt = np.array([coord2group(x, y) for x, y in move_gt])
        
        weighted_js = 0.0
        total_weight = 0
        per_shot = {}
        
        for shot in range(1, 12):
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            if np.sum(mask_gt) == 0:
                continue
            
            p = histogram(area_pred[mask_pred], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
            q = histogram(area_gt[mask_gt], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
            
            js = js_divergence(p, q)
            weight = float(np.sum(mask_gt))
            
            weighted_js += js * weight
            total_weight += weight
            per_shot[str(shot)] = {
                'js_divergence': js,
                'support_gt': int(weight),
                'support_pred': int(np.sum(mask_pred)),
            }
        
        return {
            'weighted_average': float(weighted_js / max(total_weight, 1e-8)),
            'per_shot': per_shot,
        }
    
    @staticmethod
    def per_shot_landing_mdc(land_pred: np.ndarray, land_gt: np.ndarray,
                            shot_pred: np.ndarray, shot_gt: np.ndarray) -> Dict:
        """
        For each shot type, compute Mean Distance from Center (MDC) ratio.
        Ratio < 1 means model predicts closer to center than GT.
        """
        if len(land_pred) == 0 or len(land_gt) == 0:
            return {'weighted_average': float('nan'), 'per_shot': {}}
        
        center = np.array([0.0, 0.0])
        weighted_ratio = 0.0
        total_weight = 0
        per_shot = {}
        
        for shot in range(1, 12):
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            if np.sum(mask_gt) == 0:
                continue
            
            pred_dist = float(np.mean(np.linalg.norm(land_pred[mask_pred] - center, axis=1)))
            gt_dist = float(np.mean(np.linalg.norm(land_gt[mask_gt] - center, axis=1)))
            ratio = pred_dist / max(gt_dist, 1e-8)
            
            weight = float(np.sum(mask_gt))
            weighted_ratio += ratio * weight
            total_weight += weight
            
            per_shot[str(shot)] = {
                'ratio': ratio,
                'pred_mdc': pred_dist,
                'gt_mdc': gt_dist,
                'support': int(weight),
            }
        
        return {
            'weighted_average': float(weighted_ratio / max(total_weight, 1e-8)),
            'per_shot': per_shot,
        }
    
    @staticmethod
    def per_prev_shot_transition_js(shot_pred_seqs: List[List[int]], 
                                    shot_gt_seqs: List[List[int]]) -> Dict:
        """
        For each previous shot type, compute JS divergence of next shot distributions.
        
        Analyzes conditional probability: P(next_shot | prev_shot)
        """
        # Collect transitions
        transitions_pred = defaultdict(lambda: defaultdict(int))
        transitions_gt = defaultdict(lambda: defaultdict(int))
        
        for seq in shot_pred_seqs:
            for i in range(len(seq) - 1):
                prev_shot = seq[i]
                next_shot = seq[i + 1]
                if 1 <= prev_shot <= 11 and 1 <= next_shot <= 11:
                    transitions_pred[prev_shot][next_shot] += 1
        
        for seq in shot_gt_seqs:
            for i in range(len(seq) - 1):
                prev_shot = seq[i]
                next_shot = seq[i + 1]
                if 1 <= prev_shot <= 11 and 1 <= next_shot <= 11:
                    transitions_gt[prev_shot][next_shot] += 1
        
        # Compute JS for each previous shot type
        per_prev_shot = {}
        weighted_js = 0.0
        total_weight = 0
        
        for prev_shot in range(1, 12):
            if prev_shot not in transitions_gt:
                continue
            
            # Build distributions (11 possible next shots)
            p = np.zeros(11)
            q = np.zeros(11)
            
            for next_shot in range(1, 12):
                p[next_shot - 1] = transitions_pred[prev_shot].get(next_shot, 0)
                q[next_shot - 1] = transitions_gt[prev_shot].get(next_shot, 0)
            
            # Skip if no transitions
            pred_support = int(p.sum())
            gt_support = int(q.sum())
            if gt_support == 0:
                continue
            
            # Smooth and normalize
            p = p + 1.0
            q = q + 1.0
            p = p / p.sum()
            q = q / q.sum()
            
            js = js_divergence(p, q)
            weight = float(gt_support)
            
            weighted_js += js * weight
            total_weight += weight
            
            per_prev_shot[str(prev_shot)] = {
                'js_divergence': js,
                'support_gt': gt_support,
                'support_pred': pred_support,
            }
        
        return {
            'weighted_average': float(weighted_js / max(total_weight, 1e-8)),
            'per_prev_shot': per_prev_shot,
        }
    
    @staticmethod
    def joint_area_shot_js(land_pred: np.ndarray, shot_pred: np.ndarray,
                          land_gt: np.ndarray, shot_gt: np.ndarray) -> float:
        """
        JS divergence of joint (area group, shot_type) distribution (18 x 12 bins).
        """
        if len(land_pred) == 0 or len(land_gt) == 0:
            return float('nan')
        
        area_pred = np.array([coord2group(x, y) for x, y in land_pred])
        area_gt = np.array([coord2group(x, y) for x, y in land_gt])
        
        joint_pred, _, _ = np.histogram2d(
            area_pred, shot_pred, bins=[N_GROUPS, 12], 
            range=[[0, N_GROUPS], [0, 12]]
        )
        joint_gt, _, _ = np.histogram2d(
            area_gt, shot_gt, bins=[N_GROUPS, 12],
            range=[[0, N_GROUPS], [0, 12]]
        )
        
        p = joint_pred.flatten().astype(np.float64) + 1.0
        p /= p.sum()
        q = joint_gt.flatten().astype(np.float64) + 1.0
        q /= q.sum()
        
        return js_divergence(p, q)


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
    
    def _denorm_move(self, norm_coords: np.ndarray) -> np.ndarray:
        """Convert [-1,1] move coords to [0,355]x[-480,0]."""
        if len(norm_coords) == 0:
            return norm_coords
        x = (norm_coords[:, 0] + 1.0) * 177.5
        y = norm_coords[:, 1] * 240.0 - 240.0
        return np.column_stack([x, y])
    
    def _draw_landing_court(self, ax):
        """Draw badminton court lines for landing half."""
        ax.plot([0, 355], [0, 0], 'k-', linewidth=1.2)
        ax.plot([0, 355], [480, 480], 'k-', linewidth=1.2)
        ax.plot([0, 0], [0, 480], 'k-', linewidth=1.2)
        ax.plot([355, 355], [0, 480], 'k-', linewidth=1.2)
        ax.plot([177.5, 177.5], [0, 480], 'k--', linewidth=0.8, alpha=0.5)
        ax.plot([0, 355], [141.85, 141.85], 'k-', linewidth=0.8, alpha=0.6)
        ax.plot([0, 355], [425.78, 425.78], 'k:', linewidth=0.7, alpha=0.4)
    
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
        """Plot side-by-side KDE heatmaps of landing positions."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        dp = self._denorm_landing(land_pred)
        dg = self._denorm_landing(land_gt)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Predicted
        if len(dp) >= 2:
            try:
                kde = gaussian_kde(dp.T)
                x = np.linspace(0, 355, 100)
                y = np.linspace(0, 480, 100)
                X, Y = np.meshgrid(x, y)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax1.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.8)
            except:
                pass
        self._draw_landing_court(ax1)
        ax1.set_xlim(0, 355)
        ax1.set_ylim(0, 480)
        ax1.set_aspect('equal')
        ax1.set_title(f'Predicted ({len(dp)} shots)')
        
        # Ground Truth
        if len(dg) >= 2:
            try:
                kde = gaussian_kde(dg.T)
                x = np.linspace(0, 355, 100)
                y = np.linspace(0, 480, 100)
                X, Y = np.meshgrid(x, y)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax2.contourf(X, Y, Z, levels=10, cmap='Reds', alpha=0.8)
            except:
                pass
        self._draw_landing_court(ax2)
        ax2.set_xlim(0, 355)
        ax2.set_ylim(0, 480)
        ax2.set_aspect('equal')
        ax2.set_title(f'Ground Truth ({len(dg)} shots)')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_shot_distribution(self, shot_pred: np.ndarray, shot_gt: np.ndarray,
                               title: str, filename: str):
        """Plot side-by-side bar chart of shot type distributions."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        shot_labels = [INDEX2EN_SHOT.get(i, f'Type {i}') for i in range(1, 12)]
        
        counts_pred = [int(np.sum(shot_pred == s)) for s in range(1, 12)]
        counts_gt = [int(np.sum(shot_gt == s)) for s in range(1, 12)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(shot_labels))
        ax1.bar(x, counts_pred, color='steelblue', edgecolor='white')
        ax1.set_xticks(x)
        ax1.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax1.set_title(f'Predicted ({len(shot_pred)} shots)')
        ax1.set_ylabel('Count')
        
        ax2.bar(x, counts_gt, color='firebrick', edgecolor='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax2.set_title(f'Ground Truth ({len(shot_gt)} shots)')
        ax2.set_ylabel('Count')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
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
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in range(1, 12)}
        shot_labels['ALL'] = 'All Shots'
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        shot_order = list(range(1, 12)) + ['ALL']
        
        # Create grid for KDE evaluation
        x_grid = np.linspace(0, 355, grid_size)
        y_grid = np.linspace(0, 480, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        for idx, shot in enumerate(shot_order):
            ax = axes[idx]
            
            if shot == 'ALL':
                mask_pred = np.ones(len(shot_pred), dtype=bool)
                mask_gt = np.ones(len(shot_gt), dtype=bool)
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
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in range(1, 12)}
        shot_labels['ALL'] = 'All Shots'
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        shot_order = list(range(1, 12)) + ['ALL']
        
        for idx, shot in enumerate(shot_order):
            ax = axes[idx]
            
            if shot == 'ALL':
                mask_pred = np.ones(len(shot_pred), dtype=bool)
                mask_gt = np.ones(len(shot_gt), dtype=bool)
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
        
        # Group names (shortened for display)
        group_labels = [
            'Deep L', 'Deep R', 'Rear L', 'Rear R', 'Rear C',
            'Mid L', 'Mid R', 'Mid C', 'Front L', 'Front R',
            'Front C', 'FL Line', 'FR Line', 'ML Line', 'MR Line',
            'RL Line', 'RR Line', 'Out'
        ]
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in range(1, 12)}
        
        # Create 3x4 grid
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        for idx, shot in enumerate(range(1, 12)):
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
        
        # Remove the last empty subplot
        axes[11].axis('off')
        
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
        
        group_labels = [
            'Deep L', 'Deep R', 'Rear L', 'Rear R', 'Rear C',
            'Mid L', 'Mid R', 'Mid C', 'Front L', 'Front R',
            'Front C', 'FL Line', 'FR Line', 'ML Line', 'MR Line',
            'RL Line', 'RR Line', 'Out'
        ]
        
        shot_labels = {i: INDEX2EN_SHOT.get(i, f'Shot {i}') for i in range(1, 12)}
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()
        
        for idx, shot in enumerate(range(1, 12)):
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
        
        axes[11].axis('off')
        
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
        
        # Build conditional distributions: P(Area | Shot Type)
        shot_labels_short = [INDEX2EN_SHOT.get(i, f'S{i}')[:12] for i in range(1, 12)]
        group_labels = ['DL', 'DR', 'RL', 'RR', 'RC', 'ML', 'MR', 'MC', 
                       'FL', 'FR', 'FC', 'FLi', 'FRi', 'MLi', 'MRi', 'RLi', 'RRi', 'Out']
        
        # Create matrices: (11 shot types) x (18 area groups)
        mat_pred = np.zeros((11, 18))
        mat_gt = np.zeros((11, 18))
        
        for shot in range(1, 12):
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
            
            areas_pred = area_pred[mask_pred]
            areas_gt = area_gt[mask_gt]
            
            if len(areas_pred) > 0:
                hist, _ = np.histogram(areas_pred, bins=18, range=(0, 18))
                mat_pred[shot-1] = hist / hist.sum() if hist.sum() > 0 else 0
            
            if len(areas_gt) > 0:
                hist, _ = np.histogram(areas_gt, bins=18, range=(0, 18))
                mat_gt[shot-1] = hist / hist.sum() if hist.sum() > 0 else 0
        
        # Create side-by-side heatmaps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10))
        
        vmax = max(mat_pred.max(), mat_gt.max())
        
        # Predicted
        im1 = ax1.imshow(mat_pred, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
        ax1.set_yticks(range(11))
        ax1.set_yticklabels(shot_labels_short)
        ax1.set_xticks(range(18))
        ax1.set_xticklabels(group_labels, rotation=45, ha='right')
        ax1.set_xlabel('Landing Area Group')
        ax1.set_ylabel('Shot Type')
        ax1.set_title('Predicted P(Area | Shot Type)')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Ground Truth
        im2 = ax2.imshow(mat_gt, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)
        ax2.set_yticks(range(11))
        ax2.set_yticklabels(shot_labels_short)
        ax2.set_xticks(range(18))
        ax2.set_xticklabels(group_labels, rotation=45, ha='right')
        ax2.set_xlabel('Landing Area Group')
        ax2.set_ylabel('Shot Type')
        ax2.set_title('Ground Truth P(Area | Shot Type)')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference (Pred - GT)
        diff = mat_pred - mat_gt
        im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        ax3.set_yticks(range(11))
        ax3.set_yticklabels(shot_labels_short)
        ax3.set_xticks(range(18))
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
        
        # Build transition matrices (11x11)
        T_pred = np.zeros((11, 11))
        T_gt = np.zeros((11, 11))
        
        for seq in shot_pred_seqs:
            for i in range(len(seq) - 1):
                prev, next_s = int(seq[i]), int(seq[i+1])
                if 1 <= prev <= 11 and 1 <= next_s <= 11:
                    T_pred[prev-1, next_s-1] += 1
        
        for seq in shot_gt_seqs:
            for i in range(len(seq) - 1):
                prev, next_s = int(seq[i]), int(seq[i+1])
                if 1 <= prev <= 11 and 1 <= next_s <= 11:
                    T_gt[prev-1, next_s-1] += 1
        
        # Normalize rows to get probabilities
        for i in range(11):
            row_sum = T_pred[i].sum()
            if row_sum > 0:
                T_pred[i] = T_pred[i] / row_sum
            row_sum = T_gt[i].sum()
            if row_sum > 0:
                T_gt[i] = T_gt[i] / row_sum
        
        shot_labels = [INDEX2EN_SHOT.get(i, f'S{i}')[:10] for i in range(1, 12)]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        
        vmax = max(T_pred.max(), T_gt.max())
        
        # Predicted transitions
        im1 = ax1.imshow(T_pred, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
        ax1.set_xticks(range(11))
        ax1.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax1.set_yticks(range(11))
        ax1.set_yticklabels(shot_labels)
        ax1.set_xlabel('Next Shot')
        ax1.set_ylabel('Previous Shot')
        ax1.set_title('Predicted P(Next | Previous)')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Ground Truth transitions
        im2 = ax2.imshow(T_gt, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)
        ax2.set_xticks(range(11))
        ax2.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax2.set_yticks(range(11))
        ax2.set_yticklabels(shot_labels)
        ax2.set_xlabel('Next Shot')
        ax2.set_ylabel('Previous Shot')
        ax2.set_title('Ground Truth P(Next | Previous)')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference
        diff = T_pred - T_gt
        im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        ax3.set_xticks(range(11))
        ax3.set_xticklabels(shot_labels, rotation=45, ha='right')
        ax3.set_yticks(range(11))
        ax3.set_yticklabels(shot_labels)
        ax3.set_xlabel('Next Shot')
        ax3.set_ylabel('Previous Shot')
        ax3.set_title('Difference (Pred - GT)')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
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
                          target_player_only: bool = True) -> Dict:
    """
    Main evaluation function.
    
    Args:
        trajectory_file: Path to trajectory pickle file
        output_dir: Directory for output files
        metrics: List of metrics to compute (None = all)
        visualize: Whether to generate visualizations
        target_player_only: If True, only evaluate target player's shots
    
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
    data = extract_sequences(trajectories, target_player_only=target_player_only)
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
    if compute_all or any(m in ['dtw', 'sequence'] for m in metrics_to_compute):
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
        
        # CTC
        if compute_all or 'ctc' in metrics_to_compute:
            print("  - CTC...")
            ctc_results = seq_metrics.compute_ctc(
                data['shot_pred_seqs'], data['shot_gt_seqs']
            )
            results['sequence_metrics']['ctc'] = ctc_results
            print(f"    Shot CTC: {ctc_results['shot_ctc']:.4f}")
        
        # Edit Distance
        if compute_all or 'edit_distance' in metrics_to_compute:
            print("  - Edit Distance...")
            edit_results = seq_metrics.compute_edit_distance(
                data['shot_pred_seqs'], data['shot_gt_seqs']
            )
            results['sequence_metrics']['edit_distance'] = edit_results
            print(f"    Edit Distance: {edit_results['edit_distance']:.4f}")
        
        print()
    
    # =========================================================================
    # Distribution Metrics
    # =========================================================================
    if compute_all or 'distribution' in metrics_to_compute:
        print("Computing Distribution Metrics...")
        dist_metrics = DistributionMetrics()
        
        results['distribution_metrics'] = {
            'landing_area_js': dist_metrics.area_distribution_js(
                data['all_land_pred'], data['all_land_gt']
            ),
            'move_area_js': dist_metrics.area_distribution_js(
                data['all_move_pred'], data['all_move_gt']
            ),
            'shot_distribution_js': dist_metrics.shot_distribution_js(
                data['all_shot_pred'], data['all_shot_gt']
            ),
            'landing_wasserstein': dist_metrics.wasserstein_1d(
                data['all_land_pred'], data['all_land_gt']
            ),
            'move_wasserstein': dist_metrics.wasserstein_1d(
                data['all_move_pred'], data['all_move_gt']
            ),
            'pred_mdc': dist_metrics.mean_distance_from_center(data['all_land_pred']),
            'gt_mdc': dist_metrics.mean_distance_from_center(data['all_land_gt']),
        }
        
        print(f"  Landing Area JS: {results['distribution_metrics']['landing_area_js']:.4f}")
        print(f"  Shot Dist JS: {results['distribution_metrics']['shot_distribution_js']:.4f}")
        print()
    
    # =========================================================================
    # Conditional Metrics (KEY FEATURE)
    # =========================================================================
    if compute_all or 'conditional' in metrics_to_compute:
        print("Computing Conditional Distribution Metrics...")
        cond_metrics = ConditionalMetrics()
        
        results['conditional_metrics'] = {}
        
        # Per-shot landing
        print("  - Per-shot landing distribution...")
        per_shot_land = cond_metrics.per_shot_landing_js(
            data['all_land_pred'], data['all_land_gt'],
            data['all_shot_pred'], data['all_shot_gt']
        )
        results['conditional_metrics']['per_shot_landing'] = per_shot_land
        print(f"    Weighted Avg JS: {per_shot_land['weighted_average']:.4f}")
        
        # Per-shot move
        print("  - Per-shot move distribution...")
        per_shot_move = cond_metrics.per_shot_move_js(
            data['all_move_pred'], data['all_move_gt'],
            data['all_shot_pred'], data['all_shot_gt']
        )
        results['conditional_metrics']['per_shot_move'] = per_shot_move
        print(f"    Weighted Avg JS: {per_shot_move['weighted_average']:.4f}")
        
        # Per-shot MDC
        print("  - Per-shot distance from center...")
        per_shot_mdc = cond_metrics.per_shot_landing_mdc(
            data['all_land_pred'], data['all_land_gt'],
            data['all_shot_pred'], data['all_shot_gt']
        )
        results['conditional_metrics']['per_shot_mdc'] = per_shot_mdc
        print(f"    Weighted Avg Ratio: {per_shot_mdc['weighted_average']:.4f}")
        
        # Per-previous-shot transition
        print("  - Per-previous-shot transition...")
        per_prev_transition = cond_metrics.per_prev_shot_transition_js(
            data['shot_pred_seqs'], data['shot_gt_seqs']
        )
        results['conditional_metrics']['per_prev_shot_transition'] = per_prev_transition
        print(f"    Weighted Avg JS: {per_prev_transition['weighted_average']:.4f}")
        
        # Joint distribution
        print("  - Joint area-shot distribution...")
        joint_js = cond_metrics.joint_area_shot_js(
            data['all_land_pred'], data['all_shot_pred'],
            data['all_land_gt'], data['all_shot_gt']
        )
        results['conditional_metrics']['joint_area_shot_js'] = joint_js
        print(f"    Joint JS: {joint_js:.4f}")
        
        print()
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    if visualize and MATPLOTLIB_AVAILABLE:
        print("Generating visualizations...")
        visualizer = Visualizer(output_dir)
        
        # Landing KDE
        print("  - Landing distribution KDE...")
        visualizer.plot_landing_kde(
            data['all_land_pred'], data['all_land_gt'],
            f'{target_player} — Landing Position Distribution',
            'landing_kde.png'
        )
        
        # Shot distribution
        print("  - Shot type distribution...")
        visualizer.plot_shot_distribution(
            data['all_shot_pred'], data['all_shot_gt'],
            f'{target_player} — Shot Type Distribution',
            'shot_distribution.png'
        )
        
        # Per-shot landing distribution plots
        if 'conditional_metrics' in results:
            print("  - Per-shot landing distribution (heatmap)...")
            visualizer.plot_per_shot_landing_distribution(
                data['all_land_pred'], data['all_land_gt'],
                data['all_shot_pred'], data['all_shot_gt'],
                f'{target_player} — Per-Shot Landing Distribution (2D Histogram)',
                'per_shot_landing.png',
                use_heatmap=True,
                use_contour=True,
                alpha=0.5,
                bins=25
            )
            
            print("  - Per-shot landing distribution (KDE)...")
            visualizer.plot_per_shot_landing_kde(
                data['all_land_pred'], data['all_land_gt'],
                data['all_shot_pred'], data['all_shot_gt'],
                f'{target_player} — Per-Shot Landing Distribution (KDE)',
                'per_shot_landing_kde.png',
                use_contour=False,
                alpha=0.5,
                grid_size=100
            )
            
            print("  - Per-shot landing distribution (KDE contour)...")
            visualizer.plot_per_shot_landing_kde(
                data['all_land_pred'], data['all_land_gt'],
                data['all_shot_pred'], data['all_shot_gt'],
                f'{target_player} — Per-Shot Landing Distribution (KDE Contour)',
                'per_shot_landing_kde_contour.png',
                use_contour=True,
                alpha=0.7,
                grid_size=100
            )
            
            # NEW: Actual conditional distribution visualizations
            print("  - Conditional landing area distributions...")
            visualizer.plot_conditional_landing_distributions(
                data['all_land_pred'], data['all_land_gt'],
                data['all_shot_pred'], data['all_shot_gt'],
                f'{target_player} — Conditional Landing Area Distributions (P(Area|Shot Type))',
                'conditional_landing_distributions.png'
            )
            
            print("  - Conditional landing heatmap...")
            visualizer.plot_conditional_landing_heatmap(
                data['all_land_pred'], data['all_land_gt'],
                data['all_shot_pred'], data['all_shot_gt'],
                f'{target_player} — Conditional Landing Distribution Heatmap',
                'conditional_landing_heatmap.png'
            )
            
            print("  - Conditional move area distributions...")
            visualizer.plot_conditional_move_distributions(
                data['all_move_pred'], data['all_move_gt'],
                data['all_shot_pred'], data['all_shot_gt'],
                f'{target_player} — Conditional Move Area Distributions (P(Area|Shot Type))',
                'conditional_move_distributions.png'
            )
            
            if 'per_shot_mdc' in results['conditional_metrics']:
                print("  - Conditional MDC ratios...")
                visualizer.plot_conditional_mdc_bars(
                    results['conditional_metrics']['per_shot_mdc'],
                    f'{target_player} — Mean Distance from Center by Shot Type',
                    'conditional_mdc_ratios.png'
                )
            
            # Conditional JS divergence heatmaps (summary metrics)
            print("  - Conditional JS divergence summaries...")
            if 'per_shot_landing' in results['conditional_metrics']:
                visualizer.plot_conditional_heatmap(
                    results['conditional_metrics']['per_shot_landing'],
                    'landing_js',
                    f'{target_player} — Per-Shot Landing JS Divergence',
                    'per_shot_landing_js.png'
                )
            
            if 'per_prev_shot_transition' in results['conditional_metrics']:
                visualizer.plot_transition_heatmap(
                    results['conditional_metrics']['per_prev_shot_transition'],
                    f'{target_player} — Per-Previous-Shot Transition JS',
                    'per_prev_shot_transition.png'
                )
                
                # NEW: Actual transition matrices
                print("  - Transition matrices...")
                visualizer.plot_transition_matrices(
                    data['shot_pred_seqs'], data['shot_gt_seqs'],
                    f'{target_player} — Shot Transition Matrices P(Next|Previous)',
                    'transition_matrices.png'
                )
        
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
        if 'ctc' in results['sequence_metrics']:
            summary_data['metric'].append('shot_ctc')
            summary_data['value'].append(results['sequence_metrics']['ctc']['shot_ctc'])
        if 'edit_distance' in results['sequence_metrics']:
            summary_data['metric'].append('edit_distance')
            summary_data['value'].append(results['sequence_metrics']['edit_distance']['edit_distance'])
    
    if 'distribution_metrics' in results:
        summary_data['metric'].append('landing_area_js')
        summary_data['value'].append(results['distribution_metrics']['landing_area_js'])
        summary_data['metric'].append('shot_distribution_js')
        summary_data['value'].append(results['distribution_metrics']['shot_distribution_js'])
    
    if 'conditional_metrics' in results:
        summary_data['metric'].append('per_shot_landing_js')
        summary_data['value'].append(results['conditional_metrics']['per_shot_landing']['weighted_average'])
        summary_data['metric'].append('per_shot_move_js')
        summary_data['value'].append(results['conditional_metrics']['per_shot_move']['weighted_average'])
        summary_data['metric'].append('joint_area_shot_js')
        summary_data['value'].append(results['conditional_metrics']['joint_area_shot_js'])
    
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


RALLY_LEVEL_COLUMNS = [
    'trajectory_file', 'target_player', 'generation', 'num_matches', 'num_rallies',
    'num_gt_shots', 'num_pred_shots', 'early_end_rate',
    'mean_land_dtw', 'mean_move_dtw', 'mean_shot_ctc', 'mean_edit_distance',
]

DISTRIBUTION_METRIC_COLUMNS = [
    'landing_area_js', 'move_area_js', 'shot_distribution_js',
    'landing_mean_wasserstein', 'move_mean_wasserstein',
    'per_shot_landing_js', 'per_shot_move_js', 'per_prev_shot_transition_js',
    'joint_area_shot_js', 'mdc_ratio',
]

MATCH_LEVEL_COLUMNS = [
    'trajectory_file', 'target_player', 'generation', 'num_matches', 'num_rallies',
    'num_gt_shots', 'num_pred_shots', 'early_end_rate',
] + [f'match_avg_{column}' for column in DISTRIBUTION_METRIC_COLUMNS] + [
    f'rally_weighted_{column}' for column in DISTRIBUTION_METRIC_COLUMNS
]

SUMMARY_METRIC_COLUMNS = [
    'land_dtw', 'move_dtw', 'shot_ctc', 'edit_distance',
    'landing_area_js', 'move_area_js', 'shot_distribution_js',
    'landing_mean_wasserstein', 'move_mean_wasserstein',
    'per_shot_landing_js', 'per_shot_move_js', 'per_prev_shot_transition_js',
    'joint_area_shot_js', 'mdc_ratio',
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

    rally_rows = []
    match_groups = defaultdict(list)
    match_sequence_values = defaultdict(lambda: defaultdict(list))

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
            for metric_name in ['land_dtw', 'move_dtw', 'shot_ctc', 'edit_distance']:
                match_sequence_values[key][metric_name].append(row[metric_name])

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
            'mean_land_dtw': finite_mean([row.get('land_dtw') for row in rows]),
            'mean_move_dtw': finite_mean([row.get('move_dtw') for row in rows]),
            'mean_shot_ctc': finite_mean([row.get('shot_ctc') for row in rows]),
            'mean_edit_distance': finite_mean([row.get('edit_distance') for row in rows]),
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
        for metric_name in ['land_dtw', 'move_dtw', 'shot_ctc', 'edit_distance']
    }
    match_weights = [row['num_rallies'] for row in match_rows]

    summary_rows = []
    for average_type in ['match_average', 'rally_weighted_average']:
        row = {
            'average_type': average_type,
            'num_trajectory_files': len(trajectory_files),
            'num_matches': len(match_rows),
            'num_rallies': len(rally_rows),
            'num_gt_shots': sum(int(match_row['num_gt_shots']) for match_row in match_rows),
            'num_pred_shots': sum(int(match_row['num_pred_shots']) for match_row in match_rows),
        }

        if average_type == 'match_average':
            for metric_name in ['land_dtw', 'move_dtw', 'shot_ctc', 'edit_distance']:
                row[f'mean_{metric_name}'] = finite_mean(sequence_match_means[metric_name])
            for metric_name in SUMMARY_METRIC_COLUMNS[4:]:
                row[f'mean_{metric_name}'] = finite_mean([match_row.get(metric_name) for match_row in match_rows])
        else:
            for metric_name in ['land_dtw', 'move_dtw', 'shot_ctc', 'edit_distance']:
                row[f'mean_{metric_name}'] = finite_mean([rally_row.get(metric_name) for rally_row in rally_rows])
            for metric_name in SUMMARY_METRIC_COLUMNS[4:]:
                row[f'mean_{metric_name}'] = finite_mean(
                    [match_row.get(metric_name) for match_row in match_rows],
                    weights=match_weights,
                )

        summary_rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    rally_csv = os.path.join(output_dir, 'rally_level_metrics.csv')
    match_csv = os.path.join(output_dir, 'match_level_metrics.csv')
    summary_csv = os.path.join(output_dir, 'experiment_summary.csv')
    json_path = os.path.join(output_dir, 'experiment_metrics.json')

    write_csv_rows(rally_csv, player_rally_rows, RALLY_LEVEL_COLUMNS)
    write_csv_rows(match_csv, player_match_rows, MATCH_LEVEL_COLUMNS)
    write_csv_rows(summary_csv, summary_rows, EXPERIMENT_SUMMARY_COLUMNS)

    results = {
        'trajectory_dir': trajectory_dir,
        'latest_only': latest_only,
        'target_player_only': target_player_only,
        'trajectory_files': trajectory_files,
        'rally_level_metrics': player_rally_rows,
        'match_level_metrics': player_match_rows,
        'experiment_summary': summary_rows,
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    rally_metric_specs = [
        ('land_dtw', 'mean_land_dtw', 'mean_land_dtw'),
        ('move_dtw', 'mean_move_dtw', 'mean_move_dtw'),
        ('shot_ctc', 'mean_shot_ctc', 'mean_shot_ctc'),
        ('edit_distance', 'mean_edit_distance', 'mean_edit_distance'),
    ]
    match_metric_specs = [
        ('landing_area_js', 'match_avg_landing_area_js', 'mean_landing_area_js'),
        ('move_area_js', 'match_avg_move_area_js', 'mean_move_area_js'),
        ('shot_distribution_js', 'match_avg_shot_distribution_js', 'mean_shot_distribution_js'),
        ('landing_wasserstein', 'match_avg_landing_mean_wasserstein', 'mean_landing_mean_wasserstein'),
        ('move_wasserstein', 'match_avg_move_mean_wasserstein', 'mean_move_mean_wasserstein'),
        ('per_shot_landing_js', 'match_avg_per_shot_landing_js', 'mean_per_shot_landing_js'),
        ('per_shot_move_js', 'match_avg_per_shot_move_js', 'mean_per_shot_move_js'),
        ('transition_js', 'match_avg_per_prev_shot_transition_js', 'mean_per_prev_shot_transition_js'),
        ('joint_area_shot_js', 'match_avg_joint_area_shot_js', 'mean_joint_area_shot_js'),
        ('mdc_ratio', 'match_avg_mdc_ratio', 'mean_mdc_ratio'),
    ]

    print_player_metric_table(
        'Rally-Level Metrics By Player (sequence)',
        player_rally_rows,
        rally_metric_specs,
        summary_rows,
        max_rows=print_max_rows,
    )
    print_player_metric_table(
        'Match-Level Metrics By Player (distribution)',
        player_match_rows,
        match_metric_specs,
        summary_rows,
        max_rows=print_max_rows,
    )

    print()
    print(f'Rally-level CSV: {rally_csv}')
    print(f'Match-level CSV: {match_csv}')
    print(f'Experiment summary CSV: {summary_csv}')
    print(f'Experiment metrics JSON: {json_path}')
    print('=' * 80)
    print('Experiment Evaluation Complete!')
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
        --metrics dtw ctc edit_distance

    # Evaluate only conditional distributions (key feature)
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/twin/trajectories/An_Se_Young_norm.pkl \\
        --metrics conditional \\
        --output-dir results/conditional/

    # Include both players (not just target player)
    python Scripts/single_player_eval/evaluate_trajectories.py \\
        --trajectory-file Experiments/naive_rag/trajectories/*.pkl \\
        --all-players

    # Evaluate every trajectory in an experiment directory
    python Scripts/single_player_eval/evaluate_trajectories.py \
        --trajectory-dir Experiments/twin/trajectories/norm \
        --output-dir Experiments/twin/results/trajectory_analysis/norm \
        --latest-only \
        --no-visualize

Supported Metrics:
    all              - Compute all metrics
    dtw              - Dynamic Time Warping for landing/move positions
    ctc              - Connectionist Temporal Classification for shot sequences
    edit_distance    - Levenshtein edit distance for shot sequences
    distribution     - JS divergence, Wasserstein distance
    conditional      - Per-shot-type conditional distributions (KEY FEATURE)
        """
    )
    
    parser.add_argument('--trajectory-file', type=str,
                        help='Path to one trajectory pickle file')
    parser.add_argument('--trajectory-dir', type=str,
                        help='Directory containing trajectory pickle files for experiment-level analysis')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results and visualizations')
    parser.add_argument('--metrics', nargs='+', 
                        choices=['all', 'dtw', 'ctc', 'edit_distance', 'distribution', 'conditional'],
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
    parser.add_argument('--print-max-rows', type=int, default=50,
                        help='Maximum rows to print per table in --trajectory-dir mode; use 0 to print all rows')
    
    args = parser.parse_args()

    if bool(args.trajectory_file) == bool(args.trajectory_dir):
        print("Error: specify exactly one of --trajectory-file or --trajectory-dir")
        return 1

    if args.trajectory_dir:
        if not os.path.isdir(args.trajectory_dir):
            print(f"Error: Trajectory directory not found: {args.trajectory_dir}")
            return 1
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
