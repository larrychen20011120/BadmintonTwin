"""
Extended evaluation metrics for single player assessment.

Metrics categories:
  A. Distribution   (1-5):  area JS, move JS, shot JS, mean distance from center, wasserstein
  B. Structural     (6-8):  shot transition diff, n-gram overlap, area transition diff
  C. Coverage       (9-11): area coverage, spatial entropy, shot diversity
  D. Joint          (12-14): joint area-shot JS, per-shot area JS, per-shot F1
  E. Behavioral     (15-18): rally length JS, inter-shot distance JS, serve accuracy, scoring efficiency

All area-based metrics use the 18 area groups defined in Utils/badminton_config.py
(Deep Left Corner, Rear Center, Front Left Side, etc.) for consistency with the LLM prompts.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import os
import sys
_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_project_path)
from Utils.helper import coord2area
from Utils.badminton_config import area_groups


# ---------------------------------------------------------------------------
# Area group mapping (33 fine zones → 18 semantic groups)
# ---------------------------------------------------------------------------

# Ordered list of group names (deterministic order for histograms)
_GROUP_NAMES = [
    "Deep Left Corner",       # 0
    "Deep Right Corner",      # 1
    "Rear Left Side",         # 2
    "Rear Right Side",        # 3
    "Rear Center",            # 4
    "Mid Left Side",          # 5
    "Mid Right Side",         # 6
    "Mid Center",             # 7
    "Front Left Side",        # 8
    "Front Right Side",       # 9
    "Front Center",           # 10
    "Front Left Side Line",   # 11
    "Front Right Side Line",  # 12
    "Mid Left Side Line",     # 13
    "Mid Right Side Line",    # 14
    "Rear Left Side Line",    # 15
    "Rear Right Side Line",   # 16
    "Out of Bounds",          # 17
]
N_GROUPS = len(_GROUP_NAMES)  # 18
N_VALID_GROUPS = N_GROUPS - 1  # 17 (exclude Out of Bounds)

# Build area_id (0-33) → group_id (0-17)
_AREA_TO_GROUP = {}
for gid, name in enumerate(_GROUP_NAMES):
    for area_id in area_groups.get(name, [33]):
        _AREA_TO_GROUP[area_id] = gid
# Any unmapped area → Out of Bounds (17)
for a in range(34):
    _AREA_TO_GROUP.setdefault(a, 17)


def coord2group(norm_x, norm_y):
    """Convert normalized coordinates to area group id (0-17)."""
    area_id = coord2area(norm_x, norm_y)
    return _AREA_TO_GROUP.get(area_id, 17)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def js_divergence(p, q):
    """Jensen-Shannon divergence (base 2) between two discrete distributions."""
    return float(jensenshannon(p, q, base=2.0) ** 2)


def histogram(data, bins, range_min, range_max):
    """Normalized histogram with Laplace (+1) smoothing."""
    hist, _ = np.histogram(data, bins=bins, range=(range_min, range_max))
    hist = hist.astype(np.float64) + 1.0
    return hist / hist.sum()


def build_transition_matrix(seq, n_classes):
    """Build row-normalized transition matrix from integer sequence."""
    T = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(len(seq) - 1):
        T[int(seq[i]), int(seq[i + 1])] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def _denorm_landing(norm_coords):
    """Convert normalized landing coords [-1,1] to actual court coords [0,355]x[0,480]."""
    x = (norm_coords[:, 0] + 1.0) * 177.5
    y = (norm_coords[:, 1] + 1.0) * 240.0
    return np.column_stack([x, y])


def _denorm_move(norm_coords):
    """Convert normalized move coords to actual: x in [0,355], y in [-480,0]."""
    x = (norm_coords[:, 0] + 1.0) * 177.5
    y = norm_coords[:, 1] * 240.0 - 240.0
    return np.column_stack([x, y])


# ===================================================================
# A. Distribution Metrics
# ===================================================================

def area_distribution_js(land_pred, land_gt):
    """JS divergence of landing area GROUP distributions (18 groups)."""
    area_pred = np.array([coord2group(x, y) for x, y in land_pred])
    area_gt = np.array([coord2group(x, y) for x, y in land_gt])
    p = histogram(area_pred, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
    q = histogram(area_gt, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
    return js_divergence(p, q)


def move_distribution_js(move_pred, move_gt):
    """JS divergence of move area group distributions."""
    area_pred = np.array([coord2group(x, y) for x, y in move_pred])
    area_gt = np.array([coord2group(x, y) for x, y in move_gt])
    p = histogram(area_pred, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
    q = histogram(area_gt, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
    return js_divergence(p, q)


def shot_distribution_js(shot_pred, shot_gt):
    """JS divergence of shot type distributions (12 bins, types 0-11)."""
    p = histogram(shot_pred, bins=12, range_min=0, range_max=12)
    q = histogram(shot_gt, bins=12, range_min=0, range_max=12)
    return js_divergence(p, q)


def mean_distance_from_center(land_pred, move_pred, land_gt, move_gt):
    """Mean Euclidean distance from court center (normalized space).

    Center = (0, 0) in normalized coords.
    land_ratio < 1 means model predicts closer to center than ground truth.
    """
    center = np.array([0.0, 0.0])
    land_pred_dist = float(np.mean(np.linalg.norm(land_pred - center, axis=1)))
    land_gt_dist = float(np.mean(np.linalg.norm(land_gt - center, axis=1)))
    move_pred_dist = float(np.mean(np.linalg.norm(move_pred - center, axis=1)))
    move_gt_dist = float(np.mean(np.linalg.norm(move_gt - center, axis=1)))
    return {
        'land_pred_mean_dist': land_pred_dist,
        'land_gt_mean_dist': land_gt_dist,
        'land_ratio': land_pred_dist / max(land_gt_dist, 1e-8),
        'move_pred_mean_dist': move_pred_dist,
        'move_gt_mean_dist': move_gt_dist,
        'move_ratio': move_pred_dist / max(move_gt_dist, 1e-8),
    }


def wasserstein_1d_marginals(land_pred, land_gt):
    """1D Wasserstein (Earth Mover) distance on x and y marginals."""
    x_dist = float(wasserstein_distance(land_pred[:, 0], land_gt[:, 0]))
    y_dist = float(wasserstein_distance(land_pred[:, 1], land_gt[:, 1]))
    return {
        'x_wasserstein': x_dist,
        'y_wasserstein': y_dist,
        'mean_wasserstein': (x_dist + y_dist) / 2.0,
    }


# ===================================================================
# B. Structural Metrics
# ===================================================================

def shot_transition_matrix_diff(shot_pred, shot_gt):
    """Normalised Frobenius norm between 12x12 shot transition matrices."""
    T_pred = build_transition_matrix(shot_pred, 12)
    T_gt = build_transition_matrix(shot_gt, 12)
    diff = np.linalg.norm(T_pred - T_gt, 'fro')
    max_diff = np.sqrt(2 * 12)
    return float(diff / max_diff)


def ngram_overlap(seq_pred, seq_gt, n=2):
    """BLEU-like n-gram precision, recall, F1 on shot type sequences."""
    if len(seq_pred) < n or len(seq_gt) < n:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    pred_ngrams = set(tuple(seq_pred[i:i + n]) for i in range(len(seq_pred) - n + 1))
    gt_ngrams = set(tuple(seq_gt[i:i + n]) for i in range(len(seq_gt) - n + 1))

    if len(pred_ngrams) == 0 or len(gt_ngrams) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    overlap = pred_ngrams & gt_ngrams
    precision = len(overlap) / len(pred_ngrams)
    recall = len(overlap) / len(gt_ngrams)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def area_transition_matrix_diff(land_pred, land_gt):
    """Normalised Frobenius norm between 18x18 area group transition matrices."""
    area_pred = np.array([coord2group(x, y) for x, y in land_pred])
    area_gt = np.array([coord2group(x, y) for x, y in land_gt])
    T_pred = build_transition_matrix(area_pred, N_GROUPS)
    T_gt = build_transition_matrix(area_gt, N_GROUPS)
    diff = np.linalg.norm(T_pred - T_gt, 'fro')
    max_diff = np.sqrt(2 * N_GROUPS)
    return float(diff / max_diff)


# ===================================================================
# C. Coverage / Diversity Metrics
# ===================================================================

def area_coverage(land_coords):
    """Proportion of the 17 valid court area groups covered."""
    groups = np.array([coord2group(x, y) for x, y in land_coords])
    # Exclude Out of Bounds (group 17) from coverage count
    valid_groups = groups[groups != 17]
    unique_count = len(np.unique(valid_groups))
    return {
        'unique_count': unique_count,
        'coverage_ratio': unique_count / N_VALID_GROUPS,
    }


def spatial_entropy(land_coords):
    """Shannon entropy (base 2) of area group distribution."""
    groups = np.array([coord2group(x, y) for x, y in land_coords])
    h = histogram(groups, bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
    h = h[h > 0]
    return float(-np.sum(h * np.log2(h)))


def shot_diversity(shot_types):
    """Diversity of shot type usage (types 1-10 usable, 11 is miss)."""
    valid = shot_types[shot_types != 0]
    unique_count = len(np.unique(valid))
    return {
        'unique_count': unique_count,
        'diversity_ratio': unique_count / 10.0,
    }


# ===================================================================
# D. Joint Distribution Metrics
# ===================================================================

def joint_area_shot_js(land_pred, shot_pred, land_gt, shot_gt):
    """JS divergence of joint (area group, shot_type) distribution (18 x 12 bins)."""
    area_pred = np.array([coord2group(x, y) for x, y in land_pred])
    area_gt = np.array([coord2group(x, y) for x, y in land_gt])

    joint_pred, _, _ = np.histogram2d(
        area_pred, shot_pred, bins=[N_GROUPS, 12], range=[[0, N_GROUPS], [0, 12]])
    joint_gt, _, _ = np.histogram2d(
        area_gt, shot_gt, bins=[N_GROUPS, 12], range=[[0, N_GROUPS], [0, 12]])

    p = joint_pred.flatten().astype(np.float64) + 1.0
    p /= p.sum()
    q = joint_gt.flatten().astype(np.float64) + 1.0
    q /= q.sum()

    return js_divergence(p, q)


def per_shot_type_area_js(land_pred, shot_pred, land_gt, shot_gt):
    """For each shot type (1-11), compute landing area group JS; return weighted avg."""
    area_pred = np.array([coord2group(x, y) for x, y in land_pred])
    area_gt = np.array([coord2group(x, y) for x, y in land_gt])

    weighted_js = 0.0
    total_weight = 0
    per_shot = {}

    for shot in range(1, 12):
        mask_gt = shot_gt == shot
        if np.sum(mask_gt) == 0:
            continue

        mask_pred = shot_pred == shot
        p = histogram(area_pred[mask_pred], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        q = histogram(area_gt[mask_gt], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        js = js_divergence(p, q)
        weight = float(np.sum(mask_gt))
        weighted_js += js * weight
        total_weight += weight
        per_shot[str(shot)] = js

    return {
        'weighted_average': float(weighted_js / max(total_weight, 1e-8)),
        'per_shot': per_shot,
    }


def per_shot_type_f1(shot_pred, shot_gt):
    """Precision, Recall, F1 per shot type (1-11) + macro average."""
    results = {}
    for shot in range(1, 12):
        tp = int(np.sum((shot_pred == shot) & (shot_gt == shot)))
        fp = int(np.sum((shot_pred == shot) & (shot_gt != shot)))
        fn = int(np.sum((shot_pred != shot) & (shot_gt == shot)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        results[str(shot)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(shot_gt == shot)),
        }

    valid_f1s = [v['f1'] for v in results.values() if v['support'] > 0]
    results['macro_avg_f1'] = float(np.mean(valid_f1s)) if valid_f1s else 0.0
    return results


# ===================================================================
# D2. Conditional Metrics (per shot type)
# ===================================================================

def per_shot_landing_js(land_pred, land_gt, shot_pred, shot_gt):
    """For each shot type (1-11), JS of landing area group distribution.
    Returns a dict mapping shot_type string → JS float.
    """
    area_pred = np.array([coord2group(x, y) for x, y in land_pred])
    area_gt = np.array([coord2group(x, y) for x, y in land_gt])
    results = {}
    for shot in range(1, 12):
        mask_pred = shot_pred == shot
        mask_gt = shot_gt == shot
        if np.sum(mask_gt) == 0:
            continue
        p = histogram(area_pred[mask_pred], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        q = histogram(area_gt[mask_gt], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        results[str(shot)] = js_divergence(p, q)
    return results


def per_shot_move_js(move_pred, move_gt, shot_pred, shot_gt):
    """For each shot type (1-11), JS of move area group distribution."""
    area_pred = np.array([coord2group(x, y) for x, y in move_pred])
    area_gt = np.array([coord2group(x, y) for x, y in move_gt])
    results = {}
    for shot in range(1, 12):
        mask_pred = shot_pred == shot
        mask_gt = shot_gt == shot
        if np.sum(mask_gt) == 0:
            continue
        p = histogram(area_pred[mask_pred], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        q = histogram(area_gt[mask_gt], bins=N_GROUPS, range_min=0, range_max=N_GROUPS)
        results[str(shot)] = js_divergence(p, q)
    return results


def per_shot_landing_mdc(land_pred, land_gt, shot_pred, shot_gt):
    """For each shot type (1-11), MDC ratio (pred/gt mean distance from center).
    ratio < 1 means model predicts closer to center than GT for that shot type.
    Returns a dict mapping shot_type string → float ratio.
    """
    center = np.array([0.0, 0.0])
    results = {}
    for shot in range(1, 12):
        mask_pred = shot_pred == shot
        mask_gt = shot_gt == shot
        if np.sum(mask_gt) == 0:
            continue
        pred_dist = float(np.mean(np.linalg.norm(land_pred[mask_pred] - center, axis=1)))
        gt_dist = float(np.mean(np.linalg.norm(land_gt[mask_gt] - center, axis=1)))
        results[str(shot)] = pred_dist / max(gt_dist, 1e-8)
    return results


# ===================================================================
# E. Behavioral Metrics
# ===================================================================

def rally_length_divergence(lengths_pred, lengths_gt):
    """JS divergence of rally length distributions."""
    max_len = int(max(np.max(lengths_pred), np.max(lengths_gt))) + 2
    p = histogram(lengths_pred, bins=max_len, range_min=0, range_max=max_len)
    q = histogram(lengths_gt, bins=max_len, range_min=0, range_max=max_len)
    return js_divergence(p, q)


def inter_shot_distance_js(land_pred_seqs, land_gt_seqs):
    """JS divergence of consecutive landing distances (per-rally).

    land_pred_seqs, land_gt_seqs: list of (M_i, 2) numpy arrays.
    """
    all_dists_pred = []
    all_dists_gt = []

    for seq in land_pred_seqs:
        if len(seq) >= 2:
            dists = np.sqrt(np.sum(np.diff(seq, axis=0) ** 2, axis=1))
            all_dists_pred.extend(dists.tolist())
    for seq in land_gt_seqs:
        if len(seq) >= 2:
            dists = np.sqrt(np.sum(np.diff(seq, axis=0) ** 2, axis=1))
            all_dists_gt.extend(dists.tolist())

    if len(all_dists_pred) == 0 or len(all_dists_gt) == 0:
        return float('inf')

    all_dists_pred = np.array(all_dists_pred)
    all_dists_gt = np.array(all_dists_gt)
    max_dist = max(np.max(all_dists_pred), np.max(all_dists_gt)) + 1e-8
    bins = 20

    p = histogram(all_dists_pred, bins=bins, range_min=0, range_max=max_dist)
    q = histogram(all_dists_gt, bins=bins, range_min=0, range_max=max_dist)
    return js_divergence(p, q)


def serve_accuracy(shot_pred, shot_gt, ball_rounds):
    """Shot type accuracy on serve shots (ball_round == 1) only."""
    ball_rounds = np.array(ball_rounds)
    serve_mask = ball_rounds == 1
    if np.sum(serve_mask) == 0:
        return {'serve_accuracy': None, 'serve_count': 0}
    correct = int(np.sum(shot_pred[serve_mask] == shot_gt[serve_mask]))
    return {
        'serve_accuracy': float(correct / np.sum(serve_mask)),
        'serve_count': int(np.sum(serve_mask)),
    }


def scoring_efficiency(shot_seqs, won_flags, offensive_types=(4, 9)):
    """Win rate in rallies where the player used offensive shots (smash=4, drop=9).

    shot_seqs:  list of per-rally shot type lists (int)
    won_flags:  parallel list of bool (True if player won the rally)
    """
    off_wins = 0
    off_total = 0
    all_wins = 0
    all_total = len(won_flags)

    for shot_seq, won in zip(shot_seqs, won_flags):
        if won:
            all_wins += 1
        if len(shot_seq) > 0 and any(s in offensive_types for s in shot_seq):
            off_total += 1
            if won:
                off_wins += 1

    return {
        'offensive_shot_win_rate': float(off_wins / max(off_total, 1)),
        'baseline_win_rate': float(all_wins / max(all_total, 1)),
        'offensive_rally_count': off_total,
        'total_rally_count': all_total,
    }
