"""
Visualization utilities for single player evaluation.

Generates matplotlib PNGs comparing predicted vs ground truth distributions.
Landing / Move: KDE filled contour plots overlaid on the badminton court.
Shot Distribution: side-by-side bar chart.
Transition Matrix: side-by-side 11×11 heatmap.
Joint Heatmap: area group (y) × shot type (x) with semantic labels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

_pp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _pp)


def _denorm_landing(norm_coords):
    """Convert [-1,1] landing coords to actual court coords [0,355]x[0,480]."""
    x = (norm_coords[:, 0] + 1.0) * 177.5
    y = (norm_coords[:, 1] + 1.0) * 240.0
    return np.column_stack([x, y])


def _denorm_move(norm_coords):
    """Convert [-1,1] move coords to [0,355]x[-480,0]."""
    x = (norm_coords[:, 0] + 1.0) * 177.5
    y = norm_coords[:, 1] * 240.0 - 240.0
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Court line drawing helpers
# ---------------------------------------------------------------------------

def _draw_landing_court(ax):
    """Draw badminton court lines for the landing half (y: 0 → 480, upper)."""
    ax.plot([0, 355], [0, 0], 'k-', linewidth=1.2)         # net
    ax.plot([0, 355], [480, 480], 'k-', linewidth=1.2)     # baseline
    ax.plot([0, 0], [0, 480], 'k-', linewidth=1.2)         # left sideline
    ax.plot([355, 355], [0, 480], 'k-', linewidth=1.2)     # right sideline
    ax.plot([177.5, 177.5], [0, 480], 'k--', linewidth=0.8, alpha=0.5)  # center
    # Short service line: 1.98 m from net → 141.85 px
    ax.plot([0, 355], [141.85, 141.85], 'k-', linewidth=0.8, alpha=0.6)
    # Long service line (doubles) ≈ 425.8 px
    ax.plot([0, 355], [425.78, 425.78], 'k:', linewidth=0.7, alpha=0.4)


def _draw_move_court(ax):
    """Draw badminton court lines for the move half (y: -480 → 0, lower)."""
    ax.plot([0, 355], [0, 0], 'k-', linewidth=1.2)          # net
    ax.plot([0, 355], [-480, -480], 'k-', linewidth=1.2)    # baseline
    ax.plot([0, 0], [-480, 0], 'k-', linewidth=1.2)         # left sideline
    ax.plot([355, 355], [-480, 0], 'k-', linewidth=1.2)     # right sideline
    ax.plot([177.5, 177.5], [-480, 0], 'k--', linewidth=0.8, alpha=0.5)  # center
    ax.plot([0, 355], [-141.85, -141.85], 'k-', linewidth=0.8, alpha=0.6)
    ax.plot([0, 355], [-425.78, -425.78], 'k:', linewidth=0.7, alpha=0.4)


def _draw_court_grid(ax, y_min, y_max):
    """Draw zone boundary grid lines (thin dashed) for the 18 area groups."""
    # X boundaries (normalized → court pixels): -1.0, -0.718, -0.414, 0, 0.414, 0.718, 1.0
    x_norm_bounds = [-1.0, -0.718, -0.414, 0.0, 0.414, 0.718, 1.0]
    x_court = [(xb + 1.0) * 177.5 for xb in x_norm_bounds]
    for xc in x_court[1:-1]:  # skip outer boundaries (already drawn as sidelines)
        ax.axvline(xc, color='gray', linestyle=':', linewidth=0.4, alpha=0.4)

    # Y boundaries (normalized → court pixels):
    # [-1.0, -0.375, -0.15, 0.075, 0.3, 0.525, 0.7625, 1.0]
    # Convert to y in [y_min, y_max]:  n_y 0→center of court
    y_norm_bounds = [-1.0, -0.375, -0.15, 0.075, 0.3, 0.525, 0.7625, 1.0]
    for yb in y_norm_bounds[1:-1]:
        # For landing: y_court = (yb + 1) * 240 → [0, 480]
        # For move:    y_court = yb * 240 - 240  → [-480, 0]
        y_c = yb * (y_max - y_min) / 2.0 + (y_max + y_min) / 2.0
        ax.axhline(y_c, color='gray', linestyle=':', linewidth=0.4, alpha=0.4)


# ---------------------------------------------------------------------------
# KDE helper
# ---------------------------------------------------------------------------

def _kde_mesh(coords, x_range, y_range, grid=80):
    """Compute KDE density on a mesh grid. Returns (X, Y, Z)."""
    from scipy.stats import gaussian_kde
    if len(coords) < 2:
        return None, None, None
    try:
        kde = gaussian_kde(coords.T, bw_method='scott')
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(pts).reshape(X.shape)
    return X, Y, Z


# ===================================================================
# V1. Landing Position KDE
# ===================================================================

def plot_landing_heatmap(land_pred, land_gt, output_path, title):
    """Side-by-side KDE filled contour of landing positions on the court."""
    dp = _denorm_landing(land_pred)
    dg = _denorm_landing(land_gt)

    x_range = (0, 355)
    y_range = (0, 480)

    X_p, Y_p, Z_p = _kde_mesh(dp, x_range, y_range)
    X_g, Y_g, Z_g = _kde_mesh(dg, x_range, y_range)

    vmax = 0
    if Z_p is not None:
        vmax = max(vmax, Z_p.max())
    if Z_g is not None:
        vmax = max(vmax, Z_g.max())
    if vmax == 0:
        vmax = 1e-6

    levels = np.linspace(0, vmax, 10)[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Predicted
    if Z_p is not None:
        cs1 = ax1.contourf(X_p, Y_p, Z_p, levels=levels, cmap='Blues', alpha=0.8)
        ax1.contour(X_p, Y_p, Z_p, levels=levels[::2], colors='navy', linewidths=0.4, alpha=0.5)
    _draw_landing_court(ax1)
    _draw_court_grid(ax1, 0, 480)
    ax1.set_xlim(*x_range)
    ax1.set_ylim(*y_range)
    ax1.set_aspect('equal')
    ax1.set_title(f'Predicted\n({len(land_pred)} shots)')

    # Ground Truth
    if Z_g is not None:
        cs2 = ax2.contourf(X_g, Y_g, Z_g, levels=levels, cmap='Reds', alpha=0.8)
        ax2.contour(X_g, Y_g, Z_g, levels=levels[::2], colors='darkred', linewidths=0.4, alpha=0.5)
    _draw_landing_court(ax2)
    _draw_court_grid(ax2, 0, 480)
    ax2.set_xlim(*x_range)
    ax2.set_ylim(*y_range)
    ax2.set_aspect('equal')
    ax2.set_title(f'Ground Truth\n({len(land_gt)} shots)')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# V2. Move Position KDE
# ===================================================================

def plot_move_heatmap(move_pred, move_gt, output_path, title):
    """Side-by-side KDE filled contour of move positions (player base) on the court."""
    dp = _denorm_move(move_pred)
    dg = _denorm_move(move_gt)

    x_range = (0, 355)
    y_range = (-480, 0)

    X_p, Y_p, Z_p = _kde_mesh(dp, x_range, y_range)
    X_g, Y_g, Z_g = _kde_mesh(dg, x_range, y_range)

    vmax = 0
    if Z_p is not None:
        vmax = max(vmax, Z_p.max())
    if Z_g is not None:
        vmax = max(vmax, Z_g.max())
    if vmax == 0:
        vmax = 1e-6

    levels = np.linspace(0, vmax, 10)[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if Z_p is not None:
        ax1.contourf(X_p, Y_p, Z_p, levels=levels, cmap='Blues', alpha=0.8)
        ax1.contour(X_p, Y_p, Z_p, levels=levels[::2], colors='navy', linewidths=0.4, alpha=0.5)
    _draw_move_court(ax1)
    _draw_court_grid(ax1, -480, 0)
    ax1.set_xlim(*x_range)
    ax1.set_ylim(*y_range)
    ax1.set_aspect('equal')
    ax1.set_title(f'Predicted\n({len(move_pred)} shots)')

    if Z_g is not None:
        ax2.contourf(X_g, Y_g, Z_g, levels=levels, cmap='Reds', alpha=0.8)
        ax2.contour(X_g, Y_g, Z_g, levels=levels[::2], colors='darkred', linewidths=0.4, alpha=0.5)
    _draw_move_court(ax2)
    _draw_court_grid(ax2, -480, 0)
    ax2.set_xlim(*x_range)
    ax2.set_ylim(*y_range)
    ax2.set_aspect('equal')
    ax2.set_title(f'Ground Truth\n({len(move_gt)} shots)')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# V3. Shot Type Distribution (unchanged bar chart)
# ===================================================================

_SHOT_LABELS = {
    1: 'Serve short', 2: 'Clear', 3: 'Push Shot', 4: 'Smash',
    5: 'Smash Def', 6: 'Drive', 7: 'Net Shot', 8: 'Lob',
    9: 'Drop', 10: 'Serve long', 11: 'Missed',
}


def plot_shot_distribution(shot_pred, shot_gt, output_path, title):
    """Side-by-side bar chart of shot type frequencies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    shot_types = np.arange(1, 12)
    labels = [_SHOT_LABELS.get(s, str(s)) for s in shot_types]

    counts_pred = [int(np.sum(shot_pred == s)) for s in shot_types]
    counts_gt = [int(np.sum(shot_gt == s)) for s in shot_types]

    x = np.arange(len(shot_types))
    w = 0.6

    ax1.bar(x, counts_pred, w, color='steelblue', edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_title(f'Predicted ({len(shot_pred)} shots)')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, max(max(counts_pred), max(counts_gt)) * 1.15)

    ax2.bar(x, counts_gt, w, color='firebrick', edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_title(f'Ground Truth ({len(shot_gt)} shots)')
    ax2.set_ylim(0, max(max(counts_pred), max(counts_gt)) * 1.15)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# V4. Shot Transition Matrix (unchanged)
# ===================================================================

def plot_transition_matrix(shot_pred, shot_gt, output_path, title):
    """Side-by-side 11×11 transition matrix heatmap (GT vs Pred)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    def _build_T(seq):
        T = np.zeros((11, 11))
        for i in range(len(seq) - 1):
            s1, s2 = int(seq[i]), int(seq[i + 1])
            if 1 <= s1 <= 11 and 1 <= s2 <= 11:
                T[s1 - 1, s2 - 1] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return T / row_sums

    T_pred = _build_T(shot_pred)
    T_gt = _build_T(shot_gt)

    labels = [_SHOT_LABELS.get(i, str(i)) for i in range(1, 12)]

    im1 = ax1.imshow(T_pred, cmap='Blues', vmin=0, vmax=max(T_pred.max(), T_gt.max()), aspect='auto')
    ax1.set_xticks(range(11))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(11))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_title('Predicted Transition Matrix')
    ax1.set_xlabel('Next Shot')
    ax1.set_ylabel('Current Shot')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(T_gt, cmap='Reds', vmin=0, vmax=max(T_pred.max(), T_gt.max()), aspect='auto')
    ax2.set_xticks(range(11))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(11))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_title('Ground Truth Transition Matrix')
    ax2.set_xlabel('Next Shot')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# V5. Joint (Area Group × Shot Type) Heatmap
# ===================================================================

_GROUP_NAMES = [
    "Deep Left Cnr", "Deep Right Cnr",
    "Rear Left", "Rear Right", "Rear Center",
    "Mid Left", "Mid Right", "Mid Center",
    "Front Left", "Front Right", "Front Center",
    "FL Line", "FR Line", "ML Line", "MR Line", "RL Line", "RR Line",
    "Out",
]


def _denorm_coord_to_group(land_coords):
    """Convert normalized coords → area group id (0-17)."""
    from Utils.helper import coord2area
    from Utils.badminton_config import area_groups

    _ATOG = {}
    for gid, name in enumerate([
        "Deep Left Corner", "Deep Right Corner",
        "Rear Left Side", "Rear Right Side", "Rear Center",
        "Mid Left Side", "Mid Right Side", "Mid Center",
        "Front Left Side", "Front Right Side", "Front Center",
        "Front Left Side Line", "Front Right Side Line",
        "Mid Left Side Line", "Mid Right Side Line",
        "Rear Left Side Line", "Rear Right Side Line",
        "Out of Bounds",
    ]):
        for area_id in area_groups.get(name, [33]):
            _ATOG[area_id] = gid
    for a in range(34):
        _ATOG.setdefault(a, 17)

    groups = []
    for x, y in land_coords:
        aid = coord2area(x, y)
        groups.append(_ATOG.get(aid, 17))
    return np.array(groups, dtype=int)


def plot_joint_heatmap(land_pred, shot_pred, land_gt, shot_gt, output_path, title):
    """Joint heatmap: y-axis = area group label, x-axis = shot type (1-11)."""
    area_pred = _denorm_coord_to_group(land_pred)
    area_gt = _denorm_coord_to_group(land_gt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    def _build_joint(areas, shots):
        J = np.zeros((18, 12))
        for a, s in zip(areas, shots):
            a_i = int(np.clip(a, 0, 17))
            s_i = int(np.clip(s, 0, 11))
            J[a_i, s_i] += 1
        return J

    J_pred = _build_joint(area_pred, shot_pred)
    J_gt = _build_joint(area_gt, shot_gt)
    vmax = max(J_pred.max(), J_gt.max())

    im1 = ax1.imshow(J_pred, cmap='Blues', aspect='auto', vmin=0, vmax=vmax, origin='lower')
    ax1.set_yticks(range(18))
    ax1.set_yticklabels(_GROUP_NAMES, fontsize=7)
    ax1.set_xlabel('Shot Type')
    ax1.set_title('Predicted Joint Distribution')

    im2 = ax2.imshow(J_gt, cmap='Reds', aspect='auto', vmin=0, vmax=vmax, origin='lower')
    ax2.set_yticks(range(18))
    ax2.set_yticklabels(_GROUP_NAMES, fontsize=7)
    ax2.set_xlabel('Shot Type')
    ax2.set_title('Ground Truth Joint Distribution')

    plt.colorbar(im1, ax=ax1, fraction=0.046)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# V6. Per-Shot Landing KDE Grid
# ===================================================================

def plot_per_shot_landing_kde(land_pred, land_gt, shot_pred, shot_gt, output_path, title):
    """3×4 grid: per-shot-type landing KDE contours, GT (red) + Pred (blue) overlay."""
    fig, axes = plt.subplots(3, 4, figsize=(28, 18))
    axes = axes.flatten()

    shot_order = list(range(1, 12)) + ['ALL']
    x_range = (0, 355)
    y_range = (0, 480)

    global_vmax = 0
    all_Z = []
    for shot in shot_order:
        if shot == 'ALL':
            mask_pred = np.ones(len(shot_pred), dtype=bool)
            mask_gt = np.ones(len(shot_gt), dtype=bool)
        else:
            mask_pred = shot_pred == shot
            mask_gt = shot_gt == shot
        dp = _denorm_landing(land_pred[mask_pred]) if np.sum(mask_pred) > 0 else np.zeros((0, 2))
        dg = _denorm_landing(land_gt[mask_gt]) if np.sum(mask_gt) > 0 else np.zeros((0, 2))
        _, _, Z_p = _kde_mesh(dp, x_range, y_range, grid=60)
        _, _, Z_g = _kde_mesh(dg, x_range, y_range, grid=60)
        if Z_p is not None:
            global_vmax = max(global_vmax, Z_p.max())
        if Z_g is not None:
            global_vmax = max(global_vmax, Z_g.max())
        all_Z.append((dp, dg, Z_p, Z_g))

    if global_vmax == 0:
        global_vmax = 1e-6
    levels = np.linspace(0, global_vmax, 8)[1:]

    for idx, shot in enumerate(shot_order):
        ax = axes[idx]
        dp, dg, Z_p, Z_g = all_Z[idx]

        if Z_g is not None and len(dg) >= 2:
            ax.contourf(Z_g, levels=levels, cmap='Reds', alpha=0.6,
                        extent=[0, 355, 0, 480])
            ax.contour(Z_g, levels=levels[::2], colors='darkred', linewidths=0.3,
                       alpha=0.4, extent=[0, 355, 0, 480])

        if Z_p is not None and len(dp) >= 2:
            ax.contourf(Z_p, levels=levels, cmap='Blues', alpha=0.4,
                        extent=[0, 355, 0, 480])
            ax.contour(Z_p, levels=levels[::2], colors='navy', linewidths=0.3,
                       alpha=0.4, extent=[0, 355, 0, 480])

        _draw_landing_court(ax)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_aspect('equal')

        if shot == 'ALL':
            label = 'ALL'
            n_pred, n_gt = len(land_pred), len(land_gt)
        else:
            label = _SHOT_LABELS.get(shot, f'Shot {shot}')
            n_pred = int(np.sum(shot_pred == shot))
            n_gt = int(np.sum(shot_gt == shot))
        ax.set_title(f'{label}\nGT:{n_gt} Pred:{n_pred}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
