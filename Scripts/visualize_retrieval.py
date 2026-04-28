#!/usr/bin/env python3
"""
Visualize retrieval results from NaiveRAG, TimeRAG, and Twin (I-MSDP)
against Ground Truth for a specific test match rally round.

Usage:
    python Scripts/visualize_retrieval.py \
        --player "Viktor AXELSEN" \
        --match 63 \
        --rally 1 \
        --round 3 \
        --top_k 3 \
        --experiment norm \
        --output_dir Visualizations/
"""

import argparse
import copy
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

from Utils.helper import state_col, action_col
from Utils.badminton_config import INDEX2EN_SHOT

warnings.filterwarnings("ignore")

# =============================================================================
# Coordinate Conversion
# =============================================================================

COURT_WIDTH = 177.5   # half-court width in actual units
COURT_LENGTH = 480    # half-court length in actual units
NET_Y = 0


def is_top_half_player(rally_df: pd.DataFrame, target_player: str, ball_round: int) -> bool:
    """
    Determine whether the target player is in the top or bottom half
    of the unified court for a given ball_round in the rally.

    Rule: the player who hits ball_round==1 (the server) is in the bottom half.
    Players alternate halves each shot within a rally.
    """
    rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)
    server_row = rally_df.iloc[0]
    server_name = server_row["player"]
    if ball_round == 1:
        return False
    # If target_player is the server, they are in bottom half on odd rounds, top on even
    if target_player == server_name:
        return (ball_round % 2) == 0
    else:
        # target is receiver: top half on odd rounds, bottom on even
        return (ball_round % 2) == 1


def clip_to_court(x: float, y: float) -> Tuple[float, float]:
    """Clip coordinates to court boundaries with a small margin."""
    margin = 5.0
    x = max(-COURT_WIDTH + margin, min(COURT_WIDTH - margin, x))
    y = max(-COURT_LENGTH + margin, min(COURT_LENGTH - margin, y))
    return x, y


def to_actual_x(nx: float) -> float:
    """Normalized x -> actual x (no flip)."""
    return nx * COURT_WIDTH


def player_loc_to_actual_y(ny: float, is_top: bool) -> float:
    """Player location normalized y -> actual unified y."""
    if is_top:
        return (1.0 - ny) * COURT_LENGTH
    else:
        return (ny - 1.0) * COURT_LENGTH


def opponent_loc_to_actual_y(ny: float, is_top: bool) -> float:
    """Opponent location normalized y -> actual unified y."""
    if is_top:
        return (ny - 1.0) * COURT_LENGTH
    else:
        return (1.0 - ny) * COURT_LENGTH


def hit_to_actual_y(ny: float, is_top: bool) -> float:
    """Hit position normalized y -> actual unified y.
    Hit is where the ball is (in current player's court)."""
    if is_top:
        return (1.0 - ny) * COURT_LENGTH
    else:
        return (ny - 1.0) * COURT_LENGTH


def move_to_actual_y(ny: float, is_top: bool) -> float:
    """Move position normalized y -> actual unified y."""
    if is_top:
        return -ny * COURT_LENGTH + COURT_LENGTH
    else:
        return ny * COURT_LENGTH - COURT_LENGTH


def landing_to_actual_y(ny: float, is_top: bool) -> float:
    """Landing position normalized y -> actual unified y."""
    if is_top:
        return ny * COURT_LENGTH - COURT_LENGTH
    else:
        return -ny * COURT_LENGTH + COURT_LENGTH


# =============================================================================
# Court Drawing
# =============================================================================

def draw_court(ax: plt.Axes) -> None:
    """Draw a badminton court on the given axes."""
    ax.set_xlim(-COURT_WIDTH - 20, COURT_WIDTH + 20)
    ax.set_ylim(-COURT_LENGTH - 40, COURT_LENGTH + 40)
    ax.set_aspect("equal")
    ax.axis("off")

    # Full court boundary
    court = Rectangle(
        (-COURT_WIDTH, -COURT_LENGTH),
        2 * COURT_WIDTH,
        2 * COURT_LENGTH,
        linewidth=2,
        edgecolor="black",
        facecolor="#f0f8ff",
        zorder=0,
    )
    ax.add_patch(court)

    # Center line (vertical)
    ax.plot([0, 0], [-COURT_LENGTH, COURT_LENGTH], "k-", linewidth=1.5, zorder=1)

    # Net (horizontal at y=0)
    ax.plot([-COURT_WIDTH, COURT_WIDTH], [0, 0], "k-", linewidth=3, zorder=1)
    ax.text(COURT_WIDTH + 10, 0, "Net", fontsize=10, va="center", ha="left")

    # Service lines (short service line at y = +/- 197.7, long service line at y = +/- 480... wait)
    # Actually, the service lines are at roughly y = +/- (480 - 72) = +/- 408 for singles?
    # Let's draw approximate service lines for visual reference
    service_y = COURT_LENGTH * 0.76  # approx 365
    ax.plot([-COURT_WIDTH, COURT_WIDTH], [-service_y, -service_y], "k--", linewidth=1, alpha=0.5, zorder=1)
    ax.plot([-COURT_WIDTH, COURT_WIDTH], [service_y, service_y], "k--", linewidth=1, alpha=0.5, zorder=1)

    # Singles side lines
    singles_x = COURT_WIDTH * 0.79  # approx 140
    ax.plot([-singles_x, -singles_x], [-COURT_LENGTH, COURT_LENGTH], "k-", linewidth=1, zorder=1)
    ax.plot([singles_x, singles_x], [-COURT_LENGTH, COURT_LENGTH], "k-", linewidth=1, zorder=1)

    # Labels for halves (placed inside the court to avoid overlap with title)
    ax.text(0, -COURT_LENGTH + 30, "Bottom Half", fontsize=10, ha="center", va="bottom",
            fontweight="bold", color="gray", alpha=0.7)
    ax.text(0, COURT_LENGTH - 30, "Top Half", fontsize=10, ha="center", va="top",
            fontweight="bold", color="gray", alpha=0.7)


def draw_arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float],
               color: str, linewidth: float = 2.0, linestyle: str = "-",
               label: str = None, alpha: float = 1.0, zorder: int = 5) -> None:
    """Draw an arrow from start to end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=linewidth,
                                linestyle=linestyle, alpha=alpha),
                zorder=zorder)


def draw_point(ax: plt.Axes, pos: Tuple[float, float], color: str, size: int = 80,
               marker: str = "o", label: str = None, edgecolor: str = "black", zorder: int = 5,
               label_offset_y: float = -20) -> None:
    """Draw a point on the court."""
    ax.scatter([pos[0]], [pos[1]], c=color, s=size, marker=marker,
               edgecolors=edgecolor, linewidths=1.5, zorder=zorder)
    if label:
        # Adjust label position to stay within visible area
        text_y = pos[1] + label_offset_y
        # Clamp to court interior with margin
        min_y = -COURT_LENGTH + 25
        max_y = COURT_LENGTH - 25
        if text_y < min_y:
            text_y = min_y
        if text_y > max_y:
            text_y = max_y
        ax.text(pos[0], text_y, label, fontsize=7, ha="center", va="top", color=color,
                fontweight="bold")


# =============================================================================
# Trajectory Helpers
# =============================================================================

def draw_trajectory_lines(ax, steps, is_top, color, alpha=0.5, linewidth=1.5):
    """Draw pure trajectory lines (no arrows, no points) for a sequence of steps.

    Player trajectory: player_loc -> hit -> next_player_loc -> next_hit -> ...
    Opponent trajectory: opp_loc -> landing -> next_opp_loc -> next_landing -> ...
    """
    # Player trajectory points
    player_pts = []
    for step in steps:
        if 'player_location_x' not in step or 'player_location_y' not in step:
            continue
        if 'hit_x' not in step or 'hit_y' not in step:
            continue
        p_x = to_actual_x(step['player_location_x'])
        p_y = player_loc_to_actual_y(step['player_location_y'], is_top)
        h_x = to_actual_x(step['hit_x'])
        h_y = hit_to_actual_y(step['hit_y'], is_top)
        p_x, p_y = clip_to_court(p_x, p_y)
        h_x, h_y = clip_to_court(h_x, h_y)
        player_pts.append((p_x, p_y))
        player_pts.append((h_x, h_y))

    if len(player_pts) >= 2:
        xs, ys = zip(*player_pts)
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, linestyle='-', zorder=3)

    # Opponent trajectory points
    opponent_pts = []
    for step in steps:
        if 'opponent_location_x' not in step or 'opponent_location_y' not in step:
            continue
        if 'landing_x' not in step or 'landing_y' not in step:
            continue
        o_x = to_actual_x(step['opponent_location_x'])
        o_y = opponent_loc_to_actual_y(step['opponent_location_y'], is_top)
        l_x = to_actual_x(step['landing_x'])
        l_y = landing_to_actual_y(step['landing_y'], is_top)
        o_x, o_y = clip_to_court(o_x, o_y)
        l_x, l_y = clip_to_court(l_x, l_y)
        opponent_pts.append((o_x, o_y))
        opponent_pts.append((l_x, l_y))

    if len(opponent_pts) >= 2:
        xs, ys = zip(*opponent_pts)
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, linestyle='--', zorder=3)


def draw_action_arrows_for_step(ax, step, is_top, color, alpha=0.6, linewidth=1.2, zorder=4):
    """Draw action arrows (hit->landing, player_loc->move) for a single step."""
    required = ['hit_x', 'hit_y', 'landing_x', 'landing_y',
                'player_location_x', 'player_location_y', 'moving_x', 'moving_y']
    if not all(k in step for k in required):
        return

    h_x = to_actual_x(step['hit_x'])
    h_y = hit_to_actual_y(step['hit_y'], is_top)
    l_x = to_actual_x(step['landing_x'])
    l_y = landing_to_actual_y(step['landing_y'], is_top)
    p_x = to_actual_x(step['player_location_x'])
    p_y = player_loc_to_actual_y(step['player_location_y'], is_top)
    m_x = to_actual_x(step['moving_x'])
    m_y = move_to_actual_y(step['moving_y'], is_top)

    h_x, h_y = clip_to_court(h_x, h_y)
    l_x, l_y = clip_to_court(l_x, l_y)
    p_x, p_y = clip_to_court(p_x, p_y)
    m_x, m_y = clip_to_court(m_x, m_y)

    # hit -> landing (ball trajectory)
    draw_arrow(ax, (h_x, h_y), (l_x, l_y), color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    # player_location -> move (player movement)
    draw_arrow(ax, (p_x, p_y), (m_x, m_y), color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def draw_step_points(ax, step, is_top, color, alpha=0.6, zorder=4):
    """Draw key points for a single step."""
    if 'hit_x' in step and 'hit_y' in step:
        h_x = to_actual_x(step['hit_x'])
        h_y = hit_to_actual_y(step['hit_y'], is_top)
        h_x, h_y = clip_to_court(h_x, h_y)
        ax.scatter([h_x], [h_y], c=color, s=40, marker='D', edgecolors='white', linewidths=0.5, alpha=alpha, zorder=zorder)

    if 'landing_x' in step and 'landing_y' in step:
        l_x = to_actual_x(step['landing_x'])
        l_y = landing_to_actual_y(step['landing_y'], is_top)
        l_x, l_y = clip_to_court(l_x, l_y)
        ax.scatter([l_x], [l_y], c=color, s=60, marker='*', edgecolors='white', linewidths=0.5, alpha=alpha, zorder=zorder)

    if 'player_location_x' in step and 'player_location_y' in step:
        p_x = to_actual_x(step['player_location_x'])
        p_y = player_loc_to_actual_y(step['player_location_y'], is_top)
        p_x, p_y = clip_to_court(p_x, p_y)
        ax.scatter([p_x], [p_y], c=color, s=50, marker='o', edgecolors='white', linewidths=0.5, alpha=alpha, zorder=zorder)

    if 'moving_x' in step and 'moving_y' in step:
        m_x = to_actual_x(step['moving_x'])
        m_y = move_to_actual_y(step['moving_y'], is_top)
        m_x, m_y = clip_to_court(m_x, m_y)
        ax.scatter([m_x], [m_y], c=color, s=40, marker='s', edgecolors='white', linewidths=0.5, alpha=alpha, zorder=zorder)


def draw_rally_trajectories(ax, rally_df, target_player, is_top, current_round):
    """Draw trajectory lines and action arrows for all steps up to current_round."""
    player_steps = []
    for _, row in rally_df.iterrows():
        if row['ball_round'] > current_round:
            break
        if row['player'] == target_player:
            player_steps.append(row)

    # Player trajectory lines
    player_pts = []
    for step in player_steps:
        p_x = to_actual_x(step['player_location_x'])
        p_y = player_loc_to_actual_y(step['player_location_y'], is_top)
        h_x = to_actual_x(step['hit_x'])
        h_y = hit_to_actual_y(step['hit_y'], is_top)
        p_x, p_y = clip_to_court(p_x, p_y)
        h_x, h_y = clip_to_court(h_x, h_y)
        player_pts.append((p_x, p_y))
        player_pts.append((h_x, h_y))

    if len(player_pts) >= 2:
        xs, ys = zip(*player_pts)
        ax.plot(xs, ys, color='deepskyblue', linewidth=2.5, alpha=0.9, zorder=8)

    # Opponent trajectory lines
    opponent_pts = []
    for step in player_steps:
        o_x = to_actual_x(step['opponent_location_x'])
        o_y = opponent_loc_to_actual_y(step['opponent_location_y'], is_top)
        l_x = to_actual_x(step['landing_x'])
        l_y = landing_to_actual_y(step['landing_y'], is_top)
        o_x, o_y = clip_to_court(o_x, o_y)
        l_x, l_y = clip_to_court(l_x, l_y)
        opponent_pts.append((o_x, o_y))
        opponent_pts.append((l_x, l_y))

    if len(opponent_pts) >= 2:
        xs, ys = zip(*opponent_pts)
        ax.plot(xs, ys, color='tomato', linewidth=2.5, alpha=0.9, zorder=8)

    # Action arrows and small points for all rounds
    for step in player_steps:
        is_current = (step['ball_round'] == current_round)
        alpha = 0.9 if is_current else 0.3
        linewidth = 2.5 if is_current else 1.0
        arrow_zorder = 9 if is_current else 6
        point_zorder = 10 if is_current else 5
        point_size_mult = 1.0 if is_current else 0.35

        h_x = to_actual_x(step['hit_x'])
        h_y = hit_to_actual_y(step['hit_y'], is_top)
        l_x = to_actual_x(step['landing_x'])
        l_y = landing_to_actual_y(step['landing_y'], is_top)
        p_x = to_actual_x(step['player_location_x'])
        p_y = player_loc_to_actual_y(step['player_location_y'], is_top)
        m_x = to_actual_x(step['moving_x'])
        m_y = move_to_actual_y(step['moving_y'], is_top)
        o_x = to_actual_x(step['opponent_location_x'])
        o_y = opponent_loc_to_actual_y(step['opponent_location_y'], is_top)

        h_x, h_y = clip_to_court(h_x, h_y)
        l_x, l_y = clip_to_court(l_x, l_y)
        p_x, p_y = clip_to_court(p_x, p_y)
        m_x, m_y = clip_to_court(m_x, m_y)
        o_x, o_y = clip_to_court(o_x, o_y)

        # hit -> landing (ball trajectory)
        draw_arrow(ax, (h_x, h_y), (l_x, l_y), color='black', linewidth=linewidth, alpha=alpha, zorder=arrow_zorder)
        # player_location -> move (player movement)
        draw_arrow(ax, (p_x, p_y), (m_x, m_y), color='deepskyblue', linewidth=linewidth, alpha=alpha, zorder=arrow_zorder)

        # Points
        ax.scatter([p_x], [p_y], c='deepskyblue', s=int(200*point_size_mult), marker='o',
                   edgecolors='black' if is_current else 'white',
                   linewidths=1.5 if is_current else 0.5,
                   alpha=alpha, zorder=point_zorder)
        ax.scatter([o_x], [o_y], c='tomato', s=int(200*point_size_mult), marker='o',
                   edgecolors='black' if is_current else 'white',
                   linewidths=1.5 if is_current else 0.5,
                   alpha=alpha, zorder=point_zorder)
        ax.scatter([h_x], [h_y], c='black', s=int(100*point_size_mult), marker='D',
                   edgecolors='white', linewidths=0.5, alpha=alpha, zorder=point_zorder)
        ax.scatter([l_x], [l_y], c='limegreen', s=int(150*point_size_mult), marker='*',
                   edgecolors='white', linewidths=0.5, alpha=alpha, zorder=point_zorder)
        ax.scatter([m_x], [m_y], c='deepskyblue', s=int(120*point_size_mult), marker='s',
                   edgecolors='white', linewidths=0.5, alpha=alpha, zorder=point_zorder)


# =============================================================================
# Ground Truth Visualization (current round only - prominent labels)
# =============================================================================

def draw_ground_truth_labels(ax: plt.Axes, row: pd.Series, is_top: bool,
                             player_name: str, opponent_name: str) -> None:
    """Draw prominent labels for the CURRENT round only."""
    p_x = to_actual_x(row["player_location_x"])
    p_y = player_loc_to_actual_y(row["player_location_y"], is_top)
    o_x = to_actual_x(row["opponent_location_x"])
    o_y = opponent_loc_to_actual_y(row["opponent_location_y"], is_top)
    h_x = to_actual_x(row["hit_x"])
    h_y = hit_to_actual_y(row["hit_y"], is_top)
    l_x = to_actual_x(row["landing_x"])
    l_y = landing_to_actual_y(row["landing_y"], is_top)
    m_x = to_actual_x(row["moving_x"])
    m_y = move_to_actual_y(row["moving_y"], is_top)

    p_x, p_y = clip_to_court(p_x, p_y)
    o_x, o_y = clip_to_court(o_x, o_y)
    h_x, h_y = clip_to_court(h_x, h_y)
    l_x, l_y = clip_to_court(l_x, l_y)
    m_x, m_y = clip_to_court(m_x, m_y)

    # Labels for current round (prominent)
    draw_point(ax, (p_x, p_y), "deepskyblue", size=200, marker="o",
               label=f"{player_name}\n(loc)", zorder=11, label_offset_y=-22)
    draw_point(ax, (o_x, o_y), "tomato", size=200, marker="o",
               label=f"{opponent_name}\n(opp)", zorder=11, label_offset_y=-22)

    h_label_y = h_y + 18
    h_label_y = max(h_label_y, -COURT_LENGTH + 25)
    ax.text(h_x - 18, h_label_y, "Ball\n(hit)", fontsize=7, ha="right", va="top",
            color="black", fontweight="bold", zorder=11)

    draw_point(ax, (m_x, m_y), "deepskyblue", size=120, marker="s",
               label="Move", zorder=11, label_offset_y=18)
    draw_point(ax, (l_x, l_y), "limegreen", size=150, marker="*",
               label="Landing", zorder=11, label_offset_y=18)

    # Score text
    score_text = f"Score: {int(row['player_score'])} - {int(row['opponent_score'])}"
    shot_type_name = INDEX2EN_SHOT.get(int(row.get("player_type", 0)), f"Type {int(row.get('player_type', 0))}")
    info_text = f"Shot: {shot_type_name}\n{score_text}"
    ax.text(COURT_WIDTH - 10, COURT_LENGTH - 10, info_text, fontsize=10, ha="right", va="top",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.95))


# =============================================================================
# Retrieval Result Visualization
# =============================================================================

def draw_naive_rag_results(ax: plt.Axes, experiences: List[Dict], is_top: bool) -> None:
    """Draw NaiveRAG retrieved experiences (single step each)."""
    colors = ["#2ecc71", "#27ae60", "#1e8449", "#196f3d", "#145a32"]
    for i, exp in enumerate(experiences):
        color = colors[i % len(colors)]
        alpha = 0.7 - i * 0.1

        # Draw action arrows for this experience
        draw_action_arrows_for_step(ax, exp, is_top, color, alpha=alpha, linewidth=1.5)

        # Draw points
        draw_step_points(ax, exp, is_top, color, alpha=alpha)

        # Label with rank and similarity near landing
        l_x = to_actual_x(exp["landing_x"])
        l_y = landing_to_actual_y(exp["landing_y"], is_top)
        sim = exp.get("similarity", 0)
        ax.text(l_x + 8, l_y + 8, f"#{i+1}\n{sim:.3f}", fontsize=6, color=color,
                ha="left", va="bottom")


def draw_timerag_results(ax: plt.Axes, windows: List[Dict], is_top: bool) -> None:
    """Draw TimeRAG retrieved windows (multi-step trajectories)."""
    colors = ["#9b59b6", "#8e44ad", "#7d3c98", "#6c3483", "#5b2c6f"]
    for i, window in enumerate(windows):
        color = colors[i % len(colors)]
        alpha = 0.7 - i * 0.1
        states = window.get("window_states", [])

        # Filter out padding and build step list
        steps = []
        for state in states:
            if state.get("padding"):
                continue
            steps.append(state)

        if not steps:
            continue

        # Draw trajectory lines
        draw_trajectory_lines(ax, steps, is_top, color, alpha=alpha, linewidth=1.5)

        # Draw action arrows and points for each step
        for step in steps:
            draw_action_arrows_for_step(ax, step, is_top, color, alpha=alpha, linewidth=1.2)
            draw_step_points(ax, step, is_top, color, alpha=alpha)

        # Label near last landing
        if steps:
            last_step = steps[-1]
            if 'landing_x' in last_step and 'landing_y' in last_step:
                l_x = to_actual_x(last_step["landing_x"])
                l_y = landing_to_actual_y(last_step["landing_y"], is_top)
                dist = window.get("distance", 0)
                ax.text(l_x + 8, l_y + 8, f"W#{i+1}\n{dist:.2f}", fontsize=6,
                        color=color, ha="left", va="bottom")


def draw_twin_results(ax: plt.Axes, segments: List[Dict], is_top: bool) -> None:
    """Draw Twin (I-MSDP) retrieved trajectory segments."""
    colors = ["#e67e22", "#d35400", "#ba4a00", "#a04000", "#873600"]
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        alpha = 0.7 - i * 0.1
        steps = segment.get("steps", [])

        if not steps:
            continue

        # Draw trajectory lines
        draw_trajectory_lines(ax, steps, is_top, color, alpha=alpha, linewidth=1.5)

        # Draw action arrows and points for each step
        for step in steps:
            draw_action_arrows_for_step(ax, step, is_top, color, alpha=alpha, linewidth=1.2)
            draw_step_points(ax, step, is_top, color, alpha=alpha)

        # Label near last landing
        if steps:
            last_step = steps[-1]
            if 'landing_x' in last_step and 'landing_y' in last_step:
                l_x = to_actual_x(last_step["landing_x"])
                l_y = landing_to_actual_y(last_step["landing_y"], is_top)
                sim = segment.get("similarity", 0)
                ax.text(l_x + 8, l_y + 8, f"S#{i+1}\n{sim:.3f}", fontsize=6,
                        color=color, ha="left", va="bottom")


# =============================================================================
# Retrieval Execution
# =============================================================================

def run_naive_rag_retrieval(db_path: str, query_state: Dict, top_k: int) -> List[Dict]:
    """Run NaiveRAG retrieval and return experiences."""
    from Agent.naive_rag import NaiveRAGDatabase
    database = NaiveRAGDatabase.load(db_path)
    experiences = database.retrieve(query_state, top_k=top_k)
    return experiences


def build_timerag_query_window(state_history: List[Dict], window_size: int = 3):
    """Build TimeRAG query window from state history."""
    continuous_keys = [
        'player_location_x', 'player_location_y',
        'hit_x', 'hit_y',
        'opponent_location_x', 'opponent_location_y'
    ]
    discrete_keys = ['opponent_type']

    # Build current state features
    current = state_history[-1]
    current_continuous = np.array([
        current['player_location_x'], current['player_location_y'],
        current['hit_x'], current['hit_y'],
        current['opponent_location_x'], current['opponent_location_y'],
    ], dtype=np.float32)
    current_discrete = np.array([float(current['opponent_type'])], dtype=np.float32)

    # Build window with padding
    query_continuous = np.zeros((window_size, len(continuous_keys)), dtype=np.float32)
    query_discrete = np.zeros((window_size, len(discrete_keys)), dtype=np.float32)

    n_history = min(len(state_history), window_size)
    for i in range(n_history):
        idx = len(state_history) - n_history + i
        s = state_history[idx]
        query_continuous[i] = np.array([
            s['player_location_x'], s['player_location_y'],
            s['hit_x'], s['hit_y'],
            s['opponent_location_x'], s['opponent_location_y'],
        ], dtype=np.float32)
        query_discrete[i] = np.array([float(s['opponent_type'])], dtype=np.float32)

    return query_continuous, query_discrete


def run_timerag_retrieval(db_path: str, state_history: List[Dict], top_k: int) -> List[Dict]:
    """Run TimeRAG retrieval and return windows."""
    from Agent.timerag import TimeRAGDatabase
    database = TimeRAGDatabase.load(db_path)
    query_continuous, query_discrete = build_timerag_query_window(state_history, window_size=database.window_size)
    windows = database.retrieve(query_continuous, query_discrete, top_k=top_k)
    return windows


def run_twin_retrieval(db_path: str, state_history: List[Dict], top_k: int) -> List[Dict]:
    """Run Twin (I-MSDP) incremental retrieval and return segments."""
    from Agent.twin import TwinDatabase
    database = TwinDatabase.load(db_path)

    prev_dist_tensor = None
    segments = None
    for s in state_history:
        query_state = {
            'opponent_type': s['opponent_type'],
            'player_location_x': s['player_location_x'],
            'player_location_y': s['player_location_y'],
            'hit_x': s['hit_x'],
            'hit_y': s['hit_y'],
            'opponent_location_x': s['opponent_location_x'],
            'opponent_location_y': s['opponent_location_y'],
        }
        segments, prev_dist_tensor = database.incremental_retrieve(
            query_state,
            prev_distance_tensor=prev_dist_tensor,
            top_k=top_k
        )
    return segments


# =============================================================================
# Main Visualization Function
# =============================================================================

def visualize_retrieval(player_name: str, match_id: int, set_num: int, rally_num: int, ball_round: int,
                        top_k: int, experiment: str, output_dir: str,
                        config_path: str = "Scripts/config.yaml") -> None:
    """Main function to visualize retrieval results for a specific round."""

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Paths
    test_split_path = config["paths"][f"{experiment}_split"]
    merged_csv_path = config["paths"]["merged_csv"]
    player_map_path = config["paths"]["player_map"]
    naive_rag_db_dir = config["paths"]["naive_rag"]["data"][experiment]
    timerag_db_dir = config["paths"]["timerag"]["data"][experiment]
    twin_db_dir = config["paths"]["twin"]["data"][experiment]

    # Load data
    with open(test_split_path, "r") as f:
        split_data = json.load(f)
    with open(player_map_path, "r") as f:
        player2index = json.load(f)

    # Verify player is in test set
    test_matches = split_data.get("test", {}).get(player_name, [])
    if match_id not in test_matches:
        print(f"ERROR: Match {match_id} not in test set for {player_name}")
        print(f"Available matches: {test_matches}")
        return

    # Load CSV
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    # Find the match
    match_df = df[df["match_id"] == match_id].copy()
    if len(match_df) == 0:
        print(f"ERROR: No data found for match {match_id}")
        return

    # Get opponent name
    players_in_match = match_df["player"].unique()
    opponent_name = None
    for p in players_in_match:
        if p != player_name:
            opponent_name = p
            break
    if opponent_name is None:
        print(f"ERROR: Could not find opponent for match {match_id}")
        return

    # Find the rally in the specified set
    rally_df = match_df[(match_df["set"] == set_num) & (match_df["rally"] == rally_num)].copy()
    if len(rally_df) == 0:
        print(f"ERROR: No data found for set {set_num}, rally {rally_num} in match {match_id}")
        # Try to find available sets/rallies
        avail = match_df.groupby(['set', 'rally']).size().reset_index()
        print(f"Available set/rally combinations: {avail[['set', 'rally']].values.tolist()}")
        return

    rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)

    # Find the specific round
    round_df = rally_df[rally_df["ball_round"] == ball_round]
    if len(round_df) == 0:
        print(f"ERROR: No data found for round {ball_round} in rally {rally_num}")
        print(f"Available rounds: {rally_df['ball_round'].tolist()}")
        return

    gt_row = round_df.iloc[0]
    current_player = gt_row["player"]
    is_target_turn = (current_player == player_name)

    if not is_target_turn:
        print(f"WARNING: Round {ball_round} is played by {current_player}, not {player_name}")
        print(f"Switching visualization target to {current_player}")
        player_name = current_player
        opponent_name = [p for p in players_in_match if p != player_name][0]

    # Determine which half the player is in
    is_top = is_top_half_player(rally_df, player_name, ball_round)
    half_label = "top" if is_top else "bottom"
    print(f"Player {player_name} is in the {half_label} half for round {ball_round}")

    # Build state history for this rally up to the current round
    state_history = []
    for idx in range(len(rally_df)):
        row = rally_df.iloc[idx]
        state_history.append({
            'player_location_x': row['player_location_x'],
            'player_location_y': row['player_location_y'],
            'hit_x': row['hit_x'],
            'hit_y': row['hit_y'],
            'opponent_type': int(row['opponent_type']),
            'opponent_location_x': row['opponent_location_x'],
            'opponent_location_y': row['opponent_location_y'],
        })
        if row['ball_round'] == ball_round:
            break

    # Current query state
    query_state = state_history[-1]

    # Safe names for database files
    safe_name = player_name.replace(" ", "_").replace(".", "")

    # =================================================================
    # Run Retrievals
    # =================================================================
    results = {}

    # NaiveRAG
    naive_rag_db_path = os.path.join(naive_rag_db_dir, f"{safe_name}_db.pkl")
    if os.path.exists(naive_rag_db_path):
        print(f"Running NaiveRAG retrieval (top_k={top_k})...")
        naive_rag_exps = run_naive_rag_retrieval(naive_rag_db_path, query_state, top_k)
        results["naive_rag"] = naive_rag_exps
        print(f"  Retrieved {len(naive_rag_exps)} experiences")
    else:
        print(f"WARNING: NaiveRAG database not found: {naive_rag_db_path}")
        results["naive_rag"] = []

    # TimeRAG
    timerag_db_path = os.path.join(timerag_db_dir, f"{safe_name}_db.pkl")
    if os.path.exists(timerag_db_path):
        print(f"Running TimeRAG retrieval (top_k={top_k})...")
        timerag_windows = run_timerag_retrieval(timerag_db_path, state_history, top_k)
        results["timerag"] = timerag_windows
        print(f"  Retrieved {len(timerag_windows)} windows")
    else:
        print(f"WARNING: TimeRAG database not found: {timerag_db_path}")
        results["timerag"] = []

    # Twin
    twin_db_path = os.path.join(twin_db_dir, f"{safe_name}_db.pkl")
    if os.path.exists(twin_db_path):
        print(f"Running Twin (I-MSDP) retrieval (top_k={top_k})...")
        twin_segments = run_twin_retrieval(twin_db_path, state_history, top_k)
        results["twin"] = twin_segments
        print(f"  Retrieved {len(twin_segments)} segments")
    else:
        print(f"WARNING: Twin database not found: {twin_db_path}")
        results["twin"] = []

    # =================================================================
    # Visualization
    # =================================================================
    os.makedirs(output_dir, exist_ok=True)

    methods = [
        ("naive_rag", "NaiveRAG", "#2ecc71", draw_naive_rag_results),
        ("timerag", "TimeRAG", "#9b59b6", draw_timerag_results),
        ("twin", "Twin (I-MSDP)", "#e67e22", draw_twin_results),
    ]

    for method_key, method_name, color_theme, draw_func in methods:
        if not results.get(method_key):
            print(f"Skipping visualization for {method_name} (no results)")
            continue

        fig, ax = plt.subplots(figsize=(10, 14))
        # Leave more room at top for suptitle and right side for legend
        fig.subplots_adjust(top=0.92, right=0.82)
        draw_court(ax)

        # Draw full rally trajectories (all rounds up to current)
        draw_rally_trajectories(ax, rally_df, player_name, is_top, ball_round)

        # Draw prominent labels for current round
        draw_ground_truth_labels(ax, gt_row, is_top, player_name, opponent_name)

        # Draw retrieval results
        draw_func(ax, results[method_key], is_top)

        # Title
        title = (f"{method_name} Retrieval  |  "
                 f"{player_name}  |  Match {match_id} (Set {set_num})  |  Rally {rally_num}  |  Round {ball_round}")
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

        # Legend
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="deepskyblue",
                                     markersize=10, label=f"{player_name} (GT)"),
            matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
                                     markersize=10, label=f"{opponent_name} (GT)"),
            matplotlib.lines.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                                     markersize=8, label="Ball (GT hit)"),
            matplotlib.lines.Line2D([0], [0], marker="*", color="w", markerfacecolor="limegreen",
                                     markersize=12, label="GT Landing"),
            matplotlib.lines.Line2D([0], [0], marker="s", color="w", markerfacecolor="deepskyblue",
                                     markersize=8, label="GT Move"),
            matplotlib.lines.Line2D([0], [0], marker="*", color="w", markerfacecolor=color_theme,
                                     markersize=10, label=f"{method_name} Retrieved"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, framealpha=0.9, edgecolor="gray")

        # Save
        safe_player = player_name.replace(" ", "_").replace(".", "")
        fname = f"{safe_player}_m{match_id}_s{set_num}_r{rally_num}_rd{ball_round}_{method_key}.png"
        fpath = os.path.join(output_dir, fname)
        plt.tight_layout()
        plt.savefig(fpath, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {fpath}")

    # =================================================================
    # Save Raw Retrieval Results as JSON
    # =================================================================
    json_fname = f"{safe_player}_m{match_id}_s{set_num}_r{rally_num}_rd{ball_round}_retrieval.json"
    json_path = os.path.join(output_dir, json_fname)

    # Convert numpy types to Python native types for JSON serialization
    def serialize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Deep convert
    results_serializable = copy.deepcopy(results)

    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        if isinstance(d, list):
            return [convert_dict(v) for v in d]
        return serialize(d)

    results_serializable = convert_dict(results_serializable)

    with open(json_path, "w") as f:
        json.dump({
            "query": {
                "player": player_name,
                "match_id": match_id,
                "set": set_num,
                "rally": rally_num,
                "ball_round": ball_round,
                "is_top_half": is_top,
                "opponent": opponent_name,
            },
            "ground_truth": {
                col: serialize(gt_row[col]) for col in gt_row.index
            },
            "retrieval_results": results_serializable,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved raw retrieval JSON: {json_path}")
    print("Done!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize NaiveRAG/TimeRAG/Twin retrieval against Ground Truth"
    )
    parser.add_argument("--player", type=str, required=True, help="Player name (e.g., 'Viktor AXELSEN')")
    parser.add_argument("--match", type=int, required=True, help="Match ID")
    parser.add_argument("--set", type=int, default=1, help="Set number (default: 1)")
    parser.add_argument("--rally", type=int, required=True, help="Rally number")
    parser.add_argument("--round", type=int, required=True, help="Ball round number")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top retrieval results to visualize")
    parser.add_argument("--experiment", type=str, default="norm", choices=["norm", "shifted"],
                        help="Experiment type: norm or shifted")
    parser.add_argument("--output_dir", type=str, default="Visualizations",
                        help="Output directory for images")
    parser.add_argument("--config", type=str, default="Scripts/config.yaml",
                        help="Path to config.yaml")

    args = parser.parse_args()

    visualize_retrieval(
        player_name=args.player,
        match_id=args.match,
        set_num=args.set,
        rally_num=args.rally,
        ball_round=args.round,
        top_k=args.top_k,
        experiment=args.experiment,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
