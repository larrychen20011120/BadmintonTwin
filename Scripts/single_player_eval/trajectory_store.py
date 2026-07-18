"""
Trajectory store – save / load raw rally trajectory data.

Each rally trajectory contains both ground-truth and predicted sequences
for ALL shots (both players), enabling post-hoc analysis without re-running
the agent simulation.

Trajectory Format (updated to collect both players):
{
    'match_id': str,                    # Match identifier
    'set': int,                         # Set number
    'rally': int,                       # Rally number
    'target_player': str,               # Name of the target player being evaluated
    'opponent_player': str,             # Name of the opponent player
    'gt_sequence': [                    # Complete GT sequence (both players, full length)
        {
            'player': str,              # Player name who made this shot
            'ball_round': int,
            'shot_type': int,
            'landing_x': float,
            'landing_y': float,
            'moving_x': float,
            'moving_y': float,
            'player_location_x': float,
            'player_location_y': float,
            'hit_x': float,
            'hit_y': float,
            'opponent_location_x': float,
            'opponent_location_y': float,
            'player_score': int,
            'opponent_score': int,
        },
        ...
    ],
    'pred_sequence': [                  # Predicted sequence (both players)
        {
            'player': str,              # Player name who made this shot
            'shot_type': int,
            'landing_x': float,
            'landing_y': float,
            'moving_x': float,
            'moving_y': float,
            'ball_round': int,
        },
        ...
    ],
    'target_player_won': bool,          # Whether target player won this rally
    'rally_length_gt': int,             # Length of GT sequence (complete)
    'rally_length_pred': int,           # Length of pred sequence (may be shorter)
    'simulation_ended_early': bool,     # Whether simulation terminated before rally end
}

Key Changes:
- GT sequence now contains ALL shots from the rally (both players)
- GT length is always equal to the original rally length
- Pred sequence may be shorter if simulation ends early
- Both sequences include player information for each shot
"""

import pickle
import os


def save_trajectories(trajectories, output_path):
    """Save list of rally trajectory dicts to a pickle file.

    Args:
        trajectories: list of dicts, each dict represents one rally (see format above)
        output_path: str, destination file path (e.g. Experiments/twin/trajectories/...)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Trajectories saved to {output_path} ({len(trajectories)} rallies)")


def load_trajectories(input_path):
    """Load list of rally trajectory dicts from a pickle file.

    Returns:
        list of dicts (same format as save_trajectories).
    """
    with open(input_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Trajectories loaded from {input_path} ({len(trajectories)} rallies)")
    return trajectories


def build_gt_shot_from_row(row):
    """Extract a ground-truth shot dict from a DataFrame row.

    Args:
        row: pandas Series (one row of the merged CSV).

    Returns:
        dict with keys: player, ball_round, shot_type, landing_x/y, moving_x/y,
                        player_location_x/y, hit_x/y, opponent_location_x/y,
                        player_score, opponent_score.
    """
    shot_type = int(row['player_type'])
    is_miss = shot_type == 11
    return {
        'player': str(row['player']),
        'ball_round': int(row['ball_round']),
        'shot_type': shot_type,
        'landing_x': None if is_miss else float(row['landing_x']),
        'landing_y': None if is_miss else float(row['landing_y']),
        'moving_x': None if is_miss else float(row['moving_x']),
        'moving_y': None if is_miss else float(row['moving_y']),
        'player_location_x': float(row['player_location_x']),
        'player_location_y': float(row['player_location_y']),
        'hit_x': float(row['hit_x']),
        'hit_y': float(row['hit_y']),
        'opponent_location_x': float(row['opponent_location_x']),
        'opponent_location_y': float(row['opponent_location_y']),
        'player_score': int(row['player_score']),
        'opponent_score': int(row['opponent_score']),
    }


def build_pred_shot(player_name, shot_type, land_logit_2d, move_logit_2d, ball_round):
    """Build a predicted shot dict (for either player).

    Args:
        player_name: str, name of the player who made this shot.
        shot_type: int (1-11).
        land_logit_2d: iterable of [landing_x, landing_y] or (2,) numpy/tensor.
        move_logit_2d: iterable of [moving_x, moving_y] or (2,) numpy/tensor.
        ball_round: int.

    Returns:
        dict.
    """
    shot_type = int(shot_type)
    is_miss = shot_type == 11
    return {
        'player': str(player_name),
        'shot_type': shot_type,
        'landing_x': None if is_miss else float(land_logit_2d[0]),
        'landing_y': None if is_miss else float(land_logit_2d[1]),
        'moving_x': None if is_miss else float(move_logit_2d[0]),
        'moving_y': None if is_miss else float(move_logit_2d[1]),
        'ball_round': int(ball_round),
    }
