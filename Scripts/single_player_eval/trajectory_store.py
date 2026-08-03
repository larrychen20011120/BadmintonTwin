"""
Trajectory store - save / load raw rally trajectory data.

Each rally trajectory contains both ground-truth and predicted sequences
for ALL shots (both players), enabling post-hoc analysis without re-running
the agent simulation.

Trajectory Format (updated to collect both players):
{
    'match_id': str,                    # Match identifier
    'set': int,                         # Set number
    'rally': int,                       # Rally number
    'players': [str, str],              # Both physical-rally participants
    'player_profiles': dict,            # Per-player cluster/instruction metadata
    'instruction_mode': str,            # BC/ShuttleNet use no-instruct
    'target_player': str | None,        # None for one-pass physical evaluation
    'opponent_player': str | None,
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
            'decision': {                # Present for RAG agents
                'schema_version': 2,
                'architecture': 'two_stage_v1',
                'stage1': dict,
                'stage2': dict,
                'case_ids': [dict],
                'rationale': str,
                'action': dict,
                'action_source': str,
            },
        },
        ...
    ],
    'target_player_won': bool,          # Whether target player won this rally
    'rally_length_gt': int,             # Length of GT sequence (complete)
    'rally_length_pred': int,           # Length of pred sequence (may be shorter)
    'rollout_limit': int,               # Maximum predicted shots (GT length + 2)
    'termination_reason': str,          # natural_terminate/action_none/max_rollout_reached
    'simulation_ended_early': bool,     # Whether prediction is shorter than GT
}

Key Changes:
- GT sequence now contains ALL shots from the rally (both players)
- GT length is always equal to the original rally length
- Pred sequence may be shorter if simulation ends early
- Both sequences include player information for each shot
"""

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import uuid


SCHEMA_VERSION = 1


def _safe_component(value):
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_.')
    return text or 'unknown'


def rally_filename(trajectory):
    """Return a collision-resistant filename for one rally simulation."""
    target_player = trajectory.get('target_player')
    if target_player is None:
        target = 'physical'
    else:
        digest = hashlib.sha1(str(target_player).encode('utf-8')).hexdigest()[:8]
        target = f"{_safe_component(target_player)}_{digest}"

    return (
        f"match_{_safe_component(trajectory['match_id'])}"
        f"_set_{int(trajectory['set'])}"
        f"_rally_{int(trajectory['rally'])}"
        f"_target_{target}.json"
    )


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save_trajectories(trajectories, output_dir, metadata=None):
    """Save each rally trajectory as one JSON object in ``output_dir``.

    Args:
        trajectories: list of dicts, each dict represents one rally (see format above).
        output_dir: directory that receives one JSON file per rally simulation.
        metadata: run-level fields copied into every rally JSON.
    """
    output_paths = [
        save_trajectory(trajectory, output_dir, metadata=metadata)
        for trajectory in trajectories
    ]

    print(f"Trajectories saved to {output_dir} ({len(trajectories)} rallies)")
    return output_paths


def save_trajectory(trajectory, output_dir, metadata=None, overwrite=False):
    """Atomically save one complete rally trajectory JSON."""
    os.makedirs(output_dir, exist_ok=True)
    document = {
        'schema_version': SCHEMA_VERSION,
        **(metadata or {}),
        **trajectory,
    }
    output_path = Path(output_dir) / rally_filename(document)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Rally trajectory already exists: {output_path}')

    temporary_path = output_path.with_name(
        f'.{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}'
    )
    try:
        with temporary_path.open('x', encoding='utf-8') as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return str(output_path)


def completed_rally_keys(output_dir, expected_fingerprint=None):
    """Return valid completed physical-rally keys and invalid JSON paths."""
    completed = set()
    invalid = []
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return completed, invalid

    for path in sorted(output_dir.glob('*.json')):
        try:
            with path.open('r', encoding='utf-8') as handle:
                document = json.load(handle)
            key = (
                int(document['match_id']),
                int(document['set']),
                int(document['rally']),
            )
            if document.get('evaluation_unit') != 'physical_rally':
                raise ValueError('not a physical-rally trajectory')
            if document.get('schema_version') != SCHEMA_VERSION:
                raise ValueError('unsupported trajectory schema')
            if expected_fingerprint is not None and document.get('config_fingerprint') != expected_fingerprint:
                raise ValueError('configuration fingerprint mismatch')
            players = document.get('players')
            gt_sequence = document.get('gt_sequence')
            pred_sequence = document.get('pred_sequence')
            if not isinstance(players, list) or len(players) != 2:
                raise ValueError('invalid players')
            if not isinstance(gt_sequence, list) or not isinstance(pred_sequence, list):
                raise ValueError('missing trajectory sequences')
            if document.get('rally_length_gt') != len(gt_sequence):
                raise ValueError('ground-truth length mismatch')
            if document.get('rally_length_pred') != len(pred_sequence):
                raise ValueError('prediction length mismatch')
            rollout_limit = document.get('rollout_limit')
            if not isinstance(rollout_limit, int) or rollout_limit < len(pred_sequence):
                raise ValueError('invalid rollout limit')
            if document.get('termination_reason') not in {
                'natural_terminate', 'action_none', 'max_rollout_reached'
            }:
                raise ValueError('invalid termination reason')
            required_action = {
                'player', 'ball_round', 'shot_type',
                'landing_x', 'landing_y', 'moving_x', 'moving_y',
            }
            for shot in pred_sequence:
                if not isinstance(shot, dict) or not required_action <= set(shot):
                    raise ValueError('invalid predicted shot')
                if document.get('algorithm') in {'naive_rag', 'timerag', 'twin'}:
                    decision = shot.get('decision')
                    if not isinstance(decision, dict):
                        raise ValueError('missing RAG decision trace')
                    if decision.get('schema_version') != 2:
                        raise ValueError('unsupported RAG decision schema')
                    if decision.get('architecture') != 'two_stage_v1':
                        raise ValueError('unsupported RAG architecture')
                    if document.get('algorithm') == 'twin':
                        pre_stage = decision.get('pre_stage')
                        if not isinstance(pre_stage, dict):
                            raise ValueError('missing Twin aspect pre-stage trace')
                        if pre_stage.get('architecture') != 'twin_aspect_v1':
                            raise ValueError('unsupported Twin aspect architecture')
                        if not isinstance(pre_stage.get('semantic_query_text'), str):
                            raise ValueError('invalid Twin semantic query trace')
                    if not isinstance(decision.get('case_ids'), list):
                        raise ValueError('invalid RAG case IDs')
                    if not isinstance(decision.get('rationale'), str):
                        raise ValueError('invalid RAG rationale')
                    if not isinstance(decision.get('action'), dict):
                        raise ValueError('invalid RAG action trace')
                    stage1 = decision.get('stage1')
                    stage2 = decision.get('stage2')
                    action = decision['action']
                    if action.get('shot_type') != shot.get('shot_type'):
                        raise ValueError('RAG trace shot type differs from predicted shot')
                    if not isinstance(stage1, dict) or not isinstance(stage2, dict):
                        raise ValueError('missing two-stage RAG trace')
                    if not isinstance(stage1.get('normal_case_ids'), list):
                        raise ValueError('invalid stage-1 normal evidence')
                    if not isinstance(stage1.get('miss_case_ids'), list):
                        raise ValueError('invalid stage-1 missed evidence')
                    if stage1.get('outcome') not in {'returned', 'missed'}:
                        raise ValueError('invalid stage-1 outcome')
                    if not isinstance(stage1.get('rationale'), str):
                        raise ValueError('invalid stage-1 rationale')
                    if stage1['outcome'] == 'missed':
                        if stage1.get('shot_type') is not None:
                            raise ValueError('missed stage-1 decision has a shot type')
                        if stage2.get('status') != 'skipped_missed':
                            raise ValueError('missed decision did not skip stage 2')
                        if action.get('shot_type') != 11:
                            raise ValueError('missed decision action mismatch')
                        if any(action.get(key) is not None for key in (
                            'landing_x', 'landing_y', 'moving_x', 'moving_y'
                        )):
                            raise ValueError('missed decision coordinates must be null')
                        if any(shot.get(key) is not None for key in (
                            'landing_x', 'landing_y', 'moving_x', 'moving_y'
                        )):
                            raise ValueError('missed predicted coordinates must be null')
                    else:
                        shot_type = stage1.get('shot_type')
                        if (
                            isinstance(shot_type, bool)
                            or not isinstance(shot_type, int)
                            or shot_type not in range(1, 11)
                        ):
                            raise ValueError('invalid returned stage-1 shot type')
                        if stage2.get('status') != 'completed':
                            raise ValueError('returned decision did not complete stage 2')
                        if stage2.get('conditioned_shot_type') != shot_type:
                            raise ValueError('stage-2 conditioned shot mismatch')
                        if action.get('shot_type') != shot_type:
                            raise ValueError('final action changed the stage-1 shot type')
                        if not isinstance(stage2.get('case_ids'), list):
                            raise ValueError('invalid stage-2 evidence')
                        if any(not isinstance(case, dict) for case in stage2['case_ids']):
                            raise ValueError('invalid stage-2 case reference')
                        if any(
                            case.get('conditioned_shot_type') != shot_type
                            for case in stage2['case_ids']
                        ):
                            raise ValueError('stage-2 evidence contains another shot type')
                        for coordinate_key in ('landing_x', 'landing_y', 'moving_x', 'moving_y'):
                            traced_value = action.get(coordinate_key)
                            predicted_value = shot.get(coordinate_key)
                            if (
                                isinstance(traced_value, bool)
                                or isinstance(predicted_value, bool)
                                or not isinstance(traced_value, (int, float))
                                or not isinstance(predicted_value, (int, float))
                                or not math.isfinite(float(traced_value))
                                or not math.isfinite(float(predicted_value))
                                or not -1.0 <= float(traced_value) <= 1.0
                                or not -1.0 <= float(predicted_value) <= 1.0
                                or not math.isclose(
                                    float(traced_value),
                                    float(predicted_value),
                                    rel_tol=0.0,
                                    abs_tol=1e-9,
                                )
                            ):
                                raise ValueError(
                                    f'RAG trace {coordinate_key} differs from predicted shot'
                                )
            completed.add(key)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            invalid.append(str(path))
    return completed, invalid


def load_trajectories(input_path):
    """Load a rally JSON, JSON directory, or legacy trajectory pickle.

    Returns:
        list of rally trajectory dictionaries.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        paths = sorted(input_path.rglob('*.json'))
        paths = [
            path for path in paths
            if path.name not in {'manifest.json', 'summary.json', 'experiment_metrics.json'}
            and ('rallies' in path.parts or path.parent == input_path)
        ]
        trajectories = []
        for path in paths:
            with path.open('r', encoding='utf-8') as handle:
                document = json.load(handle)
            trajectories.extend(document if isinstance(document, list) else [document])
    elif input_path.suffix.lower() == '.pkl':
        with input_path.open('rb') as handle:
            trajectories = pickle.load(handle)
    elif input_path.suffix.lower() == '.json':
        with input_path.open('r', encoding='utf-8') as handle:
            document = json.load(handle)
        trajectories = document if isinstance(document, list) else [document]
    else:
        raise ValueError(f'Unsupported trajectory path: {input_path}')

    print(f"Trajectories loaded from {input_path} ({len(trajectories)} rallies)")
    return trajectories


def create_run_directory(output_root):
    """Create and return the next ``run_NNN`` directory under an output root."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    generations = []
    for path in output_root.iterdir():
        match = re.fullmatch(r'run_(\d+)', path.name) if path.is_dir() else None
        if match:
            generations.append(int(match.group(1)))

    generation = max(generations, default=0) + 1
    while True:
        run_dir = output_root / f'run_{generation:03d}'
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            generation += 1

    (run_dir / 'rallies').mkdir()
    return str(run_dir), generation


def save_manifest(run_dir, manifest):
    """Write run-level metadata separately from rally documents."""
    output_path = Path(run_dir) / 'manifest.json'
    temporary_path = output_path.with_name(
        f'.{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}'
    )
    try:
        with temporary_path.open('x', encoding='utf-8') as handle:
            json.dump(
                {'schema_version': SCHEMA_VERSION, **manifest},
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return str(output_path)


def build_rollout_metadata(gt_sequence, pred_sequence, termination_reason, rollout_limit=None):
    """Build consistent rollout length and termination fields for one rally."""
    gt_length = len(gt_sequence)
    pred_length = len(pred_sequence)
    return {
        'rally_length_gt': gt_length,
        'rally_length_pred': pred_length,
        'rollout_limit': rollout_limit if rollout_limit is not None else gt_length + 2,
        'termination_reason': termination_reason,
        'simulation_ended_early': pred_length < gt_length,
    }


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


def build_pred_shot(player_name, shot_type, land_logit_2d, move_logit_2d, ball_round,
                    decision=None):
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
    shot = {
        'player': str(player_name),
        'shot_type': shot_type,
        'landing_x': None if is_miss else float(land_logit_2d[0]),
        'landing_y': None if is_miss else float(land_logit_2d[1]),
        'moving_x': None if is_miss else float(move_logit_2d[0]),
        'moving_y': None if is_miss else float(move_logit_2d[1]),
        'ball_round': int(ball_round),
    }
    if decision is not None:
        shot['decision'] = decision
    return shot
