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
PROMPT_TRACE_SCHEMA_VERSION = 1


def _validate_case_tags(case_tags, expected_references, label):
    """Validate exhaustive tagged references against the retrieved evidence."""
    if not isinstance(case_tags, list) or len(case_tags) != len(expected_references):
        raise ValueError(f'{label} case-tag coverage mismatch')
    for annotation, expected in zip(case_tags, expected_references):
        if not isinstance(annotation, dict):
            raise ValueError(f'invalid {label} case-tag reference')
        tags = annotation.get('tags')
        if (
            not isinstance(tags, list)
            or len(tags) > 2
            or any(not isinstance(tag, str) for tag in tags)
            or len(set(tags)) != len(tags)
        ):
            raise ValueError(f'invalid {label} tags')
        reference = dict(annotation)
        reference.pop('tags', None)
        if reference != expected:
            raise ValueError(f'{label} case-tag reference mismatch')


def _require_v4_summary(container, label, include_decision=False):
    if not isinstance(container.get('evidence_summary'), str):
        raise ValueError(f'invalid {label} evidence summary')
    if include_decision and not isinstance(container.get('decision_summary'), str):
        raise ValueError(f'invalid {label} decision summary')
    if 'selected_cases' in container or 'rationale' in container:
        raise ValueError(f'legacy fields present in {label} schema 4 trace')


def _validate_brief_notes(brief_notes, expected_references, label):
    """Validate exhaustive brief notes against retrieved references."""
    if not isinstance(brief_notes, list) or len(brief_notes) != len(expected_references):
        raise ValueError(f'{label} brief-note coverage mismatch')
    for annotation, expected in zip(brief_notes, expected_references):
        if not isinstance(annotation, dict):
            raise ValueError(f'invalid {label} brief-note reference')
        note = annotation.get('note')
        if (
            not isinstance(note, str)
            or not note.strip()
            or len(note) > 120
            or '\n' in note
            or '\r' in note
        ):
            raise ValueError(f'invalid {label} brief note')
        reference = dict(annotation)
        reference.pop('note', None)
        if reference != expected:
            raise ValueError(f'{label} brief-note reference mismatch')


def _validate_v5_summary(container, label, require_nonempty, max_length):
    summary = container.get('summary')
    if (
        not isinstance(summary, str)
        or (require_nonempty and not summary.strip())
        or len(summary) > max_length
        or '\n' in summary
        or '\r' in summary
    ):
        raise ValueError(f'invalid {label} summary')
    if any(field in container for field in (
        'case_tags', 'evidence_summary', 'decision_summary',
        'selected_cases', 'rationale',
    )):
        raise ValueError(f'legacy fields present in {label} schema 5 trace')


def _validate_case_assessments(assessments, expected_references, label):
    """Validate exhaustive v6 assessments against retrieved references."""
    if not isinstance(assessments, list) or len(assessments) != len(expected_references):
        raise ValueError(f'{label} case-assessment coverage mismatch')
    banned_words = {
        'all', 'although', 'and', 'because', 'but', 'case', 'cases',
        'aligns', 'causes', 'covers', 'differs', 'favors', 'favours',
        'leads', 'maintains', 'matches', 'or', 'other', 'others',
        'produces', 'rank', 'ranks', 'results', 'shows', 'supports',
        'targets', 'uses', 'while',
    }
    factor_pattern = re.compile(
        r"^[a-z]+(?:[-'][a-z]+)*(?: [a-z]+(?:[-'][a-z]+)*){0,3}$"
    )
    for assessment, expected in zip(assessments, expected_references):
        if not isinstance(assessment, dict):
            raise ValueError(f'invalid {label} case assessment')
        if assessment.get('impact') not in {'support', 'conflict', 'irrelevant'}:
            raise ValueError(f'invalid {label} impact')
        factors = assessment.get('factors')
        if not isinstance(factors, list) or not 1 <= len(factors) <= 2:
            raise ValueError(f'invalid {label} factors')
        normalized_factors = []
        for factor in factors:
            if not isinstance(factor, str) or '\n' in factor or '\r' in factor:
                raise ValueError(f'invalid {label} factor')
            normalized = ' '.join(factor.strip().lower().split())
            if (
                factor != normalized
                or len(normalized) > 32
                or factor_pattern.fullmatch(normalized) is None
                or banned_words.intersection(normalized.split())
            ):
                raise ValueError(f'invalid {label} factor')
            normalized_factors.append(normalized)
        if len(set(normalized_factors)) != len(normalized_factors):
            raise ValueError(f'duplicate {label} factor')
        reference = dict(assessment)
        reference.pop('impact', None)
        reference.pop('factors', None)
        if reference != expected:
            raise ValueError(f'{label} case-assessment reference mismatch')


def _require_v6_contract(container, label):
    if any(field in container for field in (
        'brief_notes', 'summary', 'case_tags', 'evidence_summary',
        'decision_summary', 'selected_cases', 'rationale',
    )):
        raise ValueError(f'legacy fields present in {label} schema 6 trace')


def _validate_single_stage_direct_decision(decision, shot, label):
    if not isinstance(decision, dict):
        raise ValueError(f'missing {label} decision trace')
    if decision.get('schema_version') != 6:
        raise ValueError(f'unsupported {label} decision schema')
    if decision.get('architecture') != 'single_stage_v1':
        raise ValueError(f'unsupported {label} architecture')
    if decision.get('case_grounding') is not False:
        raise ValueError(f'{label} must use direct output')
    _require_v6_contract(decision, label)
    if decision.get('case_assessments') != []:
        raise ValueError(f'{label} has case assessments')

    retrieval = decision.get('retrieval')
    if not isinstance(retrieval, dict) or set(retrieval) != {
        'normal_case_ids', 'miss_case_ids'
    }:
        raise ValueError(f'invalid {label} retrieval trace')
    if any(
        not isinstance(retrieval[key], list)
        or any(not isinstance(case, dict) for case in retrieval[key])
        for key in ('normal_case_ids', 'miss_case_ids')
    ):
        raise ValueError(f'invalid {label} evidence')
    expected_case_ids = retrieval['normal_case_ids'] + retrieval['miss_case_ids']
    if decision.get('case_ids') != expected_case_ids:
        raise ValueError(f'{label} evidence lists differ')
    if not isinstance(decision.get('action_source'), str):
        raise ValueError(f'invalid {label} action source')

    action = decision.get('action')
    if not isinstance(action, dict):
        raise ValueError(f'invalid {label} action trace')
    shot_type = action.get('shot_type')
    if shot_type != shot.get('shot_type'):
        raise ValueError(f'{label} trace shot type differs from predicted shot')
    if (
        isinstance(shot_type, bool)
        or not isinstance(shot_type, int)
        or shot_type not in range(1, 12)
    ):
        raise ValueError(f'invalid {label} shot type')
    expected_outcome = 'missed' if shot_type == 11 else 'returned'
    if decision.get('outcome') != expected_outcome:
        raise ValueError(f'{label} outcome differs from action')

    coordinate_keys = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
    if shot_type == 11:
        if any(action.get(key) is not None for key in coordinate_keys):
            raise ValueError(f'missed {label} trace coordinates must be null')
        if any(shot.get(key) is not None for key in coordinate_keys):
            raise ValueError('missed predicted coordinates must be null')
        return
    for coordinate_key in coordinate_keys:
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
                f'{label} trace {coordinate_key} differs from predicted shot'
            )


def _validate_twin_v7_pre_stage(decision, shot):
    pre_stage = decision.get('pre_stage')
    if not isinstance(pre_stage, dict):
        raise ValueError('missing Twin pre-stage trace')
    if pre_stage.get('architecture') != 'twin_aspect_v1':
        raise ValueError('unsupported Twin aspect architecture')
    activation_round = pre_stage.get('activation_ball_round')
    if (
        isinstance(activation_round, bool)
        or not isinstance(activation_round, int)
        or activation_round < 1
    ):
        raise ValueError('invalid Twin semantic activation round')
    ball_round = shot.get('ball_round')
    if isinstance(ball_round, bool) or not isinstance(ball_round, int):
        raise ValueError('invalid Twin decision ball round')
    semantic_enabled = pre_stage.get('semantic_enabled')
    if not isinstance(semantic_enabled, bool):
        raise ValueError('invalid Twin semantic-enabled flag')
    if semantic_enabled != (ball_round >= activation_round):
        raise ValueError('Twin semantic schedule mismatch')

    retrieval_mode = decision.get('retrieval_mode')
    case_ids = decision.get('case_ids')
    if not isinstance(case_ids, list) or any(
        not isinstance(case, dict) for case in case_ids
    ):
        raise ValueError('invalid Twin v7 case references')
    if semantic_enabled:
        if retrieval_mode != 'state_semantic':
            raise ValueError('invalid semantic Twin retrieval mode')
        semantic_query = pre_stage.get('semantic_query_text')
        aspects = pre_stage.get('aspects')
        aspect_names = {
            'my_tactical_pattern',
            'my_tempo_movement_dynamics',
            'opponent_tactical_pattern',
            'opponent_tempo_movement_dynamics',
        }
        if not isinstance(semantic_query, str) or not semantic_query.strip():
            raise ValueError('invalid Twin semantic query trace')
        if (
            not isinstance(aspects, dict)
            or set(aspects) != aspect_names
            or any(
                not isinstance(value, str) or not value.strip()
                for value in aspects.values()
            )
        ):
            raise ValueError('invalid Twin aspect values')
        source = pre_stage.get('source')
        if source not in {'llm', 'fallback', 'llm_terminal_extension'}:
            raise ValueError('invalid Twin semantic source')
        repair = pre_stage.get('repair')
        if source == 'llm_terminal_extension':
            if (
                not isinstance(repair, dict)
                or set(repair) != {
                    'type', 'aspects', 'from_round', 'to_round'
                }
                or repair.get('type') != 'extend_one_terminal_round'
                or not isinstance(repair.get('aspects'), list)
                or not repair['aspects']
                or len(set(repair['aspects'])) != len(repair['aspects'])
                or not set(repair['aspects']) <= aspect_names
                or repair.get('from_round') != ball_round - 1
                or repair.get('to_round') != ball_round
            ):
                raise ValueError('invalid Twin terminal-extension trace')
        elif repair is not None:
            raise ValueError('unexpected Twin aspect repair trace')
        if any(
            'semantic_similarity' not in case or 'distance' in case
            for case in case_ids
        ):
            raise ValueError('invalid semantic Twin case reference')
        action_source = decision.get('action_source')
        if not isinstance(action_source, str) or not action_source.startswith(
            f'{source}+'
        ):
            raise ValueError('invalid semantic Twin action source')
    else:
        if retrieval_mode != 'state_only':
            raise ValueError('invalid early-round Twin retrieval mode')
        if pre_stage.get('source') != 'skipped_early_round':
            raise ValueError('invalid early-round Twin pre-stage source')
        if set(pre_stage) != {
            'architecture', 'semantic_enabled', 'activation_ball_round', 'source'
        }:
            raise ValueError('semantic data present in early-round Twin trace')
        if any(
            'distance' not in case or 'semantic_similarity' in case
            for case in case_ids
        ):
            raise ValueError('invalid state-only Twin case reference')
        action_source = decision.get('action_source')
        if not isinstance(action_source, str) or not action_source.startswith('state_only+'):
            raise ValueError('invalid state-only Twin action source')


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


def _contains_system_prompt(value):
    if isinstance(value, dict):
        return any(
            key == 'system_prompt' or _contains_system_prompt(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_system_prompt(item) for item in value)
    return False


def _validate_prompt_trace_document(document, expected_fingerprint=None):
    required_shot_fields = {
        'agent', 'player', 'match_id', 'set', 'rally', 'ball_round', 'score',
        'launch', 'input', 'retrievals', 'generations', 'final_action',
        'reasoning', 'action_source',
    }
    required_generation_fields = {
        'stage', 'user_prompt', 'parsed_response', 'raw_output', 'usage',
        'latency_ms',
    }
    if document.get('schema_version') != PROMPT_TRACE_SCHEMA_VERSION:
        raise ValueError('unsupported prompt-trace schema')
    if expected_fingerprint is not None and (
        document.get('config_fingerprint') != expected_fingerprint
    ):
        raise ValueError('configuration fingerprint mismatch')
    key = (
        int(document['match_id']),
        int(document['set']),
        int(document['rally']),
    )
    if not isinstance(document.get('trajectory_file'), str):
        raise ValueError('missing trajectory reference')
    shots = document.get('shots')
    if not isinstance(shots, list) or not shots:
        raise ValueError('missing prompt traces')
    for shot in shots:
        if not isinstance(shot, dict) or not required_shot_fields <= set(shot):
            raise ValueError('invalid prompt trace shot')
        if _contains_system_prompt(shot):
            raise ValueError('system prompt must not be stored locally')
        if (
            int(shot['match_id']) != key[0]
            or int(shot['set']) != key[1]
            or int(shot['rally']) != key[2]
        ):
            raise ValueError('prompt trace rally identity mismatch')
        if not isinstance(shot['retrievals'], list):
            raise ValueError('invalid prompt trace retrievals')
        generations = shot['generations']
        if not isinstance(generations, list) or not generations:
            raise ValueError('missing prompt trace generations')
        for generation in generations:
            if (
                not isinstance(generation, dict)
                or set(generation) != required_generation_fields
                or not isinstance(generation['user_prompt'], str)
            ):
                raise ValueError('invalid prompt trace generation')
        if not isinstance(shot['action_source'], str):
            raise ValueError('invalid prompt trace action source')
    return key


def save_prompt_trace(trajectory, prompt_traces, output_dir, trajectory_path,
                      metadata=None, overwrite=False):
    """Atomically save one rally's local retrieval/generation trace sidecar."""
    document = {
        'schema_version': PROMPT_TRACE_SCHEMA_VERSION,
        **(metadata or {}),
        'match_id': trajectory['match_id'],
        'set': trajectory['set'],
        'rally': trajectory['rally'],
        'players': list(trajectory.get('players') or []),
        'target_player': trajectory.get('target_player'),
        'trajectory_file': os.path.relpath(trajectory_path, output_dir),
        'shots': prompt_traces,
    }
    _validate_prompt_trace_document(
        document, document.get('config_fingerprint')
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / rally_filename(document)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Prompt trace already exists: {output_path}')
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


def completed_prompt_trace_keys(output_dir, expected_fingerprint=None):
    """Return valid local prompt-trace rally keys and invalid sidecar paths."""
    completed = set()
    invalid = []
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return completed, invalid
    for path in sorted(output_dir.glob('*.json')):
        try:
            with path.open('r', encoding='utf-8') as handle:
                document = json.load(handle)
            key = _validate_prompt_trace_document(document, expected_fingerprint)
            completed.add(key)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            invalid.append(str(path))
    return completed, invalid


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
                if document.get('algorithm') in {
                    'naive_rag_single_stage', 'timerag_single_stage'
                }:
                    _validate_single_stage_direct_decision(
                        shot.get('decision'),
                        shot,
                        document['algorithm'],
                    )
                elif document.get('algorithm') == 'twin_single_stage':
                    decision = shot.get('decision')
                    if not isinstance(decision, dict):
                        raise ValueError('missing single-stage Twin decision trace')
                    decision_schema = decision.get('schema_version')
                    if decision_schema not in {2, 3, 4, 5, 6, 7}:
                        raise ValueError('unsupported single-stage Twin decision schema')
                    if decision.get('architecture') != 'single_stage_v1':
                        raise ValueError('unsupported single-stage Twin architecture')
                    pre_stage = decision.get('pre_stage')
                    if not isinstance(pre_stage, dict):
                        raise ValueError('missing Twin aspect pre-stage trace')
                    if pre_stage.get('architecture') != 'twin_aspect_v1':
                        raise ValueError('unsupported Twin aspect architecture')
                    if decision_schema == 7:
                        _validate_twin_v7_pre_stage(decision, shot)
                    elif not isinstance(pre_stage.get('semantic_query_text'), str):
                        raise ValueError('invalid Twin semantic query trace')
                    retrieval = decision.get('retrieval')
                    if not isinstance(retrieval, dict):
                        raise ValueError('missing single-stage Twin retrieval trace')
                    if set(retrieval) != {'normal_case_ids', 'miss_case_ids'}:
                        raise ValueError('invalid single-stage Twin retrieval stores')
                    if any(
                        not isinstance(retrieval.get(key), list)
                        or any(not isinstance(case, dict) for case in retrieval[key])
                        for key in ('normal_case_ids', 'miss_case_ids')
                    ):
                        raise ValueError('invalid single-stage Twin evidence')
                    if not isinstance(decision.get('case_ids'), list):
                        raise ValueError('invalid single-stage Twin case IDs')
                    if any(not isinstance(case, dict) for case in decision['case_ids']):
                        raise ValueError('invalid single-stage Twin case reference')
                    expected_case_ids = (
                        retrieval['normal_case_ids'] + retrieval['miss_case_ids']
                    )
                    if decision['case_ids'] != expected_case_ids:
                        raise ValueError('single-stage Twin evidence lists differ')
                    if decision_schema in {6, 7}:
                        _require_v6_contract(decision, 'single-stage Twin')
                        assessments = decision.get('case_assessments')
                        if decision.get('case_grounding') is True:
                            expected_assessments = [
                                {**case, 'source': 'miss'}
                                for case in retrieval['miss_case_ids']
                            ] + [
                                {**case, 'source': 'success'}
                                for case in retrieval['normal_case_ids']
                            ]
                            _validate_case_assessments(
                                assessments,
                                expected_assessments,
                                'single-stage Twin',
                            )
                        elif assessments != []:
                            raise ValueError(
                                'direct single-stage Twin has case assessments'
                            )
                    elif decision_schema == 5:
                        grounded = decision.get('case_grounding') is True
                        _validate_v5_summary(
                            decision, 'single-stage Twin', grounded, 240
                        )
                        brief_notes = decision.get('brief_notes')
                        if grounded:
                            expected_notes = [
                                {**case, 'source': 'miss'}
                                for case in retrieval['miss_case_ids']
                            ] + [
                                {**case, 'source': 'success'}
                                for case in retrieval['normal_case_ids']
                            ]
                            _validate_brief_notes(
                                brief_notes, expected_notes, 'single-stage Twin'
                            )
                        elif brief_notes != [] or decision.get('summary') != '':
                            raise ValueError('direct single-stage Twin has grounded reasoning')
                    elif decision_schema == 4:
                        _require_v4_summary(
                            decision, 'single-stage Twin', include_decision=True
                        )
                        case_tags = decision.get('case_tags')
                        if decision.get('case_grounding'):
                            expected_tags = [
                                {**case, 'source': 'miss'}
                                for case in retrieval['miss_case_ids']
                            ] + [
                                {**case, 'source': 'success'}
                                for case in retrieval['normal_case_ids']
                            ]
                            _validate_case_tags(
                                case_tags, expected_tags, 'single-stage Twin'
                            )
                        elif case_tags != []:
                            raise ValueError('direct single-stage Twin has case tags')
                    else:
                        selected_cases = decision.get('selected_cases')
                        if selected_cases is not None and (
                            not isinstance(selected_cases, list)
                            or any(not isinstance(case, dict) for case in selected_cases)
                        ):
                            raise ValueError('invalid single-stage Twin selected cases')
                        if not isinstance(decision.get('rationale'), str):
                            raise ValueError('invalid single-stage Twin rationale')
                    if not isinstance(decision.get('action_source'), str):
                        raise ValueError('invalid single-stage Twin action source')
                    action = decision.get('action')
                    if not isinstance(action, dict):
                        raise ValueError('invalid single-stage Twin action trace')
                    shot_type = action.get('shot_type')
                    if shot_type != shot.get('shot_type'):
                        raise ValueError('single-stage Twin trace shot type differs from predicted shot')
                    if (
                        isinstance(shot_type, bool)
                        or not isinstance(shot_type, int)
                        or shot_type not in range(1, 12)
                    ):
                        raise ValueError('invalid single-stage Twin shot type')
                    coordinate_keys = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
                    if shot_type == 11:
                        if any(action.get(key) is not None for key in coordinate_keys):
                            raise ValueError('missed single-stage Twin trace coordinates must be null')
                        if any(shot.get(key) is not None for key in coordinate_keys):
                            raise ValueError('missed predicted coordinates must be null')
                    else:
                        for coordinate_key in coordinate_keys:
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
                                    f'single-stage Twin trace {coordinate_key} differs from predicted shot'
                                )
                elif document.get('algorithm') in {'naive_rag', 'timerag', 'twin'}:
                    decision = shot.get('decision')
                    if not isinstance(decision, dict):
                        raise ValueError('missing RAG decision trace')
                    decision_schema = decision.get('schema_version')
                    if decision_schema not in {2, 3, 4, 5, 6, 7}:
                        raise ValueError('unsupported RAG decision schema')
                    if decision.get('architecture') != 'two_stage_v1':
                        raise ValueError('unsupported RAG architecture')
                    if document.get('algorithm') == 'twin':
                        pre_stage = decision.get('pre_stage')
                        if not isinstance(pre_stage, dict):
                            raise ValueError('missing Twin aspect pre-stage trace')
                        if pre_stage.get('architecture') != 'twin_aspect_v1':
                            raise ValueError('unsupported Twin aspect architecture')
                        if decision_schema == 7:
                            _validate_twin_v7_pre_stage(decision, shot)
                        elif not isinstance(pre_stage.get('semantic_query_text'), str):
                            raise ValueError('invalid Twin semantic query trace')
                    if not isinstance(decision.get('case_ids'), list):
                        raise ValueError('invalid RAG case IDs')
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
                    if decision_schema in {6, 7}:
                        grounded = decision.get('case_grounding') is True
                        for container, label in (
                            (decision, 'RAG'),
                            (stage1, 'stage-1'),
                            (stage2, 'stage-2'),
                        ):
                            _require_v6_contract(container, label)
                        stage1_expected_assessments = [
                            {**case, 'source': 'success'}
                            for case in stage1['normal_case_ids']
                        ] + [
                            {**case, 'source': 'miss'}
                            for case in stage1['miss_case_ids']
                        ]
                        if grounded:
                            _validate_case_assessments(
                                stage1.get('case_assessments'),
                                stage1_expected_assessments,
                                'stage-1',
                            )
                        elif stage1.get('case_assessments') != []:
                            raise ValueError('direct stage-1 has case assessments')
                    elif decision_schema == 5:
                        grounded = decision.get('case_grounding') is True
                        _validate_v5_summary(decision, 'RAG', grounded, 481)
                        _validate_v5_summary(stage1, 'stage-1', grounded, 240)
                        _validate_v5_summary(stage2, 'stage-2', False, 240)
                        stage1_expected_notes = [
                            {**case, 'source': 'success'}
                            for case in stage1['normal_case_ids']
                        ] + [
                            {**case, 'source': 'miss'}
                            for case in stage1['miss_case_ids']
                        ]
                        if grounded:
                            _validate_brief_notes(
                                stage1.get('brief_notes'),
                                stage1_expected_notes,
                                'stage-1',
                            )
                        elif (
                            stage1.get('brief_notes') != []
                            or stage1.get('summary') != ''
                        ):
                            raise ValueError('direct stage-1 has grounded reasoning')
                    elif decision_schema == 4:
                        _require_v4_summary(decision, 'RAG', include_decision=True)
                        _require_v4_summary(stage1, 'stage-1')
                        _require_v4_summary(stage2, 'stage-2')
                        stage1_expected_tags = [
                            {**case, 'source': 'success'}
                            for case in stage1['normal_case_ids']
                        ] + [
                            {**case, 'source': 'miss'}
                            for case in stage1['miss_case_ids']
                        ]
                        if decision.get('case_grounding'):
                            _validate_case_tags(
                                stage1.get('case_tags'),
                                stage1_expected_tags,
                                'stage-1',
                            )
                        elif stage1.get('case_tags') != []:
                            raise ValueError('direct stage-1 has case tags')
                    else:
                        if not isinstance(decision.get('rationale'), str):
                            raise ValueError('invalid RAG rationale')
                        if not isinstance(stage1.get('rationale'), str):
                            raise ValueError('invalid stage-1 rationale')
                        for container, label in (
                            (decision, 'decision'), (stage1, 'stage-1'), (stage2, 'stage-2')
                        ):
                            selected_cases = container.get('selected_cases')
                            if selected_cases is not None and (
                                not isinstance(selected_cases, list)
                                or any(not isinstance(case, dict) for case in selected_cases)
                            ):
                                raise ValueError(f'invalid {label} selected cases')
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
                        if decision_schema in {6, 7}:
                            if stage2.get('case_assessments') != []:
                                raise ValueError(
                                    'skipped stage-2 has case assessments'
                                )
                        elif decision_schema == 5:
                            if (
                                stage2.get('brief_notes') != []
                                or stage2.get('summary') != ''
                            ):
                                raise ValueError('skipped stage-2 has grounded reasoning')
                        elif decision_schema == 4:
                            if stage2.get('case_tags') != []:
                                raise ValueError('skipped stage-2 has case tags')
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
                        if decision_schema in {6, 7}:
                            if grounded:
                                _validate_case_assessments(
                                    stage2.get('case_assessments'),
                                    stage2['case_ids'],
                                    'stage-2',
                                )
                            elif stage2.get('case_assessments') != []:
                                raise ValueError(
                                    'direct stage-2 has case assessments'
                                )
                        elif decision_schema == 5:
                            if grounded:
                                _validate_brief_notes(
                                    stage2.get('brief_notes'),
                                    stage2['case_ids'],
                                    'stage-2',
                                )
                                if not stage2.get('summary', '').strip():
                                    raise ValueError('missing grounded stage-2 summary')
                            elif (
                                stage2.get('brief_notes') != []
                                or stage2.get('summary') != ''
                            ):
                                raise ValueError('direct stage-2 has grounded reasoning')
                        elif decision_schema == 4:
                            if decision.get('case_grounding'):
                                _validate_case_tags(
                                    stage2.get('case_tags'),
                                    stage2['case_ids'],
                                    'stage-2',
                                )
                            elif stage2.get('case_tags') != []:
                                raise ValueError('direct stage-2 has case tags')
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
                    if decision_schema in {4, 5, 6, 7}:
                        expected_case_ids = (
                            stage1['normal_case_ids']
                            + stage1['miss_case_ids']
                            + stage2.get('case_ids', [])
                        )
                        if decision['case_ids'] != expected_case_ids:
                            raise ValueError('RAG evidence lists differ')
                        if decision_schema in {6, 7}:
                            expected_assessments = (
                                stage1.get('case_assessments', [])
                                + stage2.get('case_assessments', [])
                            )
                            if decision.get('case_assessments') != expected_assessments:
                                raise ValueError('RAG case-assessment lists differ')
                        elif decision_schema == 5:
                            expected_brief_notes = (
                                stage1.get('brief_notes', [])
                                + stage2.get('brief_notes', [])
                            )
                            if decision.get('brief_notes') != expected_brief_notes:
                                raise ValueError('RAG brief-note lists differ')
                            expected_summary = ' '.join(filter(None, [
                                stage1.get('summary', ''),
                                stage2.get('summary', ''),
                            ]))
                            if decision.get('summary') != expected_summary:
                                raise ValueError('RAG summaries differ')
                        else:
                            expected_case_tags = (
                                stage1.get('case_tags', [])
                                + stage2.get('case_tags', [])
                            )
                            if decision.get('case_tags') != expected_case_tags:
                                raise ValueError('RAG case-tag lists differ')
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
