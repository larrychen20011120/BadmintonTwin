#!/usr/bin/env python3
"""Evaluate each unique physical test rally once, with parallel RAG resume support."""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Preprocessing.rally_data import load_rally_dataset, rally_key
from Scripts.single_player_eval.trajectory_store import (
    completed_rally_keys,
    create_run_directory,
    save_manifest,
    save_trajectory,
)


RAG_ALGORITHMS = {'naive_rag', 'timerag', 'twin'}
PARALLEL_LLM_ALGORITHMS = RAG_ALGORITHMS | {'pure_llm'}
_WORKER_EVALUATOR = None
_WORKER_OUTPUT_DIR = None
_WORKER_METADATA = None


def load_config():
    path = os.path.join(PROJECT_ROOT, 'Scripts', 'config.yaml')
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_settings(config, args, dataset):
    from Utils.llm import LLM

    llm = LLM.resolve_effective_settings()
    endpoint_payload = json.dumps({
        'host': llm.get('host'),
        'port': llm.get('port'),
        'base_url': llm.get('base_url'),
        'host_url': llm.get('host_url'),
    }, ensure_ascii=True, sort_keys=True)
    prompt_dir = os.path.join(PROJECT_ROOT, 'Prompts', 'New', args.algorithm)
    system_prompt_path = os.path.join(PROJECT_ROOT, 'Prompts', 'New', 'simulate_system.md')
    implementation_paths = {
        'agent': os.path.join(PROJECT_ROOT, 'Agent', f'{args.algorithm}.py'),
        'base_agent': os.path.join(PROJECT_ROOT, 'Agent', 'base_agent.py'),
        'evaluator': os.path.join(
            PROJECT_ROOT,
            'Scripts',
            'single_player_eval',
            'evaluators',
            f'{args.algorithm}_evaluator.py',
        ),
        'trajectory_store': os.path.join(
            PROJECT_ROOT, 'Scripts', 'single_player_eval', 'trajectory_store.py'
        ),
        'transition': os.path.join(PROJECT_ROOT, 'Utils', 'helper.py'),
        'llm_client': os.path.join(PROJECT_ROOT, 'Utils', 'llm.py'),
        'badminton_config': os.path.join(PROJECT_ROOT, 'Utils', 'badminton_config.py'),
    }
    if args.algorithm == 'twin':
        implementation_paths.update({
            'verbalizer': os.path.join(PROJECT_ROOT, 'Agent', 'naive_rag.py'),
            'embedding_client': os.path.join(PROJECT_ROOT, 'Utils', 'embeddings.py'),
        })
    settings = {
        'algorithm': args.algorithm,
        'dataset': 'rally',
        'experiment': 'rally',
        'evaluation_unit': 'physical_rally',
        'instruction_mode': args.instruction_mode,
        'seed': args.seed,
        'device': args.device,
        'limit': args.limit,
        'rollout_extra_steps': config.get('evaluation', {}).get('rollout_extra_steps', 2),
        'implementation_sha256': {
            name: _file_digest(path) for name, path in implementation_paths.items()
        },
        'llm_type': llm.get('llm_type'),
        'llm_provider': llm.get('provider'),
        'llm_model_name': llm.get('model_name'),
        'llm_generation': llm.get('generation_params'),
        'llm_endpoint_sha256': hashlib.sha256(endpoint_payload.encode('utf-8')).hexdigest(),
        'split_sha256': dataset['split_sha256'],
        'profiles_sha256': dataset['profiles_sha256'],
        'merged_csv_sha256': _file_digest(config['paths']['merged_csv']),
        'player_map_sha256': _file_digest(config['paths']['player_map']),
    }
    if args.algorithm == 'pure_llm':
        prompt_name = (
            'user_instruct.jinja'
            if args.instruction_mode == 'instruct'
            else 'user.jinja'
        )
        settings.update({
            'prompt_contract': (
                'zero_shot_with_tactical_profile'
                if args.instruction_mode == 'instruct'
                else 'fixed_zero_shot'
            ),
            'tactical_profile_input': args.instruction_mode == 'instruct',
            'prompt_sha256': {
                'user': _file_digest(os.path.join(prompt_dir, prompt_name)),
            },
        })
        return settings

    retrieval = config.get('retrieval', {})
    stage1 = retrieval.get('stage1', {})
    if args.algorithm == 'twin':
        from Agent.twin import embedding_model_key
        from Utils.embeddings import resolve_embedding_settings

        embedding = resolve_embedding_settings(config=config)
        embedding_key = embedding_model_key(
            embedding['model_name'],
            embedding['revision'],
            embedding['normalize_embeddings'],
            embedding['dtype'],
            embedding['provider'],
            embedding['dimension'],
        )
        memory_manifest = os.path.join(
            config['paths']['twin']['data'], 'preparation_manifest.json'
        )
        embedding_manifest = os.path.join(
            config['paths']['twin']['embeddings'],
            embedding_key,
            'preparation_manifest.json',
        )
        prompt_suffix = args.instruction_mode
        settings.update({
            'retrieval_architecture': 'twin_aspect_two_stage_v1',
            'action_architecture': retrieval.get('architecture'),
            'stage1_normal_top_k': stage1.get('normal_top_k'),
            'stage1_miss_top_k': stage1.get('miss_top_k'),
            'stage2_conditioned_top_k': args.conditioned_top_k,
            'lambda_weight': args.lambda_weight,
            'embedding': {
                key: embedding.get(key)
                for key in (
                    'provider', 'model_name', 'dimension', 'revision', 'dtype',
                    'normalize_embeddings',
                )
            },
            'prompt_sha256': {
                'system': _file_digest(system_prompt_path),
                'aspect': _file_digest(os.path.join(prompt_dir, 'aspect_user.jinja')),
                'stage1': _file_digest(os.path.join(prompt_dir, f'stage1_{prompt_suffix}.jinja')),
                'stage2': _file_digest(os.path.join(prompt_dir, f'stage2_{prompt_suffix}.jinja')),
            },
            'database_manifest_sha256': (
                _file_digest(memory_manifest) if os.path.isfile(memory_manifest) else None
            ),
            'embedding_manifest_sha256': (
                _file_digest(embedding_manifest) if os.path.isfile(embedding_manifest) else None
            ),
        })
        return settings

    prompt_suffix = 'instruct' if args.instruction_mode == 'instruct' else 'norm'
    database_manifest = os.path.join(
        config['paths'][args.algorithm]['data'], 'preparation_manifest.json'
    )
    settings.update({
        'retrieval_architecture': retrieval.get('architecture'),
        'stage1_normal_top_k': stage1.get('normal_top_k'),
        'stage1_miss_top_k': stage1.get('miss_top_k'),
        'stage2_conditioned_top_k': args.conditioned_top_k,
        'retrieval_features': [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y',
            'ball_distance_x', 'ball_distance_y',
            'opponent_type',
        ],
        'prompt_sha256': {
            'system': _file_digest(system_prompt_path),
            'stage1': _file_digest(os.path.join(prompt_dir, f'stage1_{prompt_suffix}.jinja')),
            'stage2': _file_digest(os.path.join(prompt_dir, f'stage2_{prompt_suffix}.jinja')),
        },
        'window_size': args.window_size if args.algorithm == 'timerag' else None,
        'database_manifest_sha256': (
            _file_digest(database_manifest) if os.path.isfile(database_manifest) else None
        ),
    })
    return settings


def _fingerprint(settings):
    payload = json.dumps(settings, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _resolve_instruction_mode(algorithm, requested_mode, default_mode):
    if algorithm in PARALLEL_LLM_ALGORITHMS:
        return requested_mode or default_mode
    return 'no_instruct'


def _evaluation_records(algorithm, dataset, instruction_mode='no_instruct'):
    if algorithm not in PARALLEL_LLM_ALGORITHMS or instruction_mode == 'no_instruct':
        return list(dataset['test'])
    return [
        record for record in dataset['test']
        if all((record.get('player_profiles') or {}).get(player) is not None
               for player in record.get('players', []))
    ]


def _create_parallel_evaluator(algorithm, kwargs):
    if algorithm == 'pure_llm':
        from Scripts.single_player_eval.evaluators.pure_llm_evaluator import PureLLMEvaluator

        return PureLLMEvaluator(**kwargs)
    if algorithm == 'naive_rag':
        from Scripts.single_player_eval.evaluators.naive_rag_evaluator import NaiveRAGEvaluator

        return NaiveRAGEvaluator(**kwargs)
    if algorithm == 'twin':
        from Scripts.single_player_eval.evaluators.twin_evaluator import TwinEvaluator

        return TwinEvaluator(**kwargs)
    from Scripts.single_player_eval.evaluators.timerag_evaluator import TimeRAGEvaluator

    return TimeRAGEvaluator(**kwargs)


def _init_parallel_worker(algorithm, evaluator_kwargs, output_dir, metadata):
    global _WORKER_EVALUATOR, _WORKER_OUTPUT_DIR, _WORKER_METADATA
    _WORKER_EVALUATOR = _create_parallel_evaluator(algorithm, evaluator_kwargs)
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_METADATA = metadata


def _run_parallel_rally(record):
    started_at = time.time()
    key = rally_key(record)
    try:
        seed_payload = f"{_WORKER_METADATA.get('seed', 42)}:{key}".encode('utf-8')
        rally_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], 'big')
        random.seed(rally_seed)
        np.random.seed(rally_seed)
        torch.manual_seed(rally_seed)
        trajectory = _WORKER_EVALUATOR.simulate_rally(record)
        output_path = save_trajectory(
            trajectory,
            _WORKER_OUTPUT_DIR,
            metadata=_WORKER_METADATA,
            overwrite=True,
        )
        return {
            'key': key,
            'status': 'success',
            'output_path': output_path,
            'elapsed_seconds': round(time.time() - started_at, 3),
            'error': None,
            'traceback': None,
        }
    except Exception as exc:
        return {
            'key': key,
            'status': 'failed',
            'output_path': None,
            'elapsed_seconds': round(time.time() - started_at, 3),
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }


def _failure_path(failure_dir, key):
    match_id, set_num, rally_num = key
    return os.path.join(
        failure_dir,
        f'match_{match_id}_set_{set_num}_rally_{rally_num}.json',
    )


def _write_failure(failure_dir, result):
    os.makedirs(failure_dir, exist_ok=True)
    path = _failure_path(failure_dir, result['key'])
    temporary = f'{path}.tmp-{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _clear_failure(failure_dir, key):
    path = _failure_path(failure_dir, key)
    if os.path.exists(path):
        os.remove(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate every unique physical test rally and save one JSON per rally.'
    )
    parser.add_argument(
        '--algorithm', required=True,
        choices=['bc', 'rallynet', 'shuttlenet', 'random', 'pure_llm', 'naive_rag', 'timerag', 'twin'],
    )
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--start-method', choices=['spawn', 'fork', 'forkserver'])
    instruction = parser.add_mutually_exclusive_group()
    instruction.add_argument('--instruction-mode', choices=['instruct', 'no_instruct'])
    instruction.add_argument('--instruct', dest='instruction_mode', action='store_const', const='instruct')
    instruction.add_argument('--no-instruct', dest='instruction_mode', action='store_const', const='no_instruct')
    parser.add_argument('--resume-run', help='Existing run_NNN directory to resume')
    parser.add_argument('--model-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--distribution', choices=['deterministic', 'gaussian', 'gmm'])
    parser.add_argument('--gmm-num', type=int)
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Disable RallyNet/ShuttleNet sampling',
    )
    parser.add_argument('--error-threshold', type=float)
    parser.add_argument('--conditioned-top-k', type=int)
    parser.add_argument('--lambda-weight', type=float)
    parser.add_argument('--window-size', type=int)
    parser.add_argument('--enable-tracing', action='store_true')
    parser.add_argument('--limit', type=int, help='Optional physical-rally smoke-test limit')
    return parser.parse_args()


def _run_baseline(args, config, dataset):
    if args.workers != 1:
        raise ValueError('BC/RallyNet/ShuttleNet/Random currently require --workers 1')
    if args.resume_run:
        raise ValueError('--resume-run is currently supported for LLM/RAG algorithms only')

    from Scripts.single_player_eval.evaluators.bc_evaluator import BCEvaluator
    from Scripts.single_player_eval.evaluators.random_evaluator import RandomEvaluator
    from Scripts.single_player_eval.evaluators.rallynet_evaluator import RallyNetEvaluator
    from Scripts.single_player_eval.evaluators.shuttlenet_evaluator import ShuttleNetEvaluator

    output_root = args.output_dir or os.path.join(
        'Experiments', args.algorithm, 'trajectories', 'rally_no_instruct'
    )
    run_dir, generation = create_run_directory(output_root)
    output_path = os.path.join(run_dir, 'rallies')
    metadata = {
        'algorithm': args.algorithm,
        'experiment': 'rally',
        'output_experiment': 'rally_no_instruct',
        'generation': generation,
        'instruction_mode': 'no_instruct',
        'evaluation_unit': 'physical_rally',
        'seed': args.seed,
        'device': args.device,
    }
    common = {
        'player_name': None,
        'experiment': 'rally',
        'device': args.device,
        'save_trajectories': True,
        'all_rallies': True,
    }
    if args.algorithm == 'bc':
        evaluator = BCEvaluator(
            **common,
            model_path=args.model_path,
            distribution=args.distribution,
            gmm_num=args.gmm_num,
        )
    elif args.algorithm == 'rallynet':
        evaluator = RallyNetEvaluator(
            **common,
            model_path=args.model_path,
            seed=args.seed,
            sample=not args.deterministic,
            error_threshold=args.error_threshold,
            records_limit=args.limit,
        )
    elif args.algorithm == 'shuttlenet':
        evaluator = ShuttleNetEvaluator(
            **common,
            model_path=args.model_path,
            sample=not args.deterministic,
        )
    else:
        evaluator = RandomEvaluator(**common, seed=args.seed)
    if args.algorithm == 'rallynet':
        provenance = evaluator.policy.checkpoint.get('training', {}).get('provenance', {})
        metadata.update({
            'sample': evaluator.sample,
            'error_threshold': evaluator.error_threshold,
            'limit': args.limit,
            'model_path': os.path.abspath(evaluator.model_path),
            'model_sha256': _file_digest(evaluator.model_path),
            'checkpoint_version': evaluator.policy.checkpoint.get('version'),
            'model_config': evaluator.policy.model.model_config,
            'dataset_fingerprint': provenance.get('dataset_fingerprint'),
            'split_sha256': provenance.get('split_sha256'),
        })
    evaluator.trajectory_output_path = output_path
    evaluator.trajectory_run_metadata = metadata
    save_manifest(run_dir, {**metadata, 'status': 'running'})
    trajectories = evaluator.simulate()
    if args.algorithm == 'rallynet':
        metadata['skipped_malformed_rallies'] = evaluator.skipped_malformed_rallies
        metadata['skipped_malformed_rally_count'] = len(
            evaluator.skipped_malformed_rallies
        )
    summary = {
        **metadata,
        'status': 'completed',
        'physical_test_rallies': len(trajectories),
        'player_test_assignments': dataset['summary']['player_test_assignments'],
        'run_dir': run_dir,
    }
    save_manifest(run_dir, summary)
    return summary


def _run_parallel_llm(args, config, dataset):
    records = _evaluation_records(args.algorithm, dataset, args.instruction_mode)
    skipped_missing_profiles = len(dataset['test']) - len(records)
    if args.limit is not None:
        records = records[:args.limit]

    settings = _safe_settings(config, args, dataset)
    config_fingerprint = _fingerprint(settings)
    output_label = f'rally_{args.instruction_mode}'

    if args.resume_run:
        run_dir = os.path.abspath(args.resume_run)
        manifest_path = os.path.join(run_dir, 'manifest.json')
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f'Resume manifest not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            previous_manifest = json.load(handle)
        if previous_manifest.get('config_fingerprint') != config_fingerprint:
            raise ValueError('Resume run configuration does not match the current settings')
        generation = int(previous_manifest['generation'])
    else:
        output_root = args.output_dir or os.path.join(
            'Experiments', args.algorithm, 'trajectories', output_label
        )
        run_dir, generation = create_run_directory(output_root)

    rally_dir = os.path.join(run_dir, 'rallies')
    failure_dir = os.path.join(run_dir, 'failures')
    os.makedirs(rally_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)
    metadata = {
        **settings,
        'output_experiment': output_label,
        'generation': generation,
        'config_fingerprint': config_fingerprint,
    }
    completed, invalid_paths = completed_rally_keys(rally_dir, config_fingerprint)
    for key in completed:
        _clear_failure(failure_dir, key)
    pending = [record for record in records if rally_key(record) not in completed]

    manifest = {
        **metadata,
        'status': 'running',
        'workers': args.workers,
        'planned_rallies': len(records),
        'completed_rallies': len(completed),
        'failed_rallies': 0,
        'skipped_missing_profiles': skipped_missing_profiles,
        'invalid_trajectory_files': invalid_paths,
    }
    save_manifest(run_dir, manifest)

    evaluator_kwargs = {
        'player_name': None,
        'device': args.device,
        'enable_tracing': args.enable_tracing,
        'save_trajectories': False,
        'all_rallies': True,
        'output_experiment': output_label,
        'instruction_mode': args.instruction_mode,
    }
    if args.algorithm in RAG_ALGORITHMS:
        evaluator_kwargs.update({
            'experiment': 'rally',
            'conditioned_top_k': args.conditioned_top_k,
        })
        if args.algorithm == 'timerag':
            evaluator_kwargs['window_size'] = args.window_size
        elif args.algorithm == 'twin':
            evaluator_kwargs['lambda_weight'] = args.lambda_weight

    results = []
    def record_result(result):
        results.append(result)
        if result['status'] == 'success':
            completed.add(tuple(result['key']))
            _clear_failure(failure_dir, result['key'])
        else:
            _write_failure(failure_dir, result)
        manifest.update({
            'completed_rallies': len(completed),
            'failed_rallies': sum(item['status'] == 'failed' for item in results),
        })
        save_manifest(run_dir, manifest)

    if args.workers == 1:
        _init_parallel_worker(args.algorithm, evaluator_kwargs, rally_dir, metadata)
        for record in pending:
            record_result(_run_parallel_rally(record))
    else:
        context = multiprocessing.get_context(args.start_method)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_init_parallel_worker,
            initargs=(args.algorithm, evaluator_kwargs, rally_dir, metadata),
        ) as executor:
            futures = {executor.submit(_run_parallel_rally, record): record for record in pending}
            for future in concurrent.futures.as_completed(futures):
                record = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    result = {
                        'key': rally_key(record),
                        'status': 'failed',
                        'output_path': None,
                        'elapsed_seconds': 0.0,
                        'error': str(exc),
                        'traceback': traceback.format_exc(),
                    }
                record_result(result)

    failed = [result for result in results if result['status'] == 'failed']
    completed, final_invalid_paths = completed_rally_keys(rally_dir, config_fingerprint)
    completed &= {rally_key(record) for record in records}
    final_status = (
        'completed'
        if len(completed) == len(records) and not failed and not final_invalid_paths
        else 'completed_with_errors'
    )
    summary = {
        **manifest,
        'status': final_status,
        'completed_rallies': len(completed),
        'failed_rallies': len(failed),
        'pending_rallies': len(records) - len(completed),
        'invalid_trajectory_files': final_invalid_paths,
        'elapsed_seconds': round(sum(item['elapsed_seconds'] for item in results), 3),
        'run_dir': run_dir,
        'trajectory_dir': rally_dir,
    }
    save_manifest(run_dir, summary)
    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f"Completed rallies: {len(completed)}/{len(records)}")
    print(f"Failed rallies: {len(failed)}")
    print(f"Run: {run_dir}")
    return summary


def main():
    args = parse_args()
    requested_instruction_mode = args.instruction_mode
    config = load_config()
    evaluation = config.get('evaluation', {})
    retrieval = config.get('retrieval', {})
    args.device = args.device or evaluation.get('device', 'cuda')
    args.workers = args.workers if args.workers is not None else int(evaluation.get('workers', 1))
    args.start_method = args.start_method or evaluation.get('start_method', 'spawn')
    args.instruction_mode = _resolve_instruction_mode(
        args.algorithm,
        requested_instruction_mode,
        evaluation.get('instruction_mode', 'no_instruct'),
    )
    stage2 = retrieval.get('stage2', {})
    args.conditioned_top_k = (
        args.conditioned_top_k
        if args.conditioned_top_k is not None
        else int(stage2.get('conditioned_top_k', 10))
    )
    args.window_size = (
        args.window_size if args.window_size is not None
        else config.get('timerag', {}).get('window_size', 5)
    )
    args.lambda_weight = (
        args.lambda_weight
        if args.lambda_weight is not None
        else float(config.get('twin', {}).get('lambda_weight', 0.5))
    )
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')
    if args.algorithm in RAG_ALGORITHMS and args.conditioned_top_k < 1:
        raise ValueError('--conditioned-top-k must be at least 1')
    if args.algorithm == 'twin' and not 0.0 <= args.lambda_weight <= 1.0:
        raise ValueError('--lambda-weight must be between 0 and 1')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_rally_dataset(
        config['paths']['rally_split'],
        config['paths'].get('rally_profiles'),
    )
    started_at = time.time()
    summary = (
        _run_parallel_llm(args, config, dataset)
        if args.algorithm in PARALLEL_LLM_ALGORITHMS
        else _run_baseline(args, config, dataset)
    )
    print(f"Wall time: {round(time.time() - started_at, 3)}s")
    return 0 if summary.get('status') == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
