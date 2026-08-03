#!/usr/bin/env python3
"""
Run single-player evaluators over every player in the test split.

Each execution creates one run directory and saves one JSON object per rally
simulation under its ``rallies`` directory.

PureLLM uses physical-rally evaluation through eval_all_rallies.py.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
import traceback


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)


RAG_ALGORITHMS = ('naive_rag', 'timerag', 'twin')
LLM_ALGORITHMS = ('naive_rag', 'timerag', 'twin')

from Utils.llm import LLM
from Utils.runtime_report import algorithm_runtime_rows, llm_rows, print_runtime_box
from Scripts.single_player_eval.trajectory_store import create_run_directory, save_manifest


def load_config():
    """Load Scripts/config.yaml."""
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_players(experiment):
    """Return sorted player names from the configured test split."""
    config = load_config()
    split_path = config['paths'][f'{experiment}_split']

    with open(split_path, 'r') as f:
        split = json.load(f)

    return sorted(split.get('test', {}).keys())


def default_output_dir(algorithm, experiment):
    return os.path.join('Experiments', algorithm, 'trajectories', experiment)


def output_experiment_label(experiment, style_shift_instruction_mode):
    """Return the storage label; split/data selection still uses experiment."""
    if experiment != 'shifted':
        return experiment
    if style_shift_instruction_mode == 'auto':
        return 'shifted_with_instruction'
    return 'shifted_without_instruction'


def build_evaluator_kwargs(args, task_index):
    """Build kwargs accepted by the selected evaluator."""
    kwargs = {
        'save_trajectories': True,
    }

    if args.model_path and args.algorithm in ('bc', 'rallynetv2', 'shuttlenet'):
        kwargs['model_path'] = args.model_path

    if args.algorithm == 'bc':
        if args.distribution:
            kwargs['distribution'] = args.distribution
        if args.gmm_num:
            kwargs['gmm_num'] = args.gmm_num

    if args.hyperparameters_path and args.algorithm == 'rallynetv2':
        kwargs['hyperparameters_path'] = args.hyperparameters_path

    if args.algorithm == 'random' and args.seed is not None:
        kwargs['seed'] = args.seed + task_index - 1

    if args.algorithm in RAG_ALGORITHMS:
        kwargs['top_k'] = args.top_k

    if args.algorithm in LLM_ALGORITHMS:
        kwargs['enable_tracing'] = args.enable_tracing
        kwargs['style_shift_instruction_mode'] = args.style_shift_instruction
        kwargs['verbose_style_shift'] = args.verbose_style_shift
        kwargs['output_experiment'] = args.output_experiment

    if args.algorithm == 'timerag':
        kwargs['window_size'] = args.window_size

    if args.algorithm == 'twin':
        kwargs['lambda_weight'] = args.lambda_weight

    return kwargs


def _run_player_task(task):
    """Worker entry point for one player."""
    if PROJECT_PATH not in sys.path:
        sys.path.insert(0, PROJECT_PATH)

    started_at = time.time()
    result = {
        'index': task['index'],
        'player': task['player'],
        'generation': task['generation'],
        'output_path': task['output_path'],
        'status': 'failed',
        'num_rallies': 0,
        'elapsed_seconds': None,
        'error': None,
        'traceback': None,
    }

    try:
        from Scripts.single_player_eval.eval_single_player import create_evaluator

        evaluator = create_evaluator(
            algorithm=task['algorithm'],
            player_name=task['player'],
            experiment=task['experiment'],
            device=task['device'],
            skip_first_n=task['skip_first_n'],
            **task['evaluator_kwargs'],
        )
        evaluator.trajectory_output_path = task['output_path']
        evaluator.trajectory_run_metadata = task['run_metadata']

        trajectories = evaluator.simulate()
        result['status'] = 'success'
        result['num_rallies'] = len(trajectories or [])
    except Exception as exc:
        result['error'] = str(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        result['elapsed_seconds'] = round(time.time() - started_at, 3)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run all test players and save one rally JSON per simulation in a run directory.'
    )
    parser.add_argument(
        '--algorithm',
        required=True,
        choices=['bc', 'rallynetv2', 'naive_rag', 'timerag', 'twin', 'random', 'shuttlenet'],
        help='Evaluator algorithm to run.'
    )
    parser.add_argument(
        '--experiment',
        default='norm',
        choices=['norm', 'shifted'],
        help='Experiment split to evaluate.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Maximum number of players to evaluate concurrently.'
    )
    parser.add_argument(
        '--device',
        default=None,
        choices=['cuda', 'cpu'],
        help='Device passed to each evaluator. Defaults to config evaluation.device.'
    )
    parser.add_argument(
        '--skip-first-n',
        type=int,
        default=None,
        help='Passed through to the evaluator. Defaults to config.'
    )
    parser.add_argument(
        '--output-dir',
        help='Root directory for run_NNN/rallies/*.json trajectory output.'
    )
    parser.add_argument(
        '--players',
        nargs='+',
        help='Optional subset of player names to run.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Optional limit after sorting/filtering players, useful for smoke tests.'
    )
    parser.add_argument(
        '--start-method',
        default='spawn',
        choices=['spawn', 'fork', 'forkserver'],
        help='Multiprocessing start method used when workers > 1.'
    )
    parser.add_argument(
        '--model-path',
        help='Custom model path for bc/rallynetv2/shuttlenet.'
    )
    parser.add_argument(
        '--distribution',
        default=None,
        choices=['deterministic', 'gaussian', 'gmm'],
        help='BC output distribution. Defaults to config bc_training.distribution.'
    )
    parser.add_argument(
        '--gmm-num',
        type=int,
        help='Number of GMM components for BC gmm.'
    )
    parser.add_argument(
        '--hyperparameters-path',
        help='Custom hyperparameters path for rallynetv2.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Base random seed for random evaluator. The player index is added.'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Classic RAG retrieval top-k for naive_rag/timerag/twin. Defaults to config retrieval.classic_top_k.'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=None,
        help='TimeRAG window size. Defaults to config timerag.window_size.'
    )
    parser.add_argument(
        '--max-scale',
        type=int,
        default=None,
        help='Deprecated; Twin now uses --lambda-weight.'
    )
    parser.add_argument(
        '--lambda-weight',
        type=float,
        default=None,
        help='Twin numeric/semantic retrieval mixing weight. Defaults to config twin.lambda_weight.'
    )
    parser.add_argument(
        '--enable-tracing',
        action='store_true',
        help='Enable tracing for LLM/RAG evaluators.'
    )
    parser.add_argument(
        '--style-shift-instruction',
        default='auto',
        choices=['auto', 'off'],
        help=('Style-shift instruction mode for naive_rag/timerag/twin: '
              'auto uses the shifted prompt and injects instructions; '
              'off uses the normal prompt without instructions.')
    )
    parser.add_argument(
        '--verbose-style-shift',
        action='store_true',
        help='Print style-shift instruction lookup and application details.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.algorithm in ('bc', 'shuttlenet', 'naive_rag', 'timerag', 'twin'):
        raise ValueError(
            f'{args.algorithm} uses physical-rally evaluation. Run '
            'Scripts/single_player_eval/eval_all_rallies.py instead.'
        )
    if args.workers < 1:
        raise ValueError('--workers must be >= 1')

    config = load_config()
    eval_config = config.get('evaluation', {})
    algorithm_config = config.get(args.algorithm, {})
    retrieval_config = config.get('retrieval', {})
    args.device = args.device or eval_config.get('device', 'cuda')
    args.skip_first_n = (
        args.skip_first_n
        if args.skip_first_n is not None
        else algorithm_config.get('skip_first_n', eval_config.get('skip_first_n', 0))
    )
    if args.algorithm == 'bc':
        args.distribution = args.distribution or config.get('bc_training', {}).get('distribution')
    if args.algorithm in RAG_ALGORITHMS:
        args.top_k = args.top_k if args.top_k is not None else retrieval_config.get('classic_top_k', retrieval_config.get('top_k', algorithm_config.get('top_k')))
    elif args.top_k is not None:
        print(f"Warning: --top-k is ignored for algorithm '{args.algorithm}'")
    if args.algorithm == 'timerag':
        args.window_size = args.window_size if args.window_size is not None else algorithm_config.get('window_size')
    if args.algorithm == 'twin':
        if args.max_scale is not None:
            print("Warning: --max-scale is deprecated and ignored for Twin; use --lambda-weight.")
        args.lambda_weight = args.lambda_weight if args.lambda_weight is not None else algorithm_config.get('lambda_weight')

    rows = algorithm_runtime_rows(
        config,
        args.algorithm,
        experiment=args.experiment,
        device=args.device,
        skip_first_n=args.skip_first_n,
        top_k=args.top_k,
        retrieval_mode=retrieval_config.get('mode'),
        normal_top_k=retrieval_config.get('normal_top_k'),
        miss_top_k=retrieval_config.get('miss_top_k'),
        window_size=args.window_size,
        lambda_weight=args.lambda_weight,
    ) + [
        ("workers", args.workers),
        ("limit", args.limit),
        ("style_shift_instruction", args.style_shift_instruction),
    ]
    if args.algorithm in LLM_ALGORITHMS:
        rows.extend(llm_rows(LLM.resolve_effective_settings()))
    print_runtime_box(f"Runtime Parameters: eval_all_players/{args.algorithm}", rows)

    players = load_test_players(args.experiment)
    if args.players:
        requested = set(args.players)
        players = [player for player in players if player in requested]
        missing = sorted(requested - set(players))
        if missing:
            raise ValueError(f"Players not found in test split: {missing}")

    if args.limit is not None:
        players = players[:args.limit]

    if not players:
        raise ValueError('No players selected for evaluation')

    args.output_experiment = output_experiment_label(args.experiment, args.style_shift_instruction)
    output_root = args.output_dir or default_output_dir(args.algorithm, args.output_experiment)
    run_dir, generation = create_run_directory(output_root)
    rally_output_dir = os.path.join(run_dir, 'rallies')
    run_metadata = {
        'algorithm': args.algorithm,
        'experiment': args.experiment,
        'output_experiment': args.output_experiment,
        'generation': generation,
        'style_shift_instruction_mode': args.style_shift_instruction,
    }
    save_manifest(run_dir, {**run_metadata, 'status': 'running'})

    tasks = []
    for index, player in enumerate(players, start=1):
        tasks.append({
            'index': index,
            'player': player,
            'generation': generation,
            'algorithm': args.algorithm,
            'experiment': args.experiment,
            'output_experiment': args.output_experiment,
            'device': args.device,
            'skip_first_n': args.skip_first_n,
            'output_path': rally_output_dir,
            'run_metadata': run_metadata,
            'evaluator_kwargs': build_evaluator_kwargs(args, index),
        })

    print('=' * 80)
    print('All-player single-player evaluation')
    print('=' * 80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Experiment: {args.experiment}")
    print(f"Output experiment: {args.output_experiment}")
    print(f"Style-shift instruction: {args.style_shift_instruction}")
    print(f"Players: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print(f"Run dir: {run_dir}")
    print('Output pattern: rallies/match_*_set_*_rally_*_target_*.json')
    print('=' * 80)

    results = []
    if args.workers == 1:
        for task in tasks:
            print(f"[{task['index']:04d}/{len(tasks):04d}] {task['player']} gen{task['generation']:03d}")
            result = _run_player_task(task)
            results.append(result)
            print(f"  -> {result['status']} ({result['num_rallies']} rallies, {result['elapsed_seconds']}s)")
    else:
        mp_context = multiprocessing.get_context(args.start_method)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_context,
        ) as executor:
            future_to_task = {executor.submit(_run_player_task, task): task for task in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        'index': task['index'],
                        'player': task['player'],
                        'generation': task['generation'],
                        'output_path': task['output_path'],
                        'status': 'failed',
                        'num_rallies': 0,
                        'elapsed_seconds': None,
                        'error': str(exc),
                        'traceback': traceback.format_exc(),
                    }
                results.append(result)
                print(
                    f"[{completed:04d}/{len(tasks):04d}] "
                    f"#{result['index']:04d} {result['player']} gen{result['generation']:03d} -> "
                    f"{result['status']} ({result['num_rallies']} rallies, "
                    f"{result['elapsed_seconds']}s)"
                )

    results = sorted(results, key=lambda item: item['index'])
    success_count = sum(1 for item in results if item['status'] == 'success')
    failed_count = len(results) - success_count

    summary = {
        'algorithm': args.algorithm,
        'experiment': args.experiment,
        'output_experiment': args.output_experiment,
        'style_shift_instruction_mode': args.style_shift_instruction,
        'verbose_style_shift': args.verbose_style_shift,
        'workers': args.workers,
        'device': args.device,
        'output_dir': run_dir,
        'total_players': len(results),
        'success_count': success_count,
        'failed_count': failed_count,
        'results': results,
    }

    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write('\n')
    save_manifest(run_dir, {
        **run_metadata,
        'status': 'completed' if failed_count == 0 else 'completed_with_errors',
        'total_players': len(results),
        'success_count': success_count,
        'failed_count': failed_count,
        'num_rallies': sum(item['num_rallies'] for item in results),
    })

    print('=' * 80)
    print(f"Success: {success_count}, Failed: {failed_count}")
    print(f"Summary: {summary_path}")
    print('=' * 80)

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
