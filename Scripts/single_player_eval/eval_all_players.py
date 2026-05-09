#!/usr/bin/env python3
"""
Run single-player evaluators over every player in the test split.

Each player trajectory is saved with a per-player generation index, so rerunning
the same player produces Player_norm_gen002.pkl, Player_norm_gen003.pkl, etc.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import re
import sys
import time
import traceback


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)


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


def safe_player_name(player_name):
    """Match the evaluator filename convention while keeping labels readable."""
    return player_name.replace(' ', '_')


def default_output_dir(algorithm, experiment):
    return os.path.join('Experiments', algorithm, 'trajectories', experiment)


def trajectory_base_name(player_name, experiment):
    return f"{safe_player_name(player_name)}_{experiment}"


def next_generation(output_dir, player_name, experiment):
    """Return the next generation number for this player trajectory name."""
    if not os.path.isdir(output_dir):
        return 1

    base_name = trajectory_base_name(player_name, experiment)
    escaped_base = re.escape(base_name)
    generation_pattern = re.compile(rf"^{escaped_base}_gen(\d+)\.pkl$")
    legacy_exact = f"{base_name}.pkl"
    legacy_indexed_pattern = re.compile(rf"^\d{{4}}_{escaped_base}\.pkl$")

    generations = []
    for filename in os.listdir(output_dir):
        match = generation_pattern.match(filename)
        if match:
            generations.append(int(match.group(1)))
        elif filename == legacy_exact or legacy_indexed_pattern.match(filename):
            generations.append(1)

    return max(generations, default=0) + 1


def build_output_path(output_dir, player_name, experiment, generation):
    filename = f"{trajectory_base_name(player_name, experiment)}_gen{generation:03d}.pkl"
    return os.path.join(output_dir, filename)


def build_evaluator_kwargs(args, task_index):
    """Build kwargs accepted by the selected evaluator."""
    kwargs = {
        'save_trajectories': True,
        'teacher_forcing': args.teacher_forcing,
    }

    if args.model_path and args.algorithm in ('bc', 'rallynetv2'):
        kwargs['model_path'] = args.model_path

    if args.algorithm == 'bc':
        kwargs['distribution'] = args.distribution

    if args.hyperparameters_path and args.algorithm == 'rallynetv2':
        kwargs['hyperparameters_path'] = args.hyperparameters_path

    if args.algorithm == 'random' and args.seed is not None:
        kwargs['seed'] = args.seed + task_index - 1

    if args.algorithm in ('naive_rag', 'timerag', 'twin'):
        kwargs['top_k'] = args.top_k
        kwargs['enable_tracing'] = args.enable_tracing

    if args.algorithm == 'timerag':
        kwargs['window_size'] = args.window_size

    if args.algorithm == 'twin':
        kwargs['max_scale'] = args.max_scale

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
        description='Run single-player evaluation for all test players in parallel and save generation-numbered trajectories.'
    )
    parser.add_argument(
        '--algorithm',
        required=True,
        choices=['bc', 'rallynetv2', 'naive_rag', 'timerag', 'twin', 'random'],
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
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device passed to each evaluator.'
    )
    parser.add_argument(
        '--skip-first-n',
        type=int,
        default=0,
        help='Passed through to the evaluator.'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for generation-numbered trajectory pickle files.'
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
        '--teacher-forcing',
        action='store_true',
        help='Use ground-truth states for prediction.'
    )

    parser.add_argument(
        '--model-path',
        help='Custom model path for bc/rallynetv2.'
    )
    parser.add_argument(
        '--distribution',
        default='deterministic',
        choices=['deterministic', 'gaussian'],
        help='BC output distribution.'
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
        default=5,
        help='RAG retrieval top-k for naive_rag/timerag/twin.'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=3,
        help='TimeRAG window size.'
    )
    parser.add_argument(
        '--max-scale',
        type=int,
        default=10,
        help='Twin I-MSDP max suffix length.'
    )
    parser.add_argument(
        '--enable-tracing',
        action='store_true',
        help='Enable tracing for LLM/RAG evaluators.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be >= 1')

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

    output_dir = args.output_dir or default_output_dir(args.algorithm, args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for index, player in enumerate(players, start=1):
        generation = next_generation(output_dir, player, args.experiment)
        tasks.append({
            'index': index,
            'player': player,
            'generation': generation,
            'algorithm': args.algorithm,
            'experiment': args.experiment,
            'device': args.device,
            'skip_first_n': args.skip_first_n,
            'output_path': build_output_path(output_dir, player, args.experiment, generation),
            'evaluator_kwargs': build_evaluator_kwargs(args, index),
        })

    print('=' * 80)
    print('All-player single-player evaluation')
    print('=' * 80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Experiment: {args.experiment}")
    print(f"Players: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print(f"Output dir: {output_dir}")
    print('Filename pattern: Player_experiment_genNNN.pkl')
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
        'workers': args.workers,
        'device': args.device,
        'output_dir': output_dir,
        'total_players': len(results),
        'success_count': success_count,
        'failed_count': failed_count,
        'results': results,
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('=' * 80)
    print(f"Success: {success_count}, Failed: {failed_count}")
    print(f"Summary: {summary_path}")
    print('=' * 80)

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
