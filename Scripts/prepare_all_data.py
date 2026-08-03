#!/usr/bin/env python3
"""Prepare all algorithm datasets from the centralized project config."""

import argparse
import os
import sys

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Scripts.create_style_shift_split import create_style_shift_split
from Utils.runtime_report import print_runtime_box


SUPPORTED_EXPERIMENTS = ('norm', 'shifted')
SUPPORTED_METHODS = (
    'bc',
    'naive_rag',
    'timerag',
    'twin',
    'rallynet',
    'rallynetv2',
    'shuttlenet',
)
RALLY_METHODS = {'bc', 'rallynet', 'shuttlenet', 'naive_rag', 'timerag', 'twin'}


def load_config(config_path=None):
    config_path = config_path or os.path.join(project_path, 'Scripts', 'config.yaml')
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}, config_path


def resolve_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(project_path, path)


def nested_value(config, *keys):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing config value: {'.'.join(keys)}")
        value = value[key]
    return value


def select_items(args, config):
    prep_config = config.get('data_preparation', {})

    requested_experiments = [
        name for name in SUPPORTED_EXPERIMENTS if getattr(args, name)
    ]
    experiments = requested_experiments or prep_config.get(
        'experiments', list(SUPPORTED_EXPERIMENTS)
    )

    requested_methods = [
        name for name in SUPPORTED_METHODS
        if getattr(args, name)
    ]
    methods = requested_methods or prep_config.get('methods', list(SUPPORTED_METHODS))

    experiments = list(dict.fromkeys(experiments))
    methods = list(dict.fromkeys(methods))
    unknown_experiments = sorted(set(experiments) - set(SUPPORTED_EXPERIMENTS))
    unknown_methods = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown_experiments:
        raise ValueError(f"Unsupported data preparation experiments: {unknown_experiments}")
    if unknown_methods:
        raise ValueError(f"Unsupported data preparation methods: {unknown_methods}")
    if not experiments or not methods:
        raise ValueError('At least one experiment and one method must be configured')
    return experiments, methods


def split_path(config, experiment):
    if experiment == 'rally':
        return resolve_path(nested_value(config, 'paths', 'rally_split'))
    key = 'norm_split' if experiment == 'norm' else 'shifted_split'
    return resolve_path(nested_value(config, 'paths', key))


def output_dir(config, method, experiment):
    if method in RALLY_METHODS:
        if experiment != 'rally':
            raise ValueError(f'{method} only supports the rally-level dataset')
        return resolve_path(nested_value(config, 'paths', method, 'data'))
    artifact = 'memory' if method == 'twin' else 'data'
    return resolve_path(nested_value(config, 'paths', method, artifact, experiment))


def build_jobs(experiments, methods):
    """Route rally-level methods once and keep other method experiments."""
    jobs = []
    for method in methods:
        if method in RALLY_METHODS:
            jobs.append(('rally', method))
        else:
            jobs.extend((experiment, method) for experiment in experiments)
    return jobs


def validate_config(config, experiments, methods):
    input_paths = {
        'paths.merged_csv': resolve_path(nested_value(config, 'paths', 'merged_csv')),
        'paths.player_map': resolve_path(nested_value(config, 'paths', 'player_map')),
    }
    if any(method in RALLY_METHODS for method in methods):
        input_paths['paths.rally_split'] = split_path(config, 'rally')
    if any(method in RALLY_METHODS - {'rallynet'} for method in methods):
        input_paths['paths.rally_profiles'] = resolve_path(
            nested_value(config, 'paths', 'rally_profiles')
        )
    if any(method in RALLY_METHODS for method in methods):
        for method in methods:
            if method in RALLY_METHODS:
                output_dir(config, method, 'rally')

    for method in methods:
        if method in RALLY_METHODS:
            continue
        for experiment in experiments:
            input_paths[f'paths.{experiment}_split'] = split_path(config, experiment)
            output_dir(config, method, experiment)

    twin_config = config.get('twin', {})
    needs_shifted = (
        any(method not in RALLY_METHODS for method in methods)
        and 'shifted' in experiments
    )
    prep_config = config.get('data_preparation', {})
    shifted_path = split_path(config, 'shifted') if needs_shifted else None
    create_shifted = bool(prep_config.get('create_shifted_split_if_missing', True))

    for label, path in input_paths.items():
        if label == 'paths.shifted_split' and create_shifted and not os.path.exists(path):
            style_shift_path = resolve_path(nested_value(config, 'paths', 'style_shift_json'))
            if not os.path.isfile(style_shift_path):
                raise FileNotFoundError(f"Missing input file: {style_shift_path}")
            continue
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing input file: {path}")

    if 'timerag' in methods:
        window_size = nested_value(config, 'timerag', 'window_size')
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError('timerag.window_size must be a positive integer')

    if 'twin' in methods:
        workers = twin_config.get('workers', 1)
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('twin.workers must be a positive integer')

    if 'shuttlenet' in methods:
        encode_length = nested_value(config, 'shuttlenet_training', 'encode_length')
        if not isinstance(encode_length, int) or encode_length < 1:
            raise ValueError('shuttlenet_training.encode_length must be a positive integer')

    return shifted_path


def ensure_shifted_split(config, shifted_path):
    if os.path.isfile(shifted_path):
        return
    if not config.get('data_preparation', {}).get('create_shifted_split_if_missing', True):
        raise FileNotFoundError(f"Missing shifted split: {shifted_path}")

    split_config = {'paths': dict(config['paths'])}
    split_config['paths']['style_shift_json'] = resolve_path(config['paths']['style_shift_json'])
    split_config['paths']['shifted_split'] = shifted_path
    create_style_shift_split(split_config)


def validate_output(method, directory):
    expected_files = {
        'bc': ('train_state.pkl', 'train_action.pkl'),
        'rallynet': ('rallies.pkl', 'metadata.json'),
        'rallynetv2': (
            'train_total_traj.pkl',
            'train_total_mask.pkl',
            'train_player_traj.pkl',
            'train_player_mask.pkl',
            'train_opponent_traj.pkl',
            'train_opponent_mask.pkl',
        ),
        'shuttlenet': ('train_samples.pkl',),
    }
    if method in expected_files:
        missing = [name for name in expected_files[method] if not os.path.isfile(os.path.join(directory, name))]
        if missing:
            raise RuntimeError(f"{method} did not produce expected files: {missing}")
        return

    database_files = [name for name in os.listdir(directory) if name.endswith('_db.pkl')]
    if not database_files:
        raise RuntimeError(f"{method} did not produce any player databases in {directory}")


def run_method(config, experiment, method, input_cache):
    merged_csv = resolve_path(config['paths']['merged_csv'])
    player_map = resolve_path(config['paths']['player_map'])
    current_split = split_path(config, experiment)
    destination = output_dir(config, method, experiment)

    print('\n' + '=' * 70)
    print(f"Preparing {method} data: {experiment}")
    print(f"Output: {destination}")
    print('=' * 70)

    if method == 'shuttlenet':
        from Agent.shuttlenet import prepare_shuttlenet_data

        prepare_shuttlenet_data(
            merged_csv,
            current_split,
            player_map,
            destination,
            encode_length=config['shuttlenet_training']['encode_length'],
        )
        validate_output(method, destination)
        return destination

    if method == 'rallynet' and experiment not in input_cache:
        from prepare_rallynet import prepare_rallynet_data

        prepare_rallynet_data(
            merged_csv,
            current_split,
            player_map,
            destination,
            experiment=experiment,
        )
        validate_output(method, destination)
        return destination

    from Preprocessing.data_preparation import (
        load_match_preparation_data,
        load_preparation_data,
        prepare_bc_data,
        prepare_naive_rag_data,
        prepare_rallynetv2_data,
        prepare_timerag_data,
        prepare_twin_data,
    )

    input_key = (experiment, 'twin_raw') if method == 'twin' else experiment
    if input_key not in input_cache:
        if experiment == 'rally':
            input_cache[input_key] = load_preparation_data(
                merged_csv,
                current_split,
                player_map,
                resolve_path(config['paths']['rally_profiles']),
                drop_incomplete_rows=method != 'twin',
            )
        else:
            input_cache[input_key] = load_match_preparation_data(
                merged_csv,
                current_split,
                player_map,
            )
    df, split_content, player2index = input_cache[input_key]

    if method == 'bc':
        mapped_df = df.copy()
        mapped_df['player'] = mapped_df['player'].map(player2index)
        prepare_bc_data(mapped_df, split_content, player2index, destination)
    elif method == 'rallynet':
        from prepare_rallynet import prepare_rallynet_data

        prepare_rallynet_data(
            merged_csv,
            current_split,
            player_map,
            destination,
            experiment=experiment,
            split_content=split_content,
            player_map=player2index,
        )
    elif method == 'naive_rag':
        prepare_naive_rag_data(df, split_content, player2index, destination)
    elif method == 'timerag':
        prepare_timerag_data(
            df,
            split_content,
            player2index,
            destination,
            window_size=config['timerag']['window_size'],
        )
    elif method == 'twin':
        twin_config = config.get('twin', {})
        prepare_twin_data(
            df,
            split_content,
            player2index,
            destination,
            use_llm=bool(twin_config.get('use_llm', True)),
            max_rallies_per_player=twin_config.get('max_rallies_per_player'),
            twin_workers=twin_config.get('workers', 1),
            llm_config_path=config.get('_config_path'),
        )
    elif method == 'rallynetv2':
        mapped_df = df.copy()
        mapped_df['player'] = mapped_df['player'].map(player2index)
        prepare_rallynetv2_data(mapped_df, split_content, player2index, destination)
    else:
        raise ValueError(f"Unsupported method: {method}")

    validate_output(method, destination)
    return destination


def build_parser():
    parser = argparse.ArgumentParser(
        description='Prepare all BadmintonTwin algorithm datasets using Scripts/config.yaml',
    )
    parser.add_argument('--config', help='Alternative config YAML path')
    parser.add_argument('--norm', action='store_true', help='Prepare norm data for legacy match-level methods')
    parser.add_argument('--shifted', action='store_true', help='Prepare shifted data for legacy match-level methods')
    parser.add_argument('--bc', action='store_true', help='Prepare only BC data')
    parser.add_argument('--naive-rag', dest='naive_rag', action='store_true', help='Prepare NaiveRAG rally data')
    parser.add_argument('--timerag', action='store_true', help='Prepare TimeRAG rally data')
    parser.add_argument('--twin', action='store_true', help='Prepare only Twin data')
    parser.add_argument('--rallynet', action='store_true', help='Prepare only RallyNet rally data')
    parser.add_argument('--rallynetv2', action='store_true', help='Prepare only RallyNetV2 data')
    parser.add_argument('--shuttlenet', action='store_true', help='Prepare only ShuttleNet data')
    return parser


def main():
    args = build_parser().parse_args()
    try:
        config, config_path = load_config(args.config)
        config['_config_path'] = config_path
        experiments, methods = select_items(args, config)
        shifted_path = validate_config(config, experiments, methods)
        if any(method not in RALLY_METHODS for method in methods) and (
            'shifted' in experiments
        ):
            ensure_shifted_split(config, shifted_path)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    twin_config = config.get('twin', {})
    jobs = build_jobs(experiments, methods)
    print_runtime_box(
        'Runtime Parameters: prepare_all_data',
        [
            ('config', config_path),
            ('experiments', experiments),
            ('methods', methods),
            ('jobs', jobs),
            ('timerag.window_size', config.get('timerag', {}).get('window_size')),
            ('twin.use_llm', twin_config.get('use_llm')),
            ('twin.workers', twin_config.get('workers')),
            ('twin.max_rallies_per_player', twin_config.get('max_rallies_per_player')),
            ('shuttlenet.encode_length', config.get('shuttlenet_training', {}).get('encode_length')),
        ],
    )

    input_cache = {}
    completed = []
    failures = []

    for experiment, method in jobs:
        try:
            run_method(config, experiment, method, input_cache)
            completed.append((experiment, method))
        except Exception as exc:
            failures.append((experiment, method, str(exc)))

    print('\n' + '=' * 70)
    print(f"Data preparation completed: {len(completed)} succeeded, {len(failures)} failed")
    for experiment, method in completed:
        print(f"  OK     {experiment:7s} {method}")
    for experiment, method, error in failures:
        print(f"  FAILED {experiment:7s} {method}: {error}")
    print('=' * 70)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
