#!/usr/bin/env python3
"""Prepare all algorithm datasets from the centralized project config."""

import argparse
import json
import os
import shutil
import sys
import tempfile

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Scripts.create_style_shift_split import create_style_shift_split
from Utils.runtime_report import print_runtime_box


SUPPORTED_EXPERIMENTS = ('norm', 'shifted')
SUPPORTED_METHODS = ('bc', 'naive_rag', 'timerag', 'twin', 'rallynetv2', 'shuttlenet')


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

    requested_experiments = [name for name in SUPPORTED_EXPERIMENTS if getattr(args, name)]
    experiments = requested_experiments or prep_config.get('experiments', list(SUPPORTED_EXPERIMENTS))

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
    key = 'norm_split' if experiment == 'norm' else 'shifted_split'
    return resolve_path(nested_value(config, 'paths', key))


def output_dir(config, method, experiment):
    artifact = 'memory' if method == 'twin' else 'data'
    return resolve_path(nested_value(config, 'paths', method, artifact, experiment))


def validate_config(config, experiments, methods):
    input_paths = {
        'paths.merged_csv': resolve_path(nested_value(config, 'paths', 'merged_csv')),
        'paths.player_map': resolve_path(nested_value(config, 'paths', 'player_map')),
    }
    for experiment in experiments:
        input_paths[f'paths.{experiment}_split'] = split_path(config, experiment)
        for method in methods:
            output_dir(config, method, experiment)

    prep_config = config.get('data_preparation', {})
    twin_config = config.get('twin', {})
    derive_norm_twin = (
        'twin' in methods
        and 'norm' in experiments
        and bool(twin_config.get('derive_norm_from_shifted', True))
    )
    if derive_norm_twin:
        output_dir(config, 'twin', 'shifted')
        input_paths['paths.shifted_split'] = split_path(config, 'shifted')

    needs_shifted = 'shifted' in experiments or derive_norm_twin
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

    return derive_norm_twin, shifted_path


def ensure_shifted_split(config, shifted_path):
    if os.path.isfile(shifted_path):
        return
    if not config.get('data_preparation', {}).get('create_shifted_split_if_missing', True):
        raise FileNotFoundError(f"Missing shifted split: {shifted_path}")

    split_config = {'paths': dict(config['paths'])}
    split_config['paths']['style_shift_json'] = resolve_path(config['paths']['style_shift_json'])
    split_config['paths']['shifted_split'] = shifted_path
    create_style_shift_split(split_config)


def derive_norm_twin_from_shifted(norm_split_path, shifted_twin_dir, norm_twin_dir):
    """Filter shifted Twin memories to the matches allowed by the norm split."""
    from Agent.twin import TwinDatabase, safe_player_name

    with open(norm_split_path, 'r') as f:
        train_players = json.load(f).get('train', {})

    prepared_databases = []
    source_memories = 0
    total_memories = 0
    missing_players = []

    for player_name, match_ids in train_players.items():
        safe_name = safe_player_name(player_name)
        source_path = os.path.join(shifted_twin_dir, f'{safe_name}_db.pkl')
        if not os.path.isfile(source_path):
            missing_players.append(player_name)
            continue

        allowed_match_ids = {int(match_id) for match_id in match_ids}
        database = TwinDatabase.load(source_path)
        original_memories = list(database.memories)
        filtered_memories = []
        source_memories += len(original_memories)

        for memory in original_memories:
            try:
                match_id = int(float(memory.get('match_id')))
            except (TypeError, ValueError):
                continue
            if match_id in allowed_match_ids:
                filtered_memories.append(memory)

        database.memories = filtered_memories
        database.embedding_cache = None
        database.build_metadata = {
            **getattr(database, 'build_metadata', {}),
            'expected_match_ids': sorted(allowed_match_ids),
            'target_source_row_count': int(len(filtered_memories)),
            'memory_count': int(len(filtered_memories)),
            'derived_from_shifted': True,
        }
        database._build_memory_stores()

        present_match_ids = {
            int(float(memory['match_id']))
            for memory in filtered_memories
            if memory.get('match_id') is not None
        }
        missing_match_ids = sorted(allowed_match_ids - present_match_ids)
        if missing_match_ids:
            missing_players.append(f"{player_name} (matches {missing_match_ids})")
        prepared_databases.append((database, safe_name))
        total_memories += len(filtered_memories)

    if missing_players:
        raise RuntimeError(f"Missing shifted Twin databases for: {missing_players}")
    if source_memories == 0 or total_memories == 0:
        raise RuntimeError('Twin norm filtering produced no memories')

    norm_twin_dir = os.path.abspath(norm_twin_dir)
    output_parent = os.path.dirname(norm_twin_dir)
    os.makedirs(output_parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix='.twin_norm_memory_', dir=output_parent)
    backup_dir = None
    try:
        for database, safe_name in prepared_databases:
            database.save(os.path.join(staging_dir, f'{safe_name}_db.pkl'))
        if os.path.exists(norm_twin_dir):
            backup_dir = tempfile.mkdtemp(prefix='.twin_norm_backup_', dir=output_parent)
            os.rmdir(backup_dir)
            os.replace(norm_twin_dir, backup_dir)
        os.replace(staging_dir, norm_twin_dir)
    except Exception:
        if backup_dir is not None and os.path.exists(backup_dir) and not os.path.exists(norm_twin_dir):
            os.replace(backup_dir, norm_twin_dir)
        raise
    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir is not None and os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
    print(f"Twin norm data: {len(prepared_databases)} players, {total_memories} memories")


def twin_memory_is_complete(split_file, memory_dir, use_llm=True, max_rallies_per_player=None):
    """Check whether canonical memory fully covers the configured train split."""
    from Agent.twin import TwinDatabase, safe_player_name

    if not os.path.isdir(memory_dir):
        return False
    with open(split_file, 'r') as f:
        train_players = json.load(f).get('train', {})
    expected_files = {
        f'{safe_player_name(player_name)}_db.pkl'
        for player_name in train_players
    }
    actual_files = {
        filename for filename in os.listdir(memory_dir) if filename.endswith('_db.pkl')
    }
    if not expected_files or actual_files != expected_files:
        return False

    try:
        for player_name, match_ids in train_players.items():
            database = TwinDatabase.load(
                os.path.join(memory_dir, f'{safe_player_name(player_name)}_db.pkl')
            )
            if database.player_name != str(player_name):
                return False
            metadata = getattr(database, 'build_metadata', {})
            expected_match_ids = {int(match_id) for match_id in match_ids}
            if metadata.get('use_llm') != bool(use_llm):
                return False
            if metadata.get('max_rallies_per_player') != max_rallies_per_player:
                return False
            if {int(match_id) for match_id in metadata.get('expected_match_ids', [])} != expected_match_ids:
                return False
            if int(metadata.get('target_source_row_count', -1)) != len(database.memories):
                return False
            if int(metadata.get('memory_count', -1)) != len(database.memories):
                return False
            present_match_ids = {
                int(float(memory['match_id']))
                for memory in database.memories
                if memory.get('match_id') is not None
            }
            if present_match_ids != expected_match_ids:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def validate_output(method, directory):
    expected_files = {
        'bc': ('train_state.pkl', 'train_action.pkl'),
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

    from Preprocessing.data_preparation import (
        load_preparation_data,
        prepare_bc_data,
        prepare_naive_rag_data,
        prepare_rallynetv2_data,
        prepare_timerag_data,
        prepare_twin_data,
    )

    if experiment not in input_cache:
        input_cache[experiment] = load_preparation_data(merged_csv, current_split, player_map)
    df, split_content, player2index = input_cache[experiment]

    if method == 'bc':
        mapped_df = df.copy()
        mapped_df['player'] = mapped_df['player'].map(player2index)
        prepare_bc_data(mapped_df, split_content, player2index, destination)
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
    parser.add_argument('--norm', action='store_true', help='Prepare only norm data')
    parser.add_argument('--shifted', action='store_true', help='Prepare only shifted data')
    parser.add_argument('--bc', action='store_true', help='Prepare only BC data')
    parser.add_argument('--naive-rag', dest='naive_rag', action='store_true', help='Prepare only NaiveRAG data')
    parser.add_argument('--timerag', action='store_true', help='Prepare only TimeRAG data')
    parser.add_argument('--twin', action='store_true', help='Prepare only Twin data')
    parser.add_argument('--rallynetv2', action='store_true', help='Prepare only RallyNetV2 data')
    parser.add_argument('--shuttlenet', action='store_true', help='Prepare only ShuttleNet data')
    return parser


def main():
    args = build_parser().parse_args()
    try:
        config, config_path = load_config(args.config)
        config['_config_path'] = config_path
        experiments, methods = select_items(args, config)
        derive_norm_twin, shifted_path = validate_config(config, experiments, methods)
        if 'shifted' in experiments or derive_norm_twin:
            ensure_shifted_split(config, shifted_path)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    twin_config = config.get('twin', {})
    print_runtime_box(
        'Runtime Parameters: prepare_all_data',
        [
            ('config', config_path),
            ('experiments', experiments),
            ('methods', methods),
            ('timerag.window_size', config.get('timerag', {}).get('window_size')),
            ('twin.use_llm', twin_config.get('use_llm')),
            ('twin.workers', twin_config.get('workers')),
            ('twin.max_rallies_per_player', twin_config.get('max_rallies_per_player')),
            ('twin.derive_norm_from_shifted', twin_config.get('derive_norm_from_shifted')),
            ('shuttlenet.encode_length', config.get('shuttlenet_training', {}).get('encode_length')),
        ],
    )

    input_cache = {}
    completed = []
    failures = []

    for experiment in experiments:
        for method in methods:
            if derive_norm_twin and experiment == 'norm' and method == 'twin':
                continue
            try:
                run_method(config, experiment, method, input_cache)
                completed.append((experiment, method))
            except Exception as exc:
                failures.append((experiment, method, str(exc)))

    if derive_norm_twin and 'shifted' not in experiments:
        shifted_twin_dir = output_dir(config, 'twin', 'shifted')
        shifted_split_file = split_path(config, 'shifted')
        if twin_memory_is_complete(
            shifted_split_file,
            shifted_twin_dir,
            use_llm=bool(twin_config.get('use_llm', True)),
            max_rallies_per_player=twin_config.get('max_rallies_per_player'),
        ):
            print(f"Reusing complete shifted Twin memory: {shifted_twin_dir}")
        else:
            try:
                run_method(config, 'shifted', 'twin', input_cache)
                completed.append(('shifted', 'twin'))
            except Exception as exc:
                failures.append(('shifted', 'twin', str(exc)))

    if derive_norm_twin and not any(exp == 'shifted' and method == 'twin' for exp, method, _ in failures):
        try:
            norm_twin_dir = output_dir(config, 'twin', 'norm')
            derive_norm_twin_from_shifted(
                split_path(config, 'norm'),
                output_dir(config, 'twin', 'shifted'),
                norm_twin_dir,
            )
            validate_output('twin', norm_twin_dir)
            completed.append(('norm', 'twin'))
        except Exception as exc:
            failures.append(('norm', 'twin', str(exc)))

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
