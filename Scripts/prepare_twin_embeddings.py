#!/usr/bin/env python3
"""Build model-specific Twin embedding caches from canonical memories."""

import argparse
import os
import sys

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Agent.twin import embedding_model_key, load_embedding_model
from Preprocessing.data_preparation import prepare_twin_embeddings
from Utils.runtime_report import print_runtime_box


CONFIG_PATH = os.path.join(project_path, 'Scripts', 'config.yaml')
SUPPORTED_EXPERIMENTS = ('norm', 'shifted')


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_path, path)


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f) or {}


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build Twin embeddings using the model configured in Scripts/config.yaml',
    )
    parser.add_argument('--norm', action='store_true', help='Build norm embedding caches')
    parser.add_argument('--shifted', action='store_true', help='Build shifted embedding caches')
    parser.add_argument('--players', nargs='+', help='Only build caches for these exact player names')
    parser.add_argument('--force', action='store_true', help='Rebuild caches even when they are valid')
    return parser


def main():
    args = build_parser().parse_args()
    experiments = [name for name in SUPPORTED_EXPERIMENTS if getattr(args, name)]
    if not experiments:
        experiments = list(SUPPORTED_EXPERIMENTS)

    try:
        config = load_config()
        twin_config = config['twin']
        twin_paths = config['paths']['twin']
        model_name = str(twin_config['embedding_model']).strip()
        revision = twin_config.get('embedding_revision')
        device = twin_config.get('embedding_device', 'auto')
        batch_size = twin_config.get('embedding_batch_size')
        dtype = twin_config.get('embedding_dtype', 'float32')
        normalize_embeddings = bool(twin_config.get('normalize_embeddings', True))
        if not model_name:
            raise ValueError('twin.embedding_model must not be empty')
        if device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('twin.embedding_device must be auto, cpu, or cuda')
        if dtype not in {'float16', 'float32'}:
            raise ValueError('twin.embedding_dtype must be float16 or float32')
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError('twin.embedding_batch_size must be a positive integer or null')
        for experiment in experiments:
            if experiment not in twin_paths['memory'] or experiment not in twin_paths['embeddings']:
                raise ValueError(f'Missing Twin paths for experiment: {experiment}')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f'Configuration error: {exc}', file=sys.stderr)
        return 2

    model_key = embedding_model_key(model_name, revision, normalize_embeddings, dtype)
    print_runtime_box(
        'Runtime Parameters: prepare_twin_embeddings',
        [
            ('config', CONFIG_PATH),
            ('experiments', experiments),
            ('embedding_model', model_name),
            ('embedding_revision', revision),
            ('embedding_model_key', model_key),
            ('embedding_device', device),
            ('embedding_batch_size', batch_size),
            ('embedding_dtype', dtype),
            ('normalize_embeddings', normalize_embeddings),
            ('players', args.players or 'all'),
            ('force', args.force),
        ],
    )

    failures = []
    shared_model = None

    def get_embedding_model():
        nonlocal shared_model
        if shared_model is None:
            shared_model = load_embedding_model(model_name, device, revision)
        return shared_model

    for experiment in experiments:
        memory_dir = resolve_path(twin_paths['memory'][experiment])
        cache_dir = resolve_path(twin_paths['embeddings'][experiment])
        print(f'\nPreparing Twin embeddings: {experiment}')
        print(f'Memory: {memory_dir}')
        print(f'Cache:  {cache_dir}')
        try:
            prepare_twin_embeddings(
                memory_dir,
                cache_dir,
                model_name,
                embedding_device=device,
                embedding_batch_size=batch_size,
                embedding_revision=revision,
                normalize_embeddings=normalize_embeddings,
                embedding_dtype=dtype,
                players=args.players,
                force=args.force,
                embedding_model_loader=get_embedding_model,
            )
        except Exception as exc:
            failures.append((experiment, str(exc)))

    for experiment, error in failures:
        print(f'FAILED {experiment}: {error}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
