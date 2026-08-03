#!/usr/bin/env python3
"""Build model-specific Twin embedding caches from canonical memories."""

import argparse
import os
import sys

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Agent.twin import embedding_model_key
from Preprocessing.data_preparation import prepare_twin_embeddings
from Utils.embeddings import create_embedding_client, resolve_embedding_settings
from Utils.runtime_report import print_runtime_box


CONFIG_PATH = os.path.join(project_path, 'Scripts', 'config.yaml')


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
    parser.add_argument('--players', nargs='+', help='Only build caches for these exact player names')
    parser.add_argument('--force', action='store_true', help='Rebuild caches even when they are valid')
    return parser


def main():
    args = build_parser().parse_args()
    try:
        config = load_config()
        twin_paths = config['paths']['twin']
        embedding_settings = resolve_embedding_settings(config=config)
        provider = embedding_settings['provider']
        model_name = embedding_settings['model_name']
        revision = embedding_settings['revision']
        dimension = embedding_settings['dimension']
        device = embedding_settings['device']
        batch_size = embedding_settings['batch_size']
        dtype = embedding_settings['dtype']
        normalize_embeddings = embedding_settings['normalize_embeddings']
        memory_dir = resolve_path(twin_paths['data'])
        cache_dir = resolve_path(twin_paths['embeddings'])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f'Configuration error: {exc}', file=sys.stderr)
        return 2

    model_key = embedding_model_key(
        model_name, revision, normalize_embeddings, dtype, provider, dimension,
    )
    print_runtime_box(
        'Runtime Parameters: prepare_twin_embeddings',
        [
            ('config', CONFIG_PATH),
            ('dataset', 'rally'),
            ('embedding_provider', provider),
            ('embedding_model', model_name),
            ('embedding_revision', revision),
            ('embedding_dimension', dimension),
            ('embedding_model_key', model_key),
            ('embedding_device', device),
            ('embedding_batch_size', batch_size),
            ('embedding_dtype', dtype),
            ('normalize_embeddings', normalize_embeddings),
            ('players', args.players or 'all'),
            ('force', args.force),
        ],
    )

    shared_model = None

    def get_embedding_model():
        nonlocal shared_model
        if shared_model is None:
            shared_model = create_embedding_client(embedding_settings)
        return shared_model

    print('\nPreparing Twin embeddings: rally')
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
            embedding_provider=provider,
            embedding_dimension=dimension,
        )
    except Exception as exc:
        print(f'FAILED rally: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
