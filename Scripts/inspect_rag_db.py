#!/usr/bin/env python3
"""Inspect offline-verbalized NaiveRAG and TimeRAG pickle databases."""

import argparse
import os
import sys

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


def load_config():
    with open(os.path.join(project_path, 'Scripts', 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)


def safe_player_name(player_name):
    return player_name.replace(' ', '_').replace('.', '')


def load_database(algorithm, experiment, player_name):
    config = load_config()
    directory = config['paths'][algorithm]['data'][experiment]
    path = os.path.join(project_path, directory, f'{safe_player_name(player_name)}_db.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Database not found: {path}')

    if algorithm == 'naive_rag':
        from Agent.naive_rag import NaiveRAGDatabase

        database = NaiveRAGDatabase.load(path)
    else:
        from Agent.timerag import TimeRAGDatabase

        database = TimeRAGDatabase.load(path)
    return database, path


def iter_records(algorithm, database):
    if algorithm == 'naive_rag':
        for record_id, verbalization in enumerate(database.case_verbalizations):
            row = database.data.iloc[record_id]
            yield {
                'record_id': record_id,
                'match_id': int(row['match_id']),
                'rally': int(row['rally']),
                'ball_round': int(row['ball_round']),
                **verbalization,
            }
    else:
        for window_id, verbalization in enumerate(database.window_verbalizations):
            yield {
                'record_id': window_id,
                **database.windows_metadata[window_id],
                **verbalization,
            }


def main():
    parser = argparse.ArgumentParser(description='Print precomputed RAG case descriptions without inference')
    parser.add_argument('--algorithm', choices=['naive_rag', 'timerag'], required=True)
    parser.add_argument('--experiment', choices=['norm', 'shifted'], required=True)
    parser.add_argument('--player', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()

    if args.start < 0 or args.limit < 1:
        parser.error('--start must be non-negative and --limit must be positive')

    try:
        database, path = load_database(args.algorithm, args.experiment, args.player)
    except (FileNotFoundError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1

    records = list(iter_records(args.algorithm, database))
    selected = records[args.start:args.start + args.limit]
    print(f'Database: {path}')
    print(f'Player: {database.player_name}')
    print(f'Records: {len(records)}')
    print(f'Schema: {database.schema_version}, verbalization: {database.verbalization_schema_version}')

    for record in selected:
        print('\n' + '=' * 80)
        print(
            f"Record {record['record_id']} | match {record.get('match_id')} | "
            f"rally {record.get('rally')} | ball round {record.get('ball_round')} | "
            f"shots {record.get('num_shots')}"
        )
        print('-' * 80)
        print(record['description'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
