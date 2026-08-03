import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import Scripts.single_player_eval.score as score_module
from Scripts.single_player_eval.score import (
    aggregate_hierarchies,
    compute_hierarchy,
    discover_completed_runs,
    main,
)


def build_rollout_metadata(gt_sequence, pred_sequence, termination_reason):
    return {
        'rally_length_gt': len(gt_sequence),
        'rally_length_pred': len(pred_sequence),
        'rollout_limit': len(gt_sequence) + 2,
        'termination_reason': termination_reason,
        'simulation_ended_early': len(pred_sequence) < len(gt_sequence),
    }


def shot(player, ball_round, shot_type, x=0.1, y=0.2):
    missed = shot_type == 11
    return {
        'player': player,
        'ball_round': ball_round,
        'shot_type': shot_type,
        'landing_x': None if missed else x,
        'landing_y': None if missed else y,
        'moving_x': None if missed else -x,
        'moving_y': None if missed else -y,
    }


def trajectory(pred_sequence=None, rally=1):
    gt_sequence = [
        shot('A', 1, 1, 0.0, 0.2),
        shot('B', 2, 2, 0.1, 0.3),
        shot('A', 3, 4, -0.2, 0.4),
        shot('B', 4, 7, 0.2, 0.6),
    ]
    pred_sequence = pred_sequence or [dict(step) for step in gt_sequence]
    return {
        'match_id': 1,
        'set': 1,
        'rally': rally,
        'players': ['A', 'B'],
        'target_player': None,
        'opponent_player': None,
        'player_profiles': {
            'A': {'cluster_id': 0, 'label': 'Attack', 'instruction': 'Attack more.'},
            'B': {'cluster_id': 1, 'label': 'Defense', 'instruction': 'Defend more.'},
        },
        'gt_sequence': gt_sequence,
        'pred_sequence': pred_sequence,
        **build_rollout_metadata(gt_sequence, pred_sequence, 'natural_terminate'),
    }


def create_run(root, trajectories, algorithm='bc'):
    os.makedirs(root, exist_ok=True)
    generations = [
        int(name.removeprefix('run_'))
        for name in os.listdir(root)
        if name.startswith('run_') and name.removeprefix('run_').isdigit()
    ]
    generation = max(generations, default=0) + 1
    run_dir = os.path.join(root, f'run_{generation:03d}')
    rally_dir = os.path.join(run_dir, 'rallies')
    os.makedirs(rally_dir)
    metadata = {
        'algorithm': algorithm,
        'experiment': 'rally',
        'instruction_mode': 'no-instruct',
        'generation': generation,
    }
    for index, item in enumerate(trajectories, start=1):
        with open(os.path.join(rally_dir, f'rally_{index:03d}.json'), 'w', encoding='utf-8') as handle:
            json.dump({**metadata, **item}, handle)
    manifest = {
        **metadata,
        'status': 'completed',
        'num_rallies': len(trajectories),
    }
    with open(os.path.join(run_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle)
    return run_dir


class ScoreTest(unittest.TestCase):
    def test_score_loads_without_legacy_single_player_eval_modules(self):
        spec = importlib.util.spec_from_file_location('isolated_score', score_module.__file__)
        module = importlib.util.module_from_spec(spec)
        blocked = {
            'Scripts.single_player_eval.evaluate_trajectories': None,
            'Scripts.single_player_eval.trajectory_store': None,
        }
        with patch.dict(sys.modules, blocked):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.compute_hierarchy))
        self.assertTrue(callable(module.load_trajectories))

    def test_identical_trajectory_has_perfect_scores_at_all_scopes(self):
        scores = compute_hierarchy([trajectory()])
        expected = {
            'land_dtw': 0.0,
            'move_dtw': 0.0,
            'shot_lcs_ratio': 1.0,
            'shotland_jsd': 0.0,
            'shotdist_jsd': 0.0,
            'rallylen_jsd': 0.0,
        }
        self.assertEqual(scores['overall']['scores'], expected)
        self.assertEqual(scores['players']['A']['scores'], expected)
        self.assertEqual(scores['players']['A']['clusters']['0']['scores'], expected)

    def test_average_of_n_reports_metric_mean_and_std(self):
        perfect = compute_hierarchy([trajectory()])
        changed_pred = [
            shot('A', 1, 1, 0.4, 0.4),
            shot('B', 2, 3, 0.4, 0.4),
            shot('A', 3, 3, 0.4, 0.4),
            shot('B', 4, 3, 0.4, 0.4),
        ]
        changed = compute_hierarchy([trajectory(changed_pred)])
        aggregate = aggregate_hierarchies([perfect, changed])

        lcs = aggregate['overall']['scores']['shot_lcs_ratio']
        self.assertEqual(lcs['valid_values'], 2)
        self.assertGreater(lcs['std'], 0.0)
        self.assertGreater(lcs['mean'], 0.0)
        self.assertLess(lcs['mean'], 1.0)

    def test_overall_sequence_scores_split_each_rally_by_player(self):
        crossed_pred = [
            shot('A', 1, 2, 0.0, 0.2),
            shot('B', 2, 1, 0.1, 0.3),
            shot('A', 3, 7, -0.2, 0.4),
            shot('B', 4, 4, 0.2, 0.6),
        ]

        scores = compute_hierarchy([trajectory(crossed_pred)])

        self.assertEqual(scores['players']['A']['scores']['shot_lcs_ratio'], 0.0)
        self.assertEqual(scores['players']['B']['scores']['shot_lcs_ratio'], 0.0)
        self.assertEqual(scores['overall']['scores']['shot_lcs_ratio'], 0.0)

    def test_average_of_n_computes_std_from_run_level_means(self):
        def prediction(a_first, a_second):
            return [
                shot('A', 1, a_first, 0.0, 0.2),
                shot('B', 2, 2, 0.1, 0.3),
                shot('A', 3, a_second, -0.2, 0.4),
                shot('B', 4, 7, 0.2, 0.6),
            ]

        first_run = compute_hierarchy([
            trajectory(prediction(2, 7), rally=1),
            trajectory(prediction(1, 4), rally=2),
        ])
        second_run = compute_hierarchy([
            trajectory(prediction(1, 7), rally=1),
            trajectory(prediction(1, 7), rally=2),
        ])

        aggregate = aggregate_hierarchies([first_run, second_run])
        player_lcs = aggregate['players']['A']['scores']['shot_lcs_ratio']

        self.assertEqual(player_lcs['valid_values'], 2)
        self.assertAlmostEqual(player_lcs['mean'], 0.5)
        self.assertAlmostEqual(player_lcs['std'], 0.0)

    def test_player_rally_length_js_uses_complete_rally_length(self):
        extended_pred = [
            shot('A', 1, 1, 0.0, 0.2),
            shot('B', 2, 2, 0.1, 0.3),
            shot('A', 3, 4, -0.2, 0.4),
            shot('B', 4, 7, 0.2, 0.6),
            shot('A', 5, 3, 0.0, 0.2),
        ]

        scores = compute_hierarchy([trajectory(extended_pred)])
        overall = scores['overall']['scores']['rallylen_jsd']

        self.assertAlmostEqual(overall, 1.0)
        self.assertEqual(scores['players']['A']['scores']['rallylen_jsd'], overall)
        self.assertEqual(scores['players']['B']['scores']['rallylen_jsd'], overall)

    def test_cli_writes_json_csv_and_terminal_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = os.path.join(temp_dir, 'runs')
            run_dir = create_run(run_root, [trajectory()])
            output_dir = os.path.join(temp_dir, 'scores')
            argv = [
                'score.py',
                '--run-root', run_root,
                '--output-dir', output_dir,
            ]
            with patch('sys.argv', argv), patch('builtins.print'):
                self.assertEqual(main(), 0)

            for filename in ('scores.json', 'scores.csv', 'table.txt'):
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)))
            with open(os.path.join(output_dir, 'scores.json'), encoding='utf-8') as handle:
                document = json.load(handle)
            self.assertEqual(document['mode'], 'latest')
            self.assertEqual(document['runs'], [os.path.abspath(run_dir)])
            self.assertEqual(document['scores']['overall']['scores']['shot_lcs_ratio'], 1.0)
            self.assertNotIn('_sequence_samples', document['scores']['overall'])
            with open(os.path.join(output_dir, 'table.txt'), encoding='utf-8') as handle:
                table = handle.read()
            self.assertIn('Overall Scores', table)
            self.assertIn('Per-Player Cluster Scores', table)

    def test_cli_all_writes_only_rally_player_mean_and_std(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = os.path.join(temp_dir, 'runs')
            first = create_run(run_root, [trajectory()])
            second = create_run(run_root, [trajectory()])
            output_dir = os.path.join(temp_dir, 'all')
            argv = [
                'score.py',
                '--run-root', run_root,
                '--all',
                '--output-dir', output_dir,
            ]
            with patch('sys.argv', argv), patch('builtins.print'):
                self.assertEqual(main(), 0)

            with open(os.path.join(output_dir, 'scores.json'), encoding='utf-8') as handle:
                document = json.load(handle)
            self.assertEqual(document['mode'], 'all')
            self.assertEqual(document['runs'], [os.path.abspath(first), os.path.abspath(second)])
            self.assertEqual(
                set(document['results']),
                {'average_of_n'},
            )
            self.assertEqual(
                document['results']['average_of_n']['aggregation'],
                'average_of_n',
            )
            self.assertEqual(
                document['results']['average_of_n']['scores']['overall']['scores'][
                    'shot_lcs_ratio'
                ]['valid_values'],
                2,
            )
            with open(os.path.join(output_dir, 'scores.csv'), encoding='utf-8') as handle:
                aggregations = {row['aggregation'] for row in csv.DictReader(handle)}
            self.assertEqual(aggregations, {'average_of_n'})
            self.assertFalse(os.path.exists(os.path.join(output_dir, 'best_selection.csv')))
            with open(os.path.join(output_dir, 'table.txt'), encoding='utf-8') as handle:
                table = handle.read()
            self.assertIn('Average-of-N Scores', table)
            self.assertNotIn('Best-of-N Scores', table)

    def test_run_discovery_uses_latest_completed_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = create_run(temp_dir, [trajectory()])
            second = create_run(temp_dir, [trajectory()])
            incomplete = create_run(temp_dir, [trajectory()])
            manifest_path = os.path.join(incomplete, 'manifest.json')
            with open(manifest_path, encoding='utf-8') as handle:
                manifest = json.load(handle)
            manifest['status'] = 'running'
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle)

            self.assertEqual(
                discover_completed_runs(temp_dir),
                [os.path.abspath(first), os.path.abspath(second)],
            )


if __name__ == '__main__':
    unittest.main()
