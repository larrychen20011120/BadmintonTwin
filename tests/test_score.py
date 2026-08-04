import copy
import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import Scripts.single_player_eval.score as score_module
from Scripts.single_player_eval.score import (
    DistributionMetrics,
    aggregate_hierarchies,
    aggregate_train_reference_hierarchies,
    compute_hierarchy,
    compute_train_reference_hierarchy,
    discover_completed_runs,
    load_train_reference,
    main,
    predicted_winner,
    resolve_run,
    validate_runs,
    validate_trajectories,
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


def create_run(root, trajectories, algorithm='bc', metadata_overrides=None):
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
    metadata.update(metadata_overrides or {})
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


def terminal_trajectory(rally=1, predicted_winner='A', termination_reason='natural_terminate'):
    loser = 'B' if predicted_winner == 'A' else 'A'
    winner = predicted_winner
    sequence = [
        shot(winner, 1, 1, 0.0, 0.2),
        shot(loser, 2, 11),
    ]
    item = trajectory(sequence, rally=rally)
    item['gt_sequence'] = [dict(step) for step in sequence]
    item['pred_sequence'] = [dict(step) for step in sequence]
    item.update(build_rollout_metadata(sequence, sequence, termination_reason))
    item['termination_reason'] = termination_reason
    return item


def reference_for(trajectories):
    clusters = {}
    test_assignments = {}
    rally_winners = {}
    for player, cluster_id in (('A', 0), ('B', 1)):
        actions = [
            step
            for item in trajectories
            for step in item['gt_sequence']
            if step['player'] == player
        ]
        lengths = [len(item['gt_sequence']) for item in trajectories]
        wins = [
            item['gt_sequence'][-1]['player'] != player
            for item in trajectories
        ]
        clusters[(player, cluster_id)] = {
            'label': f'{player} cluster',
            'instruction': f'Play like {player}.',
            'train_keys': [(99, 1, index + 1) for index in range(len(trajectories))],
            'lengths': lengths,
            'wins': wins,
            'shot_counts': DistributionMetrics.shot_counts(actions),
            'landing_counts': DistributionMetrics.conditioned_landing_counts(actions),
            'overall_landing_counts': DistributionMetrics.landing_counts(actions),
            'length_counts': DistributionMetrics.length_counts(lengths),
        }
    for item in trajectories:
        key = (item['match_id'], item['set'], item['rally'])
        test_assignments[('A', key)] = 0
        test_assignments[('B', key)] = 1
        rally_winners[key] = next(
            player for player in ('A', 'B')
            if player != item['gt_sequence'][-1]['player']
        )
    return {
        'merged_csv_path': '/tmp/merged.csv',
        'profiles_path': '/tmp/profiles.json',
        'merged_csv_sha256': 'merged',
        'profiles_sha256': 'profiles',
        'split_sha256': 'split',
        'clusters': clusters,
        'test_assignments': test_assignments,
        'rally_winners': rally_winners,
    }


class ScoreTest(unittest.TestCase):
    def test_dtw_reports_path_and_max_length_normalizations(self):
        fake_dtw = Mock()
        fake_dtw.distance.return_value = 12.0
        fake_dtw.warping_path.return_value = [(0, 0)] * 4

        with patch.object(score_module, 'dtw_ndim', fake_dtw):
            path_value, maxlen_value = score_module.dtw_distance_normalizations(
                [0, 0, 0], [0, 0, 0, 0, 0]
            )

        self.assertEqual(path_value, 6.0)
        self.assertEqual(maxlen_value, 2.4)
        fake_dtw.distance.assert_called_once()
        fake_dtw.warping_path.assert_called_once()

    def test_selected_run_cli_scores_non_latest_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = os.path.join(temp_dir, 'runs')
            selected = create_run(run_root, [trajectory()])
            create_run(run_root, [trajectory()])
            output_dir = os.path.join(temp_dir, 'selected')
            argv = [
                'score.py', '--run', selected, '--output-dir', output_dir,
            ]
            with patch('sys.argv', argv), patch('builtins.print'):
                self.assertEqual(main(), 0)
            with open(os.path.join(output_dir, 'scores.json'), encoding='utf-8') as handle:
                document = json.load(handle)
            self.assertEqual(document['mode'], 'selected')
            self.assertEqual(document['runs'], [os.path.abspath(selected)])

    def test_selected_run_rejects_all(self):
        argv = [
            'score.py', '--run', '/tmp/run_001', '--all', '--output-dir', '/tmp/out',
        ]
        with patch('sys.argv', argv), self.assertRaises(SystemExit):
            score_module.parse_args()

    def test_train_reference_identical_distributions_and_win_rates_are_zero(self):
        trajectories = [terminal_trajectory(rally=index) for index in range(1, 6)]
        reference = reference_for(trajectories)

        result = compute_train_reference_hierarchy(
            trajectories, reference, min_cluster_test_rallies=5
        )

        cluster = result['generated_test_vs_train']['players']['A']['clusters']['0']
        self.assertEqual(cluster['scores'], {
            'shotdist_jsd': 0.0,
            'shotland_jsd': 0.0,
            'overall_landing_jsd': 0.0,
            'rallylen_jsd': 0.0,
        })
        outcome = result['win_rate']['players']['A']['clusters']['0']
        self.assertEqual(outcome['generated_win_rate'], 1.0)
        self.assertEqual(outcome['train_win_rate'], 1.0)
        self.assertEqual(outcome['win_rate_difference'], 0.0)
        self.assertEqual(outcome['win_rate_absolute_difference'], 0.0)
        self.assertEqual(outcome['generated_win_rate_coverage'], 1.0)

    def test_overall_landing_jsd_ignores_shot_type(self):
        trajectories = [terminal_trajectory(rally=index) for index in range(1, 6)]
        reference = reference_for(trajectories)
        for item in trajectories:
            item['pred_sequence'][0]['shot_type'] = 2

        result = compute_train_reference_hierarchy(trajectories, reference, 5)

        scores = result['generated_test_vs_train']['players']['A']['clusters']['0'][
            'scores'
        ]
        self.assertEqual(scores['overall_landing_jsd'], 0.0)
        self.assertGreater(scores['shotdist_jsd'], 0.0)
        self.assertGreater(scores['shotland_jsd'], 0.0)

    def test_small_clusters_and_empty_generated_distributions_are_excluded(self):
        trajectories = [terminal_trajectory(rally=index) for index in range(1, 5)]
        for item in trajectories:
            item['pred_sequence'] = [shot('A', 1, 11)]
            item.update(build_rollout_metadata(
                item['gt_sequence'], item['pred_sequence'], 'natural_terminate'
            ))
        reference = reference_for(trajectories)

        result = compute_train_reference_hierarchy(
            trajectories, reference, min_cluster_test_rallies=5
        )

        cluster = result['generated_test_vs_train']['players']['A']['clusters']['0']
        self.assertFalse(cluster['eligible_for_macro'])
        self.assertIsNone(cluster['scores']['shotdist_jsd'])
        self.assertIsNone(cluster['scores']['shotland_jsd'])
        self.assertIsNone(cluster['scores']['overall_landing_jsd'])
        self.assertIsNone(
            result['generated_test_vs_train']['overall']['scores']['shotdist_jsd']
        )

    def test_distribution_macro_eligibility_is_metric_specific(self):
        trajectories = [terminal_trajectory(rally=index) for index in range(1, 6)]
        for item in trajectories:
            item['pred_sequence'] = [shot('A', 1, 11)]
            item.update(build_rollout_metadata(
                item['gt_sequence'], item['pred_sequence'], 'natural_terminate'
            ))
        reference = reference_for(trajectories)

        result = compute_train_reference_hierarchy(trajectories, reference, 5)

        cluster = result['generated_test_vs_train']['players']['A']['clusters']['0']
        self.assertTrue(cluster['eligible_for_macro'])
        self.assertEqual(cluster['metric_eligible_for_macro'], {
            'shotdist_jsd': False,
            'shotland_jsd': False,
            'overall_landing_jsd': False,
            'rallylen_jsd': True,
        })

    def test_predicted_winner_marks_non_natural_termination_unresolved(self):
        natural = terminal_trajectory()
        self.assertEqual(predicted_winner(natural), ('A', 'resolved'))

        action_none = terminal_trajectory(termination_reason='action_none')
        self.assertEqual(predicted_winner(action_none), (None, 'action_none'))

        max_rollout = terminal_trajectory(termination_reason='max_rollout_reached')
        self.assertEqual(
            predicted_winner(max_rollout), (None, 'max_rollout_reached')
        )

        malformed = terminal_trajectory()
        malformed['pred_sequence'][-1]['player'] = 'Unknown'
        self.assertEqual(
            predicted_winner(malformed), (None, 'invalid_natural_termination')
        )
        with self.assertRaisesRegex(ValueError, 'unknown players'):
            validate_trajectories([malformed])

    def test_predicted_winner_replays_float32_boundary_termination(self):
        item = terminal_trajectory()
        item['pred_sequence'][-1] = shot(
            'B', 2, 2, x=-0.7183098591549295, y=-0.125
        )

        self.assertEqual(predicted_winner(item), ('A', 'resolved'))

    def test_train_reference_all_aggregates_dynamic_support(self):
        trajectories = [terminal_trajectory(rally=index) for index in range(1, 6)]
        reference = reference_for(trajectories)
        first = compute_train_reference_hierarchy(trajectories, reference, 5)
        changed = [dict(item) for item in trajectories]
        changed[0] = copy.deepcopy(changed[0])
        changed[0]['termination_reason'] = 'action_none'
        second = compute_train_reference_hierarchy(changed, reference, 5)

        aggregate = aggregate_train_reference_hierarchies([first, second])

        support = aggregate['win_rate']['players']['A']['clusters']['0']['support']
        self.assertEqual(support['resolved_generated_rallies']['mean'], 4.5)
        coverage = aggregate['win_rate']['overall']['generated_win_rate_coverage']
        self.assertEqual(coverage['valid_values'], 2)
        self.assertAlmostEqual(coverage['mean'], 0.9)
        self.assertEqual(
            aggregate['win_rate']['overall']['num_clusters_scored']['mean'], 1.0
        )
        self.assertEqual(
            aggregate['win_rate']['players']['A']['num_clusters_scored']['mean'],
            0.5,
        )
        eligibility = aggregate['win_rate']['players']['A']['clusters']['0'][
            'eligible_for_macro'
        ]
        self.assertEqual(eligibility['mean'], 0.5)
        self.assertEqual(
            aggregate['win_rate']['players']['A']['clusters']['0']['support'][
                'unresolved_reasons'
            ]['action_none']['mean'],
            0.5,
        )

        first['provenance']['run'] = '/tmp/run_001'
        second['provenance']['run'] = '/tmp/run_002'
        aggregate = aggregate_train_reference_hierarchies([first, second])
        self.assertNotIn('run', aggregate['provenance'])
        self.assertEqual(
            [item['run'] for item in aggregate['provenance']['mismatches_by_run']],
            ['/tmp/run_001', '/tmp/run_002'],
        )

    def test_train_reference_rejects_missing_embedded_cluster_assignment(self):
        trajectories = [terminal_trajectory(rally=index) for index in range(1, 6)]
        reference = reference_for(trajectories)
        trajectories[0]['player_profiles']['A'] = None

        with self.assertRaisesRegex(ValueError, 'Missing embedded cluster assignment'):
            compute_train_reference_hierarchy(trajectories, reference, 5)

    def test_train_reference_loader_uses_exact_training_assignments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_path = os.path.join(temp_dir, 'merged.csv')
            profiles_path = os.path.join(temp_dir, 'profiles.json')
            fields = [
                'match_id', 'set', 'rally', 'ball_round', 'player', 'player_type',
                'landing_x', 'landing_y', 'getpoint_player', 'lose_reason',
            ]
            with open(merged_path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {
                        'match_id': 10, 'set': 1, 'rally': 1, 'ball_round': 0,
                        'player': 'A', 'player_type': '1.5', 'landing_x': 0.0,
                        'landing_y': 0.2, 'getpoint_player': 'player',
                        'lose_reason': '',
                    },
                    {
                        'match_id': 10, 'set': 1, 'rally': 1, 'ball_round': 1,
                        'player': 'A', 'player_type': 1, 'landing_x': 0.0,
                        'landing_y': 0.2, 'getpoint_player': 'player',
                        'lose_reason': '',
                    },
                    {
                        'match_id': 10, 'set': 1, 'rally': 1, 'ball_round': 2,
                        'player': 'B', 'player_type': '2.0', 'landing_x': 0.0,
                        'landing_y': 0.2, 'getpoint_player': 'player',
                        'lose_reason': 'out',
                    },
                ])
            profiles = {
                'split_sha256': 'split',
                'players': {
                    'A': {
                        'clusters': [{
                            'cluster_id': 0, 'train_size': 1, 'label': 'A0',
                            'instruction': 'Play A0.',
                        }],
                        'train_assignments': [{
                            'match_id': 10, 'set': 1, 'rally': 1, 'cluster_id': 0,
                        }],
                        'test_assignments': [{
                            'match_id': 20, 'set': 1, 'rally': 1, 'cluster_id': 0,
                        }],
                    }
                },
            }
            with open(profiles_path, 'w', encoding='utf-8') as handle:
                json.dump(profiles, handle)

            reference = load_train_reference(merged_path, profiles_path)

            cluster = reference['clusters'][('A', 0)]
            self.assertEqual(cluster['shot_counts'][0], 1)
            self.assertEqual(cluster['length_counts'][2], 1)
            self.assertEqual(cluster['wins'], [True])
            self.assertEqual(reference['test_assignments'][('A', (20, 1, 1))], 0)
            self.assertEqual(DistributionMetrics._safe_int('11.0'), 11)

    def test_train_reference_cli_writes_additive_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_path = os.path.join(temp_dir, 'merged.csv')
            profiles_path = os.path.join(temp_dir, 'profiles.json')
            fields = [
                'match_id', 'set', 'rally', 'ball_round', 'player', 'player_type',
                'landing_x', 'landing_y', 'getpoint_player',
            ]
            with open(merged_path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {
                        'match_id': 10, 'set': 1, 'rally': 1, 'ball_round': 1,
                        'player': 'A', 'player_type': 1, 'landing_x': 0.0,
                        'landing_y': 0.2, 'getpoint_player': 'A',
                    },
                    {
                        'match_id': 10, 'set': 1, 'rally': 1, 'ball_round': 2,
                        'player': 'B', 'player_type': 11, 'landing_x': '',
                        'landing_y': '', 'getpoint_player': 'A',
                    },
                    {
                        'match_id': 1, 'set': 1, 'rally': 1, 'ball_round': 1,
                        'player': 'A', 'player_type': 1, 'landing_x': 0.0,
                        'landing_y': 0.2, 'getpoint_player': 'A',
                    },
                    {
                        'match_id': 1, 'set': 1, 'rally': 1, 'ball_round': 2,
                        'player': 'B', 'player_type': 11, 'landing_x': '',
                        'landing_y': '', 'getpoint_player': 'A',
                    },
                ])
            profiles = {
                'split_sha256': 'split',
                'players': {
                    'A': {
                        'clusters': [{
                            'cluster_id': 0, 'train_size': 1, 'label': 'A0',
                            'instruction': 'Play A0.',
                        }],
                        'train_assignments': [{
                            'match_id': 10, 'set': 1, 'rally': 1, 'cluster_id': 0,
                        }],
                        'test_assignments': [{
                            'match_id': 1, 'set': 1, 'rally': 1, 'cluster_id': 0,
                        }],
                    }
                },
            }
            with open(profiles_path, 'w', encoding='utf-8') as handle:
                json.dump(profiles, handle)

            item = terminal_trajectory()
            item['player_profiles']['B'] = None
            run_root = os.path.join(temp_dir, 'runs')
            run_dir = create_run(
                run_root,
                [item],
                algorithm='twin',
                metadata_overrides={
                    'instruction_mode': 'instruct',
                    'instruction_variant': 'original',
                    'merged_csv_sha256': score_module._file_sha256(merged_path),
                    'profiles_sha256': score_module._file_sha256(profiles_path),
                    'split_sha256': 'split',
                },
            )
            output_dir = os.path.join(temp_dir, 'scores')
            argv = [
                'score.py', '--run', run_dir, '--train-reference',
                '--merged-csv', merged_path, '--rally-profiles', profiles_path,
                '--min-cluster-test-rallies', '1', '--output-dir', output_dir,
            ]

            with patch('sys.argv', argv), patch('builtins.print'):
                self.assertEqual(main(), 0)

            for filename in (
                'scores.json', 'scores.csv', 'table.txt',
                'train_reference_scores.csv', 'train_reference_outcomes.csv',
            ):
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)))
            with open(os.path.join(output_dir, 'scores.json'), encoding='utf-8') as handle:
                document = json.load(handle)
            self.assertEqual(document['mode'], 'selected')
            self.assertEqual(
                document['train_reference']['generated_test_vs_train']['overall']['scores'][
                    'shotdist_jsd'
                ],
                0.0,
            )
            with open(
                os.path.join(output_dir, 'train_reference_scores.csv'),
                encoding='utf-8',
            ) as handle:
                metric_names = {row['metric'] for row in csv.DictReader(handle)}
            self.assertEqual(metric_names, {
                'shotdist_jsd', 'shotland_jsd', 'overall_landing_jsd',
                'rallylen_jsd',
            })
            with open(
                os.path.join(output_dir, 'train_reference_outcomes.csv'),
                encoding='utf-8',
            ) as handle:
                outcome_rows = list(csv.DictReader(handle))
                outcome_names = {row['metric'] for row in outcome_rows}
            self.assertIn('win_rate_difference', outcome_names)
            self.assertIn('generated_win_rate_coverage', outcome_names)
            cluster_rows = [row for row in outcome_rows if row['scope'] == 'cluster']
            self.assertTrue(all(row['eligible_for_macro'] == 'True' for row in cluster_rows))

            without_reference = [
                'score.py', '--run', run_dir, '--output-dir', output_dir,
            ]
            with patch('sys.argv', without_reference), patch('builtins.print'):
                self.assertEqual(main(), 0)
            self.assertFalse(os.path.exists(
                os.path.join(output_dir, 'train_reference_scores.csv')
            ))
            self.assertFalse(os.path.exists(
                os.path.join(output_dir, 'train_reference_outcomes.csv')
            ))

    def test_run_rejects_trajectory_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = create_run(
                temp_dir,
                [trajectory()],
                metadata_overrides={'instruction_variant': 'original'},
            )
            rally_path = os.path.join(run_dir, 'rallies', 'rally_001.json')
            with open(rally_path, encoding='utf-8') as handle:
                document = json.load(handle)
            document['instruction_variant'] = 'opposite'
            with open(rally_path, 'w', encoding='utf-8') as handle:
                json.dump(document, handle)

            with self.assertRaisesRegex(ValueError, 'instruction_variant'):
                resolve_run(run_dir)

    def test_train_reference_all_cli_writes_aggregated_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectories = [terminal_trajectory(rally=index) for index in range(1, 6)]
            reference = reference_for(trajectories)
            metadata = {
                'instruction_mode': 'instruct',
                'instruction_variant': 'original',
                'merged_csv_sha256': 'merged',
                'profiles_sha256': 'profiles',
                'split_sha256': 'split',
            }
            run_root = os.path.join(temp_dir, 'runs')
            first = create_run(run_root, trajectories, algorithm='twin', metadata_overrides=metadata)
            second = create_run(run_root, trajectories, algorithm='twin', metadata_overrides=metadata)
            output_dir = os.path.join(temp_dir, 'scores')
            argv = [
                'score.py', '--run-root', run_root, '--all', '--train-reference',
                '--min-cluster-test-rallies', '5', '--output-dir', output_dir,
            ]

            with (
                patch('sys.argv', argv),
                patch('builtins.print'),
                patch.object(score_module, 'load_train_reference', return_value=reference),
            ):
                self.assertEqual(main(), 0)

            with open(os.path.join(output_dir, 'scores.json'), encoding='utf-8') as handle:
                document = json.load(handle)
            aggregate = document['results']['average_of_n']['train_reference']
            self.assertEqual(
                aggregate['win_rate']['overall']['num_clusters_scored']['mean'],
                2.0,
            )
            self.assertNotIn('run', aggregate['provenance'])
            self.assertEqual(
                [item['run'] for item in aggregate['provenance']['mismatches_by_run']],
                [os.path.abspath(first), os.path.abspath(second)],
            )
            with open(
                os.path.join(output_dir, 'train_reference_outcomes.csv'),
                encoding='utf-8',
            ) as handle:
                cluster_rows = [
                    row for row in csv.DictReader(handle) if row['scope'] == 'cluster'
                ]
            self.assertTrue(all(row['eligible_for_macro'] == '1.0' for row in cluster_rows))
            with open(
                os.path.join(output_dir, 'train_reference_scores.csv'),
                encoding='utf-8',
            ) as handle:
                distribution_rows = [
                    row for row in csv.DictReader(handle) if row['scope'] == 'cluster'
                ]
            eligibility_values = {
                row['eligible_for_macro'] for row in distribution_rows
            }
            self.assertEqual(eligibility_values, {'0.0', '1.0'})

    def test_runs_with_different_instruction_variants_are_rejected(self):
        common = {
            'algorithm': 'pure_llm',
            'experiment': 'rally',
            'instruction_mode': 'instruct',
        }
        runs = [
            {'manifest': {**common, 'instruction_variant': variant}, 'keys': set(), 'by_key': {}}
            for variant in ('opposite', 'rewrite')
        ]

        with self.assertRaisesRegex(ValueError, "instruction_variant"):
            validate_runs(runs)

    def test_missing_instruction_variant_is_treated_as_original(self):
        common = {
            'algorithm': 'pure_llm',
            'experiment': 'rally',
            'instruction_mode': 'instruct',
        }
        runs = [
            {'manifest': common, 'keys': set(), 'by_key': {}},
            {
                'manifest': {**common, 'instruction_variant': 'original'},
                'keys': set(),
                'by_key': {},
            },
        ]

        validate_runs(runs)

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
            'land_dtw_maxlen': 0.0,
            'move_dtw_maxlen': 0.0,
            'shot_lcs_ratio': 1.0,
            'shotland_jsd': 0.0,
            'shotdist_jsd': 0.0,
            'rallylen_jsd': 0.0,
        }
        self.assertEqual(scores['overall']['scores'], expected)
        self.assertEqual(list(scores['overall']['scores']), list(expected))
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
        for metric in ('land_dtw_maxlen', 'move_dtw_maxlen'):
            value = aggregate['overall']['scores'][metric]
            self.assertEqual(value['valid_values'], 2)
            self.assertGreater(value['mean'], 0.0)
            self.assertGreater(value['std'], 0.0)

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
            self.assertEqual(
                document['scores']['overall']['scores']['land_dtw_maxlen'], 0.0
            )
            self.assertNotIn('_sequence_samples', document['scores']['overall'])
            with open(os.path.join(output_dir, 'scores.csv'), encoding='utf-8') as handle:
                metric_names = {row['metric'] for row in csv.DictReader(handle)}
            self.assertIn('land_dtw_maxlen', metric_names)
            self.assertIn('move_dtw_maxlen', metric_names)
            with open(os.path.join(output_dir, 'table.txt'), encoding='utf-8') as handle:
                table = handle.read()
            self.assertIn('Overall Scores', table)
            self.assertIn('Per-Player Cluster Scores', table)
            self.assertLess(table.index('LandDTWMaxLen'), table.index('MoveDTWMaxLen'))

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
            self.assertEqual(
                document['results']['average_of_n']['scores']['overall']['scores'][
                    'land_dtw_maxlen'
                ],
                {'mean': 0.0, 'std': 0.0, 'valid_values': 2},
            )
            with open(os.path.join(output_dir, 'scores.csv'), encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
                aggregations = {row['aggregation'] for row in rows}
            self.assertEqual(aggregations, {'average_of_n'})
            self.assertIn('land_dtw_maxlen', {row['metric'] for row in rows})
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
