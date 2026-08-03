import json
import os
import pickle
import tempfile
import unittest

from Scripts.single_player_eval.trajectory_store import (
    build_rollout_metadata,
    completed_rally_keys,
    create_run_directory,
    load_trajectories,
    save_trajectories,
)


class TrajectoryStoreTest(unittest.TestCase):
    @staticmethod
    def decision_trace():
        stage2_case = {
            'database_player': 'A',
            'stage': 2,
            'store': 'shot_conditioned',
            'case_id': 3,
            'conditioned_shot_type': 1,
        }
        return {
            'schema_version': 2,
            'architecture': 'two_stage_v1',
            'stage1': {
                'normal_case_ids': [],
                'miss_case_ids': [],
                'outcome': 'returned',
                'shot_type': 1,
                'rationale': 'Reachable serve.',
                'source': 'llm',
            },
            'stage2': {
                'status': 'completed',
                'conditioned_shot_type': 1,
                'case_ids': [stage2_case],
                'rationale': 'Conditioned placement.',
                'source': 'llm',
            },
            'case_ids': [stage2_case],
            'rationale': 'Combined rationale.',
            'action': {
                'shot_type': 1,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            },
            'action_source': 'llm+llm',
        }

    def trajectory(self, target_player='A'):
        gt_sequence = [{'player': 'A', 'ball_round': 1, 'shot_type': 1}]
        pred_sequence = [{
            'player': 'A',
            'ball_round': 1,
            'shot_type': 1,
            'landing_x': 0.0,
            'landing_y': 0.0,
            'moving_x': 0.0,
            'moving_y': 0.0,
        }]
        return {
            'match_id': 1,
            'set': 1,
            'rally': 2,
            'players': ['A', 'B'],
            'target_player': target_player,
            'gt_sequence': gt_sequence,
            'pred_sequence': pred_sequence,
            **build_rollout_metadata(gt_sequence, pred_sequence, 'natural_terminate'),
        }

    def test_saves_one_json_object_per_rally_simulation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, generation = create_run_directory(temp_dir)
            rally_dir = os.path.join(run_dir, 'rallies')
            trajectories = [self.trajectory('A'), self.trajectory('B')]

            paths = save_trajectories(
                trajectories,
                rally_dir,
                metadata={'algorithm': 'random', 'generation': generation},
            )

            self.assertEqual(len(paths), 2)
            self.assertEqual(len(set(paths)), 2)
            for path in paths:
                with open(path, 'r', encoding='utf-8') as handle:
                    document = json.load(handle)
                self.assertIsInstance(document, dict)
                self.assertEqual(document['schema_version'], 1)
                self.assertEqual(document['rollout_limit'], 3)

            loaded = load_trajectories(rally_dir)
            self.assertEqual({item['target_player'] for item in loaded}, {'A', 'B'})

    def test_loads_legacy_pickle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'legacy.pkl')
            trajectory = self.trajectory()
            with open(path, 'wb') as handle:
                pickle.dump([trajectory], handle)

            self.assertEqual(load_trajectories(path), [trajectory])

    def test_rollout_metadata_marks_only_short_predictions_as_early(self):
        metadata = build_rollout_metadata([1, 2], [1], 'natural_terminate')
        self.assertEqual(metadata['rollout_limit'], 4)
        self.assertTrue(metadata['simulation_ended_early'])

        metadata = build_rollout_metadata([1, 2], [1, 2, 3, 4], 'max_rollout_reached')
        self.assertFalse(metadata['simulation_ended_early'])

    def test_completed_rallies_require_valid_json_and_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            with open(os.path.join(temp_dir, 'broken.json'), 'w', encoding='utf-8') as handle:
                handle.write('{')

            completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, {(1, 1, 2)})
            self.assertEqual(len(invalid), 1)

            completed, invalid = completed_rally_keys(temp_dir, 'different')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 2)

    def test_rag_trajectory_requires_per_shot_decision_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(completed, set())
        self.assertEqual(len(invalid), 1)

    def test_rag_trajectory_accepts_consistent_two_stage_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = self.decision_trace()
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_twin_trajectory_requires_and_accepts_aspect_pre_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trace = self.decision_trace()
            trace['pre_stage'] = {
                'architecture': 'twin_aspect_v1',
                'semantic_query_text': 'Current Twin semantic query.',
                'source': 'llm',
            }
            trajectory['pred_sequence'][0]['decision'] = trace
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'twin',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_rejects_outer_action_mismatch_and_bad_case_type(self):
        for mutation in ('outer_shot', 'bad_case', 'invalid_shot_type'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trace = self.decision_trace()
                trajectory['pred_sequence'][0]['decision'] = trace
                if mutation == 'outer_shot':
                    trajectory['pred_sequence'][0]['shot_type'] = 9
                elif mutation == 'bad_case':
                    trace['stage2']['case_ids'] = [1]
                else:
                    trajectory['pred_sequence'][0]['shot_type'] = 0
                    trace['stage1']['shot_type'] = 0
                    trace['stage2']['conditioned_shot_type'] = 0
                    trace['stage2']['case_ids'][0]['conditioned_shot_type'] = 0
                    trace['action']['shot_type'] = 0
                save_trajectories(
                    [trajectory],
                    temp_dir,
                    metadata={
                        'algorithm': 'naive_rag',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 1)


if __name__ == '__main__':
    unittest.main()
