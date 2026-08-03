import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from Agent.shuttlenet import ShuttleNetEvaluator
from Scripts.single_player_eval.evaluators.bc_evaluator import BCEvaluator
from Scripts.single_player_eval.evaluators.rallynet_evaluator import RallyNetEvaluator
from Scripts.single_player_eval.evaluators.random_evaluator import RandomEvaluator
from Scripts.single_player_eval.score import validate_trajectories
from Utils.helper import action_col, state_col


class DummyBCModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        landing = torch.zeros((1, 2))
        shots = torch.zeros((1, 12))
        shots[0, 1] = 10.0
        movement = torch.zeros((1, 2))
        return landing, shots, movement


class DummyShuttlePolicy:
    def __init__(self):
        self.calls = 0

    def predict_next(self, history, target_player, launch=False):
        self.calls += 1
        return {
            'shot_type': 1,
            'landing': [0.0, 0.0],
            'moving': [0.0, 0.0],
        }


class DummyRandomAgent:
    def __init__(self):
        self.calls = 0

    def predict(self, state, info, launch=False):
        self.calls += 1
        return (1 if launch else 2, state[-1], (0.0, 0.0), (0.0, 0.0), [0.1] * 10), ''


class DummyRallyNetPolicy:
    def __init__(self, max_sequence_length=10, miss=False):
        self.model = type('Model', (), {
            'model_config': {'max_sequence_length': max_sequence_length},
        })()
        self.calls = []
        self.miss = miss

    def predict_next(self, history, launch):
        self.calls.append((len(history), launch))
        if self.miss:
            return {
                'is_error': True,
                'shot_type': 11,
                'landing': None,
                'moving': None,
            }
        return {
            'is_error': False,
            'shot_type': 1 if launch else 2,
            'landing': torch.tensor([0.0, 0.0]),
            'moving': torch.tensor([0.0, 0.0]),
        }


def model_rows(round_count=2):
    rows = []
    for ball_round in range(1, round_count + 1):
        player = 'A' if ball_round % 2 else 'B'
        row = {column: 0.0 for column in state_col + action_col}
        row.update({
            'match_id': 3,
            'set': 1,
            'rally': 4,
            'ball_round': ball_round,
            'player': player,
            'player_type': 1 if ball_round == 1 else 2,
            'opponent_type': 0 if ball_round == 1 else 1,
        })
        rows.append(row)
    return pd.DataFrame(rows)


class AllRallyEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.split_path = os.path.join(self.temp_dir.name, 'split.json')
        record = {'match_id': 3, 'set': 1, 'rally': 4}
        with open(self.split_path, 'w', encoding='utf-8') as handle:
            json.dump({
                'train': {'A': [], 'B': []},
                'test': {'A': [record], 'B': [record]},
            }, handle)

    def common_attributes(self, evaluator):
        evaluator.device = 'cpu'
        evaluator.test_split_path = self.split_path
        evaluator.rally_profiles_path = None
        evaluator.all_rallies = True
        evaluator.player_name = None
        evaluator.player2index = {'A': 0, 'B': 1}
        evaluator.df = model_rows()
        evaluator.config = {'evaluation': {}}
        evaluator.save_trajectories = False
        evaluator.trajectory_output_path = None

    def test_bc_runs_duplicate_player_assignments_once(self):
        evaluator = BCEvaluator.__new__(BCEvaluator)
        self.common_attributes(evaluator)
        evaluator.distribution = 'deterministic'
        evaluator.gmm_num = 5
        evaluator.model = DummyBCModel()

        with patch(
            'Scripts.single_player_eval.evaluators.bc_evaluator.terminate',
            return_value=True,
        ):
            trajectories = evaluator.simulate()

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(evaluator.model.calls, 1)
        self.assertEqual(trajectories[0]['players'], ['A', 'B'])
        self.assertIsNone(trajectories[0]['target_player'])

    def test_shuttlenet_runs_duplicate_player_assignments_once(self):
        evaluator = ShuttleNetEvaluator.__new__(ShuttleNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.policy = DummyShuttlePolicy()

        with patch('Agent.shuttlenet.terminate', return_value=True):
            trajectories = evaluator.simulate()

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(evaluator.policy.calls, 1)
        self.assertEqual(trajectories[0]['players'], ['A', 'B'])

    def test_random_runs_duplicate_player_assignments_once(self):
        evaluator = RandomEvaluator.__new__(RandomEvaluator)
        self.common_attributes(evaluator)
        evaluator.agent = DummyRandomAgent()

        with patch(
            'Scripts.single_player_eval.evaluators.random_evaluator.terminate',
            return_value=True,
        ):
            trajectories = evaluator.simulate()

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(evaluator.agent.calls, 1)
        self.assertEqual(trajectories[0]['players'], ['A', 'B'])
        self.assertIsNone(trajectories[0]['target_player'])
        self.assertEqual(validate_trajectories(trajectories), {(3, 1, 4)})

    def test_rallynet_runs_physical_rally_once_and_resets_history(self):
        evaluator = RallyNetEvaluator.__new__(RallyNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.records_limit = None
        evaluator.policy = DummyRallyNetPolicy()

        with patch(
            'Scripts.single_player_eval.evaluators.rallynet_evaluator.terminate',
            return_value=False,
        ):
            trajectories = evaluator.simulate()

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(evaluator.policy.calls, [(1, True), (2, False), (3, False), (4, False)])
        self.assertEqual(trajectories[0]['players'], ['A', 'B'])
        self.assertEqual(trajectories[0]['instruction_mode'], 'no_instruct')
        self.assertIsNone(trajectories[0]['target_player'])
        self.assertEqual(trajectories[0]['termination_reason'], 'max_rollout_reached')

    def test_rallynet_miss_uses_null_coordinates_and_terminates_naturally(self):
        evaluator = RallyNetEvaluator.__new__(RallyNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.records_limit = None
        evaluator.policy = DummyRallyNetPolicy(miss=True)

        trajectories = evaluator.simulate()

        trajectory = trajectories[0]
        self.assertEqual(trajectory['termination_reason'], 'natural_terminate')
        self.assertEqual(trajectory['rally_length_pred'], 1)
        self.assertEqual(trajectory['pred_sequence'][0]['shot_type'], 11)
        self.assertIsNone(trajectory['pred_sequence'][0]['landing_x'])
        self.assertIsNone(trajectory['pred_sequence'][0]['moving_y'])

    def test_rallynet_checks_rollout_history_capacity_before_prediction(self):
        evaluator = RallyNetEvaluator.__new__(RallyNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.records_limit = None
        evaluator.policy = DummyRallyNetPolicy(max_sequence_length=3)

        with self.assertRaisesRegex(ValueError, 'required rollout history'):
            evaluator.simulate()
        self.assertEqual(evaluator.policy.calls, [])

    def test_rallynet_records_limit_slices_canonical_records(self):
        evaluator = RallyNetEvaluator.__new__(RallyNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.records_limit = 0
        evaluator.policy = DummyRallyNetPolicy(max_sequence_length=0)

        trajectories = evaluator.simulate()

        self.assertEqual(trajectories, [])
        self.assertEqual(evaluator.policy.calls, [])

    def test_rallynet_records_malformed_test_rallies_instead_of_aborting(self):
        evaluator = RallyNetEvaluator.__new__(RallyNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.records_limit = None
        evaluator.policy = DummyRallyNetPolicy()
        evaluator.df.loc[evaluator.df['ball_round'] == 1, 'player_type'] = 3

        trajectories = evaluator.simulate()

        self.assertEqual(trajectories, [])
        self.assertEqual(evaluator.policy.calls, [])
        self.assertEqual(len(evaluator.skipped_malformed_rallies), 1)
        self.assertIn(
            'Invalid successful serve',
            evaluator.skipped_malformed_rallies[0]['reason'],
        )

    def test_bc_rollout_limit_is_gt_length_plus_two(self):
        evaluator = BCEvaluator.__new__(BCEvaluator)
        self.common_attributes(evaluator)
        evaluator.distribution = 'deterministic'
        evaluator.gmm_num = 5
        evaluator.model = DummyBCModel()

        with patch(
            'Scripts.single_player_eval.evaluators.bc_evaluator.terminate',
            return_value=False,
        ):
            trajectories = evaluator.simulate()

        trajectory = trajectories[0]
        self.assertEqual(evaluator.model.calls, 4)
        self.assertEqual(trajectory['rollout_limit'], 4)
        self.assertEqual(trajectory['rally_length_pred'], 4)
        self.assertEqual(trajectory['termination_reason'], 'max_rollout_reached')
        self.assertFalse(trajectory['simulation_ended_early'])

    def test_shuttlenet_rollout_limit_is_gt_length_plus_two(self):
        evaluator = ShuttleNetEvaluator.__new__(ShuttleNetEvaluator)
        self.common_attributes(evaluator)
        evaluator.policy = DummyShuttlePolicy()

        with patch('Agent.shuttlenet.terminate', return_value=False):
            trajectories = evaluator.simulate()

        trajectory = trajectories[0]
        self.assertEqual(evaluator.policy.calls, 4)
        self.assertEqual(trajectory['rollout_limit'], 4)
        self.assertEqual(trajectory['rally_length_pred'], 4)
        self.assertEqual(trajectory['termination_reason'], 'max_rollout_reached')
        self.assertFalse(trajectory['simulation_ended_early'])

    def test_random_rollout_limit_preserves_gt_plus_two_with_minimum_five(self):
        evaluator = RandomEvaluator.__new__(RandomEvaluator)
        self.common_attributes(evaluator)
        evaluator.agent = DummyRandomAgent()

        with patch(
            'Scripts.single_player_eval.evaluators.random_evaluator.terminate',
            return_value=False,
        ):
            trajectories = evaluator.simulate()

        trajectory = trajectories[0]
        self.assertEqual(evaluator.agent.calls, 5)
        self.assertEqual(trajectory['rollout_limit'], 5)
        self.assertEqual(trajectory['rally_length_pred'], 5)
        self.assertEqual(trajectory['termination_reason'], 'max_rollout_reached')

    def test_random_uses_gt_plus_two_when_it_exceeds_minimum_five(self):
        evaluator = RandomEvaluator.__new__(RandomEvaluator)
        self.common_attributes(evaluator)
        evaluator.df = model_rows(round_count=4)
        evaluator.agent = DummyRandomAgent()

        with patch(
            'Scripts.single_player_eval.evaluators.random_evaluator.terminate',
            return_value=False,
        ):
            trajectories = evaluator.simulate()

        trajectory = trajectories[0]
        self.assertEqual(evaluator.agent.calls, 6)
        self.assertEqual(trajectory['rollout_limit'], 6)
        self.assertEqual(trajectory['rally_length_pred'], 6)
        self.assertEqual(trajectory['termination_reason'], 'max_rollout_reached')


if __name__ == '__main__':
    unittest.main()
