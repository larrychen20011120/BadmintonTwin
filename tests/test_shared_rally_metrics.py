import os
import tempfile
import unittest
from unittest.mock import patch

from Scripts.single_player_eval.evaluate_trajectories import (
    as_player_view,
    evaluate_trajectory_directory,
    extract_sequences,
    trajectory_players,
)
from Scripts.single_player_eval.trajectory_store import save_trajectories


def shot(player, ball_round, shot_type):
    return {
        'player': player,
        'ball_round': ball_round,
        'shot_type': shot_type,
        'landing_x': 0.1,
        'landing_y': 0.2,
        'moving_x': 0.0,
        'moving_y': -0.1,
    }


class SharedRallyMetricsTest(unittest.TestCase):
    def test_player_views_do_not_mutate_physical_trajectory(self):
        trajectory = {
            'target_player': None,
            'players': ['A', 'B'],
            'gt_sequence': [shot('A', 1, 1), shot('B', 2, 2)],
            'pred_sequence': [shot('A', 1, 1), shot('B', 2, 3)],
        }
        self.assertEqual(trajectory_players(trajectory), ['A', 'B'])

        a_data = extract_sequences([as_player_view(trajectory, 'A')], True)
        b_data = extract_sequences([as_player_view(trajectory, 'B')], True)
        self.assertEqual(a_data['all_shot_gt'].tolist(), [1])
        self.assertEqual(b_data['all_shot_gt'].tolist(), [2])
        self.assertIsNone(trajectory['target_player'])

    def test_directory_summary_counts_physical_rally_once(self):
        trajectory = {
            'match_id': 1,
            'set': 1,
            'rally': 1,
            'target_player': None,
            'opponent_player': None,
            'players': ['A', 'B'],
            'gt_sequence': [shot('A', 1, 1), shot('B', 2, 2)],
            'pred_sequence': [shot('A', 1, 1), shot('B', 2, 2)],
            'rally_length_gt': 2,
            'rally_length_pred': 2,
            'simulation_ended_early': False,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, 'input')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(input_dir)
            rally_dir = os.path.join(input_dir, 'run_001', 'rallies')
            save_trajectories(
                [trajectory],
                rally_dir,
                metadata={'algorithm': 'bc', 'experiment': 'rally', 'generation': 1},
            )
            with patch('builtins.print'):
                results = evaluate_trajectory_directory(input_dir, output_dir)

        self.assertEqual(results['experiment_summary'][0]['num_rallies'], 1)
        self.assertEqual(
            {row['target_player'] for row in results['rally_level_metrics']},
            {'A', 'B'},
        )
        self.assertEqual(len(results['physical_rally_metrics']), 1)
        self.assertEqual(len(results['instruction_level_metrics']), 2)


if __name__ == '__main__':
    unittest.main()
