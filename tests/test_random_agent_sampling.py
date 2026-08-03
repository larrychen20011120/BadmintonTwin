import unittest
from unittest.mock import Mock, call

from Agent.random_agent import RandomAgent


class RandomAgentSamplingTest(unittest.TestCase):
    def test_prediction_uses_requested_uniform_ranges(self):
        agent = RandomAgent('A')
        agent.rng = Mock()
        agent.rng.uniform.side_effect = [(-1.0, 0.25), (0.5, 0.999)]
        agent.rng.randint.return_value = 10
        state = [0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

        action, _ = agent.predict(state, {'round': [6]}, launch=False)

        shot_type, _, landing, moving, distribution = action
        self.assertEqual(shot_type, 10)
        self.assertEqual(landing, (-1.0, 0.25))
        self.assertEqual(moving, (0.5, 0.999))
        self.assertEqual(distribution, [0.1] * 10)
        self.assertEqual(
            agent.rng.uniform.call_args_list,
            [call(-1.0, 1.0, size=2), call(-1.0, 1.0, size=2)],
        )
        agent.rng.randint.assert_called_once_with(1, 11)

    def test_predictions_stay_within_coordinate_and_shot_ranges(self):
        agent = RandomAgent('A', seed=7)
        state = [0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        for ball_round in range(1, 9):
            for _ in range(100):
                action, _ = agent.predict(state, {'round': [ball_round]}, ball_round == 1)
                shot_type, _, landing, moving, _ = action
                self.assertLessEqual(1, shot_type)
                self.assertLessEqual(shot_type, 10)
                for coordinate in (*landing, *moving):
                    self.assertGreaterEqual(coordinate, -1.0)
                    self.assertLess(coordinate, 1.0)

    def test_seed_reproduces_complete_action_sequence(self):
        first = RandomAgent('A', seed=123)
        second = RandomAgent('A', seed=123)
        state = [0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        first_actions = []
        second_actions = []
        for ball_round in range(1, 9):
            info = {'round': [ball_round]}
            launch = ball_round == 1
            first_actions.append(first.predict(state, info, launch)[0])
            second_actions.append(second.predict(state, info, launch)[0])
        self.assertEqual(first_actions, second_actions)


if __name__ == '__main__':
    unittest.main()
