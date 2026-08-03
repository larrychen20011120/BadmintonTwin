import unittest

from Agent.random_agent import RandomAgent
from Utils.helper import coord2area


class RandomAgentSamplingTest(unittest.TestCase):
    def test_candidate_sets_ignore_serve_rules_and_protect_first_five(self):
        self.assertEqual(RandomAgent.shot_candidates(1, launch=True), tuple(range(1, 11)))
        self.assertEqual(RandomAgent.shot_candidates(1, launch=False), tuple(range(1, 11)))
        self.assertEqual(RandomAgent.shot_candidates(5, launch=False), tuple(range(1, 11)))
        self.assertEqual(
            RandomAgent.shot_candidates(6, launch=False),
            tuple(range(1, 12)),
        )
        self.assertEqual(RandomAgent.area_candidates(5), tuple(range(1, 25)))
        self.assertEqual(RandomAgent.area_candidates(6), tuple(range(1, 33)))

    def test_area_first_sampling_maps_coordinates_back_to_sampled_area(self):
        agent = RandomAgent('A', seed=42)
        for ball_round in (1, 5, 6):
            for _ in range(100):
                area_id, coordinates = agent.sample_area_position(ball_round)
                self.assertEqual(coord2area(*coordinates), area_id)
                if ball_round <= 5:
                    self.assertLessEqual(area_id, 24)

    def test_first_five_predictions_cannot_miss_or_use_termination_areas(self):
        agent = RandomAgent('A', seed=7)
        state = [0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        for ball_round in range(1, 6):
            for _ in range(100):
                action, _ = agent.predict(
                    state,
                    {'round': [ball_round]},
                    launch=ball_round == 1,
                )
                shot_type, _, landing, moving, _ = action
                self.assertNotEqual(shot_type, 11)
                self.assertLessEqual(coord2area(*landing), 24)
                self.assertLessEqual(coord2area(*moving), 24)

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
