import copy
import unittest

from Scripts.single_player_eval.staterag_kento_multistate_case_study import (
    _case_action_cell,
    _case_result,
    _trajectory_actions,
    assess_adherence,
    select_representative,
    select_shortest_natural,
    validate_independent_rollouts,
)


def _trajectory(index, length, termination="natural_terminate"):
    sequence = []
    for ball_round in range(1, length + 1):
        player = "Kento MOMOTA" if ball_round % 2 else "CHOU Tien Chen"
        sequence.append({
            "player": player,
            "ball_round": ball_round,
            "shot_type": 7 if ball_round < length else 11,
            "landing_x": 0.0 if ball_round < length else None,
            "landing_y": 0.8 if ball_round < length else None,
        })
    return {
        "rollout_index": index,
        "initial_state": {"player_score": 6, "opponent_score": 7},
        "rally_length": length,
        "termination_reason": termination,
        "target_player": "Kento MOMOTA",
        "pred_sequence": sequence,
    }


class KentoMultistateCaseStudyTest(unittest.TestCase):
    def test_five_rollouts_reset_to_identical_source_state(self):
        spec = {
            "source_data_point": {
                "initial_state": {"player_score": 6, "opponent_score": 7},
            },
            "rollouts_per_condition": 5,
        }
        trajectories = [_trajectory(index, 4 + index) for index in range(1, 6)]
        validate_independent_rollouts(trajectories, spec)

        carried = copy.deepcopy(trajectories)
        carried[1]["initial_state"]["player_score"] = 7
        with self.assertRaisesRegex(ValueError, "did not reset"):
            validate_independent_rollouts(carried, spec)

        scored = copy.deepcopy(trajectories)
        scored[1]["score_before"] = {"Kento MOMOTA": 7, "CHOU Tien Chen": 7}
        with self.assertRaisesRegex(ValueError, "score carryover"):
            validate_independent_rollouts(scored, spec)

    def test_shortest_selection_uses_only_natural_terminations(self):
        trajectories = [
            _trajectory(1, 2, "max_shots_reached"),
            _trajectory(2, 7),
            _trajectory(3, 4),
            _trajectory(4, 4),
            _trajectory(5, 9),
        ]
        selected = select_shortest_natural(trajectories)
        self.assertEqual(selected["rollout_index"], 3)
        self.assertEqual(selected["rally_length"], 4)

    def test_short_net_adherence_uses_successful_target_actions(self):
        trajectory = {
            "target_player": "Kento MOMOTA",
            "termination_reason": "natural_terminate",
            "rally_length": 6,
            "pred_sequence": [
                {
                    "player": "Kento MOMOTA",
                    "ball_round": 1,
                    "shot_type": 1,
                    "landing_x": 0.0,
                    "landing_y": 0.8,
                },
                {
                    "player": "CHOU Tien Chen",
                    "ball_round": 2,
                    "shot_type": 8,
                    "landing_x": 0.0,
                    "landing_y": 0.0,
                },
                {
                    "player": "Kento MOMOTA",
                    "ball_round": 3,
                    "shot_type": 7,
                    "landing_x": -0.5,
                    "landing_y": 0.8,
                },
                {
                    "player": "CHOU Tien Chen",
                    "ball_round": 4,
                    "shot_type": 8,
                    "landing_x": 0.0,
                    "landing_y": 0.0,
                },
                {
                    "player": "Kento MOMOTA",
                    "ball_round": 5,
                    "shot_type": 9,
                    "landing_x": 0.5,
                    "landing_y": 0.8,
                },
                {
                    "player": "CHOU Tien Chen",
                    "ball_round": 6,
                    "shot_type": 11,
                    "landing_x": None,
                    "landing_y": None,
                },
            ],
        }
        adherence = assess_adherence(trajectory, "short_net")
        self.assertEqual(adherence["rating"], "strong")
        self.assertIn("Front-court landings: 2 of 2", adherence["evidence"])

    def test_normal_uses_seeded_random_fallback_without_natural_termination(self):
        trajectories = [
            _trajectory(index, 24, "max_shots_reached")
            for index in range(1, 11)
        ]
        selected, reason = select_representative(trajectories, "normal", seed=42)
        self.assertIn(selected["rollout_index"], range(1, 11))
        self.assertEqual(reason, "random_fallback_no_natural_termination")
        repeated, _ = select_representative(trajectories, "normal", seed=42)
        self.assertEqual(selected["rollout_index"], repeated["rollout_index"])

    def test_case_study_actions_use_full_names(self):
        trajectory = _trajectory(1, 2)
        actions = _trajectory_actions(trajectory)
        self.assertIn("Net Shot to Front Center", actions[0])
        self.assertNotIn("FC", actions[0])

    def test_aligned_action_uses_instruction_color(self):
        trajectory = _trajectory(1, 2)
        cell = _case_action_cell(
            trajectory["pred_sequence"][0],
            trajectory,
            highlighted_rounds={1},
            color="netgreen",
        )
        self.assertIn(r"\textcolor{netgreen}", cell)
        self.assertIn("Net Shot to Front Center", cell)

    def test_missed_shot_result_is_explicit(self):
        trajectory = _trajectory(1, 2)
        trajectory.update({"winner": "Kento MOMOTA", "loser": "CHOU Tien Chen"})
        trajectory["pred_sequence"][-1]["player"] = "CHOU Tien Chen"
        result = _case_result(trajectory)
        self.assertEqual(
            result,
            "Kento MOMOTA wins after CHOU Tien Chen misses the shuttle.",
        )


if __name__ == "__main__":
    unittest.main()
