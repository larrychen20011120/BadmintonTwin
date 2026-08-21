import unittest

from Scripts.single_player_eval.staterag_carolina_case_study import (
    _terminal_outcome,
    assess_alignment,
    select_balanced_cases,
)


def _trajectory(condition, index, length, outcome="Out of bounds", winner="Carolina MARIN"):
    sequence = []
    for ball_round in range(1, length + 1):
        target_turn = ball_round % 2 == 1
        sequence.append({
            "player": "Carolina MARIN" if target_turn else "Akane YAMAGUCHI",
            "ball_round": ball_round,
            "shot_type": 11 if ball_round == length and outcome == "Missed shot" else 2,
            "landing_x": None if ball_round == length and outcome == "Missed shot" else -0.5,
            "landing_y": None if ball_round == length and outcome == "Missed shot" else -0.3,
        })
    loser = "Akane YAMAGUCHI" if winner == "Carolina MARIN" else "Carolina MARIN"
    sequence[-1]["player"] = loser
    return {
        "condition": condition,
        "rollout_index": index,
        "rally_length": length,
        "termination_reason": "natural_terminate",
        "target_player": "Carolina MARIN",
        "winner": winner,
        "loser": loser,
        "pred_sequence": sequence,
    }


class CarolinaCaseStudyTest(unittest.TestCase):
    def test_balanced_selection_keeps_outcome_diversity_then_minimizes_range(self):
        conditions = {
            "no_instruction": [
                _trajectory("no_instruction", 1, 8),
                _trajectory("no_instruction", 2, 20),
            ],
            "instruction_01_cluster_0": [
                _trajectory("instruction_01_cluster_0", 1, 9),
                _trajectory("instruction_01_cluster_0", 2, 19, "Missed shot"),
            ],
            "instruction_02_cluster_1": [
                _trajectory("instruction_02_cluster_1", 1, 18),
                _trajectory("instruction_02_cluster_1", 2, 7),
            ],
            "instruction_03_cluster_2": [
                _trajectory("instruction_03_cluster_2", 1, 17),
                _trajectory("instruction_03_cluster_2", 2, 6),
            ],
        }
        selected = select_balanced_cases(conditions)
        trajectories = [item[1] for item in selected]
        outcomes = {_terminal_outcome(trajectory) for trajectory in trajectories}
        lengths = [trajectory["rally_length"] for trajectory in trajectories]
        self.assertEqual(outcomes, {"Out of bounds", "Missed shot"})
        self.assertLessEqual(max(lengths) - min(lengths), 3)

    def test_deep_reset_highlights_back_court_reset_shots(self):
        trajectory = _trajectory("instruction_03_cluster_2", 1, 6)
        alignment = assess_alignment(trajectory, "deep_defensive_reset")
        self.assertEqual(alignment["highlight_rounds"], [1, 3, 5])
        self.assertEqual(alignment["rating"], "strong")


if __name__ == "__main__":
    unittest.main()
