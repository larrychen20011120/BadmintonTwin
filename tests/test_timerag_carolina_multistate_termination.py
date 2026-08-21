import copy
import unittest

from Scripts.single_player_eval.timerag_carolina_multistate_termination import (
    _score_context,
    _terminal_outcome,
    validate_trajectories,
)


def _trajectory(index, initial_state, shot_type=11, reason="natural_terminate"):
    return {
        "algorithm": "timerag",
        "action_architecture": "two_stage_v1",
        "rollout_index": index,
        "initial_state": copy.deepcopy(initial_state),
        "termination_reason": reason,
        "pred_sequence": [{"shot_type": shot_type}],
    }


class TimeRAGCarolinaTerminationTest(unittest.TestCase):
    def test_validates_five_independent_two_stage_rollouts(self):
        initial = {"player": "Carolina MARIN", "player_score": 3, "opponent_score": 2}
        spec = {"source_data_point": {"initial_state": initial}}
        trajectories = [_trajectory(index, initial) for index in range(1, 6)]
        validate_trajectories(trajectories, spec)

        trajectories[1]["action_architecture"] = "single_stage_v1"
        with self.assertRaisesRegex(ValueError, "not two-stage"):
            validate_trajectories(trajectories, spec)

    def test_score_context_handles_opponent_server(self):
        spec = {
            "target_player": "Carolina MARIN",
            "source_data_point": {"initial_state": {
                "player": "Akane YAMAGUCHI",
                "player_score": 8,
                "opponent_score": 5,
            }},
        }
        self.assertEqual(_score_context(spec), ("Akane YAMAGUCHI", 5, 8))

    def test_terminal_outcomes_distinguish_miss_out_and_cap(self):
        initial = {"player": "Carolina MARIN"}
        self.assertEqual(_terminal_outcome(_trajectory(1, initial)), "Missed shot")
        self.assertEqual(_terminal_outcome(_trajectory(1, initial, shot_type=2)), "Out of bounds")
        self.assertEqual(
            _terminal_outcome(_trajectory(1, initial, shot_type=2, reason="max_shots_reached")),
            "Maximum shot limit reached",
        )


if __name__ == "__main__":
    unittest.main()
