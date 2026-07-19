import json
import unittest

import pandas as pd

from Agent.base_agent import BaseAgent
from Scripts.single_player_eval.evaluate_trajectories import extract_sequences
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
)
from Utils.badminton_config import LANG_AREA_MAPPING
from Utils.helper import coord2area
from Utils.llm import LLM


class PromptCoordinateTest(unittest.TestCase):
    def test_integer_grid_round_trip(self):
        for x in range(356):
            normalized_x, _ = BaseAgent.prompt_to_normalized_coordinate(x, 0)
            prompt_x, _ = BaseAgent.normalized_to_prompt_coordinate(normalized_x, -1.0)
            self.assertEqual(prompt_x, x)

        for y in range(481):
            _, normalized_y = BaseAgent.prompt_to_normalized_coordinate(0, y)
            _, prompt_y = BaseAgent.normalized_to_prompt_coordinate(-1.0, normalized_y)
            self.assertEqual(prompt_y, y)

    def test_prompt_area_partitions_entire_grid(self):
        expected_areas = {
            "Deep Left Corner", "Deep Right Corner",
            "Rear Left Side", "Rear Center", "Rear Right Side",
            "Rear Left Side Line", "Rear Right Side Line",
            "Mid Left Side", "Mid Center", "Mid Right Side",
            "Mid Left Side Line", "Mid Right Side Line",
            "Front Left Side", "Front Center", "Front Right Side",
            "Front Left Side Line", "Front Right Side Line",
        }
        actual_areas = {
            BaseAgent.prompt_area_name(x, y)
            for x in range(356)
            for y in range(481)
        }
        self.assertEqual(actual_areas, expected_areas)

    def test_prompt_area_matches_existing_coord2area(self):
        for x in range(356):
            for y in range(481):
                normalized_x, normalized_y = BaseAgent.prompt_to_normalized_coordinate(x, y)
                existing_area = LANG_AREA_MAPPING[coord2area(normalized_x, normalized_y)]
                self.assertEqual(BaseAgent.prompt_area_name(x, y), existing_area)

    def test_offline_prompt_action_uses_integer_coordinates_and_areas(self):
        action = BaseAgent.build_prompt_action(1, 0.2, 0.3, -0.15, -0.25)
        self.assertEqual(action, {
            "shot_type": "Serve short",
            "landing_area": "Mid Center",
            "landing_x": 213,
            "landing_y": 312,
            "moving_area": "Rear Center",
            "moving_x": 150,
            "moving_y": 180,
        })

    def test_missed_prompt_action_is_null(self):
        action = BaseAgent.build_prompt_action(11, None, None, None, None)
        self.assertEqual(action["shot_type"], "Missed shot")
        self.assertTrue(all(
            action[key] is None
            for key in BaseAgent.PROMPT_ACTION_KEYS - {"shot_type"}
        ))
        self.assertIn('"landing_x": null', BaseAgent.format_prompt_action(action))

    def test_online_parser_returns_normalized_coordinates(self):
        response = {
            "rationale": "Imitate the retrieved serve.",
            "action": BaseAgent.build_prompt_action(1, 0.2, 0.3, -0.15, -0.25),
        }
        parsed, reasoning = BaseAgent.parse_prompt_action(response)
        self.assertEqual(reasoning, "Imitate the retrieved serve.")
        self.assertEqual(parsed["shot_type"], 1)
        self.assertAlmostEqual(parsed["landing_x"], 213 / 177.5 - 1)
        self.assertAlmostEqual(parsed["landing_y"], 312 / 240 - 1)

    def test_online_parser_rejects_decimal_and_unknown_area(self):
        action = BaseAgent.build_prompt_action(2, 0.2, 0.3, -0.15, -0.25)
        action["landing_x"] = 213.0
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)

        action = BaseAgent.build_prompt_action(2, 0.2, 0.3, -0.15, -0.25)
        action["landing_area"] = "Unknown Area"
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)

    def test_online_parser_accepts_area_coordinate_mismatch(self):
        action = {
            "shot_type": "Net Shot",
            "landing_area": "Front Center",
            "landing_x": 186,
            "landing_y": 440,
            "moving_area": "Mid Left Side Line",
            "moving_x": 50,
            "moving_y": 420,
        }
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["moving_area"], "Mid Left Side Line")

    def test_online_parser_accepts_grid_endpoints_only(self):
        action = BaseAgent.build_prompt_action(2, 1.0, 1.0, 1.0, 1.0)
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNotNone(parsed)

        action["landing_x"] = 356
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)

        action["landing_x"] = 355
        action["landing_y"] = 481
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)


class MissedTrajectoryTest(unittest.TestCase):
    def test_trajectory_builders_store_null_for_missed_shots(self):
        row = pd.Series({
            "player": "Player A", "ball_round": 2, "player_type": 11,
            "landing_x": 0.4, "landing_y": 0.5,
            "moving_x": -0.2, "moving_y": -0.3,
            "player_location_x": 0.0, "player_location_y": 0.0,
            "hit_x": 0.1, "hit_y": 0.2,
            "opponent_location_x": -0.1, "opponent_location_y": -0.2,
            "player_score": 1, "opponent_score": 1,
        })
        gt = build_gt_shot_from_row(row)
        pred = build_pred_shot("Player A", 11, [0.4, 0.5], [-0.2, -0.3], 2)
        for step in (gt, pred):
            self.assertIsNone(step["landing_x"])
            self.assertIsNone(step["landing_y"])
            self.assertIsNone(step["moving_x"])
            self.assertIsNone(step["moving_y"])

    def test_position_metrics_skip_missed_coordinates(self):
        normal = {
            "player": "Player A", "ball_round": 1, "shot_type": 2,
            "landing_x": 0.2, "landing_y": 0.3,
            "moving_x": -0.1, "moving_y": -0.2,
        }
        missed = {
            "player": "Player A", "ball_round": 2, "shot_type": 11,
            "landing_x": None, "landing_y": None,
            "moving_x": None, "moving_y": None,
        }
        trajectory = {
            "target_player": "Player A",
            "gt_sequence": [normal, missed],
            "pred_sequence": [normal, missed],
        }
        extracted = extract_sequences([trajectory])
        self.assertEqual(extracted["all_land_gt"].shape, (1, 2))
        self.assertEqual(extracted["all_move_pred"].shape, (1, 2))
        self.assertEqual(extracted["all_shot_gt"].tolist(), [2, 11])


class LLMActionValidationTest(unittest.TestCase):
    def test_invalid_action_is_retried_before_return(self):
        invalid_action = BaseAgent.build_prompt_action(2, 0.2, 0.3, -0.1, -0.2)
        invalid_action["landing_x"] = float(invalid_action["landing_x"])
        valid_action = BaseAgent.build_prompt_action(2, 0.2, 0.3, -0.1, -0.2)
        responses = iter([
            json.dumps({"action": invalid_action}),
            json.dumps({"action": valid_action}),
        ])

        llm = LLM.__new__(LLM)
        llm.get_response = lambda **_: (next(responses), {})
        parsed, _, _ = llm.get_verified_response(
            max_trials=2,
            validator=BaseAgent.validate_prompt_action_response,
        )
        self.assertEqual(parsed["action"], valid_action)


if __name__ == "__main__":
    unittest.main()
