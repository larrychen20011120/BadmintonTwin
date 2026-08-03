import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from Preprocessing.rally_feature import (
    _landing_opponent_distance,
    _stroke_movement_distance,
    extract_rally_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestRallyFeature(unittest.TestCase):
    def test_rally_pipeline_does_not_import_match_feature(self):
        for relative_path in (
            "Preprocessing/rally_feature.py",
            "Scripts/plot_rally_profiles.py",
        ):
            tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            self.assertNotIn("Preprocessing.match_feature", imported_modules)

    def test_distance_helpers_preserve_source_precedence(self):
        rows = pd.DataFrame({
            "ball_distance": [3.0],
            "hit_x": [0.0],
            "hit_y": [0.0],
            "player_location_x": [100.0],
            "player_location_y": [100.0],
            "moving_x": [4.0],
            "moving_y": [0.0],
            "moving_distance": [200.0],
            "landing_x": [0.0],
            "landing_y": [0.0],
            "opponent_location_x": [100.0],
            "opponent_location_y": [100.0],
            "opponent_moving_distance": [7.0],
        })

        self.assertEqual(_stroke_movement_distance(rows).iloc[0], 7.0)
        self.assertEqual(_landing_opponent_distance(rows).iloc[0], 7.0)

    def test_extract_rally_features_keeps_existing_contract(self):
        rows = pd.DataFrame([
            {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "ball_round": 1,
                "player": " Player A ",
                "type": "發短球",
                "landing_area": 17,
                "ball_distance": 1.0,
                "hit_x": 0.0,
                "hit_y": 0.0,
                "moving_x": 1.0,
                "moving_y": 0.0,
                "opponent_moving_distance": 2.0,
            },
            {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "ball_round": 2,
                "player": "Player B",
                "type": "長球",
                "landing_area": 28,
                "ball_distance": 1.0,
                "hit_x": 0.0,
                "hit_y": 0.0,
                "moving_x": 1.0,
                "moving_y": 0.0,
                "opponent_moving_distance": 2.0,
            },
            {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "ball_round": 3,
                "player": "Player A",
                "type": "殺球",
                "landing_area": 25,
                "ball_distance": 1.0,
                "hit_x": 0.0,
                "hit_y": 0.0,
                "moving_x": 1.0,
                "moving_y": 0.0,
                "opponent_moving_distance": 2.0,
            },
            {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "ball_round": 4,
                "player": "Player B",
                "type": "接不到",
                "landing_area": np.nan,
                "ball_distance": np.nan,
                "hit_x": np.nan,
                "hit_y": np.nan,
                "moving_x": np.nan,
                "moving_y": np.nan,
                "opponent_moving_distance": np.nan,
            },
        ])

        features = extract_rally_features(rows)
        player_a = features.loc[features["player"] == "Player A"].iloc[0]

        self.assertEqual(player_a["rally_length"], 4)
        self.assertEqual(player_a["valid_action_count"], 2)
        self.assertEqual(player_a["target_nonserve_shot_count"], 1)
        self.assertEqual(player_a["attack_shot_count"], 1)
        self.assertEqual(player_a["valid_landing_count"], 1)
        self.assertEqual(player_a["side_landing_count"], 1)
        self.assertEqual(player_a["frontcourt_landing_count"], 1)
        self.assertEqual(player_a["backcourt_landing_count"], 0)
        self.assertEqual(player_a["average_movement"], 2.0)
        self.assertEqual(player_a["landing_opponent_distance"], 2.0)

    def test_side_frontcourt_and_backcourt_counts_overlap(self):
        rows = pd.DataFrame([
            {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "ball_round": ball_round,
                "player": "Player A",
                "type": "長球",
                "landing_area": area,
            }
            for ball_round, area in enumerate([25, 27, 28, 1, 18], start=1)
        ])

        feature = extract_rally_features(rows).iloc[0]

        self.assertEqual(feature["valid_landing_count"], 5)
        self.assertEqual(feature["side_landing_count"], 4)
        self.assertEqual(feature["frontcourt_landing_count"], 2)
        self.assertEqual(feature["backcourt_landing_count"], 2)


if __name__ == "__main__":
    unittest.main()
