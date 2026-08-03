import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd
import yaml

from Agent.twin import TwinDatabase
from Scripts.preview_twin_memory import build_parser, build_preview, main, select_rally
from Utils.helper import action_col, state_col


def make_rally():
    rows = []
    for ball_round, player, player_type in [
        (1, "Player A", 1),
        (2, "Player B", 2),
        (3, "Player A", 4),
    ]:
        row = {column: 0.0 for column in state_col + action_col}
        row.update({
            "match_id": 7,
            "set": 2,
            "rally": 3,
            "ball_round": ball_round,
            "player": player,
            "opponent_type": 2,
            "player_type": player_type,
            "landing_x": 0.1,
            "landing_y": 0.2,
            "moving_x": 0.0,
            "moving_y": -0.1,
        })
        rows.append(row)
    return pd.DataFrame(rows)


class PreviewTwinMemoryTest(unittest.TestCase):
    def test_parser_supports_preview_controls(self):
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--no-llm", option_strings)
        self.assertIn("--show-prompt", option_strings)
        self.assertIn("--show-annotation", option_strings)
        self.assertIn("--output", option_strings)

    def test_select_rally_requires_set_when_ambiguous(self):
        dataframe = pd.concat([
            make_rally(),
            make_rally().assign(set=1),
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "provide --set"):
            select_rally(dataframe, 7, 3)

    def test_memory_annotation_accepts_only_range_and_phase(self):
        rally = make_rally()
        segments = {
            aspect: [{"range": [1, 3], "phase": "Sustained control phase"}]
            for aspect in (
                "my_tactical_pattern",
                "my_tempo_movement_dynamics",
                "opponent_tactical_pattern",
                "opponent_tempo_movement_dynamics",
            )
        }
        annotation = {"aspect_segments": segments}
        self.assertTrue(TwinDatabase._validate_rally_annotation(annotation, rally))

        segments["my_tactical_pattern"][0]["evidence_strokes"] = [1, 3]
        self.assertFalse(TwinDatabase._validate_rally_annotation(annotation, rally))

    def test_fallback_preview_uses_target_player_strokes(self):
        rally, set_number = select_rally(make_rally(), 7, 3, 2)
        preview = build_preview(
            rally,
            "Player A",
            1,
            "Opening rally of the match",
            use_llm=False,
        )

        self.assertEqual(set_number, 2)
        self.assertEqual(len(preview["memories"]), 2)
        self.assertEqual(preview["memories"][0]["memory_id"], "7:2:3:1")
        memory = preview["memories"][0]
        self.assertIn("My tactical pattern:", memory["semantic_text"])
        self.assertEqual(
            memory["my_tactical_pattern"],
            "Full-rally fallback phase",
        )
        self.assertNotIn("Phase:", memory["my_tactical_pattern"])
        self.assertNotIn("Narrative:", memory["my_tactical_pattern"])
        segment = preview["annotation"]["aspect_segments"]["my_tactical_pattern"][0]
        self.assertEqual(segment, {
            "range": [1, 3],
            "phase": "Full-rally fallback phase",
        })
        self.assertIn("Annotate one completed badminton rally", preview["llm_user_prompt"])

    def test_main_can_write_json_without_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "merged.csv")
            map_path = os.path.join(directory, "players.json")
            config_path = os.path.join(directory, "config.yaml")
            output_path = os.path.join(directory, "preview.json")
            make_rally().to_csv(csv_path, index=False)
            with open(map_path, "w", encoding="utf-8") as file:
                json.dump({"Player A": 1, "Player B": 2}, file)
            with open(config_path, "w", encoding="utf-8") as file:
                yaml.safe_dump({
                    "paths": {
                        "merged_csv": csv_path,
                        "player_map": map_path,
                    }
                }, file)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main([
                    "--config", config_path,
                    "--player", "Player A",
                    "--match-id", "7",
                    "--set", "2",
                    "--rally", "3",
                    "--no-llm",
                    "--show-annotation",
                    "--output", output_path,
                ])

            self.assertEqual(exit_code, 0, stderr.getvalue())
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            self.assertEqual(payload["selection"]["annotation_mode"], "fallback")
            self.assertEqual(len(payload["memories"]), 2)
            self.assertIn("Rally annotation", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
