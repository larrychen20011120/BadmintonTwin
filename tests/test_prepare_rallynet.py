import json
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from prepare_rallynet import (
    compute_rallynet_dataset_fingerprint,
    prepare_rallynet_data,
)
from Scripts.prepare_all_data import run_method, validate_config
from Utils.helper import action_col, state_col


class PrepareRallyNetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.csv_path = os.path.join(self.temp_dir.name, "merged.csv")
        self.split_path = os.path.join(self.temp_dir.name, "split.json")
        self.player_map_path = os.path.join(self.temp_dir.name, "players.json")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        self.train_key = {"match_id": 7, "set": 1, "rally": 1}
        self.test_key = {"match_id": 7, "set": 1, "rally": 2}
        self.split = {
            "schema_version": 2,
            "train": {"A": [self.train_key], "B": [self.train_key]},
            "test": {"A": [self.test_key], "B": [self.test_key]},
        }
        self.player_map = {"A": 0, "B": 1}
        self._write_json(self.split_path, self.split)
        self._write_json(self.player_map_path, self.player_map)
        self.rows = self._make_rows()
        self.rows.to_csv(self.csv_path, index=False)

    @staticmethod
    def _write_json(path, value):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle)

    @staticmethod
    def _make_rows():
        rows = []
        specifications = {
            1: [("A", 1), ("B", 4), ("A", 11)],
            2: [("A", 10), ("B", 2)],
        }
        for rally_number, shots in specifications.items():
            for ball_round, (player, shot_type) in enumerate(shots, start=1):
                row = {column: 0.0 for column in state_col + action_col}
                row.update({
                    "match_id": 7,
                    "set": 1,
                    "rally": rally_number,
                    "ball_round": ball_round,
                    "player": player,
                    "player_type": shot_type,
                    "landing_x": 0.25,
                    "landing_y": 0.5,
                    "moving_x": -0.25,
                    "moving_y": -0.5,
                })
                if shot_type == 11:
                    row.update({
                        "landing_x": np.nan,
                        "landing_y": np.nan,
                        "moving_x": np.nan,
                        "moving_y": np.nan,
                    })
                rows.append(row)
        return pd.DataFrame(rows)

    def test_prepares_exact_physical_train_rallies_with_provenance(self):
        metadata = prepare_rallynet_data(
            self.csv_path,
            self.split_path,
            self.player_map_path,
            self.output_dir,
            experiment="rally",
        )

        with open(os.path.join(self.output_dir, "rallies.pkl"), "rb") as handle:
            payload = pickle.load(handle)
        rally = payload["rallies"][0]
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual((rally["match_id"], rally["set"], rally["rally"]), (7, 1, 1))
        self.assertEqual(rally["serve_mask"].tolist(), [True, False, False])
        self.assertEqual(rally["error_targets"].tolist(), [False, False, True])
        self.assertEqual(rally["coordinate_mask"].tolist(), [True, True, False])
        np.testing.assert_array_equal(rally["actions"][-1, [0, 1, 3, 4]], 0.0)
        fingerprint = compute_rallynet_dataset_fingerprint(
            payload["rallies"], self.player_map
        )
        self.assertEqual(payload["dataset_fingerprint"], fingerprint)
        self.assertEqual(metadata["dataset_fingerprint"], fingerprint)
        self.assertEqual(metadata["train_rally_keys"], [self.train_key])
        self.assertEqual(metadata["excluded_test_rally_count"], 1)
        self.assertEqual(metadata["counts"]["rallies"], 1)
        self.assertEqual(metadata["counts"]["skipped_train_rallies"], 0)
        self.assertTrue(metadata["split_sha256"])
        self.assertEqual(metadata["provenance"]["player_map"], self.player_map)

    def test_rejects_physical_train_test_overlap(self):
        self.split["test"]["A"] = [self.train_key]
        self.split["test"]["B"] = [self.train_key]
        self._write_json(self.split_path, self.split)

        with self.assertRaisesRegex(ValueError, "both train and test"):
            prepare_rallynet_data(
                self.csv_path,
                self.split_path,
                self.player_map_path,
                self.output_dir,
            )

    def test_reports_key_for_invalid_row_inside_requested_rally(self):
        self.rows.loc[
            (self.rows["rally"] == 1) & (self.rows["ball_round"] == 2),
            "player_type",
        ] = 1
        self.rows.to_csv(self.csv_path, index=False)

        with self.assertRaisesRegex(ValueError, "7/1/1.*non-serve"):
            prepare_rallynet_data(
                self.csv_path,
                self.split_path,
                self.player_map_path,
                self.output_dir,
            )

    def test_orchestrator_dispatches_without_unrelated_split_or_profile_paths(self):
        config = {
            "paths": {
                "merged_csv": self.csv_path,
                "player_map": self.player_map_path,
                "rally_split": self.split_path,
                "rallynet": {"data": self.output_dir},
            }
        }
        shifted_path = validate_config(
            config, ["norm", "shifted"], ["rallynet"]
        )
        self.assertIsNone(shifted_path)

        destination = run_method(config, "rally", "rallynet", {})
        self.assertEqual(destination, self.output_dir)
        self.assertTrue(os.path.isfile(os.path.join(destination, "rallies.pkl")))
        self.assertTrue(os.path.isfile(os.path.join(destination, "metadata.json")))


if __name__ == "__main__":
    unittest.main()
