import copy
import unittest

import pandas as pd

from Preprocessing.rally_split import build_rally_split, validate_rally_split


def add_rallies(rows, count, players, match_ids, start_rally=1, lengths=(3, 7, 13)):
    for offset in range(count):
        rally_number = start_rally + offset
        rally_length = lengths[offset % len(lengths)]
        match_id = match_ids[offset % len(match_ids)]
        for ball_round in range(1, rally_length + 1):
            rows.append({
                "match_id": match_id,
                "set": 1,
                "rally": rally_number,
                "ball_round": ball_round,
                "player": players[(ball_round - 1) % len(players)],
                "type": "發短球" if ball_round == 1 else "長球",
                "landing_area": 28,
                "landing_x": 0.5,
                "landing_y": 0.5,
                "ball_distance": 1.0,
                "hit_x": 0.0,
                "hit_y": 0.0,
                "moving_x": 1.0,
                "moving_y": 0.0,
                "opponent_moving_distance": 2.0,
            })


def make_rally_rows(count=63):
    rows = []
    add_rallies(rows, count, ("Player A", "Player B"), (1, 2, 3))
    return pd.DataFrame(rows)


def record_keys(records):
    return {
        (record["match_id"], record["set"], record["rally"])
        for record in records
    }


class RallySplitTest(unittest.TestCase):
    def test_split_is_deterministic_atomic_and_length_stratified(self):
        data = make_rally_rows()
        first = build_rally_split(data, seed=42)
        second = build_rally_split(data, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(first["optimization"]["status"], "optimal")
        self.assertEqual(first["eligible_players"], ["Player A", "Player B"])
        self.assertEqual(
            first["eligibility_settings"]["min_eligible_usable_rallies"],
            63,
        )

        for player in ("Player A", "Player B"):
            train_keys = record_keys(first["train"][player])
            test_keys = record_keys(first["test"][player])
            self.assertFalse(train_keys & test_keys)
            self.assertEqual(len(train_keys | test_keys), 63)
            self.assertGreaterEqual(first["player_stats"][player]["usable_train"], 50)

            stats = first["player_stats"][player]
            self.assertLessEqual(abs(stats["usable_test"] - round(0.2 * 63)), 1)
            for label in ("short", "medium", "long"):
                self.assertLessEqual(
                    abs(stats["length_actuals"][label] - stats["length_targets"][label]),
                    1,
                )

        self.assertEqual(
            record_keys(first["train"]["Player A"]),
            record_keys(first["train"]["Player B"]),
        )
        self.assertEqual(
            record_keys(first["test"]["Player A"]),
            record_keys(first["test"]["Player B"]),
        )
        for values in first["physical_length_stats"].values():
            self.assertLessEqual(abs(values["actual"] - values["target"]), 1)

    def test_ineligible_opponents_force_shared_rallies_to_train(self):
        rows = []
        add_rallies(rows, 20, ("Player A", "Player B"), (1, 2), start_rally=1)
        add_rallies(rows, 3, ("Player A", "Sparse C"), (3, 4), start_rally=100)
        add_rallies(rows, 20, ("Player A", "One Match D"), (5,), start_rally=200)
        split = build_rally_split(
            pd.DataFrame(rows),
            seed=7,
            min_train_rallies=10,
            min_matches=2,
        )

        self.assertEqual(split["eligible_players"], ["Player A", "Player B"])
        self.assertIn("insufficient_usable_rallies", split["train_only_players"]["Sparse C"]["reasons"])
        self.assertIn("insufficient_usable_matches", split["train_only_players"]["One Match D"]["reasons"])
        self.assertEqual(split["test"]["Sparse C"], [])
        self.assertEqual(split["test"]["One Match D"], [])

        forced_keys = record_keys(split["train"]["Sparse C"]) | record_keys(
            split["train"]["One Match D"]
        )
        self.assertTrue(forced_keys.issubset(record_keys(split["train"]["Player A"])))
        self.assertFalse(forced_keys & record_keys(split["test"]["Player A"]))
        self.assertGreaterEqual(split["player_stats"]["Player A"]["usable_train"], 10)
        self.assertGreaterEqual(split["player_stats"]["Player B"]["usable_train"], 10)

    def test_usability_comes_from_nonserve_feature_count(self):
        rows = []
        add_rallies(
            rows,
            20,
            ("Serve Only", "Eligible Receiver"),
            (1, 2),
            lengths=(2,),
        )
        split = build_rally_split(
            pd.DataFrame(rows),
            min_train_rallies=10,
            min_matches=2,
        )

        self.assertEqual(split["player_stats"]["Serve Only"]["usable_total"], 0)
        self.assertFalse(split["player_stats"]["Serve Only"]["eligible"])
        self.assertTrue(split["player_stats"]["Eligible Receiver"]["eligible"])
        self.assertEqual(split["test"]["Serve Only"], [])
        self.assertEqual(split["test"]["Eligible Receiver"], [])
        self.assertEqual(split["player_stats"]["Eligible Receiver"]["usable_train"], 20)

    def test_validation_rejects_train_only_test_entries(self):
        split = build_rally_split(make_rally_rows(), seed=42)
        broken = copy.deepcopy(split)
        broken["train_only_players"]["Player A"] = {"reasons": ["test"]}
        with self.assertRaisesRegex(ValueError, "Train-only player"):
            validate_rally_split(broken)


if __name__ == "__main__":
    unittest.main()
