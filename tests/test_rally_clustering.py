from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from jinja2 import Environment, StrictUndefined

from Preprocessing.rally_clustering import (
    _valid_instruction_response,
    add_instruction_variants,
    candidate_ks,
    centroid_feature_payload,
    fit_player_profile,
    generate_cluster_instruction,
    generate_instruction_variant,
    select_elbow_k,
)
from Preprocessing.rally_feature import CLUSTER_FEATURE_COLUMNS, extract_rally_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RALLY_PROFILE_PROMPT = PROJECT_ROOT / "Prompts" / "Preprocessing" / "rally_profile_user.jinja"
INSTRUCTION_VARIANT_PROMPT = (
    PROJECT_ROOT / "Prompts" / "Preprocessing" / "rally_instruction_variant_user.jinja"
)


def make_feature_row(rally_number, cluster_id):
    common = {
        "player": "Player A",
        "match_id": 1,
        "set": 1,
        "rally": rally_number,
        "valid_action_count": 6,
        "target_nonserve_shot_count": 6,
        "valid_landing_count": 6,
        "front_back_transition_count": 5,
        "left_right_transition_count": 5,
    }
    variants = (
        {
            "rally_length": 4 + rally_number % 2,
            "attack_shot_count": 5,
            "defense_shot_count": 0,
            "side_landing_count": 5,
            "frontcourt_landing_count": 5,
            "backcourt_landing_count": 0,
            "left_landing_count": 0,
            "right_landing_count": 6,
            "front_back_switch_count": 0,
            "left_right_switch_count": 0,
            "average_movement": 1.0 + 0.02 * (rally_number % 3),
            "landing_opponent_distance": 4.0,
        },
        {
            "rally_length": 14 + rally_number % 2,
            "attack_shot_count": 0,
            "defense_shot_count": 5,
            "side_landing_count": 1,
            "frontcourt_landing_count": 0,
            "backcourt_landing_count": 5,
            "left_landing_count": 6,
            "right_landing_count": 0,
            "front_back_switch_count": 1,
            "left_right_switch_count": 0,
            "average_movement": 3.0 + 0.02 * (rally_number % 3),
            "landing_opponent_distance": 6.0,
        },
        {
            "rally_length": 8 + rally_number % 2,
            "attack_shot_count": 1,
            "defense_shot_count": 1,
            "side_landing_count": 2,
            "frontcourt_landing_count": 3,
            "backcourt_landing_count": 3,
            "left_landing_count": 3,
            "right_landing_count": 3,
            "front_back_switch_count": 5,
            "left_right_switch_count": 5,
            "average_movement": 2.0 + 0.02 * (rally_number % 3),
            "landing_opponent_distance": 8.0,
        },
    )
    return {**common, **variants[cluster_id]}


def make_cluster_data():
    rows = []
    train_records = []
    test_records = []
    rally_number = 1
    for cluster_id in range(3):
        for _ in range(15):
            rows.append(make_feature_row(rally_number, cluster_id))
            train_records.append({"match_id": 1, "set": 1, "rally": rally_number})
            rally_number += 1
    for cluster_id in range(3):
        for _ in range(3):
            rows.append(make_feature_row(rally_number, cluster_id))
            test_records.append({"match_id": 1, "set": 1, "rally": rally_number})
            rally_number += 1
    return pd.DataFrame(rows), train_records, test_records


SETTINGS = {
    "min_target_nonserve_shots": 1,
    "smoothing_alpha": 2.0,
    "clip_quantiles": [0.01, 0.99],
    "min_train_rallies": 30,
    "min_cluster_size": 5,
    "n_init": 20,
    "random_state": 42,
    "elbow_window_radius": 0,
}


class RallyClusteringTest(unittest.TestCase):
    def test_candidate_k_boundary(self):
        self.assertEqual(candidate_ks(7, 7), [])
        self.assertEqual(candidate_ks(8, 8), [2])
        self.assertEqual(candidate_ks(18, 18), [2, 3])
        self.assertEqual(candidate_ks(18, 18, min_cluster_size=10), [2, 3])
        self.assertEqual(candidate_ks(18, 2), [2])

    def test_elbow_selection_uses_normalized_chord_distance(self):
        self.assertEqual(select_elbow_k({1: 100, 2: 70, 3: 40, 4: 35}), 3)
        self.assertEqual(select_elbow_k({1: 10, 2: 5}), 2)
        self.assertEqual(select_elbow_k({1: 10, 2: 10, 3: 10}), 2)
        self.assertEqual(select_elbow_k({1: 10, 2: 5, 3: 0}), 2)

    def test_train_only_clustering_and_test_assignment(self):
        features, train_records, test_records = make_cluster_data()
        profile = fit_player_profile(
            "Player A", features, train_records, test_records, SETTINGS
        )
        self.assertEqual(profile["status"], "clustered")
        self.assertEqual(profile["selected_k"], 2)
        self.assertEqual(profile["elbow_k"], 2)
        self.assertEqual(profile["k_candidates"], [2, 3, 4])
        self.assertEqual(set(profile["inertia_by_k"]), {"1", "2", "3", "4"})
        self.assertEqual(
            profile["rejected_ks"]["3"]["reasons"],
            ["outside_elbow_window"],
        )
        self.assertEqual(profile["rejected_ks"]["3"]["cluster_sizes"], [15, 15, 15])
        self.assertEqual(profile["elbow_window"], {"radius": 0, "lower": 2, "upper": 2})
        self.assertEqual(
            profile["selection_method"],
            "elbow_window_then_maximum_silhouette",
        )
        self.assertNotIn("stability", profile["silhouette_scores"]["2"])
        self.assertNotIn("evaluation_eligible", profile)
        self.assertEqual(sorted(cluster["train_size"] for cluster in profile["clusters"]), [15, 30])
        for cluster in profile["clusters"]:
            self.assertEqual(cluster["train_match_count"], 1)
            attack_summary = cluster["feature_summary_z"]["attack_shot_ratio"]
            self.assertEqual(
                set(attack_summary),
                {"mean", "median", "q25", "q75", "iqr", "direction_support"},
            )
        self.assertEqual(len(profile["train_assignments"]), 45)
        self.assertEqual(len(profile["test_assignments"]), 9)
        self.assertEqual(
            {item["cluster_id"] for item in profile["test_assignments"]},
            {0, 1},
        )

        changed_test = features.copy()
        test_keys = {record["rally"] for record in test_records}
        mask = changed_test["rally"].isin(test_keys)
        changed_test.loc[mask, "rally_length"] = 1000
        changed_test.loc[mask, "average_movement"] = 1000
        changed_profile = fit_player_profile(
            "Player A", changed_test, train_records, test_records, SETTINGS
        )
        self.assertEqual(
            [cluster["center_z"] for cluster in profile["clusters"]],
            [cluster["center_z"] for cluster in changed_profile["clusters"]],
        )
        self.assertEqual(
            [cluster["feature_summary_z"] for cluster in profile["clusters"]],
            [cluster["feature_summary_z"] for cluster in changed_profile["clusters"]],
        )
        for k, inertia in profile["inertia_by_k"].items():
            self.assertAlmostEqual(inertia, changed_profile["inertia_by_k"][k], places=10)
        self.assertEqual(profile["silhouette_scores"], changed_profile["silhouette_scores"])
        self.assertEqual(profile["selected_k"], changed_profile["selected_k"])

    def test_elbow_window_uses_strict_maximum_silhouette(self):
        features, train_records, test_records = make_cluster_data()
        settings = {
            **SETTINGS,
            "n_init": 5,
            "min_cluster_size": 1,
            "elbow_window_radius": 1,
        }

        def fit_with_scores(scores):
            def fake_silhouette(_, labels):
                return scores[len(set(labels))]

            with (
                patch(
                    "Preprocessing.rally_clustering.select_elbow_k",
                    return_value=3,
                ),
                patch(
                    "Preprocessing.rally_clustering.silhouette_score",
                    side_effect=fake_silhouette,
                ),
            ):
                return fit_player_profile(
                    "Player A", features, train_records, test_records, settings
                )

        strict_profile = fit_with_scores({2: 0.7, 3: 0.681, 4: 0.69})
        self.assertEqual(strict_profile["selected_k"], 2)
        upper_window_profile = fit_with_scores({2: 0.6, 3: 0.65, 4: 0.7})
        self.assertEqual(upper_window_profile["selected_k"], 4)
        tied_profile = fit_with_scores({2: 0.7, 3: 0.7, 4: 0.6})
        self.assertEqual(tied_profile["selected_k"], 2)

    def test_default_elbow_window_radius_is_one(self):
        features, train_records, test_records = make_cluster_data()
        settings = {
            key: value
            for key, value in SETTINGS.items()
            if key != "elbow_window_radius"
        }
        with patch(
            "Preprocessing.rally_clustering.select_elbow_k",
            return_value=3,
        ):
            profile = fit_player_profile(
                "Player A", features, train_records, test_records, settings
            )

        self.assertEqual(profile["elbow_window"], {"radius": 1, "lower": 2, "upper": 4})

    def test_rally_feature_extraction_uses_physical_rally_length(self):
        rows = []
        shot_types = ["發短球", "長球", "殺球", "網前球"]
        players = ["Player A", "Player B", "Player A", "Player B"]
        areas = [17, 28, 25, 32]
        for index in range(4):
            rows.append({
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "ball_round": index + 1,
                "player": players[index],
                "type": shot_types[index],
                "landing_area": areas[index],
                "landing_x": 0.5,
                "landing_y": 0.5,
                "ball_distance": 1.0,
                "hit_x": 0.0,
                "hit_y": 0.0,
                "moving_x": 1.0,
                "moving_y": 0.0,
                "opponent_moving_distance": 2.0,
            })
        features = extract_rally_features(pd.DataFrame(rows))
        player_a = features[features["player"] == "Player A"].iloc[0]
        self.assertEqual(player_a["rally_length"], 4)
        self.assertEqual(player_a["target_nonserve_shot_count"], 1)
        self.assertEqual(player_a["attack_shot_count"], 1)


class FakeLLM:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    def get_verified_response(
        self,
        schema=None,
        system_prompt="",
        user_prompt="",
        validator=None,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        data = {
            "label": "Right-side attack",
            "description": "An attacking mode that favors right-side targets.",
            "instruction": "Use more attacking shots and favor right-side targets.",
        }
        if not validator(data):
            raise AssertionError("Test response should satisfy the validator")
        return data, "raw", {}


class RallyProfilePromptTest(unittest.TestCase):
    def test_centroid_payload_describes_all_features_in_order(self):
        center_z = {
            feature: index / 10
            for index, feature in enumerate(CLUSTER_FEATURE_COLUMNS)
        }

        payload = centroid_feature_payload(center_z)

        self.assertEqual(
            [item["feature"] for item in payload],
            CLUSTER_FEATURE_COLUMNS,
        )
        self.assertEqual(len(payload), 11)
        self.assertEqual(
            set(payload[0]),
            {"feature", "label", "description", "z_score"},
        )
        payload_text = " ".join(
            item["description"] for item in payload
        ).lower()
        for exact_detail in (
            "smash",
            "drop",
            "push",
            "clear",
            "lob",
            "drive",
            "net shot",
            "serve",
            "deep",
            "corner",
            "baseline",
            "rear",
        ):
            self.assertNotIn(exact_detail, payload_text)
        self.assertIn("attacking shots", payload[0]["description"])
        self.assertIn("defensive or rally-resetting shots", payload[1]["description"])
        backcourt = payload[CLUSTER_FEATURE_COLUMNS.index("backcourt_ratio")]
        self.assertIn("back court", backcourt["description"])
        signed_bias = payload[CLUSTER_FEATURE_COLUMNS.index("signed_side_bias")]
        self.assertIn("positive z-score", signed_bias["description"])
        self.assertIn("negative value", signed_bias["description"])

    def test_instruction_is_generated_from_complete_centroid(self):
        template = Environment(undefined=StrictUndefined).from_string(
            RALLY_PROFILE_PROMPT.read_text(encoding="utf-8")
        )
        cluster = {
            "cluster_id": 0,
            "train_size": 20,
            "center_z": {
                feature: index / 10 - 0.5
                for index, feature in enumerate(CLUSTER_FEATURE_COLUMNS)
            },
        }
        llm = FakeLLM()
        result, source = generate_cluster_instruction(
            cluster, llm, template
        )
        self.assertEqual(source, "llm")
        self.assertEqual(result["label"], "Right-side attack")
        self.assertEqual(llm.system_prompt, "")
        for feature in CLUSTER_FEATURE_COLUMNS:
            self.assertIn(f'"feature": "{feature}"', llm.user_prompt)
        self.assertEqual(llm.user_prompt.count('"z_score"'), 11)
        self.assertEqual(llm.user_prompt.count('"description": "The'), 11)
        for excluded_field in (
            "median_z",
            "q25_z",
            "q75_z",
            "direction_support",
            "strength",
        ):
            self.assertNotIn(excluded_field, llm.user_prompt)
        self.assertNotIn("Player A", llm.user_prompt)
        for required_text in (
            "badminton tactical analyst and coach",
            "Examine all eleven features",
            "Prioritize features with larger absolute deviations",
            "Analyze related features jointly",
            "direct coach-style commands",
        ):
            self.assertIn(required_text, llm.user_prompt)
        self.assertFalse(
            (RALLY_PROFILE_PROMPT.parent / "rally_profile_system.md").exists()
        )

    def test_deterministic_fallback_uses_centroid_directions(self):
        center_z = {feature: 0.0 for feature in CLUSTER_FEATURE_COLUMNS}
        center_z["attack_shot_ratio"] = 1.2
        center_z["signed_side_bias"] = -0.8
        cluster = {"center_z": center_z}
        template = Environment(undefined=StrictUndefined).from_string(
            "{{ centroid_features }}"
        )

        result, source = generate_cluster_instruction(
            cluster,
            None,
            template,
        )

        self.assertEqual(source, "deterministic")
        self.assertIn("use attacking shots more frequently", result["instruction"])
        self.assertIn("shift targeting toward the left side", result["instruction"])

    def test_instruction_validator_allows_free_tactical_wording(self):
        detailed = {
            "label": "Deep-corner defense",
            "description": "Use clears and lobs to extend play.",
            "instruction": "Drive deep into the front court and play drop shots more often.",
        }
        self.assertTrue(_valid_instruction_response(detailed))

        for field in detailed:
            with self.subTest(field=field):
                invalid = {**detailed, field: " "}
                self.assertFalse(_valid_instruction_response(invalid))

    def test_opposite_fallback_reverses_centroid_directions(self):
        center_z = {feature: 0.0 for feature in CLUSTER_FEATURE_COLUMNS}
        center_z["attack_shot_ratio"] = 1.2
        center_z["signed_side_bias"] = -0.8
        cluster = {
            "center_z": center_z,
            "label": "Right-side attack",
            "description": "Attack more and favor the right side.",
            "instruction": "Use more attacking shots and favor right-side targets.",
        }
        template = Environment(undefined=StrictUndefined).from_string(
            INSTRUCTION_VARIANT_PROMPT.read_text(encoding="utf-8")
        )

        result, source = generate_instruction_variant(
            cluster,
            "opposite",
            None,
            template,
        )

        self.assertEqual(source, "deterministic")
        self.assertIn("use attacking shots less frequently", result["instruction"])
        self.assertIn("shift targeting toward the right side", result["instruction"])

    def test_rewrite_prompt_preserves_structured_generation_contract(self):
        cluster = {
            "center_z": {feature: 0.0 for feature in CLUSTER_FEATURE_COLUMNS},
            "label": "Right-side attack",
            "description": "Attack more and favor the right side.",
            "instruction": "Use more attacking shots and favor right-side targets.",
        }
        template = Environment(undefined=StrictUndefined).from_string(
            INSTRUCTION_VARIANT_PROMPT.read_text(encoding="utf-8")
        )
        llm = FakeLLM()

        result, source = generate_instruction_variant(
            cluster,
            "rewrite",
            llm,
            template,
        )

        self.assertEqual(source, "llm")
        self.assertEqual(result["instruction"], "Use more attacking shots and favor right-side targets.")
        self.assertIn("Change only the wording", llm.user_prompt)
        self.assertIn('"label": "Right-side attack"', llm.user_prompt)

    def test_random_other_is_deterministic_and_never_selects_source_cluster(self):
        clusters = []
        for cluster_id in range(3):
            clusters.append({
                "cluster_id": cluster_id,
                "center_z": {feature: 0.0 for feature in CLUSTER_FEATURE_COLUMNS},
                "label": f"Mode {cluster_id}",
                "description": f"Description {cluster_id}",
                "instruction": f"Instruction {cluster_id}",
            })
        profile = {"clusters": clusters}
        template = Environment(undefined=StrictUndefined).from_string(
            INSTRUCTION_VARIANT_PROMPT.read_text(encoding="utf-8")
        )

        add_instruction_variants("Player A", profile, None, template, random_seed=17)
        first_targets = [
            cluster["instruction_variants"]["random_other"]["target_cluster_id"]
            for cluster in clusters
        ]

        add_instruction_variants("Player A", profile, None, template, random_seed=17)
        second_targets = [
            cluster["instruction_variants"]["random_other"]["target_cluster_id"]
            for cluster in clusters
        ]
        self.assertEqual(first_targets, second_targets)
        for cluster, target in zip(clusters, first_targets):
            self.assertNotEqual(cluster["cluster_id"], target)


if __name__ == "__main__":
    unittest.main()
