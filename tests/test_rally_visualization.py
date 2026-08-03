import builtins
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Preprocessing.rally_feature import CLUSTER_FEATURE_COLUMNS
from Scripts.plot_rally_profiles import (
    _profile_centers,
    _plot_umap,
    _select_players,
    aggregate_cluster_counts,
    build_assignment_map,
    coverage_ellipse,
    map_rows_to_assignments,
    sanitize_filename,
)


class RallyVisualizationHelpersTest(unittest.TestCase):
    def test_umap_fits_train_and_only_transforms_model_centers(self):
        class FakeUMAP:
            instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fit_values = None
                self.transform_values = []
                FakeUMAP.instance = self

            def fit_transform(self, values):
                self.fit_values = values.copy()
                return values[:, :2]

            def transform(self, values):
                self.transform_values.append(values.copy())
                return values[:, :2]

        module = types.ModuleType("umap")
        module.UMAP = FakeUMAP
        train = pd.DataFrame(
            np.arange(6 * len(CLUSTER_FEATURE_COLUMNS), dtype=float).reshape(6, -1),
            columns=CLUSTER_FEATURE_COLUMNS,
        )
        train["cluster_id"] = [0, 0, 0, 1, 1, 1]
        profile = {
            "clusters": [
                {
                    "cluster_id": cluster_id,
                    "center_z": {
                        feature: float(cluster_id)
                        for feature in CLUSTER_FEATURE_COLUMNS
                    },
                }
                for cluster_id in range(2)
            ]
        }
        with (
            patch.dict(sys.modules, {"umap": module}),
            patch("Scripts.plot_rally_profiles._plot_embedding") as plot_embedding,
        ):
            _plot_umap(None, train, profile, ["blue", "red"])

        self.assertEqual(FakeUMAP.instance.fit_values.shape, (6, 11))
        self.assertEqual(
            [values.shape for values in FakeUMAP.instance.transform_values],
            [(2, 11)],
        )
        self.assertEqual(FakeUMAP.instance.kwargs["random_state"], 42)
        plot_embedding.assert_called_once()
        call_args = plot_embedding.call_args.args
        np.testing.assert_allclose(
            call_args[2],
            [train.iloc[:3, :2].mean().to_numpy(), train.iloc[3:, :2].mean().to_numpy()],
        )
        np.testing.assert_allclose(
            plot_embedding.call_args.kwargs["projected_model_center_xy"],
            [[0.0, 0.0], [1.0, 1.0]],
        )

    def test_coverage_ellipse_and_profile_centers_are_pure_helpers(self):
        points = np.array([
            [-2.0, -0.2],
            [-1.0, 0.1],
            [0.0, 0.0],
            [1.0, -0.1],
            [2.0, 0.2],
        ])
        ellipse = coverage_ellipse(points, coverage=0.8)
        np.testing.assert_allclose(ellipse["center"], [0.0, 0.0])
        self.assertGreater(ellipse["width"], ellipse["height"])
        self.assertEqual(ellipse["coverage"], 0.8)
        with self.assertRaisesRegex(ValueError, "coverage"):
            coverage_ellipse(points, coverage=1.0)

        center = {
            feature: index / 10
            for index, feature in enumerate(CLUSTER_FEATURE_COLUMNS)
        }
        profile = {
            "clusters": [
                {
                    "cluster_id": 1,
                    "center_z": {
                        key: value + 1 for key, value in center.items()
                    },
                },
                {"cluster_id": 0, "center_z": center},
            ]
        }
        centers = _profile_centers(profile)
        self.assertEqual(centers.shape, (2, len(center)))
        np.testing.assert_allclose(centers[1] - centers[0], 1.0)

    def test_default_player_selection_skips_excluded_profiles(self):
        split = {"eligible_players": ["Clustered", "Excluded"]}
        profiles = {
            "Clustered": {"status": "clustered"},
            "Excluded": {"status": "excluded", "reason": "unstable"},
        }
        self.assertEqual(_select_players(None, split, profiles), ["Clustered"])
        with self.assertRaisesRegex(ValueError, "unstable"):
            _select_players(["Excluded"], split, profiles)

    def test_import_and_filename_sanitization_do_not_require_matplotlib(self):
        real_import = builtins.__import__

        def reject_matplotlib(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise AssertionError("Pure helpers must not import matplotlib")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_matplotlib):
            self.assertEqual(sanitize_filename("  PUSARLA V. Sindhu  "), "PUSARLA_V_Sindhu")
            self.assertEqual(sanitize_filename("A/B:C*D?"), "A_B_C_D")
            self.assertEqual(sanitize_filename("選手"), "player")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            sanitize_filename("  ")

    def test_assignment_mapping_preserves_fold_and_rejects_duplicates(self):
        train = [
            {"match_id": 1, "set": 1, "rally": 4, "cluster_id": 0},
            {"match_id": 1, "set": 1, "rally": 5, "cluster_id": 1},
        ]
        test = [{"match_id": 2, "set": 2, "rally": 1, "cluster_id": 1}]
        rows = pd.DataFrame(
            [
                {"match_id": 1, "set": 1, "rally": 4, "value": "a"},
                {"match_id": 1, "set": 1, "rally": 4, "value": "b"},
                {"match_id": 2, "set": 2, "rally": 1, "value": "c"},
                {"match_id": 9, "set": 1, "rally": 1, "value": "ignored"},
            ]
        )
        mapped = map_rows_to_assignments(rows, train[:1], test, require_complete=True)
        self.assertEqual(mapped["fold"].tolist(), ["train", "train", "test"])
        self.assertEqual(mapped["cluster_id"].tolist(), [0, 0, 1])
        self.assertEqual(mapped["value"].tolist(), ["a", "b", "c"])

        with self.assertRaisesRegex(ValueError, "Duplicate rally key"):
            build_assignment_map([train[0], train[0]])
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            build_assignment_map([
                {"match_id": 1, "set": 1, "rally": 1, "cluster_id": 1.5}
            ])
        with self.assertRaisesRegex(ValueError, "no matching data row"):
            map_rows_to_assignments(rows.iloc[:1], train, [], require_complete=True)

    def test_cluster_category_aggregation_returns_counts_and_proportions(self):
        assignments = [
            {"match_id": 1, "set": 1, "rally": 1, "cluster_id": 0},
            {"match_id": 1, "set": 1, "rally": 2, "cluster_id": 1},
        ]
        strokes = pd.DataFrame(
            [
                {"match_id": 1, "set": 1, "rally": 1, "shot": "Smash"},
                {"match_id": 1, "set": 1, "rally": 1, "shot": "Smash"},
                {"match_id": 1, "set": 1, "rally": 1, "shot": "Drop"},
                {"match_id": 1, "set": 1, "rally": 2, "shot": "Clear"},
                {"match_id": 9, "set": 1, "rally": 9, "shot": "ignored"},
            ]
        )
        counts, proportions = aggregate_cluster_counts(strokes, assignments, "shot")
        self.assertEqual(counts.loc[0, "Smash"], 2)
        self.assertEqual(counts.loc[1, "Clear"], 1)
        self.assertAlmostEqual(proportions.loc[0, "Smash"], 2 / 3)
        self.assertAlmostEqual(proportions.loc[0].sum(), 1.0)
        self.assertAlmostEqual(proportions.loc[1].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
