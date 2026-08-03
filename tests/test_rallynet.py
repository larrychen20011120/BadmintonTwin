import json
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch
import yaml

from Agent.rallynet import RallyNet, load_checkpoint
from prepare_rallynet import compute_rallynet_dataset_fingerprint
from Scripts.train_rallynet import (
    build_parser,
    collate_rally_prefixes,
    train_rallynet,
)


class RallyNetTest(unittest.TestCase):
    def test_model_validates_configuration_and_history_limit(self):
        with self.assertRaisesRegex(ValueError, "exactly 12"):
            RallyNet(player_count=2, shot_type_count=11)
        with self.assertRaisesRegex(ValueError, "solver_substeps"):
            RallyNet(player_count=2, solver_substeps=0)

        model = RallyNet(
            player_count=2,
            latent_size=2,
            context_size=2,
            hidden_size=4,
            action_hidden_size=4,
            state_embedding_dim=2,
            selected_shot_embedding_dim=2,
            gmm_components=1,
            max_sequence_length=3,
            solver_substeps=1,
        )
        states = torch.zeros(4, 1, 18)
        with self.assertRaisesRegex(ValueError, "max_sequence_length 3"):
            model.error_model.predict_error(states, sample=False)

    def test_training_cli_defaults_to_physical_rally(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.experiment, "rally")
        self.assertEqual(args.stage, "all")

    def test_sde_collation_uses_observed_prefixes(self):
        torch.manual_seed(3)
        states = torch.arange(5 * 18, dtype=torch.float32).reshape(5, 18)
        errors = torch.arange(5, dtype=torch.float32)
        batch_states, batch_errors, valid_mask = collate_rally_prefixes([
            (states, errors),
        ])
        prefix_length = int(valid_mask[:, 0].sum().item())
        self.assertGreaterEqual(prefix_length, 1)
        self.assertLessEqual(prefix_length, 5)
        torch.testing.assert_close(batch_states[:prefix_length, 0], states[:prefix_length])
        torch.testing.assert_close(batch_errors[:prefix_length, 0], errors[:prefix_length])

    def test_tiny_two_stage_training_writes_loadable_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = os.path.join(directory, "data")
            checkpoint_dir = os.path.join(directory, "checkpoints")
            model_path = os.path.join(directory, "model.pth")
            player_map_path = os.path.join(directory, "players.json")
            config_path = os.path.join(directory, "config.yaml")
            os.makedirs(data_dir)

            player_map = {"A": 0, "B": 1}
            with open(player_map_path, "w", encoding="utf-8") as handle:
                json.dump(player_map, handle)

            length = 11
            states = np.zeros((length, 18), dtype=np.float32)
            states[:, 0] = 1
            states[:, 1] = np.arange(1, length + 1)
            states[:, 12] = np.arange(length) % 12
            states[:, 17] = np.arange(length) % 2
            actions = np.zeros((length, 5), dtype=np.float32)
            actions[:, 2] = np.arange(1, 12)
            errors = np.zeros(length, dtype=np.bool_)
            errors[-1] = True
            coordinate_mask = ~errors
            serve_mask = np.zeros(length, dtype=np.bool_)
            serve_mask[0] = True
            rally = {
                "match_id": 1,
                "set": 1,
                "rally": 1,
                "length": length,
                "states": states,
                "actions": actions,
                "error_targets": errors,
                "serve_mask": serve_mask,
                "coordinate_mask": coordinate_mask,
            }
            fingerprint = compute_rallynet_dataset_fingerprint([rally], player_map)
            payload = {
                "schema_version": 1,
                "player_map": player_map,
                "dataset_fingerprint": fingerprint,
                "rallies": [rally],
            }
            metadata = {
                "schema_version": 1,
                "experiment": "rally",
                "player_map": player_map,
                "shot_type_count": 12,
                "dataset_fingerprint": fingerprint,
                "split_sha256": "test-split",
                "train_rally_keys": [{"match_id": 1, "set": 1, "rally": 1}],
                "requested_train_rally_keys": [
                    {"match_id": 1, "set": 1, "rally": 1}
                ],
                "skipped_train_rallies": [],
                "excluded_test_match_ids": [],
            }
            with open(os.path.join(data_dir, "rallies.pkl"), "wb") as handle:
                pickle.dump(payload, handle)
            with open(
                os.path.join(data_dir, "metadata.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(metadata, handle)

            config = {
                "paths": {
                    "player_map": player_map_path,
                    "rallynet": {
                        "data": data_dir,
                        "checkpoint_dir": checkpoint_dir,
                        "model": model_path,
                    },
                },
                "evaluation": {"rollout_extra_steps": 2},
                "rallynet_training": {
                    "seed": 7,
                    "bc_epochs": 1,
                    "bc_batch_size": 16,
                    "bc_learning_rate": 0.001,
                    "action_hidden_size": 4,
                    "rallynet_num_iters": 1,
                    "rallynet_batch_size": 1,
                    "rallynet_learning_rate": 0.001,
                    "latent_size": 2,
                    "context_size": 2,
                    "hidden_size": 4,
                    "state_embedding_dim": 2,
                    "selected_shot_embedding_dim": 2,
                    "gmm_components": 1,
                    "min_sigma": 0.001,
                    "max_sample_sigma": 1.0,
                    "normalization_epsilon": 0.000001,
                    "reconstruction_noise_std": 0.1,
                    "max_sequence_length": 13,
                    "time_horizon": 1.0,
                    "solver_method": "euler",
                    "solver_substeps": 1,
                    "lr_gamma": 1.0,
                    "kl_anneal_steps": 1,
                    "checkpoint_interval": 1,
                    "kl_weight": 0.01,
                    "reconstruction_weight": 0.01,
                    "gradient_clip_norm": 5.0,
                    "num_workers": 0,
                    "pin_memory": False,
                    "error_pos_weight": None,
                },
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)

            result = train_rallynet(config_path=config_path, device="cpu")
            self.assertEqual(result, model_path)
            model, checkpoint = load_checkpoint(model_path)
            self.assertIsInstance(model, RallyNet)
            self.assertTrue(checkpoint["training"]["complete"])
            self.assertTrue(checkpoint["training"]["bc_trained"])
            self.assertTrue(checkpoint["training"]["sde_trained"])
            self.assertEqual(
                checkpoint["training"]["provenance"]["split_sha256"],
                "test-split",
            )


if __name__ == "__main__":
    unittest.main()
