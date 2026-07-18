import argparse
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from Agent.bc import train_model
import Scripts.train_bc as train_bc_script


class BCTrainingConfigTest(unittest.TestCase):
    def test_training_wrapper_forwards_architecture_config(self):
        config = {
            'paths': {
                'bc': {
                    'data': {'norm': 'data/norm', 'shifted': 'data/shifted'},
                    'models': {'norm': 'weights/norm.pth', 'shifted': 'weights/shifted.pth'},
                    'weight_dir': 'weights',
                }
            },
            'bc_training': {
                'epochs': 1,
                'batch_size': 2,
                'learning_rate': 0.001,
                'input_dim': 32,
                'hidden_dim': 64,
                'player_id_len': 43,
                'shot_type_len': 12,
                'distribution': 'gmm',
                'gmm_num': 5,
            },
        }

        with (
            patch.object(train_bc_script, 'load_config', return_value=config),
            patch.object(train_bc_script.os.path, 'exists', return_value=True),
            patch.object(train_bc_script.os, 'makedirs'),
            patch.object(train_bc_script, 'print_runtime_box'),
            patch.object(
                train_bc_script.subprocess,
                'run',
                return_value=SimpleNamespace(returncode=0),
            ) as run,
        ):
            success = train_bc_script.train_bc('norm', distribution='gmm')

        self.assertTrue(success)
        command = run.call_args.args[0]
        expected_arguments = {
            '--input_dim': '32',
            '--hidden_dim': '64',
            '--player_id_len': '43',
            '--shot_type_len': '12',
        }
        for argument, expected_value in expected_arguments.items():
            index = command.index(argument)
            self.assertEqual(command[index + 1], expected_value)

    def test_train_model_uses_forwarded_architecture(self):
        state = np.zeros((2, 18), dtype=np.float32)
        state[:, 12] = 1
        action = np.zeros((2, 5), dtype=np.float32)
        action[:, 2] = 1

        with tempfile.TemporaryDirectory() as data_dir:
            with open(os.path.join(data_dir, 'train_state.pkl'), 'wb') as file:
                pickle.dump(state, file)
            with open(os.path.join(data_dir, 'train_action.pkl'), 'wb') as file:
                pickle.dump(action, file)

            output_path = os.path.join(data_dir, 'model.pth')
            args = argparse.Namespace(
                data_dir=data_dir,
                batch_size=2,
                input_dim=32,
                hidden_dim=7,
                player_id_len=3,
                shot_type_len=6,
                distribution='deterministic',
                gmm_num=5,
                lr=0.001,
                epochs=0,
                output=output_path,
            )

            with patch('Agent.bc.compute_shot_type_weights', return_value=torch.ones(6)):
                train_model(args)

            state_dict = torch.load(output_path, map_location='cpu', weights_only=True)

        self.assertEqual(tuple(state_dict['fc.weight'].shape), (7, 32))
        self.assertEqual(tuple(state_dict['player_embedding.weight'].shape), (3, 8))
        self.assertEqual(tuple(state_dict['shot_embedding.weight'].shape), (6, 8))
        self.assertEqual(tuple(state_dict['predict_shot_type.weight'].shape), (6, 7))


if __name__ == '__main__':
    unittest.main()
