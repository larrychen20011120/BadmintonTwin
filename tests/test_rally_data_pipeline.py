import hashlib
import json
import os
import pickle
import tempfile
import unittest

import pandas as pd

from Agent.shuttlenet import prepare_shuttlenet_data
from Preprocessing.data_preparation import prepare_bc_data
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset
from Scripts.prepare_all_data import build_jobs
from Utils.helper import action_col, state_col


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def make_rows():
    rows = []
    for rally, players in ((1, ('A', 'B')), (2, ('A', 'B'))):
        for ball_round, player in enumerate(players, start=1):
            row = {column: 0.0 for column in state_col + action_col}
            row.update({
                'match_id': 7,
                'set': 1,
                'rally': rally,
                'ball_round': ball_round,
                'player': player,
                'server': 1,
                'player_type': 1 if ball_round == 1 else 2,
                'opponent_type': 0 if ball_round == 1 else 1,
            })
            rows.append(row)
    return pd.DataFrame(rows)


class RallyDataPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.split_path = os.path.join(self.temp_dir.name, 'split.json')
        self.profiles_path = os.path.join(self.temp_dir.name, 'profiles.json')
        assignment_train = {'match_id': 7, 'set': 1, 'rally': 1}
        assignment_test = {'match_id': 7, 'set': 1, 'rally': 2}
        self.split = {
            'schema_version': 2,
            'train': {'A': [assignment_train], 'B': [assignment_train]},
            'test': {'A': [assignment_test], 'B': [assignment_test]},
        }
        write_json(self.split_path, self.split)
        profile = {
            'status': 'clustered',
            'clusters': [{
                'cluster_id': 0,
                'label': 'Attack',
                'description': 'Attack often',
                'instruction': 'Attack more.',
                'instruction_source': 'deterministic',
            }],
            'train_assignments': [{**assignment_train, 'cluster_id': 0}],
            'test_assignments': [{
                **assignment_test,
                'cluster_id': 0,
                'in_train_support': True,
            }],
        }
        write_json(self.profiles_path, {
            'split_sha256': file_digest(self.split_path),
            'players': {'A': profile, 'B': profile},
        })

    def test_loader_deduplicates_physical_rallies_and_joins_profiles(self):
        dataset = load_rally_dataset(self.split_path, self.profiles_path)

        self.assertEqual(dataset['summary']['physical_train_rallies'], 1)
        self.assertEqual(dataset['summary']['physical_test_rallies'], 1)
        self.assertEqual(dataset['summary']['player_test_assignments'], 2)
        self.assertEqual(dataset['test'][0]['players'], ['A', 'B'])
        self.assertEqual(
            dataset['test'][0]['player_profiles']['A']['instruction'],
            'Attack more.',
        )

    def test_prepare_jobs_run_rally_methods_once(self):
        jobs = build_jobs(
            ['norm', 'shifted'],
            ['bc', 'rallynet', 'shuttlenet', 'naive_rag', 'twin'],
        )
        self.assertEqual(jobs.count(('rally', 'bc')), 1)
        self.assertEqual(jobs.count(('rally', 'rallynet')), 1)
        self.assertEqual(jobs.count(('rally', 'shuttlenet')), 1)
        self.assertEqual(jobs.count(('rally', 'naive_rag')), 1)
        self.assertEqual(jobs.count(('rally', 'twin')), 1)
        self.assertNotIn(('norm', 'rallynet'), jobs)
        self.assertNotIn(('shifted', 'rallynet'), jobs)
        self.assertNotIn(('norm', 'naive_rag'), jobs)
        self.assertNotIn(('shifted', 'naive_rag'), jobs)
        self.assertNotIn(('norm', 'twin'), jobs)
        self.assertNotIn(('shifted', 'twin'), jobs)

    def test_loader_rejects_old_match_list_format(self):
        write_json(self.split_path, {'train': {'A': [7]}, 'test': {'A': []}})
        with self.assertRaisesRegex(ValueError, 'Rally assignments must be objects'):
            load_rally_dataset(self.split_path)

    def test_bc_and_shuttlenet_include_each_training_shot_once(self):
        dataset = load_rally_dataset(self.split_path)
        rows = make_rows()
        selected = filter_rally_rows(rows, dataset['train'])
        self.assertEqual(len(selected), 2)

        player_map = {'A': 0, 'B': 1}
        mapped = rows.copy()
        mapped['player'] = mapped['player'].map(player_map)
        bc_dir = os.path.join(self.temp_dir.name, 'bc')
        prepare_bc_data(mapped, dataset, player_map, bc_dir)
        with open(os.path.join(bc_dir, 'train_state.pkl'), 'rb') as handle:
            states = pickle.load(handle)
        with open(os.path.join(bc_dir, 'train_action.pkl'), 'rb') as handle:
            actions = pickle.load(handle)
        with open(os.path.join(bc_dir, 'preparation_summary.json'), encoding='utf-8') as handle:
            bc_summary = json.load(handle)
        self.assertEqual(states.shape, (2, len(state_col)))
        self.assertEqual(actions.shape, (2, len(action_col)))
        self.assertEqual(bc_summary['physical_train_rallies_matched'], 1)
        self.assertEqual(bc_summary['physical_train_rallies_skipped'], 0)

        csv_path = os.path.join(self.temp_dir.name, 'merged.csv')
        map_path = os.path.join(self.temp_dir.name, 'players.json')
        rows.to_csv(csv_path, index=False)
        write_json(map_path, player_map)
        shuttle_dir = os.path.join(self.temp_dir.name, 'shuttlenet')
        prepare_shuttlenet_data(csv_path, self.split_path, map_path, shuttle_dir)
        with open(os.path.join(shuttle_dir, 'train_samples.pkl'), 'rb') as handle:
            payload = pickle.load(handle)
        self.assertEqual(len(payload['samples']), 2)
        self.assertEqual(payload['metadata']['physical_train_rallies_matched'], 1)
        self.assertEqual(payload['metadata']['physical_train_rallies_skipped'], 0)


if __name__ == '__main__':
    unittest.main()
