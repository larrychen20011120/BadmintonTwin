import json
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from Agent.twin import (
    TwinDatabase,
    TwinEmbeddingCache,
    embedding_model_key,
)
from Agent.base_agent import BaseAgent
from Preprocessing.data_preparation import prepare_twin_data, prepare_twin_embeddings
from Scripts.prepare_twin_embeddings import build_parser
from Scripts.prepare_all_data import twin_memory_is_complete
from Utils.helper import action_col, state_col


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **kwargs):
        rows = []
        for text in texts:
            values = np.array([len(text), sum(map(ord, text)) % 17, 1.0], dtype=np.float32)
            if normalize_embeddings:
                values /= np.linalg.norm(values)
            rows.append(values)
        return np.asarray(rows, dtype=np.float32)


def make_database(player_name='Test Player'):
    database = TwinDatabase(pd.DataFrame(), player_name, 7)
    database.memories = [
        {
            'memory_id': '1:1:1:1',
            'match_id': 1,
            'semantic_text': 'attacking clear under pressure',
            'player_location_x': 0.0,
            'player_location_y': 0.0,
            'hit_x': 0.1,
            'hit_y': 0.2,
            'opponent_location_x': -0.1,
            'opponent_location_y': -0.2,
            'opponent_type': 2,
            'landing_x': 0.3,
            'landing_y': 0.4,
            'player_type': 2,
            'moving_x': 0.0,
            'moving_y': -0.1,
        },
        {
            'memory_id': '1:1:1:3',
            'match_id': 1,
            'semantic_text': 'late defensive miss',
            'player_location_x': 0.2,
            'player_location_y': 0.1,
            'hit_x': 0.3,
            'hit_y': 0.2,
            'opponent_location_x': -0.2,
            'opponent_location_y': -0.1,
            'opponent_type': 4,
            'landing_x': None,
            'landing_y': None,
            'player_type': 11,
            'moving_x': None,
            'moving_y': None,
        },
    ]
    for memory in database.memories:
        prompt_action = BaseAgent.build_prompt_action(
            memory['player_type'],
            memory['landing_x'],
            memory['landing_y'],
            memory['moving_x'],
            memory['moving_y'],
        )
        memory['action_verbalization'] = BaseAgent.format_prompt_action(prompt_action)
    database.build_metadata = {
        'use_llm': True,
        'max_rallies_per_player': None,
        'expected_match_ids': [1],
        'target_source_row_count': 2,
        'memory_count': 2,
    }
    database._build_memory_stores()
    return database


class TwinEmbeddingPipelineTest(unittest.TestCase):
    def test_embedding_cli_has_no_config_or_model_override(self):
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn('--config', option_strings)
        self.assertNotIn('--embedding-model', option_strings)

    def test_canonical_memory_does_not_store_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'Test_Player_db.pkl')
            make_database().save(path)

            with open(path, 'rb') as f:
                payload = pickle.load(f)
            self.assertEqual(payload['schema_version'], 4)
            self.assertNotIn('semantic_embeddings', payload)
            self.assertNotIn('embedding_model_name', payload)

            loaded = TwinDatabase.load(path)
            self.assertEqual(len(loaded.memories), 2)
            self.assertIsNone(loaded.embedding_cache)

    def test_legacy_inline_embedding_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'legacy_db.pkl')
            with open(path, 'wb') as f:
                pickle.dump({'schema_version': 3, 'player_name': 'Test Player', 'player_id': 7}, f)
            with self.assertRaisesRegex(ValueError, 'Rebuild canonical Twin memory'):
                TwinDatabase.load(path)

    def test_model_specific_cache_round_trip_and_retrieval(self):
        database = make_database()
        model = FakeEmbeddingModel()
        cache = TwinEmbeddingCache.build(database, model, 'fake/model')

        with tempfile.TemporaryDirectory() as directory:
            cache.save(directory)
            loaded = TwinEmbeddingCache.load(directory, database, 'fake/model')
            database.attach_embedding_cache(loaded)
            query = model.encode(['attacking clear under pressure'])[0]
            results = database.retrieve(
                {
                    'player_location_x': 0.0,
                    'player_location_y': 0.0,
                    'hit_x': 0.1,
                    'hit_y': 0.2,
                    'opponent_location_x': -0.1,
                    'opponent_location_y': -0.2,
                    'opponent_type': 2,
                },
                query,
                top_k=1,
            )
            self.assertEqual(results[0]['memory_id'], '1:1:1:1')
            self.assertIn('"landing_x": 230', results[0]['action_verbalization'])
            self.assertNotIn('action', results[0])

    def test_stale_cache_is_rejected(self):
        database = make_database()
        cache = TwinEmbeddingCache.build(database, FakeEmbeddingModel(), 'fake/model')
        with tempfile.TemporaryDirectory() as directory:
            cache.save(directory)
            database.memories[0]['semantic_text'] = 'changed text'
            with self.assertRaisesRegex(ValueError, 'stale'):
                TwinEmbeddingCache.load(directory, database, 'fake/model')

    def test_action_verbalization_does_not_change_embedding_digest(self):
        database = make_database()
        cache = TwinEmbeddingCache.build(database, FakeEmbeddingModel(), 'fake/model')
        database.memories[0]['action_verbalization'] = 'Action: changed display only'
        cache.validate(database)

    def test_different_models_have_different_cache_keys(self):
        self.assertNotEqual(
            embedding_model_key('fake/model-a'),
            embedding_model_key('fake/model-b'),
        )
        self.assertNotEqual(
            embedding_model_key('fake/model-a', dtype='float32'),
            embedding_model_key('fake/model-a', dtype='float16'),
        )

    def test_embedding_stage_skips_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_dir = os.path.join(directory, 'memory')
            cache_dir = os.path.join(directory, 'embeddings')
            os.makedirs(memory_dir)
            make_database().save(os.path.join(memory_dir, 'Test_Player_db.pkl'))
            model = FakeEmbeddingModel()

            first = prepare_twin_embeddings(
                memory_dir,
                cache_dir,
                'fake/model',
                embedding_model=model,
            )
            second = prepare_twin_embeddings(
                memory_dir,
                cache_dir,
                'fake/model',
                embedding_model=model,
            )
            self.assertEqual(first, {'built': 1, 'skipped': 0})
            self.assertEqual(second, {'built': 0, 'skipped': 1})

    def test_memory_completeness_rejects_stale_player_files(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_dir = os.path.join(directory, 'memory')
            os.makedirs(memory_dir)
            split_path = os.path.join(directory, 'split.json')
            with open(split_path, 'w') as f:
                json.dump({'train': {'Test Player': [1]}}, f)
            make_database().save(os.path.join(memory_dir, 'Test_Player_db.pkl'))
            self.assertTrue(twin_memory_is_complete(split_path, memory_dir))

            database_with_extra_match = make_database()
            extra_memory = dict(database_with_extra_match.memories[0])
            extra_memory['memory_id'] = '2:1:1:1'
            extra_memory['match_id'] = 2
            database_with_extra_match.memories.append(extra_memory)
            database_with_extra_match.build_metadata['target_source_row_count'] = 3
            database_with_extra_match.build_metadata['memory_count'] = 3
            database_with_extra_match._build_memory_stores()
            database_with_extra_match.save(os.path.join(memory_dir, 'Test_Player_db.pkl'))
            self.assertFalse(twin_memory_is_complete(split_path, memory_dir))

            make_database().save(os.path.join(memory_dir, 'Test_Player_db.pkl'))
            make_database('Stale Player').save(os.path.join(memory_dir, 'Stale_Player_db.pkl'))
            self.assertFalse(twin_memory_is_complete(split_path, memory_dir))

    def test_memory_build_atomically_replaces_stale_player_files(self):
        row = {column: 0.0 for column in state_col + action_col}
        row.update({
            'player': 'Test Player',
            'match_id': 1,
            'rally': 1,
            'ball_round': 1,
            'opponent_type': 2,
            'player_type': 2,
        })
        with tempfile.TemporaryDirectory() as directory:
            output_dir = os.path.join(directory, 'memory')
            os.makedirs(output_dir)
            make_database('Stale Player').save(os.path.join(output_dir, 'Stale_Player_db.pkl'))

            prepare_twin_data(
                pd.DataFrame([row]),
                {'train': {'Test Player': [1]}},
                {'Test Player': 7},
                output_dir,
                use_llm=False,
            )

            self.assertEqual(os.listdir(output_dir), ['Test_Player_db.pkl'])
            loaded = TwinDatabase.load(os.path.join(output_dir, 'Test_Player_db.pkl'))
            self.assertEqual(loaded.build_metadata['expected_match_ids'], [1])
            self.assertIn('action_verbalization', loaded.memories[0])
            self.assertIn('"landing_x": 177', loaded.memories[0]['action_verbalization'])


if __name__ == '__main__':
    unittest.main()
