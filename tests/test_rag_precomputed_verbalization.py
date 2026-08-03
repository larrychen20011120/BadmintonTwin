import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Agent.naive_rag import NaiveRAGAgent, NaiveRAGDatabase, NaiveRAGVerbalizer
from Agent.timerag import TimeRAGDatabase, TimeRAGVerbalizer
from Preprocessing.data_preparation import prepare_naive_rag_data, prepare_timerag_data
from Utils.artifact_names import player_database_filename
from Utils.langfuse_tracing import (
    simplify_naive_rag_experiences,
    simplify_timerag_windows,
)


NAIVE_STAGE1_SERVE_DESCRIPTION = '''Current score: 3-2. I lead by 1 points.
Match Context: Rally 1, currently at Shot 1. This is the first shot of the rally.

Serve situation: player at Rear Center, opponent at Mid Center.
Outcome: Returned with Serve short.'''

NAIVE_STAGE2_SERVE_DESCRIPTION = '''Current score: 3-2. I lead by 1 points.
Match Context: Rally 1, currently at Shot 1. This is the first shot of the rally.

It is my turn to serve. I am at Rear Center, opponent at Mid Center.
I serve a Serve short to Mid Center, then move to Rear Center.
Action: {"shot_type": "Serve short", "landing_area": "Mid Center", "landing_x": 213, "landing_y": 312, "moving_area": "Rear Center", "moving_x": 150, "moving_y": 180}'''

TIMERAG_STAGE1_THREE_SHOT_DESCRIPTION = '''Current score: 3-2. I lead by 1 points.
Match Context: Rally 1, currently at Shot 3. 2 shots have been played in this rally.

Shot 1:
It is my turn. I am at Rear Center, opponent at Mid Center.
I serve a Serve short to Mid Center, then move to Rear Center.
Shot 2:
It is opponent's turn. I am at Mid Center, opponent at Rear Center.
My Serve short is arriving at Mid Center.
Opponent moves to Mid Center and hits a Clear to Mid Center, then moves to Rear Center.
Shot 3:
It is my turn. I am at Rear Center, opponent at Mid Center.
Opponent's Clear is arriving at Mid Center.
[Next shot to be decided]
Outcome: Returned with Smash.'''

TIMERAG_STAGE2_THREE_SHOT_DESCRIPTION = '''Current score: 3-2. I lead by 1 points.
Match Context: Rally 1, currently at Shot 3. 2 shots have been played in this rally.

Shot 1:
It is my turn. I am at Rear Center, opponent at Mid Center.
I serve a Serve short to Mid Center, then move to Rear Center.
Shot 2:
It is opponent's turn. I am at Mid Center, opponent at Rear Center.
My Serve short is arriving at Mid Center.
Opponent moves to Mid Center and hits a Clear to Mid Center, then moves to Rear Center.
Shot 3:
It is my turn. I am at Rear Center, opponent at Mid Center.
Opponent's Clear is arriving at Mid Center.
I move to Mid Center and hit a Smash to Mid Center, then move to Rear Center.
Action: {"shot_type": "Smash", "landing_area": "Mid Center", "landing_x": 216, "landing_y": 316, "moving_area": "Rear Center", "moving_x": 154, "moving_y": 184}'''


def make_row(ball_round, player, player_type, opponent_type, offset=0.0):
    return {
        'player': player,
        'match_id': 1,
        'set': 1,
        'rally': 1,
        'ball_round': ball_round,
        'player_score': 3,
        'opponent_score': 2,
        'score_status': 1,
        'player_location_x': 0.10 + offset,
        'player_location_y': -0.20 + offset,
        'ball_distance': 0.30,
        'ball_distance_x': 0.10,
        'ball_distance_y': 0.20,
        'hit_x': 0.15 + offset,
        'hit_y': 0.25 + offset,
        'opponent_type': opponent_type,
        'opponent_location_x': -0.10 + offset,
        'opponent_location_y': 0.20 + offset,
        'opponent_move_x': 0.05,
        'opponent_move_y': -0.05,
        'landing_x': 0.20 + offset,
        'landing_y': 0.30 + offset,
        'player_type': player_type,
        'moving_x': -0.15 + offset,
        'moving_y': -0.25 + offset,
    }


def make_dataframe():
    rows = [
        make_row(1, 'Player A', 1, 0, 0.00),
        make_row(2, 'Player B', 2, 1, 0.01),
        make_row(3, 'Player A', 4, 2, 0.02),
        make_row(4, 'Player B', 7, 4, 0.03),
        make_row(5, 'Player A', 11, 7, 0.04),
    ]
    return pd.DataFrame(rows)


class RAGDataPreparationTest(unittest.TestCase):
    def test_string_dtype_player_names_are_not_treated_as_ids(self):
        dataframe = make_dataframe()
        dataframe['player'] = dataframe['player'].astype('string')
        record = {
            'match_id': 1,
            'set': 1,
            'rally': 1,
            'players': ['Player A', 'Player B'],
            'player_profiles': {'Player A': None, 'Player B': None},
        }
        split = {
            'train': [record],
            'test': [],
            'split_sha256': 'test-split',
            'profiles_sha256': None,
        }
        player_map = {'Player A': 0, 'Player B': 1}

        with tempfile.TemporaryDirectory() as directory:
            naive_dir = os.path.join(directory, 'naive')
            timerag_dir = os.path.join(directory, 'timerag')
            self.assertEqual(
                prepare_naive_rag_data(dataframe, split, player_map, naive_dir),
                2,
            )
            self.assertEqual(
                prepare_timerag_data(dataframe, split, player_map, timerag_dir, window_size=3),
                2,
            )

    def test_rally_preparation_does_not_include_test_rally_from_same_match(self):
        train = make_dataframe()
        test = make_dataframe().copy()
        test['rally'] = 2
        dataframe = pd.concat([train, test], ignore_index=True)
        train_record = {
            'match_id': 1, 'set': 1, 'rally': 1,
            'players': ['Player A', 'Player B'], 'player_profiles': {},
        }
        test_record = {
            'match_id': 1, 'set': 1, 'rally': 2,
            'players': ['Player A', 'Player B'], 'player_profiles': {},
        }
        split = {
            'train': [train_record],
            'test': [test_record],
            'split_sha256': 'test-split',
            'profiles_sha256': None,
        }
        player_map = {'Player A': 0, 'Player B': 1}

        with tempfile.TemporaryDirectory() as directory:
            naive_dir = os.path.join(directory, 'naive')
            timerag_dir = os.path.join(directory, 'timerag')
            prepare_naive_rag_data(dataframe, split, player_map, naive_dir)
            prepare_timerag_data(dataframe, split, player_map, timerag_dir, window_size=3)
            naive = NaiveRAGDatabase.load(os.path.join(
                naive_dir, player_database_filename('Player A')
            ))
            timerag = TimeRAGDatabase.load(os.path.join(
                timerag_dir, player_database_filename('Player A')
            ))

        self.assertEqual(set(naive.data['rally'].astype(int)), {1})
        self.assertEqual({int(row['rally']) for row in timerag.windows_metadata}, {1})
        self.assertEqual(naive.build_metadata['train_rally_keys'], [[1, 1, 1]])

    def test_failed_build_does_not_replace_existing_database_directory(self):
        dataframe = make_dataframe()
        record = {
            'match_id': 1, 'set': 1, 'rally': 1,
            'players': ['Player A', 'Player B'], 'player_profiles': {},
        }
        split = {
            'train': [record], 'test': [],
            'split_sha256': 'test-split', 'profiles_sha256': None,
        }
        player_map = {'Player A': 0, 'Player B': 1}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = os.path.join(directory, 'naive')
            os.makedirs(output_dir)
            marker = os.path.join(output_dir, 'existing.txt')
            with open(marker, 'w', encoding='utf-8') as handle:
                handle.write('keep')
            with patch.object(NaiveRAGDatabase, 'save', side_effect=RuntimeError('boom')):
                with self.assertRaisesRegex(RuntimeError, 'Failed to build'):
                    prepare_naive_rag_data(dataframe, split, player_map, output_dir)
            with open(marker, encoding='utf-8') as handle:
                self.assertEqual(handle.read(), 'keep')


class NaiveRAGOfflineVerbalizationTest(unittest.TestCase):
    def setUp(self):
        self.database = NaiveRAGDatabase(make_dataframe(), 'Player A', 0)

    def test_offline_text_matches_canonical_renderer(self):
        verbalizer = NaiveRAGVerbalizer()
        expected = [
            {
                'record_id': int(row.name),
                **verbalizer.build_case_verbalization(self.database._build_static_experience(row)),
            }
            for _, row in self.database.data.iterrows()
        ]
        self.assertEqual(self.database.case_verbalizations, expected)
        verbalization = self.database.case_verbalizations[0]
        self.assertNotIn('description', verbalization)
        self.assertEqual(
            verbalization['stage1_description'], NAIVE_STAGE1_SERVE_DESCRIPTION
        )
        self.assertEqual(
            verbalization['stage2_description'], NAIVE_STAGE2_SERVE_DESCRIPTION
        )

    def test_retrieved_stage_cases_do_not_call_renderer(self):
        row = self.database.data.iloc[0]
        query = {key: row[key] for key in self.database.retrieval_keys}
        experiences = self.database.retrieve(query, top_k=2)
        with patch.object(NaiveRAGVerbalizer, 'build_case_verbalization', side_effect=AssertionError):
            verbalizer = NaiveRAGVerbalizer()
            stage1_cases = verbalizer.build_stage1_cases(experiences)
            stage2_cases = verbalizer.build_stage2_cases(experiences)
        self.assertEqual([case['index'] for case in stage1_cases], [1, 2])
        self.assertEqual(
            stage1_cases[0]['stage1_description'],
            experiences[0]['case_verbalization']['stage1_description'],
        )
        self.assertEqual(
            stage2_cases[0]['stage2_description'],
            experiences[0]['case_verbalization']['stage2_description'],
        )
        self.assertNotIn('description', stage1_cases[0])
        self.assertNotIn('description', stage2_cases[0])
        self.assertNotIn('distance', stage1_cases[0])
        self.assertNotIn('distance', stage2_cases[0])
        self.assertNotIn('landing_x', stage1_cases[0]['stage1_description'])
        self.assertIn('landing_x', stage2_cases[0]['stage2_description'])

    def test_dual_retrieval_uses_aligned_cached_cases(self):
        row = self.database.data.iloc[-1]
        query = {key: row[key] for key in self.database.retrieval_keys}
        result = self.database.retrieve_dual(query, normal_top_k=2, miss_top_k=1)
        state_dict = self.database._build_static_experience(row)
        with patch.object(NaiveRAGVerbalizer, 'build_case_verbalization', side_effect=AssertionError):
            context = NaiveRAGVerbalizer().build_dual_prompt_context_nl(state_dict, result)
        self.assertEqual(len(context['normal_cases']), 2)
        self.assertEqual(len(context['miss_cases']), 1)
        self.assertIn('MISSED SHOT', context['miss_cases'][0]['stage2_description'])

    def test_two_stage_retrieval_uses_eight_features_and_filters_before_ranking(self):
        self.assertEqual(self.database.continuous_keys, [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y',
            'ball_distance_x', 'ball_distance_y',
        ])
        row = self.database.data.iloc[1]
        query = {key: row[key] for key in self.database.retrieval_keys}
        stage1 = self.database.retrieve_stage1(
            query, normal_top_k=5, miss_top_k=5, launch=False
        )
        self.assertTrue(stage1['normal_cases'])
        self.assertTrue(stage1['miss_cases'])
        self.assertEqual(
            {int(case['player_type']) for case in stage1['normal_cases']}, {4}
        )
        conditioned = self.database.retrieve_by_shot_type(
            query, 4, top_k=5, distances=stage1['distances']
        )
        self.assertTrue(conditioned)
        self.assertEqual({int(case['player_type']) for case in conditioned}, {4})
        self.assertEqual(self.database.retrieve_by_shot_type(query, 9, top_k=5), [])
        distractor = self.database.data.iloc[1]
        distractor_query = {
            key: distractor[key] for key in self.database.retrieval_keys
        }
        self.assertEqual(
            int(self.database.retrieve(distractor_query, top_k=1)[0]['player_type']),
            4,
        )
        conditioned_serve = self.database.retrieve_by_shot_type(
            distractor_query, 1, top_k=1
        )
        self.assertEqual(int(conditioned_serve[0]['player_type']), 1)
        type_four_id = self.database.data.index[
            self.database.data['player_type'].astype(int) == 4
        ][0]
        self.database.data.loc[type_four_id, 'landing_x'] = 1.5
        self.assertEqual(
            self.database.retrieve_by_shot_type(distractor_query, 4, top_k=1),
            [],
        )

    def test_missed_shot_coordinates_are_canonicalized(self):
        missed = self.database.data[self.database.data['player_type'].astype(int) == 11]
        coordinate_keys = list(self.database.missed_coordinate_keys)
        self.assertTrue(missed[coordinate_keys].isna().all().all())
        self.assertEqual(float(self.database.data.iloc[0]['landing_x']), 0.2)
        self.assertIn(
            'Action: {"shot_type": "Missed shot", "landing_area": null, "landing_x": null, '
            '"landing_y": null, "moving_area": null, "moving_x": null, "moving_y": null}',
            self.database.case_verbalizations[-1]['stage2_description'],
        )
        self.assertNotIn(
            'landing_x', self.database.case_verbalizations[-1]['stage1_description']
        )

    def test_stage_specific_trace_descriptions_match_prompt_evidence(self):
        row = self.database.data.iloc[0]
        query = {key: row[key] for key in self.database.retrieval_keys}
        experience = self.database.retrieve(query, top_k=1)[0]

        stage1 = simplify_naive_rag_experiences([experience], stage='stage1')[0]
        stage2 = simplify_naive_rag_experiences([experience], stage='stage2')[0]

        self.assertEqual(
            stage1['stage1_description'],
            experience['case_verbalization']['stage1_description'],
        )
        self.assertNotIn('landing_x', stage1['stage1_description'])
        self.assertIn('distance', stage1)
        self.assertEqual(
            stage2['stage2_description'],
            experience['case_verbalization']['stage2_description'],
        )
        self.assertIn('landing_x', stage2['stage2_description'])
        self.assertIn('distance', stage2)

    def test_full_history_is_rendered_from_each_agents_viewpoint(self):
        raw_history = [dict(row) for row in make_dataframe().iloc[:3].to_dict('records')]

        player_a = NaiveRAGAgent.__new__(NaiveRAGAgent)
        player_a.player_name = 'Player A'
        a_history = player_a._normalize_history_for_player(raw_history)
        a_text = NaiveRAGVerbalizer().build_trajectory_text(a_history, a_history[-1])
        self.assertIn("Shot 1:\nIt is my turn.", a_text)
        self.assertIn("Shot 2:\nIt is opponent's turn.", a_text)
        self.assertEqual(a_text.count('Shot 2:'), 1)

        player_b = NaiveRAGAgent.__new__(NaiveRAGAgent)
        player_b.player_name = 'Player B'
        b_history = player_b._normalize_history_for_player(raw_history)
        b_text = NaiveRAGVerbalizer().build_trajectory_text(b_history, b_history[-1])
        self.assertIn("Shot 1:\nIt is opponent's turn.", b_text)
        self.assertIn("Shot 2:\nIt is my turn.", b_text)
        self.assertEqual(b_text.count('Shot 2:'), 1)

    def test_round_trip_and_old_schema_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'db.pkl')
            self.database.save(path)
            loaded = NaiveRAGDatabase.load(path)
            self.assertEqual(loaded.schema_version, 8)
            self.assertEqual(loaded.case_verbalizations, self.database.case_verbalizations)

            loaded.case_verbalizations[0], loaded.case_verbalizations[1] = (
                loaded.case_verbalizations[1],
                loaded.case_verbalizations[0],
            )
            with self.assertRaisesRegex(ValueError, 'wrong row'):
                loaded.validate()

            with open(path, 'rb') as f:
                payload = pickle.load(f)
            payload['schema_version'] = 7
            with open(path, 'wb') as f:
                pickle.dump(payload, f)
            with self.assertRaisesRegex(ValueError, 'rally dataset'):
                NaiveRAGDatabase.load(path)

    def test_validation_rejects_feature_and_stats_corruption(self):
        self.database.feature_stores['classic']['continuous'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'wrong rows'):
            self.database.validate()

        database = NaiveRAGDatabase(make_dataframe(), 'Player A', 0)
        database.stats['normal_count'] += 1
        with self.assertRaisesRegex(ValueError, 'stats'):
            database.validate()

        database = NaiveRAGDatabase(make_dataframe(), 'Player A', 0)
        miss_index = database.data.index[database.data['player_type'].astype(int) == 11][0]
        database.data.loc[miss_index, 'landing_x'] = 0.5
        database.data_stores['miss'].loc[miss_index, 'landing_x'] = 0.5
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            database.validate()

    def test_atomic_save_preserves_existing_database_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'db.pkl')
            self.database.save(path)
            with open(path, 'rb') as f:
                original = f.read()
            with patch('Agent.naive_rag.pickle.dump', side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    self.database.save(path)
            with open(path, 'rb') as f:
                self.assertEqual(f.read(), original)


class TimeRAGOfflineVerbalizationTest(unittest.TestCase):
    def setUp(self):
        self.database = TimeRAGDatabase(make_dataframe(), 'Player A', 0, window_size=3)

    def test_offline_text_matches_canonical_renderer(self):
        verbalizer = TimeRAGVerbalizer()
        expected = [
            {
                'window_id': window_id,
                **verbalizer.build_window_verbalization(self.database._build_static_window_data(window_id)),
            }
            for window_id in range(len(self.database.windows))
        ]
        self.assertEqual(self.database.window_verbalizations, expected)
        verbalization = self.database.window_verbalizations[1]
        self.assertNotIn('description', verbalization)
        self.assertEqual(
            verbalization['stage1_description'], TIMERAG_STAGE1_THREE_SHOT_DESCRIPTION
        )
        self.assertEqual(
            verbalization['stage2_description'], TIMERAG_STAGE2_THREE_SHOT_DESCRIPTION
        )

    def test_retrieved_stage_cases_do_not_call_renderer(self):
        query_continuous = np.asarray(self.database.windows_continuous[0], dtype=np.float32)
        query_discrete = np.asarray(self.database.windows_discrete[0], dtype=np.float32)
        windows = self.database.retrieve(query_continuous, query_discrete, top_k=2)
        with patch.object(TimeRAGVerbalizer, 'build_window_verbalization', side_effect=AssertionError):
            verbalizer = TimeRAGVerbalizer()
            stage1_cases = verbalizer.build_stage1_cases(windows)
            stage2_cases = verbalizer.build_stage2_cases(windows)
        self.assertEqual([case['index'] for case in stage1_cases], [1, 2])
        self.assertEqual(
            stage1_cases[0]['stage1_description'],
            windows[0]['case_verbalization']['stage1_description'],
        )
        self.assertEqual(
            stage2_cases[0]['stage2_description'],
            windows[0]['case_verbalization']['stage2_description'],
        )
        self.assertNotIn('description', stage1_cases[0])
        self.assertNotIn('description', stage2_cases[0])
        self.assertNotIn('distance', stage1_cases[0])
        self.assertNotIn('distance', stage2_cases[0])
        self.assertNotIn('Action:', stage1_cases[0]['stage1_description'])
        self.assertIn('Action:', stage2_cases[0]['stage2_description'])

    def test_dual_retrieval_uses_aligned_cached_cases(self):
        query_continuous = np.asarray(self.database.windows_continuous[-1], dtype=np.float32)
        query_discrete = np.asarray(self.database.windows_discrete[-1], dtype=np.float32)
        result = self.database.retrieve_dual(
            query_continuous,
            query_discrete,
            normal_top_k=2,
            miss_top_k=1,
        )
        state_dict = result['miss_cases'][0]['window_states'][-1]
        with patch.object(TimeRAGVerbalizer, 'build_window_verbalization', side_effect=AssertionError):
            context = TimeRAGVerbalizer().build_dual_prompt_context_nl(state_dict, result)
        self.assertEqual(len(context['normal_cases']), 2)
        self.assertEqual(len(context['miss_cases']), 1)
        self.assertIn('MISSED SHOT', context['miss_cases'][0]['stage2_description'])

    def test_two_stage_retrieval_uses_eight_features_and_exact_shot_windows(self):
        self.assertEqual(self.database.continuous_keys, [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y',
            'ball_distance_x', 'ball_distance_y',
        ])
        query_continuous = np.asarray(self.database.windows_continuous[1], dtype=np.float32)
        query_discrete = np.asarray(self.database.windows_discrete[1], dtype=np.float32)
        stage1 = self.database.retrieve_stage1(
            query_continuous,
            query_discrete,
            normal_top_k=5,
            miss_top_k=5,
            launch=False,
        )
        self.assertTrue(stage1['normal_cases'])
        self.assertTrue(stage1['miss_cases'])
        self.assertEqual(
            {int(case['action']['player_type']) for case in stage1['normal_cases']}, {4}
        )
        conditioned = self.database.retrieve_by_shot_type(
            query_continuous,
            query_discrete,
            4,
            top_k=5,
            distances=stage1['distances'],
        )
        self.assertEqual(
            {int(case['action']['player_type']) for case in conditioned}, {4}
        )
        self.assertEqual(
            self.database.retrieve_by_shot_type(
                query_continuous, query_discrete, 9, top_k=5
            ),
            [],
        )
        self.assertEqual(
            int(self.database.retrieve(
                query_continuous, query_discrete, top_k=1
            )[0]['action']['player_type']),
            4,
        )
        conditioned_serve = self.database.retrieve_by_shot_type(
            query_continuous, query_discrete, 1, top_k=1
        )
        self.assertEqual(int(conditioned_serve[0]['action']['player_type']), 1)
        type_four_id = next(
            window_id
            for window_id, action in enumerate(self.database.windows_actions)
            if int(action['player_type']) == 4
        )
        self.database.windows_actions[type_four_id]['landing_x'] = 1.5
        self.assertEqual(
            self.database.retrieve_by_shot_type(
                query_continuous, query_discrete, 4, top_k=1
            ),
            [],
        )

    def test_missed_shot_coordinates_are_canonicalized(self):
        coordinate_keys = self.database.missed_coordinate_keys
        missed_actions = [
            action for action in self.database.windows_actions
            if int(float(action.get('player_type', 0))) == 11
        ]
        self.assertTrue(missed_actions)
        self.assertTrue(all(
            all(action[key] is None for key in coordinate_keys)
            for action in missed_actions
        ))
        missed_states = [
            state
            for window in self.database.windows
            for state in window
            if state is not None and int(float(state.get('player_type', 0))) == 11
        ]
        self.assertTrue(missed_states)
        self.assertTrue(all(
            all(pd.isna(state[key]) for key in coordinate_keys)
            for state in missed_states
        ))
        self.assertIn(
            'Action: {"shot_type": "Missed shot", "landing_area": null, "landing_x": null, '
            '"landing_y": null, "moving_area": null, "moving_x": null, "moving_y": null}',
            self.database.window_verbalizations[-1]['stage2_description'],
        )
        self.assertNotIn(
            'landing_x', self.database.window_verbalizations[-1]['stage1_description']
        )

    def test_stage_specific_trace_hides_only_final_window_placement(self):
        query_continuous = np.asarray(self.database.windows_continuous[1], dtype=np.float32)
        query_discrete = np.asarray(self.database.windows_discrete[1], dtype=np.float32)
        window = self.database.retrieve(query_continuous, query_discrete, top_k=1)[0]

        stage1 = simplify_timerag_windows([window], stage='stage1')[0]
        stage2 = simplify_timerag_windows([window], stage='stage2')[0]

        self.assertEqual(
            stage1['stage1_description'],
            window['case_verbalization']['stage1_description'],
        )
        self.assertIn('Opponent moves to Mid Center', stage1['stage1_description'])
        self.assertNotIn('Action:', stage1['stage1_description'])
        self.assertIn('distance', stage1)
        self.assertEqual(
            stage2['stage2_description'],
            window['case_verbalization']['stage2_description'],
        )
        self.assertIn('Action:', stage2['stage2_description'])
        self.assertIn('distance', stage2)

    def test_round_trip_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'db.pkl')
            self.database.save(path)
            loaded = TimeRAGDatabase.load(path)
            self.assertEqual(loaded.schema_version, 10)
            self.assertEqual(loaded.window_verbalizations, self.database.window_verbalizations)

            loaded.window_verbalizations[0], loaded.window_verbalizations[1] = (
                loaded.window_verbalizations[1],
                loaded.window_verbalizations[0],
            )
            with self.assertRaisesRegex(ValueError, 'wrong window'):
                loaded.validate()

            loaded = TimeRAGDatabase.load(path)

            loaded.window_verbalizations.pop()
            with self.assertRaisesRegex(ValueError, 'not aligned'):
                loaded.validate()

            with open(path, 'rb') as f:
                payload = pickle.load(f)
            payload['schema_version'] = 9
            with open(path, 'wb') as f:
                pickle.dump(payload, f)
            with self.assertRaisesRegex(ValueError, 'rally dataset'):
                TimeRAGDatabase.load(path)

    def test_validation_rejects_feature_and_stats_corruption(self):
        self.database.windows_continuous[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'wrong windows'):
            self.database.validate()

        database = TimeRAGDatabase(make_dataframe(), 'Player A', 0, window_size=3)
        database.stats['miss_count'] += 1
        with self.assertRaisesRegex(ValueError, 'stats'):
            database.validate()

        database = TimeRAGDatabase(make_dataframe(), 'Player A', 0, window_size=3)
        miss_window_id = database.window_stores['miss'][0]
        database.windows_actions[miss_window_id]['landing_x'] = 0.5
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            database.validate()

    def test_atomic_save_preserves_existing_database_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'db.pkl')
            self.database.save(path)
            with open(path, 'rb') as f:
                original = f.read()
            with patch('Agent.timerag.pickle.dump', side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    self.database.save(path)
            with open(path, 'rb') as f:
                self.assertEqual(f.read(), original)


if __name__ == '__main__':
    unittest.main()
