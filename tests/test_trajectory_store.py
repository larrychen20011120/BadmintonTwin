import json
import os
import pickle
import tempfile
import unittest

from Scripts.single_player_eval.trajectory_store import (
    build_rollout_metadata,
    completed_prompt_trace_keys,
    completed_rally_keys,
    create_run_directory,
    load_trajectories,
    save_prompt_trace,
    save_trajectory,
    save_trajectories,
)


class TrajectoryStoreTest(unittest.TestCase):
    @staticmethod
    def decision_trace():
        stage2_case = {
            'database_player': 'A',
            'stage': 2,
            'store': 'shot_conditioned',
            'case_id': 3,
            'conditioned_shot_type': 1,
        }
        return {
            'schema_version': 2,
            'architecture': 'two_stage_v1',
            'stage1': {
                'normal_case_ids': [],
                'miss_case_ids': [],
                'outcome': 'returned',
                'shot_type': 1,
                'rationale': 'Reachable serve.',
                'source': 'llm',
            },
            'stage2': {
                'status': 'completed',
                'conditioned_shot_type': 1,
                'case_ids': [stage2_case],
                'rationale': 'Conditioned placement.',
                'source': 'llm',
            },
            'case_ids': [stage2_case],
            'rationale': 'Combined rationale.',
            'action': {
                'shot_type': 1,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            },
            'action_source': 'llm+llm',
        }

    @staticmethod
    def decision_trace_v4():
        normal_case = {
            'database_player': 'A', 'stage': 1, 'store': 'normal',
            'case_id': 1, 'rank': 1,
        }
        miss_case = {
            'database_player': 'A', 'stage': 1, 'store': 'miss',
            'case_id': 2, 'rank': 1,
        }
        stage2_case = {
            'database_player': 'A', 'stage': 2, 'store': 'shot_conditioned',
            'case_id': 3, 'rank': 1, 'conditioned_shot_type': 1,
        }
        stage1_tags = [
            {**normal_case, 'source': 'success', 'tags': ['incoming_shot_match']},
            {**miss_case, 'source': 'miss', 'tags': []},
        ]
        stage2_tags = [{**stage2_case, 'tags': ['landing_area_pattern']}]
        return {
            'schema_version': 4,
            'architecture': 'two_stage_v1',
            'case_grounding': True,
            'stage1': {
                'normal_case_ids': [normal_case],
                'miss_case_ids': [miss_case],
                'case_tags': stage1_tags,
                'outcome': 'returned',
                'shot_type': 1,
                'evidence_summary': 'Case assessment.',
                'source': 'llm',
            },
            'stage2': {
                'status': 'completed',
                'conditioned_shot_type': 1,
                'case_ids': [stage2_case],
                'case_tags': stage2_tags,
                'evidence_summary': 'Placement assessment.',
                'source': 'llm',
            },
            'case_ids': [normal_case, miss_case, stage2_case],
            'case_tags': stage1_tags + stage2_tags,
            'evidence_summary': 'Case assessment. Placement assessment.',
            'decision_summary': 'Returned Serve short.',
            'action': {
                'shot_type': 1,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            },
            'action_source': 'llm+llm',
        }

    @staticmethod
    def decision_trace_v5():
        normal_case = {
            'database_player': 'A', 'stage': 1, 'store': 'normal',
            'case_id': 1, 'rank': 1,
        }
        miss_case = {
            'database_player': 'A', 'stage': 1, 'store': 'miss',
            'case_id': 2, 'rank': 1,
        }
        stage2_case = {
            'database_player': 'A', 'stage': 2, 'store': 'shot_conditioned',
            'case_id': 3, 'rank': 1, 'conditioned_shot_type': 1,
        }
        stage1_notes = [
            {**normal_case, 'source': 'success',
             'note': 'Reachable arrival matches this successful historical serve.'},
            {**miss_case, 'source': 'miss',
             'note': 'Historical miss had substantially worse shuttle reachability.'},
        ]
        stage2_notes = [
            {**stage2_case,
             'note': 'Landing and recovery positions follow the same pattern.'},
        ]
        stage1_summary = 'Successful reachability evidence outweighs the less comparable missed serve.'
        stage2_summary = 'The conditioned case supports the selected landing and recovery positions.'
        return {
            'schema_version': 5,
            'architecture': 'two_stage_v1',
            'case_grounding': True,
            'stage1': {
                'normal_case_ids': [normal_case],
                'miss_case_ids': [miss_case],
                'brief_notes': stage1_notes,
                'outcome': 'returned',
                'shot_type': 1,
                'summary': stage1_summary,
                'source': 'llm',
            },
            'stage2': {
                'status': 'completed',
                'conditioned_shot_type': 1,
                'case_ids': [stage2_case],
                'brief_notes': stage2_notes,
                'summary': stage2_summary,
                'source': 'llm',
            },
            'case_ids': [normal_case, miss_case, stage2_case],
            'brief_notes': stage1_notes + stage2_notes,
            'summary': f'{stage1_summary} {stage2_summary}',
            'action': {
                'shot_type': 1,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            },
            'action_source': 'llm+llm',
        }

    @staticmethod
    def decision_trace_v6():
        normal_case = {
            'database_player': 'A', 'stage': 1, 'store': 'normal',
            'case_id': 1, 'rank': 1,
        }
        miss_case = {
            'database_player': 'A', 'stage': 1, 'store': 'miss',
            'case_id': 2, 'rank': 1,
        }
        stage2_case = {
            'database_player': 'A', 'stage': 2, 'store': 'shot_conditioned',
            'case_id': 3, 'rank': 1, 'conditioned_shot_type': 1,
        }
        stage1_assessments = [
            {**normal_case, 'source': 'success', 'impact': 'support',
             'factors': ['same serve', 'same position']},
            {**miss_case, 'source': 'miss', 'impact': 'irrelevant',
             'factors': ['lower reachability']},
        ]
        stage2_assessments = [
            {**stage2_case, 'impact': 'support',
             'factors': ['same landing', 'same recovery']},
        ]
        return {
            'schema_version': 6,
            'architecture': 'two_stage_v1',
            'case_grounding': True,
            'stage1': {
                'normal_case_ids': [normal_case],
                'miss_case_ids': [miss_case],
                'case_assessments': stage1_assessments,
                'outcome': 'returned',
                'shot_type': 1,
                'source': 'llm',
            },
            'stage2': {
                'status': 'completed',
                'conditioned_shot_type': 1,
                'case_ids': [stage2_case],
                'case_assessments': stage2_assessments,
                'source': 'llm',
            },
            'case_ids': [normal_case, miss_case, stage2_case],
            'case_assessments': stage1_assessments + stage2_assessments,
            'action': {
                'shot_type': 1,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            },
            'action_source': 'llm+llm',
        }

    @classmethod
    def decision_trace_v7(cls, semantic_enabled=False):
        trace = cls.decision_trace_v6()
        trace['schema_version'] = 7
        trace['retrieval_mode'] = (
            'state_semantic' if semantic_enabled else 'state_only'
        )
        reference_field = (
            'semantic_similarity' if semantic_enabled else 'distance'
        )

        def add_reference_field(value):
            if isinstance(value, dict):
                if 'database_player' in value:
                    value[reference_field] = 0.75
                for nested in value.values():
                    add_reference_field(nested)
            elif isinstance(value, list):
                for nested in value:
                    add_reference_field(nested)

        add_reference_field(trace)
        if semantic_enabled:
            trace['pre_stage'] = {
                'architecture': 'twin_aspect_v1',
                'semantic_enabled': True,
                'activation_ball_round': 1,
                'source': 'llm',
                'semantic_query_text': 'Current Twin semantic query.',
                'aspects': {
                    'my_tactical_pattern': 'Controlled return',
                    'my_tempo_movement_dynamics': 'Stable recovery',
                    'opponent_tactical_pattern': 'Rear pressure',
                    'opponent_tempo_movement_dynamics': 'Delayed recovery',
                },
            }
            trace['action_source'] = 'llm+llm+llm'
        else:
            trace['pre_stage'] = {
                'architecture': 'twin_aspect_v1',
                'semantic_enabled': False,
                'activation_ball_round': 3,
                'source': 'skipped_early_round',
            }
            trace['action_source'] = 'state_only+llm+llm'
        return trace
    def trajectory(self, target_player='A'):
        gt_sequence = [{'player': 'A', 'ball_round': 1, 'shot_type': 1}]
        pred_sequence = [{
            'player': 'A',
            'ball_round': 1,
            'shot_type': 1,
            'landing_x': 0.0,
            'landing_y': 0.0,
            'moving_x': 0.0,
            'moving_y': 0.0,
        }]
        return {
            'match_id': 1,
            'set': 1,
            'rally': 2,
            'players': ['A', 'B'],
            'target_player': target_player,
            'gt_sequence': gt_sequence,
            'pred_sequence': pred_sequence,
            **build_rollout_metadata(gt_sequence, pred_sequence, 'natural_terminate'),
        }

    def test_saves_one_json_object_per_rally_simulation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, generation = create_run_directory(temp_dir)
            rally_dir = os.path.join(run_dir, 'rallies')
            trajectories = [self.trajectory('A'), self.trajectory('B')]

            paths = save_trajectories(
                trajectories,
                rally_dir,
                metadata={'algorithm': 'random', 'generation': generation},
            )

            self.assertEqual(len(paths), 2)
            self.assertEqual(len(set(paths)), 2)
            for path in paths:
                with open(path, 'r', encoding='utf-8') as handle:
                    document = json.load(handle)
                self.assertIsInstance(document, dict)
                self.assertEqual(document['schema_version'], 1)
                self.assertEqual(document['rollout_limit'], 3)

            loaded = load_trajectories(rally_dir)
            self.assertEqual({item['target_player'] for item in loaded}, {'A', 'B'})

    def test_loads_legacy_pickle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'legacy.pkl')
            trajectory = self.trajectory()
            with open(path, 'wb') as handle:
                pickle.dump([trajectory], handle)

            self.assertEqual(load_trajectories(path), [trajectory])

    def test_rollout_metadata_marks_only_short_predictions_as_early(self):
        metadata = build_rollout_metadata([1, 2], [1], 'natural_terminate')
        self.assertEqual(metadata['rollout_limit'], 4)
        self.assertTrue(metadata['simulation_ended_early'])

        metadata = build_rollout_metadata([1, 2], [1, 2, 3, 4], 'max_rollout_reached')
        self.assertFalse(metadata['simulation_ended_early'])

    def test_completed_rallies_require_valid_json_and_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            with open(os.path.join(temp_dir, 'broken.json'), 'w', encoding='utf-8') as handle:
                handle.write('{')

            completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, {(1, 1, 2)})
            self.assertEqual(len(invalid), 1)

            completed, invalid = completed_rally_keys(temp_dir, 'different')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 2)

    def test_prompt_trace_sidecar_is_validated_and_excludes_system_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rally_dir = os.path.join(temp_dir, 'rallies')
            trace_dir = os.path.join(temp_dir, 'prompt_traces')
            trajectory = self.trajectory(None)
            trajectory_path = save_trajectory(
                trajectory,
                rally_dir,
                metadata={
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            shot_trace = {
                'agent': 'naive_rag',
                'player': 'A',
                'match_id': 1,
                'set': 1,
                'rally': 2,
                'ball_round': 1,
                'score': [0, 0],
                'launch': True,
                'input': {'shot_type': 'Serve short'},
                'retrievals': [],
                'generations': [{
                    'stage': 'stage1',
                    'user_prompt': 'Choose an action.',
                    'parsed_response': {'outcome': 'returned', 'shot_type': 'Lob'},
                    'raw_output': '{"outcome":"returned","shot_type":"Lob"}',
                    'usage': {'total_tokens': 12},
                    'latency_ms': 8,
                }],
                'final_action': {'shot_type': 8},
                'reasoning': 'Generated action.',
                'action_source': 'llm+llm',
                'fallback_source': None,
            }
            path = save_prompt_trace(
                trajectory,
                [shot_trace],
                trace_dir,
                trajectory_path,
                metadata={
                    'algorithm': 'naive_rag',
                    'config_fingerprint': 'abc',
                    'prompt_sha256': {'system': 'sha256'},
                },
            )
            completed, invalid = completed_prompt_trace_keys(trace_dir, 'abc')
            with open(path, encoding='utf-8') as handle:
                document = json.load(handle)

            self.assertEqual(completed, {(1, 1, 2)})
            self.assertEqual(invalid, [])
            self.assertNotIn('system_prompt', json.dumps(document))
            document['shots'][0]['generations'][0]['system_prompt'] = 'secret'
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(document, handle)
            completed, invalid = completed_prompt_trace_keys(trace_dir, 'abc')
            self.assertEqual(completed, set())
            self.assertEqual(invalid, [path])

    def test_rag_trajectory_requires_per_shot_decision_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(completed, set())
        self.assertEqual(len(invalid), 1)

    def test_rag_trajectory_accepts_consistent_two_stage_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = self.decision_trace()
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_accepts_schema_4_all_case_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = self.decision_trace_v4()
            save_trajectories(
                [trajectory], temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_rejects_schema_4_coverage_and_legacy_fields(self):
        for mutation in ('missing', 'wrong_reference', 'legacy'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trace = self.decision_trace_v4()
                if mutation == 'missing':
                    trace['stage1']['case_tags'].pop()
                    trace['case_tags'] = (
                        trace['stage1']['case_tags'] + trace['stage2']['case_tags']
                    )
                elif mutation == 'wrong_reference':
                    trace['stage2']['case_tags'][0]['case_id'] = 99
                    trace['case_tags'] = (
                        trace['stage1']['case_tags'] + trace['stage2']['case_tags']
                    )
                else:
                    trace['rationale'] = 'Obsolete.'
                trajectory['pred_sequence'][0]['decision'] = trace
                save_trajectories(
                    [trajectory], temp_dir,
                    metadata={
                        'algorithm': 'naive_rag',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 1)

    def test_rag_trajectory_accepts_schema_5_brief_notes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = self.decision_trace_v5()
            save_trajectories(
                [trajectory], temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_rejects_invalid_schema_5_notes_and_summary(self):
        for mutation in ('missing', 'wrong_reference', 'multiline', 'legacy'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trace = self.decision_trace_v5()
                if mutation == 'missing':
                    trace['stage1']['brief_notes'].pop()
                    trace['brief_notes'] = (
                        trace['stage1']['brief_notes'] + trace['stage2']['brief_notes']
                    )
                elif mutation == 'wrong_reference':
                    trace['stage2']['brief_notes'][0]['case_id'] = 99
                    trace['brief_notes'] = (
                        trace['stage1']['brief_notes'] + trace['stage2']['brief_notes']
                    )
                elif mutation == 'multiline':
                    trace['stage1']['summary'] = 'First line.\nSecond line.'
                    trace['summary'] = (
                        trace['stage1']['summary'] + ' ' + trace['stage2']['summary']
                    )
                else:
                    trace['case_tags'] = []
                trajectory['pred_sequence'][0]['decision'] = trace
                save_trajectories(
                    [trajectory], temp_dir,
                    metadata={
                        'algorithm': 'naive_rag',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 1)

    def test_rag_trajectory_accepts_schema_6_case_assessments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = self.decision_trace_v6()
            save_trajectories(
                [trajectory], temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_rejects_invalid_schema_6_assessments(self):
        for mutation in (
            'missing', 'wrong_reference', 'invalid_factor', 'duplicate', 'legacy'
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trace = self.decision_trace_v6()
                if mutation == 'missing':
                    trace['stage1']['case_assessments'].pop()
                    trace['case_assessments'] = (
                        trace['stage1']['case_assessments']
                        + trace['stage2']['case_assessments']
                    )
                elif mutation == 'wrong_reference':
                    trace['stage2']['case_assessments'][0]['case_id'] = 99
                    trace['case_assessments'] = (
                        trace['stage1']['case_assessments']
                        + trace['stage2']['case_assessments']
                    )
                elif mutation == 'invalid_factor':
                    trace['stage1']['case_assessments'][0]['factors'] = [
                        'same shot and position'
                    ]
                    trace['case_assessments'] = (
                        trace['stage1']['case_assessments']
                        + trace['stage2']['case_assessments']
                    )
                elif mutation == 'duplicate':
                    trace['stage1']['case_assessments'][0]['factors'] = [
                        'same shot', 'same shot'
                    ]
                    trace['case_assessments'] = (
                        trace['stage1']['case_assessments']
                        + trace['stage2']['case_assessments']
                    )
                else:
                    trace['summary'] = 'Obsolete.'
                trajectory['pred_sequence'][0]['decision'] = trace
                save_trajectories(
                    [trajectory], temp_dir,
                    metadata={
                        'algorithm': 'naive_rag',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 1)

    def test_rag_trajectory_accepts_optional_selected_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trace = self.decision_trace()
            trace['stage1']['selected_cases'] = [{
                'source': 'success', 'retrieval_rank': 1, 'reason': 'Similar reach.',
            }]
            trace['stage2']['selected_cases'] = [{
                'retrieval_rank': 1, 'reason': 'Similar placement.',
            }]
            trace['selected_cases'] = (
                trace['stage1']['selected_cases'] + trace['stage2']['selected_cases']
            )
            trace['schema_version'] = 3
            trajectory['pred_sequence'][0]['decision'] = trace
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_rejects_non_list_selected_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trace = self.decision_trace()
            trace['stage1']['selected_cases'] = {'retrieval_rank': 1}
            trajectory['pred_sequence'][0]['decision'] = trace
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'naive_rag',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, set())
        self.assertEqual(len(invalid), 1)

    def test_twin_trajectory_requires_and_accepts_aspect_pre_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trace = self.decision_trace()
            trace['pre_stage'] = {
                'architecture': 'twin_aspect_v1',
                'semantic_query_text': 'Current Twin semantic query.',
                'source': 'llm',
            }
            trajectory['pred_sequence'][0]['decision'] = trace
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'twin',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')
        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_twin_v7_accepts_state_only_and_semantic_retrieval_contracts(self):
        for semantic_enabled in (False, True):
            with self.subTest(semantic_enabled=semantic_enabled), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trajectory['pred_sequence'][0]['decision'] = self.decision_trace_v7(
                    semantic_enabled
                )
                save_trajectories(
                    [trajectory], temp_dir,
                    metadata={
                        'algorithm': 'twin',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')

                self.assertEqual(completed, {(1, 1, 2)})
                self.assertEqual(invalid, [])

    def test_twin_v7_rejects_cross_mode_semantic_artifacts(self):
        for mutation in ('semantic_score', 'missing_distance', 'wrong_mode'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trace = self.decision_trace_v7(False)
                trajectory['pred_sequence'][0]['decision'] = trace
                if mutation == 'semantic_score':
                    trace['case_ids'][0]['semantic_similarity'] = 0.8
                elif mutation == 'missing_distance':
                    trace['case_ids'][0].pop('distance')
                else:
                    trace['retrieval_mode'] = 'state_semantic'
                save_trajectories(
                    [trajectory], temp_dir,
                    metadata={
                        'algorithm': 'twin',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')

                self.assertEqual(completed, set())
                self.assertEqual(len(invalid), 1)

    def test_twin_v7_validates_terminal_extension_metadata(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trajectory['pred_sequence'][0]['ball_round'] = 3
                trace = self.decision_trace_v7(True)
                trace['pre_stage']['activation_ball_round'] = 3
                trace['pre_stage']['source'] = 'llm_terminal_extension'
                trace['pre_stage']['repair'] = {
                    'type': 'extend_one_terminal_round',
                    'aspects': ['my_tactical_pattern'],
                    'from_round': 2,
                    'to_round': 3 if valid else 4,
                }
                trace['action_source'] = 'llm_terminal_extension+llm+llm'
                trajectory['pred_sequence'][0]['decision'] = trace
                save_trajectories(
                    [trajectory], temp_dir,
                    metadata={
                        'algorithm': 'twin',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')

                if valid:
                    self.assertEqual(completed, {(1, 1, 2)})
                    self.assertEqual(invalid, [])
                else:
                    self.assertEqual(completed, set())
                    self.assertEqual(len(invalid), 1)

    def test_single_stage_twin_v7_accepts_state_only_contract(self):
        case = {
            'database_player': 'A', 'stage': 1, 'store': 'normal',
            'case_id': 1, 'rank': 1, 'score': 0.9,
            'state_similarity': 0.9, 'distance': 0.1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = {
                'schema_version': 7,
                'architecture': 'single_stage_v1',
                'case_grounding': False,
                'outcome': 'returned',
                'retrieval_mode': 'state_only',
                'pre_stage': {
                    'architecture': 'twin_aspect_v1',
                    'semantic_enabled': False,
                    'activation_ball_round': 3,
                    'source': 'skipped_early_round',
                },
                'retrieval': {
                    'normal_case_ids': [case],
                    'miss_case_ids': [],
                },
                'case_ids': [case],
                'case_assessments': [],
                'action': {
                    'shot_type': 1,
                    'landing_x': 0.0,
                    'landing_y': 0.0,
                    'moving_x': 0.0,
                    'moving_y': 0.0,
                },
                'action_source': 'state_only+llm',
            }
            save_trajectories(
                [trajectory], temp_dir,
                metadata={
                    'algorithm': 'twin_single_stage',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_single_stage_twin_trajectory_accepts_direct_action_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = {
                'schema_version': 2,
                'architecture': 'single_stage_v1',
                'pre_stage': {
                    'architecture': 'twin_aspect_v1',
                    'semantic_query_text': 'Current Twin semantic query.',
                    'source': 'llm',
                },
                'retrieval': {
                    'normal_case_ids': [],
                    'miss_case_ids': [],
                },
                'case_ids': [],
                'rationale': 'One complete action decision.',
                'action': {
                    'shot_type': 1,
                    'landing_x': 0.0,
                    'landing_y': 0.0,
                    'moving_x': 0.0,
                    'moving_y': 0.0,
                },
                'action_source': 'llm+llm',
            }
            save_trajectories(
                [trajectory],
                temp_dir,
                metadata={
                    'algorithm': 'twin_single_stage',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_single_stage_direct_rag_trajectory_accepts_retrieval_trace(self):
        for algorithm in ('naive_rag_single_stage', 'timerag_single_stage'):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                case = {
                    'database_player': 'A',
                    'stage': 1,
                    'store': 'normal',
                    'case_id': 1,
                    'rank': 1,
                    'distance': 0.1,
                }
                trajectory['pred_sequence'][0]['decision'] = {
                    'schema_version': 6,
                    'architecture': 'single_stage_v1',
                    'case_grounding': False,
                    'outcome': 'returned',
                    'retrieval': {
                        'normal_case_ids': [case],
                        'miss_case_ids': [],
                    },
                    'case_ids': [case],
                    'case_assessments': [],
                    'action': {
                        'shot_type': 1,
                        'landing_x': 0.0,
                        'landing_y': 0.0,
                        'moving_x': 0.0,
                        'moving_y': 0.0,
                    },
                    'action_source': 'llm',
                }
                save_trajectories(
                    [trajectory],
                    temp_dir,
                    metadata={
                        'algorithm': algorithm,
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')

                self.assertEqual(completed, {(1, 1, 2)})
                self.assertEqual(invalid, [])

    def test_single_stage_twin_trajectory_accepts_schema_5_brief_notes(self):
        normal_case = {
            'database_player': 'A', 'stage': 1, 'store': 'normal',
            'case_id': 1, 'rank': 1,
        }
        miss_case = {
            'database_player': 'A', 'stage': 1, 'store': 'miss',
            'case_id': 2, 'rank': 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = {
                'schema_version': 5,
                'architecture': 'single_stage_v1',
                'case_grounding': True,
                'pre_stage': {
                    'architecture': 'twin_aspect_v1',
                    'semantic_query_text': 'Current Twin semantic query.',
                    'source': 'llm',
                },
                'retrieval': {
                    'normal_case_ids': [normal_case],
                    'miss_case_ids': [miss_case],
                },
                'case_ids': [normal_case, miss_case],
                'brief_notes': [
                    {**miss_case, 'source': 'miss',
                     'note': 'Historical miss had substantially worse shuttle reachability.'},
                    {**normal_case, 'source': 'success',
                     'note': 'Reachable arrival supports the same complete serve action.'},
                ],
                'summary': 'Successful reachability evidence supports the complete serve action.',
                'action': {
                    'shot_type': 1,
                    'landing_x': 0.0,
                    'landing_y': 0.0,
                    'moving_x': 0.0,
                    'moving_y': 0.0,
                },
                'action_source': 'llm+llm',
            }
            save_trajectories(
                [trajectory], temp_dir,
                metadata={
                    'algorithm': 'twin_single_stage',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_single_stage_twin_trajectory_accepts_schema_6_assessments(self):
        normal_case = {
            'database_player': 'A', 'stage': 1, 'store': 'normal',
            'case_id': 1, 'rank': 1,
        }
        miss_case = {
            'database_player': 'A', 'stage': 1, 'store': 'miss',
            'case_id': 2, 'rank': 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory = self.trajectory(None)
            trajectory['pred_sequence'][0]['decision'] = {
                'schema_version': 6,
                'architecture': 'single_stage_v1',
                'case_grounding': True,
                'pre_stage': {
                    'architecture': 'twin_aspect_v1',
                    'semantic_query_text': 'Current Twin semantic query.',
                    'source': 'llm',
                },
                'retrieval': {
                    'normal_case_ids': [normal_case],
                    'miss_case_ids': [miss_case],
                },
                'case_ids': [normal_case, miss_case],
                'case_assessments': [
                    {**miss_case, 'source': 'miss', 'impact': 'irrelevant',
                     'factors': ['worse reachability']},
                    {**normal_case, 'source': 'success', 'impact': 'support',
                     'factors': ['same serve action']},
                ],
                'action': {
                    'shot_type': 1,
                    'landing_x': 0.0,
                    'landing_y': 0.0,
                    'moving_x': 0.0,
                    'moving_y': 0.0,
                },
                'action_source': 'llm+llm',
            }
            save_trajectories(
                [trajectory], temp_dir,
                metadata={
                    'algorithm': 'twin_single_stage',
                    'evaluation_unit': 'physical_rally',
                    'config_fingerprint': 'abc',
                },
            )
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(completed, {(1, 1, 2)})
        self.assertEqual(invalid, [])

    def test_rag_trajectory_rejects_outer_action_mismatch_and_bad_case_type(self):
        for mutation in ('outer_shot', 'bad_case', 'invalid_shot_type'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                trajectory = self.trajectory(None)
                trace = self.decision_trace()
                trajectory['pred_sequence'][0]['decision'] = trace
                if mutation == 'outer_shot':
                    trajectory['pred_sequence'][0]['shot_type'] = 9
                elif mutation == 'bad_case':
                    trace['stage2']['case_ids'] = [1]
                else:
                    trajectory['pred_sequence'][0]['shot_type'] = 0
                    trace['stage1']['shot_type'] = 0
                    trace['stage2']['conditioned_shot_type'] = 0
                    trace['stage2']['case_ids'][0]['conditioned_shot_type'] = 0
                    trace['action']['shot_type'] = 0
                save_trajectories(
                    [trajectory],
                    temp_dir,
                    metadata={
                        'algorithm': 'naive_rag',
                        'evaluation_unit': 'physical_rally',
                        'config_fingerprint': 'abc',
                    },
                )
                completed, invalid = completed_rally_keys(temp_dir, 'abc')
            self.assertEqual(completed, set())
            self.assertEqual(len(invalid), 1)


if __name__ == '__main__':
    unittest.main()
