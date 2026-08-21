import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from Scripts.single_player_eval import eval_all_rallies
from Scripts.single_player_eval.trajectory_store import (
    completed_prompt_trace_keys,
    completed_rally_keys,
)


class DummyEvaluator:
    def simulate_rally(self, record):
        return {
            **record,
            'evaluation_unit': 'physical_rally',
            'target_player': None,
            'opponent_player': None,
            'gt_sequence': [],
            'pred_sequence': [],
            'rally_length_gt': 0,
            'rally_length_pred': 0,
            'rollout_limit': 2,
            'termination_reason': 'max_rollout_reached',
            'simulation_ended_early': False,
        }


class FailingEvaluator:
    def simulate_rally(self, record):
        raise RuntimeError('test failure')


class TracedDummyEvaluator(DummyEvaluator):
    def simulate_rally(self, record):
        trajectory = super().simulate_rally(record)
        trajectory['_prompt_traces'] = [{
            'agent': 'pure_llm',
            'player': 'A',
            'match_id': 7,
            'set': 1,
            'rally': 2,
            'ball_round': 1,
            'score': [0, 0],
            'launch': True,
            'input': {},
            'retrievals': [],
            'generations': [{
                'stage': 'generation',
                'user_prompt': 'prompt',
                'parsed_response': {'shot_type': 'Serve short'},
                'raw_output': '{}',
                'usage': {},
                'latency_ms': 1,
            }],
            'final_action': {'shot_type': 1},
            'reasoning': '',
            'action_source': 'llm',
            'fallback_source': None,
        }]
        return trajectory


class EvalAllRalliesResumeTest(unittest.TestCase):
    def setUp(self):
        eval_all_rallies._WORKER_PROMPT_TRACE_DIR = None

    def record(self):
        return {
            'match_id': 7,
            'set': 1,
            'rally': 2,
            'players': ['A', 'B'],
            'player_profiles': {},
        }

    def test_worker_atomically_saves_one_physical_rally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            eval_all_rallies._WORKER_EVALUATOR = DummyEvaluator()
            eval_all_rallies._WORKER_OUTPUT_DIR = temp_dir
            eval_all_rallies._WORKER_METADATA = {'config_fingerprint': 'abc'}

            result = eval_all_rallies._run_parallel_rally(self.record())
            completed, invalid = completed_rally_keys(temp_dir, 'abc')

        self.assertEqual(result['status'], 'success')
        self.assertEqual(completed, {(7, 1, 2)})
        self.assertEqual(invalid, [])

    def test_worker_saves_prompt_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rally_dir = os.path.join(temp_dir, 'rallies')
            trace_dir = os.path.join(temp_dir, 'prompt_traces')
            eval_all_rallies._WORKER_EVALUATOR = TracedDummyEvaluator()
            eval_all_rallies._WORKER_OUTPUT_DIR = rally_dir
            eval_all_rallies._WORKER_PROMPT_TRACE_DIR = trace_dir
            eval_all_rallies._WORKER_METADATA = {'config_fingerprint': 'abc'}

            result = eval_all_rallies._run_parallel_rally(self.record())
            trajectories, _ = completed_rally_keys(rally_dir, 'abc')
            traces, invalid = completed_prompt_trace_keys(trace_dir, 'abc')
            completed, _, _ = eval_all_rallies._completed_output_keys(
                rally_dir, trace_dir, 'abc'
            )

        self.assertEqual(result['status'], 'success')
        self.assertEqual(trajectories & traces, {(7, 1, 2)})
        self.assertEqual(completed, {(7, 1, 2)})
        self.assertEqual(invalid, [])
        self.assertTrue(result['prompt_trace_path'].endswith('.json'))

    def test_traced_completion_requires_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rally_dir = os.path.join(temp_dir, 'rallies')
            trace_dir = os.path.join(temp_dir, 'prompt_traces')
            eval_all_rallies._WORKER_EVALUATOR = DummyEvaluator()
            eval_all_rallies._WORKER_OUTPUT_DIR = rally_dir
            eval_all_rallies._WORKER_METADATA = {'config_fingerprint': 'abc'}
            result = eval_all_rallies._run_parallel_rally(self.record())
            completed, invalid_trajectories, invalid_traces = (
                eval_all_rallies._completed_output_keys(rally_dir, trace_dir, 'abc')
            )

        self.assertEqual(result['status'], 'success')
        self.assertEqual(completed, set())
        self.assertEqual(invalid_trajectories, [])
        self.assertEqual(invalid_traces, [])

    def test_failure_record_can_be_cleared_after_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            eval_all_rallies._WORKER_EVALUATOR = FailingEvaluator()
            eval_all_rallies._WORKER_OUTPUT_DIR = temp_dir
            eval_all_rallies._WORKER_METADATA = {'config_fingerprint': 'abc'}
            result = eval_all_rallies._run_parallel_rally(self.record())
            failure_dir = os.path.join(temp_dir, 'failures')
            eval_all_rallies._write_failure(failure_dir, result)
            failure_path = eval_all_rallies._failure_path(failure_dir, result['key'])
            with open(failure_path, encoding='utf-8') as handle:
                failure = json.load(handle)
            self.assertEqual(failure['error'], 'test failure')

            eval_all_rallies._clear_failure(failure_dir, result['key'])
            self.assertFalse(os.path.exists(failure_path))

    def test_config_fingerprint_is_order_independent(self):
        first = eval_all_rallies._fingerprint({'a': 1, 'b': 2})
        second = eval_all_rallies._fingerprint({'b': 2, 'a': 1})
        changed = eval_all_rallies._fingerprint({'a': 1, 'b': 3})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_pure_llm_supports_instruction_modes(self):
        self.assertEqual(
            eval_all_rallies._resolve_instruction_mode('pure_llm', None, 'instruct'),
            'instruct',
        )
        self.assertEqual(
            eval_all_rallies._resolve_instruction_mode(
                'pure_llm', 'no_instruct', 'instruct'
            ),
            'no_instruct',
        )

    def test_case_grounding_resolves_only_for_rag_algorithms(self):
        self.assertTrue(
            eval_all_rallies._resolve_case_grounding('naive_rag', None, True)
        )
        self.assertFalse(
            eval_all_rallies._resolve_case_grounding('naive_rag', False, True)
        )
        self.assertFalse(
            eval_all_rallies._resolve_case_grounding('pure_llm', None, True)
        )
        self.assertFalse(eval_all_rallies._resolve_case_grounding('bc', None, True))
        self.assertFalse(eval_all_rallies._resolve_case_grounding(
            'naive_rag_single_stage', None, True
        ))

    def test_case_grounding_cli_is_tristate_and_rag_only(self):
        args = eval_all_rallies.parse_args(['--algorithm', 'naive_rag'])
        self.assertIsNone(args.case_grounding)
        args = eval_all_rallies.parse_args([
            '--algorithm', 'naive_rag', '--no-case-grounding',
        ])
        self.assertFalse(args.case_grounding)
        with self.assertRaises(SystemExit):
            eval_all_rallies.parse_args(['--algorithm', 'pure_llm', '--case-grounding'])

    def test_prompt_trace_and_progress_cli_defaults(self):
        args = eval_all_rallies.parse_args(['--algorithm', 'naive_rag'])
        self.assertFalse(args.save_prompt_traces)
        self.assertFalse(args.no_progress)
        args = eval_all_rallies.parse_args([
            '--algorithm', 'naive_rag', '--save-prompt-traces', '--no-progress',
        ])
        self.assertTrue(args.save_prompt_traces)
        self.assertTrue(args.no_progress)
        with self.assertRaises(SystemExit):
            eval_all_rallies.parse_args(['--algorithm', 'bc', '--save-prompt-traces'])

    def test_disabled_case_grounding_gets_separate_output_label(self):
        grounded = SimpleNamespace(
            algorithm='twin', instruction_mode='instruct',
            instruction_variant='original', case_grounding=True,
        )
        self.assertEqual(eval_all_rallies._output_label(grounded), 'rally_instruct')
        grounded.case_grounding = False
        self.assertEqual(
            eval_all_rallies._output_label(grounded),
            'rally_instruct_no_case_grounding',
        )

    def test_single_stage_rag_settings_use_one_direct_action_contract(self):
        dataset = {'split_sha256': 'split', 'profiles_sha256': 'profiles'}
        for algorithm in ('naive_rag_single_stage', 'timerag_single_stage'):
            with self.subTest(algorithm=algorithm):
                args = SimpleNamespace(
                    algorithm=algorithm,
                    instruction_mode='no_instruct',
                    instruction_variant='original',
                    seed=42,
                    device='cpu',
                    limit=None,
                    conditioned_top_k=7,
                    window_size=5,
                    case_grounding=False,
                )
                settings = eval_all_rallies._safe_settings(
                    eval_all_rallies.load_config(), args, dataset
                )

                self.assertEqual(settings['action_architecture'], 'single_stage_v1')
                self.assertEqual(settings['retrieval_architecture'], 'single_stage_v1')
                self.assertEqual(set(settings['prompt_sha256']), {'system', 'action'})
                self.assertEqual(
                    set(settings['structured_output_schema_sha256']), {'action'}
                )
                self.assertNotIn('stage2_conditioned_top_k', settings)
                args.instruction_mode = 'instruct'
                instructed = eval_all_rallies._safe_settings(
                    eval_all_rallies.load_config(), args, dataset
                )
                self.assertEqual(instructed['instruction_mode'], 'instruct')
                self.assertFalse(instructed['case_grounding'])
                self.assertNotEqual(
                    settings['prompt_sha256']['action'],
                    instructed['prompt_sha256']['action'],
                )
                self.assertEqual(
                    set(instructed['structured_output_schema_sha256']), {'action'}
                )
                self.assertNotIn('stage2_conditioned_top_k', instructed)

    def test_no_instruct_keeps_rallies_without_profiles(self):
        record = self.record()
        dataset = {'test': [record]}
        self.assertEqual(
            eval_all_rallies._evaluation_records('pure_llm', dataset),
            [record],
        )
        self.assertEqual(
            eval_all_rallies._evaluation_records('naive_rag', dataset),
            [record],
        )
        self.assertEqual(
            eval_all_rallies._evaluation_records('twin', dataset, 'no_instruct'),
            [record],
        )
        self.assertEqual(
            eval_all_rallies._evaluation_records('twin', dataset, 'instruct'),
            [],
        )
        self.assertEqual(
            eval_all_rallies._evaluation_records('pure_llm', dataset, 'instruct'),
            [],
        )

    def test_instruction_variant_is_applied_without_mutating_source_record(self):
        record = self.record()
        record['player_profiles'] = {
            player: {
                'cluster_id': index,
                'label': f'Original {player}',
                'description': f'Original description {player}',
                'instruction': f'Original instruction {player}',
                'instruction_variants': {
                    'random_other': {
                        'label': f'Other {player}',
                        'description': f'Other description {player}',
                        'instruction': f'Other instruction {player}',
                        'source': 'same_player_cluster',
                        'target_cluster_id': index + 2,
                        'seed': 17,
                    },
                },
            }
            for index, player in enumerate(record['players'])
        }

        transformed = eval_all_rallies._apply_instruction_variant(
            [record], 'random_other'
        )

        source_profile = record['player_profiles']['A']
        applied_profile = transformed[0]['player_profiles']['A']
        self.assertEqual(source_profile['instruction'], 'Original instruction A')
        self.assertEqual(applied_profile['instruction'], 'Other instruction A')
        self.assertEqual(applied_profile['original_instruction'], 'Original instruction A')
        self.assertEqual(applied_profile['target_cluster_id'], 2)
        self.assertEqual(applied_profile['variant_seed'], 17)
        self.assertEqual(applied_profile['instruction_variant'], 'random_other')
        self.assertNotIn('instruction_variants', applied_profile)

    def test_missing_instruction_variant_is_rejected(self):
        record = self.record()
        record['player_profiles'] = {
            player: {'instruction_variants': {}}
            for player in record['players']
        }
        with self.assertRaisesRegex(ValueError, "Missing 'rewrite' instruction variant"):
            eval_all_rallies._apply_instruction_variant([record], 'rewrite')

    def test_pure_llm_settings_exclude_retrieval_contract(self):
        args = SimpleNamespace(
            algorithm='pure_llm',
            instruction_mode='no_instruct',
            seed=42,
            device='cpu',
            limit=None,
            conditioned_top_k=10,
            window_size=5,
            case_grounding=False,
        )
        dataset = {'split_sha256': 'split', 'profiles_sha256': 'profiles'}
        settings = eval_all_rallies._safe_settings(
            eval_all_rallies.load_config(), args, dataset
        )
        self.assertEqual(settings['prompt_contract'], 'fixed_zero_shot')
        self.assertFalse(settings['tactical_profile_input'])
        self.assertFalse(settings['case_grounding'])
        self.assertNotIn('retrieval_architecture', settings)
        self.assertNotIn('database_manifest_sha256', settings)
        self.assertEqual(set(settings['prompt_sha256']), {'system', 'user'})
        self.assertEqual(
            set(settings['structured_output_schema_sha256']), {'action'}
        )

        args.instruction_mode = 'instruct'
        instructed_settings = eval_all_rallies._safe_settings(
            eval_all_rallies.load_config(), args, dataset
        )
        self.assertEqual(
            instructed_settings['prompt_contract'],
            'zero_shot_with_tactical_profile',
        )
        self.assertTrue(instructed_settings['tactical_profile_input'])
        self.assertNotEqual(
            settings['prompt_sha256']['user'],
            instructed_settings['prompt_sha256']['user'],
        )

    def test_twin_settings_fingerprint_aspect_and_two_stage_contracts(self):
        args = SimpleNamespace(
            algorithm='twin',
            instruction_mode='no_instruct',
            seed=42,
            device='cpu',
            limit=None,
            conditioned_top_k=10,
            window_size=5,
            lambda_weight=0.5,
            case_grounding=True,
        )
        dataset = {'split_sha256': 'split', 'profiles_sha256': 'profiles'}
        settings = eval_all_rallies._safe_settings(
            eval_all_rallies.load_config(), args, dataset
        )
        self.assertEqual(settings['retrieval_architecture'], 'twin_aspect_two_stage_v1')
        self.assertEqual(settings['action_architecture'], 'two_stage_v1')
        self.assertEqual(settings['lambda_weight'], 0.5)
        self.assertTrue(settings['case_grounding'])
        self.assertEqual(settings['semantic_start_ball_round'], 3)
        self.assertEqual(
            settings['retrieval_schedule'], 'state_only_before_semantic_start'
        )
        self.assertEqual(settings['early_round_retrieval_metric'], 'twin_state_v1')
        self.assertEqual(
            settings['online_terminal_aspect_repair'], 'extend_one_round'
        )
        self.assertEqual(
            set(settings['prompt_sha256']),
            {'system', 'aspect', 'stage1', 'stage2'},
        )
        self.assertIn('embedding_manifest_sha256', settings)
        self.assertEqual(
            set(settings['structured_output_schema_sha256']), {'stage1', 'stage2'}
        )

    def test_single_stage_twin_fingerprint_excludes_stage2_contract(self):
        args = SimpleNamespace(
            algorithm='twin_single_stage',
            instruction_mode='no_instruct',
            seed=42,
            device='cpu',
            limit=None,
            conditioned_top_k=10,
            window_size=5,
            lambda_weight=0.5,
            case_grounding=True,
        )
        dataset = {'split_sha256': 'split', 'profiles_sha256': 'profiles'}
        settings = eval_all_rallies._safe_settings(
            eval_all_rallies.load_config(), args, dataset
        )

        self.assertEqual(
            settings['retrieval_architecture'],
            'twin_aspect_single_stage_v1',
        )
        self.assertEqual(settings['action_architecture'], 'single_stage_v1')
        self.assertEqual(settings['semantic_start_ball_round'], 3)
        self.assertEqual(
            settings['retrieval_schedule'], 'state_only_before_semantic_start'
        )
        self.assertEqual(
            settings['online_terminal_aspect_repair'], 'extend_one_round'
        )
        self.assertNotIn('stage2_conditioned_top_k', settings)
        self.assertEqual(
            set(settings['prompt_sha256']),
            {'system', 'aspect', 'action'},
        )
        self.assertIn('embedding_manifest_sha256', settings)
        self.assertEqual(
            set(settings['structured_output_schema_sha256']), {'action'}
        )

        args.instruction_mode = 'instruct'
        args.case_grounding = False
        direct_instruct = eval_all_rallies._safe_settings(
            eval_all_rallies.load_config(), args, dataset
        )
        args.instruction_mode = 'no_instruct'
        direct_no_instruct = eval_all_rallies._safe_settings(
            eval_all_rallies.load_config(), args, dataset
        )
        self.assertFalse(direct_instruct['case_grounding'])
        self.assertNotEqual(
            direct_instruct['prompt_sha256']['action'],
            direct_no_instruct['prompt_sha256']['action'],
        )


if __name__ == '__main__':
    unittest.main()
