import unittest
from unittest.mock import patch

import pandas as pd
import torch

from Scripts.single_player_eval.evaluators.naive_rag_evaluator import NaiveRAGEvaluator
from Scripts.single_player_eval.evaluators.pure_llm_evaluator import PureLLMEvaluator
from Scripts.single_player_eval.evaluators.rallynetv2_evaluator import RallyNetV2Evaluator
from Scripts.single_player_eval.evaluators.random_evaluator import RandomEvaluator
from Scripts.single_player_eval.evaluators.timerag_evaluator import TimeRAGEvaluator
from Scripts.single_player_eval.evaluators.twin_evaluator import TwinEvaluator
from Utils.helper import action_col, state_col


def model_rows():
    rows = []
    for ball_round, player in ((1, 'A'), (2, 'B')):
        row = {column: 0.0 for column in state_col + action_col}
        row.update({
            'match_id': 3,
            'set': 1,
            'rally': 4,
            'ball_round': ball_round,
            'player': player,
            'player_type': 1 if ball_round == 1 else 2,
            'opponent_type': 0 if ball_round == 1 else 1,
        })
        rows.append(row)
    return pd.DataFrame(rows)


class DummyAgent:
    def __init__(self):
        self.calls = 0
        self.last_decision_trace = None
        self.instructions = []

    def reset_history(self):
        self.last_decision_trace = None

    def predict(self, state, info, launch):
        self.calls += 1
        self.instructions.append(info.get('rally_profile_instruction', ''))
        self.last_decision_trace = {
            'schema_version': 2,
            'architecture': 'two_stage_v1',
            'stage1': {
                'normal_case_ids': [{
                    'database_player': info.get('current_player_name', 'A'),
                    'stage': 1,
                    'store': 'normal',
                    'case_id': self.calls,
                }],
                'miss_case_ids': [],
                'outcome': 'returned',
                'shot_type': 1,
                'rationale': 'test rationale',
                'source': 'llm',
            },
            'stage2': {
                'status': 'completed',
                'conditioned_shot_type': 1,
                'case_ids': [{
                    'database_player': info.get('current_player_name', 'A'),
                    'stage': 2,
                    'store': 'shot_conditioned',
                    'case_id': self.calls,
                    'conditioned_shot_type': 1,
                }],
                'rationale': 'test placement rationale',
                'source': 'llm',
            },
            'case_ids': [{
                'database_player': info.get('current_player_name', 'A'),
                'stage': 2,
                'store': 'shot_conditioned',
                'case_id': self.calls,
                'conditioned_shot_type': 1,
            }],
            'rationale': 'test rationale',
            'action': {
                'shot_type': 1,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            },
            'action_source': 'llm',
        }
        return (1, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0), 'test rationale'

    def on_rally_end(self, history):
        return None


class DummyRallyNetModel:
    def __init__(self):
        self.calls = 0

    def sample(self, **kwargs):
        self.calls += 1
        action = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        shot_logits = torch.zeros(12)
        return None, None, None, action, shot_logits


class PlayerRolloutLimitTest(unittest.TestCase):
    def configure(self, evaluator):
        evaluator.device = 'cpu'
        evaluator.player_name = 'A'
        evaluator.player_id = 0
        evaluator.player2index = {'A': 0, 'B': 1}
        evaluator.df = model_rows()
        evaluator.save_trajectories = False
        evaluator.style_shift_instruction_mode = 'off'
        evaluator.instruction_mode = 'no_instruct'
        evaluator.config = {'evaluation': {'rollout_extra_steps': 2}}
        evaluator.get_player_test_matches = lambda: [3]
        evaluator.attach_style_shift_instruction_for_rollout = lambda *args, **kwargs: ''

    def assert_rollout_limit(self, trajectory):
        self.assertEqual(trajectory['rally_length_gt'], 2)
        self.assertEqual(trajectory['rollout_limit'], 4)
        self.assertEqual(trajectory['rally_length_pred'], 4)
        self.assertEqual(trajectory['termination_reason'], 'max_rollout_reached')
        self.assertFalse(trajectory['simulation_ended_early'])

    def test_random_uses_total_gt_length_plus_two(self):
        evaluator = RandomEvaluator.__new__(RandomEvaluator)
        self.configure(evaluator)
        evaluator.agent = DummyAgent()

        with patch(
            'Scripts.single_player_eval.evaluators.random_evaluator.terminate',
            return_value=False,
        ):
            trajectories = evaluator.simulate()

        self.assert_rollout_limit(trajectories[0])
        self.assertEqual(evaluator.agent.calls, 4)

    def test_rallynetv2_uses_total_gt_length_plus_two(self):
        evaluator = RallyNetV2Evaluator.__new__(RallyNetV2Evaluator)
        self.configure(evaluator)
        evaluator.model = DummyRallyNetModel()
        evaluator.ts = torch.linspace(0, 2, steps=71)
        evaluator.hyperparameters = {'player_ids_len': 2, 'shot_type_len': 12}

        with patch(
            'Scripts.single_player_eval.evaluators.rallynetv2_evaluator.terminate',
            return_value=False,
        ):
            trajectories = evaluator.simulate()

        self.assert_rollout_limit(trajectories[0])
        self.assertEqual(evaluator.model.calls, 4)

    def test_rag_evaluators_use_total_gt_length_plus_two(self):
        record = {
            'match_id': 3,
            'set': 1,
            'rally': 4,
            'players': ['A', 'B'],
            'player_profiles': {'A': None, 'B': None},
        }
        for evaluator_type in (NaiveRAGEvaluator, TimeRAGEvaluator):
            with self.subTest(evaluator=evaluator_type.__name__):
                evaluator = evaluator_type.__new__(evaluator_type)
                self.configure(evaluator)
                agents = {'A': DummyAgent(), 'B': DummyAgent()}
                evaluator._get_agent = lambda player, agents=agents: agents[player]

                with patch(
                    'Scripts.single_player_eval.evaluators.naive_rag_evaluator.terminate',
                    return_value=False,
                ):
                    trajectory = evaluator.simulate_rally(record)

                self.assert_rollout_limit(trajectory)
                self.assertEqual(sum(agent.calls for agent in agents.values()), 4)
                self.assertTrue(all('decision' in shot for shot in trajectory['pred_sequence']))
                self.assertTrue(all(
                    shot['decision']['rationale'] == 'test rationale'
                    for shot in trajectory['pred_sequence']
                ))

        evaluator = TwinEvaluator.__new__(TwinEvaluator)
        self.configure(evaluator)
        agents = {'A': DummyAgent(), 'B': DummyAgent()}
        evaluator._get_agent = lambda player: agents[player]
        with patch(
            'Scripts.single_player_eval.evaluators.twin_evaluator.terminate',
            return_value=False,
        ):
            trajectory = evaluator.simulate_rally(record)
        self.assert_rollout_limit(trajectory)
        self.assertEqual(sum(agent.calls for agent in agents.values()), 4)

    def test_pure_llm_physical_rollout_does_not_apply_profiles(self):
        evaluator = PureLLMEvaluator.__new__(PureLLMEvaluator)
        self.configure(evaluator)
        agents = {'A': DummyAgent(), 'B': DummyAgent()}
        evaluator._get_agent = lambda player: agents[player]
        record = {
            'match_id': 3,
            'set': 1,
            'rally': 4,
            'players': ['A', 'B'],
            'player_profiles': {
                'A': {'label': 'Attack', 'instruction': 'Attack more.'},
                'B': {'label': 'Defense', 'instruction': 'Defend deeper.'},
            },
        }

        with patch(
            'Scripts.single_player_eval.evaluators.naive_rag_evaluator.terminate',
            return_value=False,
        ):
            trajectory = evaluator.simulate_rally(record)

        self.assert_rollout_limit(trajectory)
        self.assertTrue(all(not instruction for agent in agents.values()
                            for instruction in agent.instructions))
        self.assertEqual(trajectory['applied_instructions'], {'A': '', 'B': ''})

    def test_instructed_physical_rally_applies_each_players_profile(self):
        record = {
            'match_id': 3,
            'set': 1,
            'rally': 4,
            'players': ['A', 'B'],
            'player_profiles': {
                'A': {'label': 'Attack', 'instruction': 'Attack more.'},
                'B': {'label': 'Defense', 'instruction': 'Defend deeper.'},
            },
        }

        for evaluator_type in (NaiveRAGEvaluator, TimeRAGEvaluator, TwinEvaluator):
            with self.subTest(evaluator=evaluator_type.__name__):
                evaluator = evaluator_type.__new__(evaluator_type)
                self.configure(evaluator)
                evaluator.instruction_mode = 'instruct'
                agents = {'A': DummyAgent(), 'B': DummyAgent()}
                evaluator._get_agent = lambda player, agents=agents: agents[player]

                with patch(
                    'Scripts.single_player_eval.evaluators.naive_rag_evaluator.terminate',
                    return_value=False,
                ):
                    trajectory = evaluator.simulate_rally(record)

                self.assertEqual(set(agents['A'].instructions), {'Attack. Attack more.'})
                self.assertEqual(set(agents['B'].instructions), {'Defense. Defend deeper.'})
                self.assertEqual(trajectory['applied_instructions'], {
                    'A': 'Attack. Attack more.',
                    'B': 'Defense. Defend deeper.',
                })
                self.assertIsNone(trajectory['target_player'])


if __name__ == '__main__':
    unittest.main()
