import unittest

import pandas as pd

from Agent.base_agent import BaseAgent
from Agent.naive_rag import NaiveRAGAgent, NaiveRAGDatabase
from Agent.timerag import TimeRAGAgent, TimeRAGDatabase
from Agent.twin import TwinAgent, TwinDatabase, TwinEmbeddingCache
from Agent.twin_single_stage import TwinSingleStageAgent


def row(ball_round, player_type, offset=0.0, player="Player A"):
    player_x = 0.1 + offset
    player_y = -0.2 + offset
    hit_x = 0.2 + offset
    hit_y = 0.1 + offset
    return {
        "player": player,
        "match_id": 1,
        "set": 1,
        "rally": 1,
        "ball_round": ball_round,
        "player_score": 2,
        "opponent_score": 2,
        "score_status": 0,
        "player_location_x": player_x,
        "player_location_y": player_y,
        "ball_distance": 0.0,
        "ball_distance_x": 999.0,
        "ball_distance_y": 999.0,
        "hit_x": hit_x,
        "hit_y": hit_y,
        "opponent_type": 4,
        "opponent_location_x": -0.1 + offset,
        "opponent_location_y": 0.2 + offset,
        "opponent_move_x": 0.0,
        "opponent_move_y": 0.0,
        "landing_x": 0.3 + offset,
        "landing_y": 0.4 + offset,
        "player_type": player_type,
        "moving_x": -0.2 + offset,
        "moving_y": -0.1 + offset,
    }


def dataframe():
    return pd.DataFrame([
        row(2, 2, 0.00),
        row(3, 9, 0.05),
        row(4, 11, 0.10),
    ])


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.validators = []

    def get_verified_response(self, system_prompt, user_prompt, validator=None, **_):
        self.prompts.append(user_prompt)
        self.validators.append(validator)
        response = self.responses.pop(0)
        if response is not None and validator is not None:
            assert validator(response), response
        return response, "raw", {}


class RaisingLLM:
    def __init__(self):
        self.calls = 0

    def get_verified_response(self, **_):
        self.calls += 1
        raise TimeoutError("provider timeout")


def placement_response():
    placement = BaseAgent.build_prompt_action(2, 0.3, 0.4, -0.2, -0.1)
    placement.pop("shot_type")
    return {
        "rationale": "The Clear-conditioned cases support this placement.",
        "placement": placement,
    }


def info():
    return {
        "rally": 1,
        "round": [2],
        "score": [2, 2],
        "rally_history": [],
        "current_player_name": "Player A",
        "rally_profile_instruction": "",
    }


STATE = [4, (0.1, -0.2), (-0.1, 0.2), (0.2, 0.1)]


class TwoStageRAGTest(unittest.TestCase):
    def naive_agent(self, responses):
        database = NaiveRAGDatabase(dataframe(), "Player A", 0)
        agent = NaiveRAGAgent(
            "Player A", database, enable_tracing=False, conditioned_top_k=3
        )
        agent.llm = FakeLLM(responses)
        return agent

    def test_returned_decision_uses_two_calls_and_exact_shot_evidence(self):
        agent = self.naive_agent([
            {
                "rationale": "The player can return and most likely uses a Clear.",
                "outcome": "returned",
                "shot_type": "Clear",
            },
            placement_response(),
        ])
        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertIn("Successful Return Evidence", agent.llm.prompts[0])
        self.assertIn("evidence-driven behavior simulation", agent.llm.prompts[0])
        self.assertIn("score pressure", agent.llm.prompts[0])
        self.assertIn("Serve short or Serve long", agent.llm.prompts[0])
        self.assertNotIn('"landing_x"', agent.llm.prompts[0])
        self.assertNotIn("then move to", agent.llm.prompts[0])
        self.assertNotIn("(distance", agent.llm.prompts[0])
        self.assertNotIn("Locked Stage-1 Decision", agent.llm.prompts[1])
        self.assertIn('Action: {"shot_type": "Clear"', agent.llm.prompts[1])
        self.assertIn('"landing_x"', agent.llm.prompts[1])
        self.assertNotIn("(distance", agent.llm.prompts[1])
        self.assertNotIn(
            "The player can return and most likely uses a Clear.",
            agent.llm.prompts[1],
        )
        self.assertNotIn('Action: {"shot_type": "Drop"', agent.llm.prompts[1])
        trace = agent.last_decision_trace
        self.assertEqual(trace["stage1"]["shot_type"], 2)
        self.assertEqual(trace["stage2"]["conditioned_shot_type"], 2)
        self.assertTrue(all(
            case["conditioned_shot_type"] == 2
            for case in trace["stage2"]["case_ids"]
        ))

    def test_missed_decision_skips_stage2(self):
        agent = self.naive_agent([{
            "rationale": "The player cannot complete a legal return.",
            "outcome": "missed",
            "shot_type": None,
        }])
        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertEqual(agent.last_decision_trace["stage2"]["status"], "skipped_missed")
        self.assertIsNone(agent.last_decision_trace["action"]["landing_x"])

    def test_stage2_failure_copies_nearest_conditioned_case(self):
        agent = self.naive_agent([
            {
                "rationale": "A Clear is most likely.",
                "outcome": "returned",
                "shot_type": "Clear",
            },
            None,
        ])
        action, _ = agent.predict(STATE, info(), launch=False)
        self.assertEqual(action[0], 2)
        self.assertEqual(
            agent.last_decision_trace["stage2"]["source"],
            "fallback_nearest_conditioned_case",
        )

    def test_transport_failures_use_deterministic_fallbacks(self):
        agent = self.naive_agent([])
        agent.llm = RaisingLLM()
        action, _ = agent.predict(STATE, info(), launch=False)
        self.assertEqual(action[0], 2)
        self.assertEqual(agent.llm.calls, 2)
        self.assertEqual(
            agent.last_decision_trace["stage1"]["source"],
            "fallback_nearest_case",
        )
        self.assertEqual(
            agent.last_decision_trace["stage2"]["source"],
            "fallback_nearest_conditioned_case",
        )

    def test_timerag_uses_the_same_two_stage_contract(self):
        database = TimeRAGDatabase(dataframe(), "Player A", 0, window_size=2)
        agent = TimeRAGAgent(
            "Player A", database, enable_tracing=False, conditioned_top_k=3
        )
        agent.llm = FakeLLM([
            {
                "rationale": "The temporal evidence supports a Clear.",
                "outcome": "returned",
                "shot_type": "Clear",
            },
            placement_response(),
        ])
        action, _ = agent.predict(STATE, info(), launch=False)
        self.assertEqual(action[0], 2)
        self.assertEqual(agent.last_decision_trace["architecture"], "two_stage_v1")
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertNotIn("(distance", agent.llm.prompts[0])
        self.assertNotIn("(distance", agent.llm.prompts[1])
        self.assertIn("historical windows", agent.llm.prompts[0])

    def twin_agent(self, responses):
        database = TwinDatabase(dataframe(), "Player A", 0)
        database.build_with_fallback()
        model = FakeEmbeddingModel()
        cache = TwinEmbeddingCache.build(database, model, "fake/model")
        agent = TwinAgent(
            "Player A",
            database,
            cache,
            embedding_model=model,
            embedding_model_name="fake/model",
            enable_tracing=False,
            conditioned_top_k=3,
        )
        agent.llm = FakeLLM(responses)
        return agent

    def twin_single_stage_agent(self, responses):
        database = TwinDatabase(dataframe(), "Player A", 0)
        database.build_with_fallback()
        model = FakeEmbeddingModel()
        cache = TwinEmbeddingCache.build(database, model, "fake/model")
        agent = TwinSingleStageAgent(
            "Player A",
            database,
            cache,
            embedding_model=model,
            embedding_model_name="fake/model",
            enable_tracing=False,
        )
        agent.llm = FakeLLM(responses)
        return agent

    def test_stage2_calls_do_not_use_semantic_validator(self):
        stage1 = {
            "rationale": "The evidence supports a legal Clear.",
            "outcome": "returned",
            "shot_type": "Clear",
        }
        timerag_database = TimeRAGDatabase(dataframe(), "Player A", 0, window_size=2)
        timerag_agent = TimeRAGAgent(
            "Player A", timerag_database, enable_tracing=False, conditioned_top_k=3
        )
        timerag_agent.llm = FakeLLM([stage1, placement_response()])
        agents = [
            self.naive_agent([stage1, placement_response()]),
            timerag_agent,
            self.twin_agent([aspect_response(), stage1, placement_response()]),
        ]

        for agent in agents:
            with self.subTest(agent=type(agent).__name__):
                agent.predict(STATE, info(), launch=False)
                self.assertIsNone(agent.llm.validators[-1])

    def test_twin_keeps_aspect_pre_stage_and_two_stage_action_contract(self):
        agent = self.twin_agent([
            aspect_response(),
            {
                "rationale": "The Twin evidence supports a legal Clear.",
                "outcome": "returned",
                "shot_type": "Clear",
            },
            placement_response(),
        ])
        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(len(agent.llm.prompts), 3)
        self.assertIn("Estimate the current Twin semantic state", agent.llm.prompts[0])
        self.assertIn("Successful Return Evidence", agent.llm.prompts[1])
        self.assertNotIn('"landing_x"', agent.llm.prompts[1])
        self.assertIn('Action: {"shot_type": "Clear"', agent.llm.prompts[2])
        self.assertIn("Locked Stage-1 Shot Type\nClear", agent.llm.prompts[2])
        trace = agent.last_decision_trace
        self.assertEqual(trace["architecture"], "two_stage_v1")
        self.assertEqual(trace["pre_stage"]["architecture"], "twin_aspect_v1")
        self.assertIn(
            "My tactical pattern: Controlled return construction.",
            trace["pre_stage"]["semantic_query_text"],
        )
        self.assertEqual(trace["stage2"]["conditioned_shot_type"], 2)

    def test_twin_missed_action_skips_stage2_after_two_calls(self):
        agent = self.twin_agent([
            aspect_response(),
            {
                "rationale": "The Twin missed evidence indicates failed contact.",
                "outcome": "missed",
                "shot_type": None,
            },
        ])
        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertEqual(agent.last_decision_trace["stage2"]["status"], "skipped_missed")

    def test_single_stage_twin_generates_the_complete_action_after_one_retrieval(self):
        response = {
            "rationale": "The Twin evidence supports a complete Clear action.",
            "action": BaseAgent.build_prompt_action(2, 0.3, 0.4, -0.2, -0.1),
        }
        agent = self.twin_single_stage_agent([aspect_response(), response])
        agent.database.retrieve_by_shot_type = lambda *_args, **_kwargs: self.fail(
            "single-stage Twin must not perform shot-conditioned retrieval"
        )

        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertIn("Retrieved Missed-Shot Twin Memories", agent.llm.prompts[1])
        self.assertIn("Retrieved Successful Return Twin Memories", agent.llm.prompts[1])
        self.assertIn('"landing_x"', agent.llm.prompts[1])
        self.assertEqual(agent.last_decision_trace["architecture"], "single_stage_v1")
        self.assertIn("retrieval", agent.last_decision_trace)
        self.assertNotIn("stage2", agent.last_decision_trace)

    def test_single_stage_twin_supports_missed_action_in_the_same_call(self):
        response = {
            "rationale": "The missed memories indicate failed contact.",
            "action": BaseAgent.build_prompt_action(11, None, None, None, None),
        }
        agent = self.twin_single_stage_agent([aspect_response(), response])

        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertIsNone(agent.last_decision_trace["action"]["landing_x"])


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **_):
        import numpy as np

        rows = []
        for text in texts:
            row = np.array([len(text), sum(map(ord, text)) % 17, 1.0], dtype=np.float32)
            if normalize_embeddings:
                row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows)


def aspect_response():
    return {
        "my_tactical_pattern": "Controlled return construction.",
        "my_tempo_movement_dynamics": "Balanced recovery pressure.",
        "opponent_tactical_pattern": "Sustained attacking pressure.",
        "opponent_tempo_movement_dynamics": "Forward recovery pressure.",
    }


if __name__ == "__main__":
    unittest.main()
