import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Agent.base_agent import BaseAgent
from Agent.twin import TwinAgent, TwinDatabase, TwinEmbeddingCache
from Agent.twin_single_stage import TwinSingleStageAgent


STATE = [4, (0.1, -0.2), (-0.1, 0.2), (0.2, 0.1)]


def _row(ball_round, player_type, offset=0.0):
    return {
        "player": "Player A",
        "match_id": 1,
        "set": 1,
        "rally": 1,
        "ball_round": ball_round,
        "player_score": 2,
        "opponent_score": 2,
        "score_status": 0,
        "player_location_x": 0.1 + offset,
        "player_location_y": -0.2 + offset,
        "ball_distance": 0.0,
        "ball_distance_x": 0.1,
        "ball_distance_y": 0.3,
        "hit_x": 0.2 + offset,
        "hit_y": 0.1 + offset,
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


def _aspect_response():
    return {
        "aspect_segments": {
            "my_tactical_pattern": [{"range": [2, 2], "phase": "Controlled return"}],
            "my_tempo_movement_dynamics": [{"range": [2, 2], "phase": "Stable recovery"}],
            "opponent_tactical_pattern": [{"range": [2, 2], "phase": "Smash pressure"}],
            "opponent_tempo_movement_dynamics": [{"range": [2, 2], "phase": "Balanced base"}],
        }
    }


def _action_response(shot_type, landing_x, landing_y, moving_x, moving_y):
    evidence_action = BaseAgent.build_prompt_action(
        shot_type, landing_x, landing_y, moving_x, moving_y
    )
    return {
        key: evidence_action[key]
        for key in BaseAgent.PROMPT_ACTION_KEYS
    }


def _placement_response(grounded):
    placement = _action_response(2, 0.3, 0.4, -0.2, -0.1)
    placement.pop("shot_type")
    response = {"placement": placement}
    if grounded:
        response["case_assessments"] = [{
            "retrieval_rank": 1,
            "impact": "support",
            "factors": ["same placement", "same recovery"],
        }]
    return response


def _single_stage_response(grounded=True, shot_type=2, source="success"):
    del source
    if not grounded:
        if shot_type == 11:
            return {
                "outcome": "missed",
                "shot_type": None,
                "landing_x": None,
                "landing_y": None,
                "moving_x": None,
                "moving_y": None,
            }
        action = _action_response(shot_type, 0.3, 0.4, -0.2, -0.1)
        return {
            "outcome": "returned",
            **action,
        }
    response = {
        "action": _action_response(
            shot_type,
            None if shot_type == 11 else 0.3,
            None if shot_type == 11 else 0.4,
            None if shot_type == 11 else -0.2,
            None if shot_type == 11 else -0.1,
        ),
    }
    if grounded:
        response["case_assessments"] = [
            {"source": "miss", "retrieval_rank": 1,
             "impact": "support" if shot_type == 11 else "irrelevant",
             "factors": ["failed contact"] if shot_type == 11 else
                        ["worse reachability"]},
            {"source": "success", "retrieval_rank": 1,
             "impact": "conflict" if shot_type == 11 else "support",
             "factors": ["easier reachability"] if shot_type == 11 else
                        ["same clear action"]},
            {"source": "success", "retrieval_rank": 2,
             "impact": "conflict", "factors": ["different movement tempo"]},
        ]
    return response


def _info():
    return {
        "rally": 1,
        "round": [2],
        "score": [2, 2],
        "rally_history": [],
        "current_player_name": "Player A",
        "rally_profile_instruction": "",
    }


class _EmbeddingModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **_):
        rows = []
        for text in texts:
            row = np.array([len(text), sum(map(ord, text)) % 17, 1.0], dtype=np.float32)
            if normalize_embeddings:
                row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows)


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get_verified_response(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        validator = kwargs.get("validator")
        if response is not None and validator is not None:
            assert validator(response), response
        return response, "raw", {}


class TwinCaseGroundingTest(unittest.TestCase):
    def _agent(self, responses, case_grounding=True):
        database = TwinDatabase(pd.DataFrame([
            _row(2, 2),
            _row(3, 9, 0.05),
            _row(4, 11, 0.1),
        ]), "Player A", 0)
        database.build_with_fallback()
        model = _EmbeddingModel()
        cache = TwinEmbeddingCache.build(database, model, "fake/model")
        agent = TwinAgent(
            "Player A",
            database,
            cache,
            embedding_model=model,
            embedding_model_name="fake/model",
            enable_tracing=False,
            conditioned_top_k=3,
            case_grounding=case_grounding,
            semantic_start_ball_round=1,
        )
        agent.llm = _LLM(responses)
        return agent

    def test_grounded_calls_use_schemas_and_map_all_case_assessments(self):
        agent = self._agent([
            _aspect_response(),
            {
                "case_assessments": [
                    {"source": "success", "retrieval_rank": 1,
                     "impact": "support", "factors": ["same clear return"]},
                    {"source": "success", "retrieval_rank": 2,
                     "impact": "conflict", "factors": ["different shot choice"]},
                    {"source": "miss", "retrieval_rank": 1,
                     "impact": "irrelevant", "factors": ["worse reachability"]},
                ],
                "outcome": "returned",
                "shot_type": "Clear",
            },
            _placement_response(True),
        ])

        action, _ = agent.predict(STATE, _info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertNotIn("schema", agent.llm.calls[0])
        self.assertEqual(agent.llm.calls[0]["max_trials"], 5)
        self.assertEqual(agent.llm.calls[1]["schema_name"], "rag_stage1_grounded")
        self.assertEqual(agent.llm.calls[2]["schema_name"], "rag_stage2_grounded")
        self.assertEqual(agent.llm.calls[1]["max_trials"], 2)
        self.assertEqual(agent.llm.calls[2]["max_trials"], 2)
        trace = agent.last_decision_trace
        self.assertTrue(trace["case_grounding"])
        self.assertEqual(trace["schema_version"], 7)
        self.assertEqual(len(trace["case_assessments"]), len(trace["case_ids"]))
        self.assertEqual(
            trace["stage1"]["case_assessments"][0]["source"], "success"
        )
        self.assertIn("case_id", trace["stage2"]["case_assessments"][0])
        self.assertNotIn("summary", trace)
        self.assertNotIn("evidence_summary", trace)
        self.assertNotIn("decision_summary", trace)
        self.assertIn("semantic_query_text", trace["pre_stage"])

    def test_direct_mode_keeps_retrieval_and_has_empty_selection(self):
        agent = self._agent([
            _aspect_response(),
            {"outcome": "returned", "shot_type": "Clear"},
            _placement_response(False),
        ], case_grounding=False)

        action, _ = agent.predict(STATE, _info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(agent.llm.calls[1]["schema_name"], "rag_stage1_direct")
        self.assertEqual(agent.llm.calls[2]["schema_name"], "rag_stage2_direct")
        self.assertNotIn('"case_tags"', agent.llm.calls[1]["user_prompt"])
        self.assertNotIn('"case_tags"', agent.llm.calls[2]["user_prompt"])
        self.assertTrue(agent.last_decision_trace["case_ids"])
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])

    def test_fallback_action_is_the_same_in_both_modes(self):
        grounded = self._agent([_aspect_response(), None, None], case_grounding=True)
        direct = self._agent([_aspect_response(), None, None], case_grounding=False)

        grounded_action, _ = grounded.predict(STATE, _info(), launch=False)
        direct_action, _ = direct.predict(STATE, _info(), launch=False)

        self.assertEqual(grounded_action, direct_action)
        self.assertTrue(grounded.last_decision_trace["case_assessments"])
        self.assertEqual(direct.last_decision_trace["case_assessments"], [])


class TwinSingleStageCaseGroundingTest(unittest.TestCase):
    def _agent(self, responses, case_grounding=True,
               instruction_mode="no_instruct"):
        database = TwinDatabase(pd.DataFrame([
            _row(2, 2),
            _row(3, 9, 0.05),
            _row(4, 11, 0.1),
        ]), "Player A", 0)
        database.build_with_fallback()
        model = _EmbeddingModel()
        cache = TwinEmbeddingCache.build(database, model, "fake/model")
        agent = TwinSingleStageAgent(
            "Player A",
            database,
            cache,
            embedding_model=model,
            embedding_model_name="fake/model",
            enable_tracing=False,
            case_grounding=case_grounding,
            instruction_mode=instruction_mode,
            semantic_start_ball_round=1,
        )
        agent.llm = _LLM(responses)
        return agent

    def test_grounded_mode_uses_schema_and_maps_all_references(self):
        agent = self._agent([
            _aspect_response(),
            _single_stage_response(),
        ])

        action, _ = agent.predict(STATE, _info(), launch=False)

        self.assertEqual(action[0], 2)
        call = agent.llm.calls[1]
        self.assertEqual(call["schema_name"], "twin_single_stage_grounded")
        self.assertEqual(call["max_trials"], 2)
        self.assertIn("source=success, retrieval_rank=1", call["user_prompt"])
        self.assertIn("source=miss, retrieval_rank=1", call["user_prompt"])
        trace = agent.last_decision_trace
        self.assertTrue(trace["case_grounding"])
        self.assertEqual(trace["case_assessments"][0]["source"], "miss")
        self.assertIn("case_id", trace["case_assessments"][0])
        self.assertEqual(len(trace["case_ids"]), len(trace["case_assessments"]))

    def test_direct_mode_keeps_retrieval_but_omits_grounding_output(self):
        agent = self._agent([
            _aspect_response(),
            _single_stage_response(grounded=False),
        ], case_grounding=False)

        action, _ = agent.predict(STATE, _info(), launch=False)

        self.assertEqual(action[0], 2)
        call = agent.llm.calls[1]
        self.assertEqual(call["schema_name"], "rag_single_stage_direct")
        self.assertNotIn('"case_tags"', call["user_prompt"])
        self.assertIn("Successful Return Twin Memories", call["user_prompt"])
        self.assertNotIn('"case_assessments"', call["user_prompt"])
        self.assertTrue(agent.last_decision_trace["case_ids"])
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])

    def test_direct_instruct_uses_instruction_only_in_action_prompt(self):
        instruction = "Attack Profile. Prefer attacking pressure."
        agent = self._agent([
            _aspect_response(),
            _single_stage_response(grounded=False),
        ], case_grounding=False, instruction_mode="instruct")
        current_info = _info()
        current_info["rally_profile_instruction"] = instruction

        action, _ = agent.predict(STATE, current_info, launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(len(agent.llm.calls), 2)
        self.assertNotIn(instruction, agent.llm.calls[0]["user_prompt"])
        action_call = agent.llm.calls[1]
        self.assertEqual(action_call["schema_name"], "rag_single_stage_direct")
        self.assertIn("Tactical Imitation Instruction", action_call["user_prompt"])
        self.assertIn(instruction, action_call["user_prompt"])
        self.assertIn("weak tie-breaker", action_call["user_prompt"])
        self.assertNotIn('"case_assessments"', action_call["user_prompt"])
        self.assertFalse(agent.last_decision_trace["case_grounding"])
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])

    def test_parser_accepts_mixed_evidence_but_validates_factors_and_action(self):
        normal_cases = [{"player_type": 2}, {"player_type": 9}]
        miss_cases = [{"player_type": 11}]
        missed_without_miss = _single_stage_response(shot_type=11)
        returned_with_wrong_type = _single_stage_response()
        normal_cases[0]["player_type"] = 9

        self.assertIsNotNone(TwinSingleStageAgent._parse_action_response(
            missed_without_miss, False, [2], True, normal_cases, miss_cases
        )[0])
        self.assertIsNotNone(TwinSingleStageAgent._parse_action_response(
            returned_with_wrong_type, False, [2], True, normal_cases, miss_cases
        )[0])

        normal_cases[0]["player_type"] = 2
        mixed_shots = _single_stage_response()
        mixed_shots["case_assessments"][2]["factors"] = ["different shot choice"]
        self.assertIsNotNone(TwinSingleStageAgent._parse_action_response(
            mixed_shots, False, [2, 9], True, normal_cases, miss_cases
        )[0])

        mixed_sources = _single_stage_response()
        mixed_sources["case_assessments"][0]["factors"] = ["different miss outcome"]
        self.assertIsNotNone(TwinSingleStageAgent._parse_action_response(
            mixed_sources, False, [2], True, normal_cases, miss_cases
        )[0])

        empty_factor = _single_stage_response()
        empty_factor["case_assessments"][0]["factors"] = [""]
        self.assertIsNone(TwinSingleStageAgent._parse_action_response(
            empty_factor, False, [2], True, normal_cases, miss_cases
        )[0])

        sentence_factor = _single_stage_response()
        sentence_factor["case_assessments"][0]["factors"] = [
            "first line has five words"
        ]
        self.assertIsNone(TwinSingleStageAgent._parse_action_response(
            sentence_factor, False, [2], True, normal_cases, miss_cases
        )[0])

        missing_case = _single_stage_response()
        missing_case["case_assessments"].pop()
        self.assertIsNone(TwinSingleStageAgent._parse_action_response(
            missing_case, False, [2], True, normal_cases, miss_cases
        )[0])

        wrong_order = _single_stage_response()
        wrong_order["case_assessments"].reverse()
        self.assertIsNone(TwinSingleStageAgent._parse_action_response(
            wrong_order, False, [2], True, normal_cases, miss_cases
        )[0])

    def test_grounded_schema_allows_exhaustive_case_assessments(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "Schemas"
            / "Agent"
            / "twin_single_stage_grounded_schema.json"
        )
        assessments = json.loads(schema_path.read_text())["properties"][
            "case_assessments"
        ]
        self.assertEqual(assessments["minItems"], 0)
        self.assertNotIn("maxItems", assessments)
        factors = assessments["items"]["properties"]["factors"]
        self.assertEqual(factors["minItems"], 1)
        self.assertEqual(factors["maxItems"], 2)
        self.assertEqual(factors["items"]["maxLength"], 32)

    def test_fallback_action_is_identical_between_modes(self):
        grounded = self._agent([_aspect_response(), None], case_grounding=True)
        direct = self._agent([_aspect_response(), None], case_grounding=False)

        grounded_action, _ = grounded.predict(STATE, _info(), launch=False)
        direct_action, _ = direct.predict(STATE, _info(), launch=False)

        self.assertEqual(grounded_action, direct_action)
        self.assertTrue(grounded.last_decision_trace["case_assessments"])
        self.assertEqual(direct.last_decision_trace["case_assessments"], [])


if __name__ == "__main__":
    unittest.main()
