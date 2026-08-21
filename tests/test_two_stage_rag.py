import copy
import json
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
        self.calls = []

    def get_verified_response(self, system_prompt, user_prompt, validator=None, **kwargs):
        self.prompts.append(user_prompt)
        self.validators.append(validator)
        self.calls.append(kwargs)
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


def llm_action_response(shot_type, landing_x, landing_y, moving_x, moving_y):
    evidence_action = BaseAgent.build_prompt_action(
        shot_type, landing_x, landing_y, moving_x, moving_y
    )
    return {
        key: evidence_action[key]
        for key in BaseAgent.PROMPT_ACTION_KEYS
    }


def placement_response(case_grounding=True):
    placement = llm_action_response(2, 0.3, 0.4, -0.2, -0.1)
    placement.pop("shot_type")
    response = {"placement": placement}
    if case_grounding:
        response["case_assessments"] = [{
            "retrieval_rank": 1,
            "impact": "support",
            "factors": ["same landing", "same recovery"],
        }]
    return response


def returned_response(reason="The selected evidence supports a Clear."):
    del reason
    return {
        "case_assessments": [
            {"source": "success", "retrieval_rank": 1,
             "impact": "support", "factors": ["same shuttle", "clear return"]},
            {"source": "success", "retrieval_rank": 2,
             "impact": "conflict", "factors": ["different shot choice"]},
            {"source": "miss", "retrieval_rank": 1,
             "impact": "irrelevant", "factors": ["lower reachability"]},
        ],
        "outcome": "returned",
        "shot_type": "Clear",
    }


def missed_response(reason="The selected evidence supports failed contact."):
    del reason
    return {
        "case_assessments": [
            {"source": "success", "retrieval_rank": 1,
             "impact": "conflict", "factors": ["easier shuttle"]},
            {"source": "success", "retrieval_rank": 2,
             "impact": "conflict", "factors": ["easier reachability"]},
            {"source": "miss", "retrieval_rank": 1,
             "impact": "support", "factors": ["difficult arrival", "failed contact"]},
        ],
        "outcome": "missed",
        "shot_type": None,
    }


def single_stage_response(shot_type=2):
    missed = shot_type == 11
    return {
        "case_assessments": [
            {"source": "miss", "retrieval_rank": 1,
             "impact": "support" if missed else "irrelevant",
             "factors": ["failed contact"] if missed else ["worse reachability"]},
            {"source": "success", "retrieval_rank": 1,
             "impact": "conflict" if missed else "support",
             "factors": ["easier reachability"] if missed else
                        ["clear return", "same placement"]},
            {"source": "success", "retrieval_rank": 2,
             "impact": "conflict", "factors": ["different movement"]},
        ],
        "action": llm_action_response(
            shot_type,
            None if missed else 0.3,
            None if missed else 0.4,
            None if missed else -0.2,
            None if missed else -0.1,
        ),
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
    def naive_agent(self, responses, case_grounding=True):
        database = NaiveRAGDatabase(dataframe(), "Player A", 0)
        agent = NaiveRAGAgent(
            "Player A", database, enable_tracing=False, conditioned_top_k=3,
            case_grounding=case_grounding,
        )
        agent.llm = FakeLLM(responses)
        return agent

    def test_returned_decision_uses_two_calls_and_exact_shot_evidence(self):
        agent = self.naive_agent([
            returned_response(),
            placement_response(case_grounding=True),
        ])
        action, reasoning = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(reasoning, "")
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertIn("Successful Return Cases", agent.llm.prompts[0])
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
        self.assertNotIn('Action: {"shot_type": "Drop"', agent.llm.prompts[1])
        trace = agent.last_decision_trace
        self.assertEqual(trace["stage1"]["shot_type"], 2)
        self.assertEqual(trace["stage2"]["conditioned_shot_type"], 2)
        self.assertTrue(all(
            case["conditioned_shot_type"] == 2
            for case in trace["stage2"]["case_ids"]
        ))
        self.assertEqual(agent.llm.calls[0]["schema_name"], "rag_stage1_grounded")
        self.assertEqual(agent.llm.calls[1]["schema_name"], "rag_stage2_grounded")
        self.assertEqual(agent.llm.calls[0]["max_trials"], 2)
        self.assertEqual(trace["schema_version"], 6)
        self.assertEqual(trace["stage1"]["case_assessments"][0]["case_id"], 0)
        self.assertEqual(trace["stage2"]["case_assessments"][0]["case_id"], 0)
        self.assertEqual(len(trace["case_assessments"]), len(trace["case_ids"]))
        self.assertNotIn("rationale", trace)

    def test_missed_decision_skips_stage2(self):
        agent = self.naive_agent([missed_response()])
        action, reasoning = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(reasoning, "")
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertEqual(agent.last_decision_trace["stage2"]["status"], "skipped_missed")
        self.assertIsNone(agent.last_decision_trace["action"]["landing_x"])

    def test_stage2_failure_copies_nearest_conditioned_case(self):
        agent = self.naive_agent([
            returned_response(),
            None,
        ])
        action, _ = agent.predict(STATE, info(), launch=False)
        self.assertEqual(action[0], 2)
        self.assertEqual(
            agent.last_decision_trace["stage2"]["source"],
            "fallback_nearest_conditioned_case",
        )
        stage2 = agent.last_decision_trace["stage2"]
        self.assertEqual(len(stage2["case_assessments"]), len(stage2["case_ids"]))
        self.assertTrue(all(
            item["impact"] == "irrelevant"
            and item["factors"] == ["assessment unavailable"]
            for item in stage2["case_assessments"]
        ))

    def test_direct_mode_uses_direct_schemas_and_no_case_tags(self):
        agent = self.naive_agent([
            {"outcome": "returned", "shot_type": "Clear"},
            placement_response(case_grounding=False),
        ], case_grounding=False)

        action, reasoning = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(reasoning, "")
        self.assertNotIn('"case_tags"', agent.llm.prompts[0])
        self.assertNotIn('"case_tags"', agent.llm.prompts[1])
        self.assertEqual(agent.llm.calls[0]["schema_name"], "rag_stage1_direct")
        self.assertEqual(agent.llm.calls[1]["schema_name"], "rag_stage2_direct")
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])

    def test_direct_and_grounded_transport_fallback_actions_match(self):
        grounded = self.naive_agent([])
        grounded.llm = RaisingLLM()
        direct = self.naive_agent([], case_grounding=False)
        direct.llm = RaisingLLM()

        grounded_action, _ = grounded.predict(STATE, info(), launch=False)
        direct_action, _ = direct.predict(STATE, info(), launch=False)

        self.assertEqual(grounded_action, direct_action)
        self.assertTrue(grounded.last_decision_trace["case_assessments"])
        self.assertEqual(direct.last_decision_trace["case_assessments"], [])

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
            returned_response("The temporal evidence supports a Clear."),
            placement_response(),
        ])
        action, _ = agent.predict(STATE, info(), launch=False)
        self.assertEqual(action[0], 2)
        self.assertEqual(agent.last_decision_trace["architecture"], "two_stage_v1")
        self.assertTrue(agent.last_decision_trace["case_grounding"])
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertNotIn("(distance", agent.llm.prompts[0])
        self.assertNotIn("(distance", agent.llm.prompts[1])
        self.assertIn("historical windows", agent.llm.prompts[0])

    def test_action_context_horizons_match_each_retrieval_unit(self):
        current_info = info()
        current_info["rally_history"] = [row(1, 10, player="Player B")]

        naive = self.naive_agent([
            returned_response(), placement_response(),
        ])
        naive.predict(STATE, current_info, launch=False)
        for prompt in naive.llm.prompts:
            current_context = prompt.split("## Successful Return", 1)[0].split(
                "## Similar Cases", 1
            )[0]
            self.assertNotIn("Shot 1:", current_context)
            self.assertIn("currently at Shot 2", current_context)

        timerag_info = info()
        timerag_info["round"] = [3]
        timerag_info["rally_history"] = [
            row(1, 10, player="Player B"),
            row(2, 2, player="Player A"),
        ]
        timerag_database = TimeRAGDatabase(dataframe(), "Player A", 0, window_size=2)
        timerag = TimeRAGAgent(
            "Player A", timerag_database, enable_tracing=False, conditioned_top_k=3
        )
        timerag.llm = FakeLLM([returned_response(), placement_response()])
        timerag.predict(STATE, timerag_info, launch=False)
        for prompt in timerag.llm.prompts:
            current_context = prompt.split("## Successful Return", 1)[0].split(
                "## Similar Windows", 1
            )[0]
            self.assertNotIn("Shot 1:", current_context)
            self.assertIn("Shot 2:", current_context)
            self.assertIn("Shot 3:", current_context)

        twin = self.twin_agent([
            aspect_response(1, 2), returned_response(), placement_response(),
        ])
        twin.predict(STATE, current_info, launch=False)
        self.assertIn("Shot 1:", twin.llm.prompts[0])
        self.assertNotIn("Shot 1:", twin.llm.prompts[1])
        self.assertNotIn("Shot 1:", twin.llm.prompts[2])
        self.assertIn("Twin semantic context:", twin.llm.prompts[1])

        single_response = single_stage_response()
        single = self.twin_single_stage_agent([
            aspect_response(1, 2), single_response,
        ])
        single.predict(STATE, current_info, launch=False)
        self.assertIn("Shot 1:", single.llm.prompts[0])
        current_context = single.llm.prompts[1].split(
            "## Current Twin Semantic Query", 1
        )[0]
        self.assertNotIn("Shot 1:", current_context)
        self.assertIn("Historical decision context:", single.llm.prompts[1])

    def twin_agent(self, responses, instruction_mode="no_instruct",
                   semantic_start_ball_round=1, save_prompt_traces=False):
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
            instruction_mode=instruction_mode,
            semantic_start_ball_round=semantic_start_ball_round,
            save_prompt_traces=save_prompt_traces,
        )
        agent.llm = FakeLLM(responses)
        return agent

    def twin_single_stage_agent(self, responses, instruction_mode="no_instruct",
                                semantic_start_ball_round=1,
                                save_prompt_traces=False):
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
            instruction_mode=instruction_mode,
            semantic_start_ball_round=semantic_start_ball_round,
            save_prompt_traces=save_prompt_traces,
        )
        agent.llm = FakeLLM(responses)
        return agent

    def test_stage2_calls_use_grounding_validator(self):
        stage1 = returned_response("The evidence supports a legal Clear.")
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
                _, reasoning = agent.predict(STATE, info(), launch=False)
                self.assertIsNotNone(agent.llm.validators[-1])
                self.assertEqual(reasoning, "")

    def test_twin_keeps_aspect_pre_stage_and_two_stage_action_contract(self):
        agent = self.twin_agent([
            aspect_response(),
            returned_response("The Twin evidence supports a legal Clear."),
            placement_response(),
        ])
        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(len(agent.llm.prompts), 3)
        self.assertIn("Annotate the observed badminton rally prefix", agent.llm.prompts[0])
        self.assertIn('"aspect_segments"', agent.llm.prompts[0])
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
            missed_response("The Twin missed evidence indicates failed contact."),
        ])
        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertEqual(agent.last_decision_trace["stage2"]["status"], "skipped_missed")

    def test_single_stage_twin_generates_the_complete_action_after_one_retrieval(self):
        response = single_stage_response()
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

    def test_twin_selects_each_aspect_segment_at_the_current_round(self):
        response = {
            "aspect_segments": {
                "my_tactical_pattern": [
                    {"range": [1, 2], "phase": "Earlier defensive construction"},
                    {"range": [3, 4], "phase": "Current counterattacking construction"},
                ],
                "my_tempo_movement_dynamics": [
                    {"range": [1, 4], "phase": "Stable recovery across the prefix"},
                ],
                "opponent_tactical_pattern": [
                    {"range": [1, 3], "phase": "Earlier rear-court pressure"},
                    {"range": [4, 4], "phase": "Current defensive reset"},
                ],
                "opponent_tempo_movement_dynamics": [
                    {"range": [1, 1], "phase": "Initial stable recovery"},
                    {"range": [2, 4], "phase": "Current delayed recovery"},
                ],
            }
        }
        agent = self.twin_agent([response])
        current_info = info()
        current_info["round"] = [4]
        state_dict = agent._build_state_dict_from_env(STATE, current_info)
        history = [
            {"ball_round": ball_round, "player_type": 2}
            for ball_round in (1, 2, 3)
        ]

        aspects, source, repair = agent._estimate_aspects(state_dict, history)

        self.assertEqual(source, "llm")
        self.assertIsNone(repair)
        self.assertEqual(aspects["my_tactical_pattern"], "Current counterattacking construction")
        self.assertEqual(aspects["my_tempo_movement_dynamics"], "Stable recovery across the prefix")
        self.assertEqual(aspects["opponent_tactical_pattern"], "Current defensive reset")
        self.assertEqual(aspects["opponent_tempo_movement_dynamics"], "Current delayed recovery")
        self.assertNotIn("Earlier defensive construction", aspects["semantic_query_text"])

    def test_twin_aspect_segment_validator_rejects_invalid_coverage_and_types(self):
        valid = aspect_response(1, 4)
        valid["aspect_segments"]["my_tactical_pattern"] = [
            {"range": [1, 2], "phase": "Initial phase"},
            {"range": [3, 4], "phase": "Current phase"},
        ]
        self.assertTrue(TwinAgent._validate_aspect_response(valid, 1, 4))

        gap = copy.deepcopy(valid)
        gap["aspect_segments"]["my_tactical_pattern"][1]["range"] = [4, 4]
        self.assertFalse(TwinAgent._validate_aspect_response(gap, 1, 4))

        boolean_range = copy.deepcopy(valid)
        boolean_range["aspect_segments"]["my_tactical_pattern"][0]["range"] = [True, 2]
        self.assertFalse(TwinAgent._validate_aspect_response(boolean_range, 1, 4))

        wrong_end = copy.deepcopy(valid)
        wrong_end["aspect_segments"]["opponent_tactical_pattern"][0]["range"] = [1, 3]
        self.assertFalse(TwinAgent._validate_aspect_response(wrong_end, 1, 4))

    def test_twin_instruction_stays_out_of_retrieval_and_in_action_stages(self):
        instruction = "Aggressive Profile. Prefer attacking pressure."
        current_info = info()
        current_info["rally_profile_instruction"] = instruction
        stage1 = returned_response("The Twin evidence supports a legal Clear.")
        agent = self.twin_agent(
            [aspect_response(), stage1, placement_response()],
            instruction_mode="instruct",
        )

        agent.predict(STATE, current_info, launch=False)

        self.assertNotIn(instruction, agent.llm.prompts[0])
        self.assertIn(instruction, agent.llm.prompts[1])
        self.assertIn(instruction, agent.llm.prompts[2])

        action_response = single_stage_response()
        single_stage = self.twin_single_stage_agent(
            [aspect_response(), action_response],
            instruction_mode="instruct",
        )

        single_stage.predict(STATE, current_info, launch=False)

        self.assertNotIn(instruction, single_stage.llm.prompts[0])
        self.assertIn(instruction, single_stage.llm.prompts[1])

    def test_single_stage_twin_supports_missed_action_in_the_same_call(self):
        response = single_stage_response(11)
        agent = self.twin_single_stage_agent([aspect_response(), response])

        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(len(agent.llm.prompts), 2)
        self.assertIsNone(agent.last_decision_trace["action"]["landing_x"])

    def test_twin_rounds_one_and_two_use_state_only_retrieval(self):
        for ball_round in (1, 2):
            with self.subTest(ball_round=ball_round):
                agent = self.twin_agent(
                    [returned_response(), placement_response()],
                    semantic_start_ball_round=3,
                )
                agent.embedding_model.calls.clear()
                current_info = info()
                current_info["round"] = [ball_round]

                action, _ = agent.predict(STATE, current_info, launch=False)

                self.assertEqual(action[0], 2)
                self.assertEqual(len(agent.llm.prompts), 2)
                self.assertEqual(agent.embedding_model.calls, [])
                for prompt in agent.llm.prompts:
                    self.assertNotIn("Current Twin Semantic Query", prompt)
                    self.assertNotIn("Twin semantic context", prompt)
                trace = agent.last_decision_trace
                self.assertEqual(trace["schema_version"], 7)
                self.assertEqual(trace["retrieval_mode"], "state_only")
                self.assertEqual(trace["pre_stage"], {
                    "architecture": "twin_aspect_v1",
                    "semantic_enabled": False,
                    "activation_ball_round": 3,
                    "source": "skipped_early_round",
                })
                self.assertTrue(trace["case_ids"])
                self.assertTrue(all(
                    "distance" in case and "semantic_similarity" not in case
                    for case in trace["case_ids"]
                ))

    def test_twin_round_three_enables_aspects_and_semantic_embedding(self):
        agent = self.twin_agent(
            [aspect_response(3, 3), returned_response(), placement_response()],
            semantic_start_ball_round=3,
        )
        agent.embedding_model.calls.clear()
        current_info = info()
        current_info["round"] = [3]

        action, _ = agent.predict(STATE, current_info, launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(len(agent.llm.prompts), 3)
        self.assertEqual(len(agent.embedding_model.calls), 1)
        self.assertIn("Annotate the observed badminton rally prefix", agent.llm.prompts[0])
        self.assertIn("Current Twin Semantic Query", agent.llm.prompts[1])
        self.assertIn("Twin semantic context", agent.llm.prompts[1])
        trace = agent.last_decision_trace
        self.assertEqual(trace["retrieval_mode"], "state_semantic")
        self.assertTrue(trace["pre_stage"]["semantic_enabled"])
        self.assertTrue(all(
            "semantic_similarity" in case and "distance" not in case
            for case in trace["case_ids"]
        ))

    def test_single_stage_twin_uses_the_same_round_boundary(self):
        early = self.twin_single_stage_agent(
            [single_stage_response()],
            semantic_start_ball_round=3,
        )
        early.embedding_model.calls.clear()
        early_info = info()
        early_info["round"] = [2]
        early.predict(STATE, early_info, launch=False)

        self.assertEqual(len(early.llm.prompts), 1)
        self.assertEqual(early.embedding_model.calls, [])
        self.assertEqual(early.last_decision_trace["retrieval_mode"], "state_only")
        self.assertNotIn("Current Twin Semantic Query", early.llm.prompts[0])
        self.assertNotIn("Twin semantic context", early.llm.prompts[0])

        semantic = self.twin_single_stage_agent(
            [aspect_response(3, 3), single_stage_response()],
            semantic_start_ball_round=3,
        )
        semantic.embedding_model.calls.clear()
        semantic_info = info()
        semantic_info["round"] = [3]
        semantic.predict(STATE, semantic_info, launch=False)

        self.assertEqual(len(semantic.llm.prompts), 2)
        self.assertEqual(len(semantic.embedding_model.calls), 1)
        self.assertEqual(
            semantic.last_decision_trace["retrieval_mode"], "state_semantic"
        )
        self.assertIn("Current Twin Semantic Query", semantic.llm.prompts[1])

    def test_online_aspect_repair_extends_only_the_terminal_round(self):
        original = aspect_response(1, 2)
        raw_original = copy.deepcopy(original)
        agent = self.twin_agent(
            [original, returned_response(), placement_response()],
            semantic_start_ball_round=3,
            save_prompt_traces=True,
        )

        class RawPreservingLLM(FakeLLM):
            def get_verified_response(self, system_prompt, user_prompt,
                                      validator=None, **kwargs):
                self.prompts.append(user_prompt)
                self.validators.append(validator)
                self.calls.append(kwargs)
                response = self.responses.pop(0)
                raw = json.dumps(response)
                if response is not None and validator is not None:
                    assert validator(response), response
                return response, raw, {}

        agent.llm = RawPreservingLLM([
            original, returned_response(), placement_response(),
        ])
        current_info = info()
        current_info["round"] = [3]
        current_info["rally_history"] = [row(1, 2), row(2, 9)]

        agent.predict(STATE, current_info, launch=False)

        pre_stage = agent.last_decision_trace["pre_stage"]
        self.assertEqual(pre_stage["source"], "llm_terminal_extension")
        self.assertEqual(pre_stage["repair"], {
            "type": "extend_one_terminal_round",
            "aspects": [
                "my_tactical_pattern",
                "my_tempo_movement_dynamics",
                "opponent_tactical_pattern",
                "opponent_tempo_movement_dynamics",
            ],
            "from_round": 2,
            "to_round": 3,
        })
        local_trace = agent.tracer.consume_local_trace()
        aspect_generation = local_trace["generations"][0]
        parsed = aspect_generation["parsed_response"]
        self.assertTrue(all(
            segments[-1]["range"][-1] == 3
            for segments in parsed["aspect_segments"].values()
        ))
        self.assertEqual(json.loads(aspect_generation["raw_output"]), raw_original)

    def test_online_aspect_repair_rejects_non_terminal_coverage_errors(self):
        missing_two = aspect_response(1, 1)
        gap = aspect_response(1, 3)
        gap["aspect_segments"]["my_tactical_pattern"] = [
            {"range": [1, 1], "phase": "Opening phase"},
            {"range": [3, 3], "phase": "Late phase"},
        ]
        wrong_start = aspect_response(2, 2)
        for annotation in (missing_two, gap, wrong_start):
            with self.subTest(annotation=annotation):
                original = copy.deepcopy(annotation)
                repaired, aspects = TwinAgent._repair_online_terminal_aspects(
                    annotation, 1, 3
                )
                self.assertEqual(repaired, original)
                self.assertEqual(aspects, [])


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False, **_):
        import numpy as np

        self.calls.append(list(texts))
        rows = []
        for text in texts:
            row = np.array([len(text), sum(map(ord, text)) % 17, 1.0], dtype=np.float32)
            if normalize_embeddings:
                row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows)


def aspect_response(first_round=2, last_round=2):
    return {
        "aspect_segments": {
            "my_tactical_pattern": [{
                "range": [first_round, last_round],
                "phase": "Controlled return construction.",
            }],
            "my_tempo_movement_dynamics": [{
                "range": [first_round, last_round],
                "phase": "Balanced recovery pressure.",
            }],
            "opponent_tactical_pattern": [{
                "range": [first_round, last_round],
                "phase": "Sustained attacking pressure.",
            }],
            "opponent_tempo_movement_dynamics": [{
                "range": [first_round, last_round],
                "phase": "Forward recovery pressure.",
            }],
        },
    }


if __name__ == "__main__":
    unittest.main()
