import unittest

from Agent.base_agent import BaseAgent
from Agent.naive_rag import NaiveRAGAgent, NaiveRAGDatabase
from Agent.timerag import TimeRAGAgent, TimeRAGDatabase
from tests.test_two_stage_rag import FakeLLM, STATE, dataframe, info


def direct_response(shot_type=2):
    if shot_type == 11:
        return {
            "outcome": "missed",
            "shot_type": None,
            "landing_x": None,
            "landing_y": None,
            "moving_x": None,
            "moving_y": None,
        }
    prompt_action = BaseAgent.build_prompt_action(
        shot_type, 0.3, 0.4, -0.2, -0.1
    )
    return {
        "outcome": "returned",
        **{
            key: prompt_action[key]
            for key in BaseAgent.PROMPT_ACTION_KEYS
        },
    }


class SingleStageDirectParserTest(unittest.TestCase):
    def test_returned_and_missed_responses_are_consistent(self):
        returned = BaseAgent.parse_rag_single_stage_response(
            direct_response(), launch=False, available_types=[2, 9]
        )
        missed = BaseAgent.parse_rag_single_stage_response(
            direct_response(11), launch=False, available_types=[2, 9]
        )

        self.assertEqual(returned["shot_type"], 2)
        self.assertEqual(returned["outcome"], "returned")
        self.assertEqual(missed["shot_type"], 11)
        self.assertEqual(missed["outcome"], "missed")

    def test_response_rejects_outcome_action_mismatches_and_illegal_serve(self):
        mismatched = direct_response()
        mismatched["outcome"] = "missed"
        illegal_serve = direct_response()

        self.assertIsNone(BaseAgent.parse_rag_single_stage_response(mismatched))
        self.assertIsNone(BaseAgent.parse_rag_single_stage_response(
            illegal_serve, launch=True, available_types=[1, 10]
        ))


class SingleStageDirectAgentTest(unittest.TestCase):
    def test_naive_rag_uses_one_retrieval_and_one_generation(self):
        database = NaiveRAGDatabase(dataframe(), "Player A", 0)
        agent = NaiveRAGAgent(
            "Player A",
            database,
            enable_tracing=False,
            case_grounding=False,
            action_architecture="single_stage_v1",
        )
        agent.llm = FakeLLM([direct_response()])
        agent.database.retrieve_by_shot_type = lambda *_args, **_kwargs: self.fail(
            "single-stage NaiveRAG must not perform shot-conditioned retrieval"
        )

        action, reasoning = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(reasoning, "")
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertIn("Successful Return Cases", agent.llm.prompts[0])
        self.assertIn('Action: {"shot_type": "Clear"', agent.llm.prompts[0])
        self.assertNotIn("Tactical Imitation Instruction", agent.llm.prompts[0])
        self.assertNotIn('"case_assessments":', agent.llm.prompts[0])
        self.assertEqual(agent.llm.calls[0]["schema_name"], "rag_single_stage_direct")
        self.assertEqual(agent.last_decision_trace["architecture"], "single_stage_v1")
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])
        self.assertNotIn("stage2", agent.last_decision_trace)

    def test_timerag_uses_one_retrieval_and_one_generation(self):
        database = TimeRAGDatabase(dataframe(), "Player A", 0, window_size=2)
        agent = TimeRAGAgent(
            "Player A",
            database,
            enable_tracing=False,
            case_grounding=False,
            action_architecture="single_stage_v1",
        )
        agent.llm = FakeLLM([direct_response()])
        agent.database.retrieve_by_shot_type = lambda *_args, **_kwargs: self.fail(
            "single-stage TimeRAG must not perform shot-conditioned retrieval"
        )

        action, reasoning = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(reasoning, "")
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertIn("Successful Return Windows", agent.llm.prompts[0])
        self.assertIn('Action: {"shot_type": "Clear"', agent.llm.prompts[0])
        self.assertNotIn("Tactical Imitation Instruction", agent.llm.prompts[0])
        self.assertNotIn('"case_assessments":', agent.llm.prompts[0])
        self.assertEqual(agent.llm.calls[0]["schema_name"], "rag_single_stage_direct")
        self.assertEqual(agent.last_decision_trace["architecture"], "single_stage_v1")
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])
        self.assertNotIn("stage2", agent.last_decision_trace)

    def test_naive_rag_instruction_only_changes_the_action_prompt(self):
        database = NaiveRAGDatabase(dataframe(), "Player A", 0)
        retrieval_calls = []
        retrieve_stage1 = database.retrieve_stage1
        database.retrieve_stage1 = lambda *args, **kwargs: (
            retrieval_calls.append((args, kwargs))
            or retrieve_stage1(*args, **kwargs)
        )
        agent = NaiveRAGAgent(
            "Player A",
            database,
            enable_tracing=False,
            prompt_variant="shifted",
            case_grounding=False,
            action_architecture="single_stage_v1",
        )
        agent.llm = FakeLLM([direct_response()])
        agent.database.retrieve_by_shot_type = lambda *_args, **_kwargs: self.fail(
            "instructed single-stage NaiveRAG must not use conditioned retrieval"
        )
        current_info = info()
        instruction = "Front-Court Pressure. Prefer front-court pressure."
        current_info["rally_profile_instruction"] = instruction

        action, reasoning = agent.predict(STATE, current_info, launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(reasoning, "")
        self.assertEqual(len(retrieval_calls), 1)
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertIn("Tactical Imitation Instruction", agent.llm.prompts[0])
        self.assertIn(instruction, agent.llm.prompts[0])
        self.assertIn("only to break a tie", agent.llm.prompts[0])
        self.assertEqual(agent.llm.calls[0]["schema_name"], "rag_single_stage_direct")
        self.assertFalse(agent.last_decision_trace["case_grounding"])
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])
        self.assertNotIn("stage2", agent.last_decision_trace)

    def test_timerag_instruction_only_changes_the_action_prompt(self):
        database = TimeRAGDatabase(dataframe(), "Player A", 0, window_size=2)
        retrieval_calls = []
        retrieve_stage1 = database.retrieve_stage1
        database.retrieve_stage1 = lambda *args, **kwargs: (
            retrieval_calls.append((args, kwargs))
            or retrieve_stage1(*args, **kwargs)
        )
        agent = TimeRAGAgent(
            "Player A",
            database,
            enable_tracing=False,
            prompt_variant="shifted",
            case_grounding=False,
            action_architecture="single_stage_v1",
        )
        agent.llm = FakeLLM([direct_response()])
        agent.database.retrieve_by_shot_type = lambda *_args, **_kwargs: self.fail(
            "instructed single-stage TimeRAG must not use conditioned retrieval"
        )
        current_info = info()
        instruction = "Front-Court Pressure. Prefer front-court pressure."
        current_info["rally_profile_instruction"] = instruction

        action, reasoning = agent.predict(STATE, current_info, launch=False)

        self.assertEqual(action[0], 2)
        self.assertEqual(reasoning, "")
        self.assertEqual(len(retrieval_calls), 1)
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertIn("Tactical Imitation Instruction", agent.llm.prompts[0])
        self.assertIn(instruction, agent.llm.prompts[0])
        self.assertIn("only to break a tie", agent.llm.prompts[0])
        self.assertEqual(agent.llm.calls[0]["schema_name"], "rag_single_stage_direct")
        self.assertFalse(agent.last_decision_trace["case_grounding"])
        self.assertEqual(agent.last_decision_trace["case_assessments"], [])
        self.assertNotIn("stage2", agent.last_decision_trace)

    def test_missed_response_finishes_without_second_stage(self):
        database = NaiveRAGDatabase(dataframe(), "Player A", 0)
        agent = NaiveRAGAgent(
            "Player A",
            database,
            enable_tracing=False,
            case_grounding=False,
            action_architecture="single_stage_v1",
        )
        agent.llm = FakeLLM([direct_response(11)])

        action, _ = agent.predict(STATE, info(), launch=False)

        self.assertEqual(action[0], 11)
        self.assertEqual(len(agent.llm.prompts), 1)
        self.assertEqual(agent.last_decision_trace["outcome"], "missed")
        self.assertIsNone(agent.last_decision_trace["action"]["landing_x"])


if __name__ == "__main__":
    unittest.main()
