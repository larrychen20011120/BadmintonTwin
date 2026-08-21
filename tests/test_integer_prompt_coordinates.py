import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from Agent.base_agent import BaseAgent
from Scripts.single_player_eval.evaluate_trajectories import extract_sequences
from Scripts.single_player_eval.trajectory_store import (
    build_gt_shot_from_row,
    build_pred_shot,
)
from Utils.badminton_config import LANG_AREA_MAPPING
from Utils.helper import coord2area
from Utils.llm import LLM


def llm_response_action(shot_type, landing_x, landing_y, moving_x, moving_y):
    evidence_action = BaseAgent.build_prompt_action(
        shot_type, landing_x, landing_y, moving_x, moving_y
    )
    return {
        key: evidence_action[key]
        for key in BaseAgent.PROMPT_ACTION_KEYS
    }


class PromptCoordinateTest(unittest.TestCase):
    def test_integer_grid_round_trip(self):
        for x in range(356):
            normalized_x, _ = BaseAgent.prompt_to_normalized_coordinate(x, 0)
            prompt_x, _ = BaseAgent.normalized_to_prompt_coordinate(normalized_x, -1.0)
            self.assertEqual(prompt_x, x)

        for y in range(481):
            _, normalized_y = BaseAgent.prompt_to_normalized_coordinate(0, y)
            _, prompt_y = BaseAgent.normalized_to_prompt_coordinate(-1.0, normalized_y)
            self.assertEqual(prompt_y, y)

    def test_prompt_area_partitions_entire_grid(self):
        expected_areas = {
            "Deep Left Corner", "Deep Right Corner",
            "Rear Left Side", "Rear Center", "Rear Right Side",
            "Rear Left Side Line", "Rear Right Side Line",
            "Mid Left Side", "Mid Center", "Mid Right Side",
            "Mid Left Side Line", "Mid Right Side Line",
            "Front Left Side", "Front Center", "Front Right Side",
            "Front Left Side Line", "Front Right Side Line",
        }
        actual_areas = {
            BaseAgent.prompt_area_name(x, y)
            for x in range(356)
            for y in range(481)
        }
        self.assertEqual(actual_areas, expected_areas)

    def test_prompt_area_matches_existing_coord2area(self):
        for x in range(356):
            for y in range(481):
                normalized_x, normalized_y = BaseAgent.prompt_to_normalized_coordinate(x, y)
                existing_area = LANG_AREA_MAPPING[coord2area(normalized_x, normalized_y)]
                self.assertEqual(BaseAgent.prompt_area_name(x, y), existing_area)

    def test_offline_prompt_action_uses_integer_coordinates_and_areas(self):
        action = BaseAgent.build_prompt_action(1, 0.2, 0.3, -0.15, -0.25)
        self.assertEqual(action, {
            "shot_type": "Serve short",
            "landing_area": "Mid Center",
            "landing_x": 213,
            "landing_y": 312,
            "moving_area": "Rear Center",
            "moving_x": 150,
            "moving_y": 180,
        })

    def test_normalized_action_summary_uses_derived_areas(self):
        self.assertEqual(
            BaseAgent.summarize_normalized_action(1, 0.2, 0.3, -0.15, -0.25),
            "Returned Serve short; Mid Center landing with Rear Center recovery.",
        )
        self.assertEqual(
            BaseAgent.summarize_normalized_action(11),
            "Missed return.",
        )

    def test_missed_prompt_action_is_null(self):
        action = BaseAgent.build_prompt_action(11, None, None, None, None)
        self.assertEqual(action["shot_type"], "Missed shot")
        self.assertTrue(all(
            value is None for key, value in action.items() if key != "shot_type"
        ))
        self.assertIn('"landing_x": null', BaseAgent.format_prompt_action(action))

    def test_online_parser_returns_normalized_coordinates(self):
        response = {
            "rationale": "Imitate the retrieved serve.",
            "action": llm_response_action(1, 0.2, 0.3, -0.15, -0.25),
        }
        parsed, reasoning = BaseAgent.parse_prompt_action(response)
        self.assertEqual(reasoning, "Imitate the retrieved serve.")
        self.assertEqual(parsed["shot_type"], 1)
        self.assertAlmostEqual(parsed["landing_x"], 213 / 177.5 - 1)
        self.assertAlmostEqual(parsed["landing_y"], 312 / 240 - 1)

    def test_online_parser_rejects_decimal_and_extra_fields(self):
        action = llm_response_action(2, 0.2, 0.3, -0.15, -0.25)
        action["landing_x"] = 213.0
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)

        action = llm_response_action(2, 0.2, 0.3, -0.15, -0.25)
        action["landing_area"] = "Unknown Area"
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)

    def test_online_parser_derives_areas_from_coordinates(self):
        action = {
            "shot_type": "Net Shot",
            "landing_x": 186,
            "landing_y": 440,
            "moving_x": 50,
            "moving_y": 420,
        }
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertEqual(parsed["landing_area"], "Front Center")
        self.assertEqual(parsed["moving_area"], "Front Left Side")

    def test_online_parser_accepts_grid_endpoints_only(self):
        action = llm_response_action(2, 1.0, 1.0, 1.0, 1.0)
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNotNone(parsed)

        action["landing_x"] = 356
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)

        action["landing_x"] = 355
        action["landing_y"] = 481
        parsed, _ = BaseAgent.parse_prompt_action({"action": action})
        self.assertIsNone(parsed)


class MissedTrajectoryTest(unittest.TestCase):
    def test_trajectory_builders_store_null_for_missed_shots(self):
        row = pd.Series({
            "player": "Player A", "ball_round": 2, "player_type": 11,
            "landing_x": 0.4, "landing_y": 0.5,
            "moving_x": -0.2, "moving_y": -0.3,
            "player_location_x": 0.0, "player_location_y": 0.0,
            "hit_x": 0.1, "hit_y": 0.2,
            "opponent_location_x": -0.1, "opponent_location_y": -0.2,
            "player_score": 1, "opponent_score": 1,
        })
        gt = build_gt_shot_from_row(row)
        pred = build_pred_shot("Player A", 11, [0.4, 0.5], [-0.2, -0.3], 2)
        for step in (gt, pred):
            self.assertIsNone(step["landing_x"])
            self.assertIsNone(step["landing_y"])
            self.assertIsNone(step["moving_x"])
            self.assertIsNone(step["moving_y"])

    def test_position_metrics_skip_missed_coordinates(self):
        normal = {
            "player": "Player A", "ball_round": 1, "shot_type": 2,
            "landing_x": 0.2, "landing_y": 0.3,
            "moving_x": -0.1, "moving_y": -0.2,
        }
        missed = {
            "player": "Player A", "ball_round": 2, "shot_type": 11,
            "landing_x": None, "landing_y": None,
            "moving_x": None, "moving_y": None,
        }
        trajectory = {
            "target_player": "Player A",
            "gt_sequence": [normal, missed],
            "pred_sequence": [normal, missed],
        }
        extracted = extract_sequences([trajectory])
        self.assertEqual(extracted["all_land_gt"].shape, (1, 2))
        self.assertEqual(extracted["all_move_pred"].shape, (1, 2))
        self.assertEqual(extracted["all_shot_gt"].tolist(), [2, 11])


class LLMActionValidationTest(unittest.TestCase):
    def test_two_stage_grounded_contract_locks_shot_and_derives_areas(self):
        normal_cases = [{"player_type": 2}, {"player_type": 9}]
        miss_cases = [{"player_type": 11}]
        returned = {
            "case_assessments": [
                {"source": "success", "retrieval_rank": 1,
                 "impact": "support", "factors": ["same incoming shot"]},
                {"source": "success", "retrieval_rank": 2,
                 "impact": "conflict", "factors": ["different score pressure"]},
                {"source": "miss", "retrieval_rank": 1,
                 "impact": "irrelevant", "factors": ["lower reachability"]},
            ],
            "outcome": "returned",
            "shot_type": "Clear",
        }
        parsed = BaseAgent.parse_rag_stage1_response(
            returned,
            launch=False,
            available_types={2, 9},
            case_grounding=True,
            normal_cases=normal_cases,
            miss_cases=miss_cases,
        )
        self.assertEqual(parsed["shot_type"], 2)
        self.assertEqual(len(parsed["case_assessments"]), 3)
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(
            {**returned, "shot_type": "Serve short"},
            launch=False,
            available_types={1, 2},
            case_grounding=True,
            normal_cases=normal_cases,
            miss_cases=miss_cases,
        ))
        self.assertEqual(
            BaseAgent.parse_rag_stage1_response({
                "case_assessments": [
                    {"source": "success", "retrieval_rank": 1,
                     "impact": "conflict", "factors": ["lower reachability"]},
                    {"source": "success", "retrieval_rank": 2,
                     "impact": "conflict", "factors": ["harder shuttle arrival"]},
                    {"source": "miss", "retrieval_rank": 1,
                     "impact": "support", "factors": ["failed contact"]},
                ],
                "outcome": "missed",
                "shot_type": None,
            }, case_grounding=True, normal_cases=normal_cases,
                miss_cases=miss_cases)["shot_type"],
            11,
        )

        placement = llm_response_action(2, 0.2, 0.3, -0.15, -0.25)
        placement.pop("shot_type")
        stage2 = BaseAgent.parse_rag_stage2_response({
            "case_assessments": [
                {"retrieval_rank": 1,
                 "impact": "support", "factors": ["same landing"]},
                {"retrieval_rank": 2,
                 "impact": "conflict", "factors": ["different recovery"]},
            ],
            "placement": placement,
        }, case_grounding=True, conditioned_cases=normal_cases)
        self.assertIsNotNone(stage2)
        self.assertEqual(stage2["landing_area"], "Mid Center")
        self.assertEqual(stage2["moving_area"], "Rear Center")
        placement["moving_area"] = "Rear Center"
        extra_area = BaseAgent.parse_rag_stage2_response({
            "case_assessments": stage2["case_assessments"],
            "placement": placement,
        }, case_grounding=True, conditioned_cases=normal_cases)
        self.assertIsNone(extra_area)

    def test_two_stage_direct_contract_has_empty_compatibility_fields(self):
        direct = BaseAgent.parse_rag_stage1_response(
            {"outcome": "returned", "shot_type": "Drop"},
            launch=False,
            available_types={9},
            case_grounding=False,
        )
        self.assertEqual(direct["case_assessments"], [])

        placement = llm_response_action(9, 0.2, 0.3, -0.15, -0.25)
        placement.pop("shot_type")
        parsed = BaseAgent.parse_rag_stage2_response(
            {"placement": placement}, case_grounding=False
        )
        self.assertEqual(parsed["case_assessments"], [])

    def test_grounded_case_assessments_require_exhaustive_ordered_coverage(self):
        normal_cases = [{"player_type": 2}, {"player_type": 9}]
        miss_cases = [{"player_type": 11}]
        base = {
            "case_assessments": [
                {"source": "success", "retrieval_rank": 1,
                 "impact": "support", "factors": ["same incoming shot"]},
                {"source": "success", "retrieval_rank": 2,
                 "impact": "conflict", "factors": ["different shot choice"]},
                {"source": "miss", "retrieval_rank": 1,
                 "impact": "irrelevant", "factors": ["different shuttle arrival"]},
            ],
            "outcome": "returned",
            "shot_type": "Clear",
        }
        kwargs = {
            "case_grounding": True,
            "normal_cases": normal_cases,
            "miss_cases": miss_cases,
        }
        self.assertIsNotNone(BaseAgent.parse_rag_stage1_response(base, **kwargs))

        invalid_rank = json.loads(json.dumps(base))
        invalid_rank["case_assessments"][0]["retrieval_rank"] = 3
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(invalid_rank, **kwargs))

        missing_case = json.loads(json.dumps(base))
        missing_case["case_assessments"].pop()
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(missing_case, **kwargs))

        wrong_order = json.loads(json.dumps(base))
        wrong_order["case_assessments"][0], wrong_order["case_assessments"][1] = (
            wrong_order["case_assessments"][1], wrong_order["case_assessments"][0]
        )
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(wrong_order, **kwargs))

        wrong_shot = {**base, "shot_type": "Drop"}
        self.assertIsNotNone(BaseAgent.parse_rag_stage1_response(wrong_shot, **kwargs))

        no_miss = {**base, "outcome": "missed", "shot_type": None}
        self.assertIsNotNone(BaseAgent.parse_rag_stage1_response(no_miss, **kwargs))

        for invalid_factor in (
            "", "five words exceed factor limit", "same shot and position",
            "case match", "rank one", "coordinate 220", "trailing punctuation.",
            "landing zone differs",
        ):
            invalid = json.loads(json.dumps(base))
            invalid["case_assessments"][0]["factors"] = [invalid_factor]
            self.assertIsNone(BaseAgent.parse_rag_stage1_response(invalid, **kwargs))

        duplicate = json.loads(json.dumps(base))
        duplicate["case_assessments"][0]["factors"] = ["same shot", "Same  shot"]
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(duplicate, **kwargs))

        too_many = json.loads(json.dumps(base))
        too_many["case_assessments"][0]["factors"] = ["same shot", "same player", "same score"]
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(too_many, **kwargs))

        invalid_impact = json.loads(json.dumps(base))
        invalid_impact["case_assessments"][0]["impact"] = "mixed"
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(invalid_impact, **kwargs))

        obsolete_reason = {**base, "decision_reason": "Obsolete."}
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(obsolete_reason, **kwargs))

        obsolete = {
            "rationale": "Obsolete.", "outcome": "returned", "shot_type": "Clear"
        }
        self.assertIsNone(BaseAgent.parse_rag_stage1_response(
            obsolete, case_grounding=False
        ))

    def test_parseable_stage2_response_is_not_retried_without_validator(self):
        placement = llm_response_action(2, 0.2, 0.3, -0.15, -0.25)
        placement.pop("shot_type")
        response = json.dumps({
            "rationale": "Parseable coordinate-only placement.",
            "placement": placement,
        })
        calls = []

        llm = LLM.__new__(LLM)
        llm.get_response = lambda **_: (calls.append(1) or response, {})
        parsed, _, _ = llm.get_verified_response(max_trials=5)

        self.assertEqual(len(calls), 1)
        self.assertEqual(parsed["placement"], placement)

    def test_invalid_action_is_retried_before_return(self):
        invalid_action = llm_response_action(2, 0.2, 0.3, -0.1, -0.2)
        invalid_action["landing_x"] = float(invalid_action["landing_x"])
        valid_action = llm_response_action(2, 0.2, 0.3, -0.1, -0.2)
        responses = iter([
            json.dumps({"action": invalid_action}),
            json.dumps({"action": valid_action}),
        ])

        llm = LLM.__new__(LLM)
        llm.get_response = lambda **_: (next(responses), {})
        parsed, _, _ = llm.get_verified_response(
            max_trials=2,
            validator=BaseAgent.validate_prompt_action_response,
        )
        self.assertEqual(parsed["action"], valid_action)


class LLMStructuredOutputTest(unittest.TestCase):
    @staticmethod
    def _llm_with_response(content):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=None,
            )

        llm = LLM.__new__(LLM)
        llm.model_name = "test-model"
        llm.generation_params = {
            "temperature": 0,
            "extra_body": {"nested": {"configured": True}},
        }
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return llm, requests

    def test_schema_is_sent_as_standard_openai_response_format(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        generation_kwargs = {"extra_body": {"nested": {"requested": True}}}
        llm, requests = self._llm_with_response('{"answer":"ok"}')

        parsed, _, _ = llm.get_verified_response(
            schema=schema,
            schema_name="answer_schema",
            generation_kwargs=generation_kwargs,
        )

        self.assertEqual(parsed, {"answer": "ok"})
        self.assertEqual(requests[0]["response_format"], {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "schema": schema,
                "strict": True,
            },
        })
        self.assertEqual(requests[0]["extra_body"]["nested"], {
            "configured": True,
            "requested": True,
        })
        self.assertEqual(generation_kwargs, {
            "extra_body": {"nested": {"requested": True}}
        })
        self.assertNotIn("response_format", llm.generation_params)

    def test_schema_none_preserves_unstructured_request(self):
        llm, requests = self._llm_with_response('{"answer":"ok"}')
        llm.get_verified_response(schema=None)
        self.assertNotIn("response_format", requests[0])

    def test_unnamed_schema_has_deterministic_name(self):
        first = {"type": "object", "properties": {"value": {"type": "string"}}}
        second = {"properties": {"value": {"type": "string"}}, "type": "object"}
        first_name = LLM._schema_response_format(first)["json_schema"]["name"]
        second_name = LLM._schema_response_format(second)["json_schema"]["name"]
        self.assertEqual(first_name, second_name)
        self.assertRegex(first_name, r"^structured_response_[0-9a-f]{16}$")

    def test_json_repair_follows_strict_parse_without_regex_rewriting(self):
        repair_calls = []

        def repair(text):
            repair_calls.append(text)
            return {"repaired": True}

        with patch("Utils.llm.repair_json_loads", repair):
            self.assertEqual(LLM._extract_json('{"strict":true}'), {"strict": True})
            self.assertEqual(LLM._extract_json("{broken}"), {"repaired": True})
        self.assertEqual(repair_calls, ["{broken}"])

        with patch("Utils.llm.repair_json_loads", None):
            self.assertIsNone(LLM._extract_json("{unquoted: 'value',}"))

    def test_rag_schemas_separate_grounded_assessments_from_direct_output(self):
        schema_root = Path(__file__).resolve().parents[1] / "Schemas" / "Agent"
        for stage in (1, 2):
            grounded = json.loads(
                (schema_root / f"rag_stage{stage}_grounded_schema.json").read_text()
            )
            assessments = grounded["properties"]["case_assessments"]
            citation = assessments["items"]
            rank = citation["properties"]["retrieval_rank"]
            self.assertEqual(rank, {"type": "integer", "minimum": 1})
            self.assertNotIn("maximum", rank)
            self.assertNotIn("enum", rank)
            self.assertEqual(assessments["minItems"], 0)
            self.assertNotIn("maxItems", assessments)
            self.assertNotIn("decision_reason", grounded["properties"])
            self.assertNotIn("summary", grounded["properties"])
            self.assertEqual(
                citation["properties"]["impact"]["enum"],
                ["support", "conflict", "irrelevant"],
            )
            factors = citation["properties"]["factors"]
            self.assertEqual(factors["minItems"], 1)
            self.assertEqual(factors["maxItems"], 2)
            self.assertEqual(factors["items"]["maxLength"], 32)
            self.assertIn("^[a-z]+", factors["items"]["pattern"])
            self.assertNotIn("uniqueItems", factors)
            if stage == 1:
                self.assertEqual(
                    citation["required"],
                    ["source", "retrieval_rank", "impact", "factors"],
                )
                self.assertEqual(
                    citation["properties"]["source"]["enum"],
                    ["success", "miss"],
                )
            else:
                self.assertEqual(
                    citation["required"],
                    ["retrieval_rank", "impact", "factors"],
                )

        stage1_direct = json.loads(
            (schema_root / "rag_stage1_direct_schema.json").read_text()
        )
        stage2_direct = json.loads(
            (schema_root / "rag_stage2_direct_schema.json").read_text()
        )
        stage2_grounded = json.loads(
            (schema_root / "rag_stage2_grounded_schema.json").read_text()
        )
        self.assertEqual(
            stage1_direct["required"], ["outcome", "shot_type"]
        )
        self.assertEqual(stage2_direct["required"], ["placement"])
        coordinate_keys = {"landing_x", "landing_y", "moving_x", "moving_y"}
        for schema in (stage2_grounded, stage2_direct):
            placement = schema["properties"]["placement"]
            self.assertEqual(set(placement["required"]), coordinate_keys)
            self.assertEqual(set(placement["properties"]), coordinate_keys)

        for mode in ("grounded", "direct"):
            schema = json.loads(
                (schema_root / f"twin_single_stage_{mode}_schema.json").read_text()
            )
            action = schema["properties"]["action"]
            self.assertEqual(
                set(action["required"]), coordinate_keys | {"shot_type"}
            )
            self.assertEqual(
                set(action["properties"]), coordinate_keys | {"shot_type"}
            )
            if mode == "grounded":
                self.assertIn("case_assessments", schema["properties"])
                self.assertNotIn("summary", schema["properties"])
                self.assertNotIn("decision_reason", schema["properties"])

    def test_rag_schema_locks_assessment_count_per_prompt(self):
        schema, _ = BaseAgent.get_rag_response_schema(1, True)
        constrained = BaseAgent.constrain_case_assessment_count(schema, 14)

        self.assertEqual(
            constrained["properties"]["case_assessments"]["minItems"], 14
        )
        self.assertEqual(
            constrained["properties"]["case_assessments"]["maxItems"], 14
        )
        self.assertEqual(schema["properties"]["case_assessments"]["minItems"], 0)
        self.assertNotIn("maxItems", schema["properties"]["case_assessments"])


if __name__ == "__main__":
    unittest.main()
