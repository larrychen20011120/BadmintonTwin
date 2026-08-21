import copy
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from dtaidistance import dtw

from Utils.badminton_config import INDEX2EN_SHOT, shot_type_to_index

class BaseAgent:
    PROMPT_ACTION_KEYS = {
        "shot_type",
        "landing_x",
        "landing_y",
        "moving_x",
        "moving_y",
    }
    RAG_STAGE1_GROUNDED_KEYS = {"case_assessments", "outcome", "shot_type"}
    RAG_STAGE1_DIRECT_KEYS = {"outcome", "shot_type"}
    RAG_STAGE2_GROUNDED_KEYS = {"case_assessments", "placement"}
    RAG_STAGE2_DIRECT_KEYS = {"placement"}
    RAG_SINGLE_STAGE_DIRECT_KEYS = {
        "outcome", "shot_type", "landing_x", "landing_y", "moving_x", "moving_y"
    }
    RAG_STAGE1_ASSESSMENT_KEYS = {
        "source", "retrieval_rank", "impact", "factors"
    }
    RAG_STAGE2_ASSESSMENT_KEYS = {"retrieval_rank", "impact", "factors"}
    RAG_IMPACTS = {"support", "conflict", "irrelevant"}
    RAG_FACTOR_BANNED_WORDS = {
        "all", "although", "and", "because", "but", "case", "cases",
        "aligns", "causes", "covers", "differs", "favors", "favours",
        "leads", "maintains", "matches", "or", "other", "others",
        "produces", "rank", "ranks", "results", "shows", "supports",
        "targets", "uses", "while",
    }
    RAG_PLACEMENT_KEYS = {
        "landing_x",
        "landing_y",
        "moving_x",
        "moving_y",
    }

    def __init__(self, player_name, is_infernce=True):

        self.player_name = player_name
        # is_infernce       --> [-177.5~177.5, 0~-480]
        # not_is_inference  --> [-1~1. -1~1]
        self.is_infernce = is_infernce

        self.tw_to_en_type = {
            "發短球": "Serve short",
            "發長球": "Serve long",
            "長球": "Clear",
            "推撲球": "Push Shot",
            "殺球": "Smash",
            "接殺防守": "Smash Defence",
            "平球": "Drive",
            "接不到": "Missed shot",
            "網前球": "Net Shot",
            "挑球": "Lob",
            "切球": "Drop",
        }

        self.type_to_index = {
            "Serve short": 1,
            "Clear": 2,
            "Push Shot": 3,
            "Smash": 4,
            "Smash Defence": 5,
            "Drive": 6,
            "Net Shot": 7,
            "Lob": 8,
            "Drop": 9,
            "Serve long": 10,
            "Missed shot": 11,
        }
        self.index_to_type = {
            v: k for k, v in self.type_to_index.items()
        }

    def predict(self, state, info, launch):
        raise NotImplementedError("Subclasses must implement evaluate")

    @staticmethod
    def normalized_to_prompt_coordinate(x, y):
        """Quantize normalized coordinates to the integer prompt grid."""
        x = float(x)
        y = float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Prompt coordinates must be finite")
        return (
            math.floor((x + 1.0) * 177.5 + 1e-9),
            math.floor((y + 1.0) * 240.0 + 1e-9),
        )

    @staticmethod
    def prompt_to_normalized_coordinate(x, y):
        """Convert integer prompt-grid coordinates back to normalized values."""
        return float(x) / 177.5 - 1.0, float(y) / 240.0 - 1.0

    @staticmethod
    def prompt_area_name(x, y):
        """Classify one integer prompt-grid position using the prompt definition."""
        if x < 0 or x > 355 or y < 0 or y > 480:
            return "Out of Bounds"

        if x <= 49:
            if y <= 204:
                return "Rear Left Side Line"
            if y <= 365:
                return "Mid Left Side Line"
            return "Front Left Side Line"

        if x >= 306 or (x == 305 and y >= 151):
            if y <= 204:
                return "Rear Right Side Line"
            if y <= 365:
                return "Mid Right Side Line"
            return "Front Right Side Line"

        if y <= 150:
            return "Deep Left Corner" if x <= 177 else "Deep Right Corner"

        if x <= 103:
            side = "Left Side"
        elif x <= 250:
            side = "Center"
        else:
            side = "Right Side"

        if y <= 257:
            return f"Rear {side}"
        if y <= 365:
            return f"Mid {side}"
        return f"Front {side}"

    @classmethod
    def build_prompt_action(cls, shot_type, landing_x, landing_y, moving_x, moving_y):
        """Build the canonical integer action shown in offline LLM evidence."""
        shot_type_index = shot_type_to_index(shot_type)
        shot_type_name = INDEX2EN_SHOT[shot_type_index]
        if shot_type_index == 11:
            return {
                "shot_type": shot_type_name,
                "landing_area": None,
                "landing_x": None,
                "landing_y": None,
                "moving_area": None,
                "moving_x": None,
                "moving_y": None,
            }

        prompt_landing_x, prompt_landing_y = cls.normalized_to_prompt_coordinate(
            landing_x, landing_y
        )
        prompt_moving_x, prompt_moving_y = cls.normalized_to_prompt_coordinate(
            moving_x, moving_y
        )
        return {
            "shot_type": shot_type_name,
            "landing_area": cls.prompt_area_name(prompt_landing_x, prompt_landing_y),
            "landing_x": prompt_landing_x,
            "landing_y": prompt_landing_y,
            "moving_area": cls.prompt_area_name(prompt_moving_x, prompt_moving_y),
            "moving_x": prompt_moving_x,
            "moving_y": prompt_moving_y,
        }

    @staticmethod
    def format_prompt_action(action):
        return f"Action: {json.dumps(action, ensure_ascii=True)}"

    @classmethod
    def summarize_normalized_action(
        cls, shot_type, landing_x=None, landing_y=None, moving_x=None, moving_y=None
    ):
        """Build a compact trace summary from the final normalized action."""
        shot_type_index = shot_type_to_index(shot_type)
        if shot_type_index == 11:
            return "Missed return."
        prompt_landing = cls.normalized_to_prompt_coordinate(landing_x, landing_y)
        prompt_moving = cls.normalized_to_prompt_coordinate(moving_x, moving_y)
        return (
            f"Returned {INDEX2EN_SHOT[shot_type_index]}; "
            f"{cls.prompt_area_name(*prompt_landing)} landing with "
            f"{cls.prompt_area_name(*prompt_moving)} recovery."
        )

    @classmethod
    def parse_prompt_action(cls, response):
        """Validate an LLM action and convert its integer coordinates to normalized values."""
        if not isinstance(response, dict):
            return None, ""

        reasoning = response.get("rationale", response.get("reasoning", ""))
        action = response.get("action", response)
        if not isinstance(action, dict) or set(action) != cls.PROMPT_ACTION_KEYS:
            return None, reasoning

        try:
            shot_type = shot_type_to_index(action["shot_type"])
        except (TypeError, ValueError):
            return None, reasoning

        coordinate_keys = ("landing_x", "landing_y", "moving_x", "moving_y")
        if shot_type == 11:
            if any(action[key] is not None for key in coordinate_keys):
                return None, reasoning
            return {
                "shot_type": shot_type,
                "landing_area": None,
                "landing_x": None,
                "landing_y": None,
                "moving_area": None,
                "moving_x": None,
                "moving_y": None,
            }, reasoning

        coordinates = [action[key] for key in coordinate_keys]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates):
            return None, reasoning

        landing_x, landing_y, moving_x, moving_y = coordinates
        if not (0 <= landing_x <= 355 and 0 <= moving_x <= 355):
            return None, reasoning
        if not (0 <= landing_y <= 480 and 0 <= moving_y <= 480):
            return None, reasoning
        normalized_landing_x, normalized_landing_y = cls.prompt_to_normalized_coordinate(
            landing_x, landing_y
        )
        normalized_moving_x, normalized_moving_y = cls.prompt_to_normalized_coordinate(
            moving_x, moving_y
        )
        return {
            "shot_type": shot_type,
            "landing_area": cls.prompt_area_name(landing_x, landing_y),
            "landing_x": normalized_landing_x,
            "landing_y": normalized_landing_y,
            "moving_area": cls.prompt_area_name(moving_x, moving_y),
            "moving_x": normalized_moving_x,
            "moving_y": normalized_moving_y,
        }, reasoning

    @classmethod
    def validate_prompt_action_response(cls, response):
        parsed, _ = cls.parse_prompt_action(response)
        return parsed is not None

    @staticmethod
    @lru_cache(maxsize=1)
    def get_rag_single_stage_response_schema():
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "Schemas"
            / "Agent"
            / "rag_single_stage_direct_schema.json"
        )
        with schema_path.open("r", encoding="utf-8") as schema_file:
            return json.load(schema_file), "rag_single_stage_direct"

    @classmethod
    def parse_rag_single_stage_response(
        cls, response, launch=False, available_types=None
    ):
        """Validate one direct outcome/action response and normalize coordinates."""
        if (
            not isinstance(response, dict)
            or set(response) != cls.RAG_SINGLE_STAGE_DIRECT_KEYS
        ):
            return None

        outcome = response.get("outcome")
        coordinate_keys = ("landing_x", "landing_y", "moving_x", "moving_y")
        if outcome == "missed":
            if response.get("shot_type") is not None:
                return None
            if any(response.get(key) is not None for key in coordinate_keys):
                return None
            return {
                "outcome": "missed",
                "shot_type": 11,
                "landing_area": None,
                "landing_x": None,
                "landing_y": None,
                "moving_area": None,
                "moving_x": None,
                "moving_y": None,
            }
        if outcome != "returned":
            return None

        try:
            shot_type = shot_type_to_index(response.get("shot_type"))
        except (TypeError, ValueError):
            return None
        legal_types = {1, 10} if launch else set(range(2, 10))
        if shot_type not in legal_types:
            return None
        if available_types is not None and shot_type not in set(available_types):
            return None

        coordinates = [response.get(key) for key in coordinate_keys]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinates
        ):
            return None
        landing_x, landing_y, moving_x, moving_y = coordinates
        if not (0 <= landing_x <= 355 and 0 <= moving_x <= 355):
            return None
        if not (0 <= landing_y <= 480 and 0 <= moving_y <= 480):
            return None
        normalized_landing = cls.prompt_to_normalized_coordinate(landing_x, landing_y)
        normalized_moving = cls.prompt_to_normalized_coordinate(moving_x, moving_y)
        return {
            "outcome": "returned",
            "shot_type": shot_type,
            "landing_area": cls.prompt_area_name(landing_x, landing_y),
            "landing_x": normalized_landing[0],
            "landing_y": normalized_landing[1],
            "moving_area": cls.prompt_area_name(moving_x, moving_y),
            "moving_x": normalized_moving[0],
            "moving_y": normalized_moving[1],
        }

    @classmethod
    def validate_rag_single_stage_response(
        cls, response, launch=False, available_types=None
    ):
        return cls.parse_rag_single_stage_response(
            response, launch=launch, available_types=available_types
        ) is not None

    @staticmethod
    @lru_cache(maxsize=4)
    def get_rag_response_schema(stage, case_grounding=True):
        """Load one fixed RAG response schema and its stable API name."""
        if stage not in {1, 2}:
            raise ValueError(f"Unsupported RAG stage: {stage}")
        if not isinstance(case_grounding, bool):
            raise TypeError("case_grounding must be a bool")
        mode = "grounded" if case_grounding else "direct"
        schema_name = f"rag_stage{stage}_{mode}"
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "Schemas"
            / "Agent"
            / f"{schema_name}_schema.json"
        )
        with schema_path.open("r", encoding="utf-8") as schema_file:
            return json.load(schema_file), schema_name

    @staticmethod
    def constrain_case_assessment_count(schema, count):
        """Lock structured decoding to the assessments displayed in one prompt."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("case assessment count must be a non-negative integer")
        constrained = copy.deepcopy(schema)
        assessments = constrained.get("properties", {}).get("case_assessments")
        if not isinstance(assessments, dict):
            raise ValueError("schema does not define case_assessments")
        assessments["minItems"] = count
        assessments["maxItems"] = count
        return constrained

    @staticmethod
    def _parse_rag_case_assessments(
        case_assessments, item_keys, expected_identities, sources=None
    ):
        if (
            not isinstance(case_assessments, list)
            or len(case_assessments) != len(expected_identities)
        ):
            return None
        parsed = []
        actual_identities = []
        factor_pattern = re.compile(
            r"^[a-z]+(?:[-'][a-z]+)*(?: [a-z]+(?:[-'][a-z]+)*){0,3}$"
        )
        for annotation in case_assessments:
            if not isinstance(annotation, dict) or set(annotation) != item_keys:
                return None
            rank = annotation.get("retrieval_rank")
            impact = annotation.get("impact")
            factors = annotation.get("factors")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                return None
            if impact not in BaseAgent.RAG_IMPACTS:
                return None
            if not isinstance(factors, list) or not 1 <= len(factors) <= 2:
                return None
            parsed_factors = []
            for factor in factors:
                if not isinstance(factor, str) or "\n" in factor or "\r" in factor:
                    return None
                normalized = " ".join(factor.strip().lower().split())
                if (
                    not normalized
                    or len(normalized) > 32
                    or factor_pattern.fullmatch(normalized) is None
                    or BaseAgent.RAG_FACTOR_BANNED_WORDS.intersection(
                        normalized.split()
                    )
                ):
                    return None
                parsed_factors.append(normalized)
            if len(set(parsed_factors)) != len(parsed_factors):
                return None
            parsed_item = {
                "retrieval_rank": rank,
                "impact": impact,
                "factors": parsed_factors,
            }
            if sources is not None:
                source = annotation.get("source")
                if source not in sources:
                    return None
                parsed_item = {"source": source, **parsed_item}
                identity = (source, rank)
            else:
                identity = rank
            actual_identities.append(identity)
            parsed.append(parsed_item)
        if actual_identities != list(expected_identities):
            return None
        return parsed

    @classmethod
    def parse_rag_stage1_response(
        cls,
        response,
        launch=False,
        available_types=None,
        case_grounding=True,
        normal_cases=None,
        miss_cases=None,
    ):
        """Validate the reachability/shot-type decision from RAG stage 1."""
        if not isinstance(case_grounding, bool):
            return None
        expected_keys = (
            cls.RAG_STAGE1_GROUNDED_KEYS
            if case_grounding
            else cls.RAG_STAGE1_DIRECT_KEYS
        )
        if not isinstance(response, dict) or set(response) != expected_keys:
            return None
        outcome = response.get("outcome")
        if outcome not in {"returned", "missed"}:
            return None
        if outcome == "missed":
            if response.get("shot_type") is not None:
                return None
            shot_type = 11
        else:
            try:
                shot_type = shot_type_to_index(response.get("shot_type"))
            except (TypeError, ValueError):
                return None
            legal_types = {1, 10} if launch else set(range(2, 10))
            if shot_type not in legal_types:
                return None
            if available_types is not None and shot_type not in set(available_types):
                return None

        case_assessments = []
        if case_grounding:
            normal_cases = list(normal_cases or [])
            miss_cases = list(miss_cases or [])
            expected_identities = [
                ("success", rank) for rank in range(1, len(normal_cases) + 1)
            ] + [
                ("miss", rank) for rank in range(1, len(miss_cases) + 1)
            ]
            case_assessments = cls._parse_rag_case_assessments(
                response.get("case_assessments"),
                cls.RAG_STAGE1_ASSESSMENT_KEYS,
                expected_identities,
                sources={"success", "miss"},
            )
            if case_assessments is None:
                return None

        return {
            "case_assessments": case_assessments,
            "outcome": outcome,
            "shot_type": shot_type,
        }

    @classmethod
    def validate_rag_stage1_response(
        cls,
        response,
        launch=False,
        available_types=None,
        case_grounding=True,
        normal_cases=None,
        miss_cases=None,
    ):
        return cls.parse_rag_stage1_response(
            response,
            launch=launch,
            available_types=available_types,
            case_grounding=case_grounding,
            normal_cases=normal_cases,
            miss_cases=miss_cases,
        ) is not None

    @classmethod
    def parse_rag_stage2_response(
        cls,
        response,
        case_grounding=True,
        conditioned_cases=None,
    ):
        """Validate placement output while keeping the stage-1 shot immutable."""
        if not isinstance(case_grounding, bool):
            return None
        expected_keys = (
            cls.RAG_STAGE2_GROUNDED_KEYS
            if case_grounding
            else cls.RAG_STAGE2_DIRECT_KEYS
        )
        if not isinstance(response, dict) or set(response) != expected_keys:
            return None
        placement = response.get("placement")
        if not isinstance(placement, dict) or set(placement) != cls.RAG_PLACEMENT_KEYS:
            return None

        case_assessments = []
        if case_grounding:
            candidate_count = len(conditioned_cases or [])
            case_assessments = cls._parse_rag_case_assessments(
                response.get("case_assessments"),
                cls.RAG_STAGE2_ASSESSMENT_KEYS,
                range(1, candidate_count + 1),
            )
            if case_assessments is None:
                return None

        coordinate_keys = ("landing_x", "landing_y", "moving_x", "moving_y")
        coordinates = [placement[key] for key in coordinate_keys]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates):
            return None
        landing_x, landing_y, moving_x, moving_y = coordinates
        if not (0 <= landing_x <= 355 and 0 <= moving_x <= 355):
            return None
        if not (0 <= landing_y <= 480 and 0 <= moving_y <= 480):
            return None
        normalized_landing = cls.prompt_to_normalized_coordinate(landing_x, landing_y)
        normalized_moving = cls.prompt_to_normalized_coordinate(moving_x, moving_y)
        return {
            "case_assessments": case_assessments,
            "landing_area": cls.prompt_area_name(landing_x, landing_y),
            "landing_x": normalized_landing[0],
            "landing_y": normalized_landing[1],
            "moving_area": cls.prompt_area_name(moving_x, moving_y),
            "moving_x": normalized_moving[0],
            "moving_y": normalized_moving[1],
        }

    @classmethod
    def validate_rag_stage2_response(
        cls,
        response,
        case_grounding=True,
        conditioned_cases=None,
    ):
        return cls.parse_rag_stage2_response(
            response,
            case_grounding=case_grounding,
            conditioned_cases=conditioned_cases,
        ) is not None

    @staticmethod
    def get_verified_response_or_none(llm, **kwargs):
        """Turn provider/transport failures into the agent's deterministic fallback path."""
        try:
            return llm.get_verified_response(**kwargs)
        except Exception as exc:
            print(f"Warning: LLM request failed: {type(exc).__name__}: {exc}")
            return None, "", {"error": type(exc).__name__}

    def action(self, state, info, launch):
        normalized_action = self.predict(state, info, launch)
        return self.decode_action(normalized_action)
    

    def encode_state(self, states):
        # Normalize positions
        player_x = (states[1][0]/177.5)
        player_y = (states[1][1]/240+1)
        hit_x = (states[3][0]/177.5)
        hit_y = (states[3][1]/240+1)
        opponent_x = -(states[2][0]/177.5)
        opponent_y = -(states[2][1]/240-1)
        pre_type = states[0]
        return [pre_type, player_x, player_y, opponent_x, opponent_y, hit_x, hit_y]

    def decode_action(self, action, reason=""):
        output_shot, hit, output_land, output_move, dist = action

        output_land[0] = -float(output_land[0]) * 177.5
        output_land[1] = -240 * float(output_land[1]) + 240 
        output_move[0] = float(output_move[0]) * 177.5 
        output_move[1] = 240 * float(output_move[1]) - 240

        return (output_shot, hit, tuple(output_land), tuple(output_move), dist), reason
