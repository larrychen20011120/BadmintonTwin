import json
import math

from dtaidistance import dtw

from Utils.badminton_config import INDEX2EN_SHOT, area_groups, shot_type_to_index

class BaseAgent:
    PROMPT_ACTION_KEYS = {
        "shot_type",
        "landing_area",
        "landing_x",
        "landing_y",
        "moving_area",
        "moving_x",
        "moving_y",
    }
    PROMPT_AREA_NAMES = frozenset(area_groups) - {"Out of Bounds"}
    RAG_STAGE1_KEYS = {"rationale", "outcome", "shot_type"}
    RAG_STAGE2_KEYS = {"rationale", "placement"}
    RAG_PLACEMENT_KEYS = {
        "landing_area",
        "landing_x",
        "landing_y",
        "moving_area",
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
        area_keys = ("landing_area", "moving_area")
        if shot_type == 11:
            if any(action[key] is not None for key in coordinate_keys + area_keys):
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

        areas = [action[key] for key in area_keys]
        if any(not isinstance(value, str) or value not in cls.PROMPT_AREA_NAMES for value in areas):
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
            "landing_area": action["landing_area"],
            "landing_x": normalized_landing_x,
            "landing_y": normalized_landing_y,
            "moving_area": action["moving_area"],
            "moving_x": normalized_moving_x,
            "moving_y": normalized_moving_y,
        }, reasoning

    @classmethod
    def validate_prompt_action_response(cls, response):
        parsed, _ = cls.parse_prompt_action(response)
        return parsed is not None

    @classmethod
    def parse_rag_stage1_response(cls, response, launch=False, available_types=None):
        """Validate the reachability/shot-type decision from RAG stage 1."""
        if not isinstance(response, dict) or set(response) != cls.RAG_STAGE1_KEYS:
            return None
        rationale = response.get("rationale")
        outcome = response.get("outcome")
        if not isinstance(rationale, str) or outcome not in {"returned", "missed"}:
            return None
        if outcome == "missed":
            if response.get("shot_type") is not None:
                return None
            return {
                "rationale": rationale,
                "outcome": outcome,
                "shot_type": 11,
            }

        try:
            shot_type = shot_type_to_index(response.get("shot_type"))
        except (TypeError, ValueError):
            return None
        legal_types = {1, 10} if launch else set(range(2, 10))
        if shot_type not in legal_types:
            return None
        if available_types is not None and shot_type not in set(available_types):
            return None
        return {
            "rationale": rationale,
            "outcome": outcome,
            "shot_type": shot_type,
        }

    @classmethod
    def validate_rag_stage1_response(cls, response, launch=False, available_types=None):
        return cls.parse_rag_stage1_response(
            response,
            launch=launch,
            available_types=available_types,
        ) is not None

    @classmethod
    def parse_rag_stage2_response(cls, response):
        """Validate placement output while keeping the stage-1 shot immutable."""
        if not isinstance(response, dict) or set(response) != cls.RAG_STAGE2_KEYS:
            return None
        rationale = response.get("rationale")
        placement = response.get("placement")
        if not isinstance(rationale, str) or not isinstance(placement, dict):
            return None
        if set(placement) != cls.RAG_PLACEMENT_KEYS:
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
            "rationale": rationale,
            "landing_area": placement["landing_area"],
            "landing_x": normalized_landing[0],
            "landing_y": normalized_landing[1],
            "moving_area": placement["moving_area"],
            "moving_x": normalized_moving[0],
            "moving_y": normalized_moving[1],
        }

    @classmethod
    def validate_rag_stage2_response(cls, response):
        return cls.parse_rag_stage2_response(response) is not None

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
