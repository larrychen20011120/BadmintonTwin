import os
import sys
import time
import numpy as np
from jinja2 import Template

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Agent.naive_rag import create_naive_rag_verbalizer
from Utils.llm import LLM
from Utils.helper import coord2area
from Utils.badminton_config import LANG_AREA_MAPPING
from Utils.langfuse_tracing import BadmintonTracer


class PureLLMAgent(BaseAgent):
    """Zero-shot LLM agent without retrieval or player-specific examples."""

    def __init__(self, player_name, model_name=None,
                 api_key=None, host=None, port=None,
                 base_url=None, host_url=None, enable_tracing=True,
                 prompt_variant="norm"):
        super().__init__(player_name)

        if prompt_variant not in {'norm', 'shifted'}:
            raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")
        self.prompt_variant = prompt_variant

        self.tracer = BadmintonTracer("pure_llm", player_name, enable_tracing)
        self.trajectory = []
        self._current_rally = None
        self.verbalizer = create_naive_rag_verbalizer()

        self.llm = LLM(
            model_name=model_name,
            host=host,
            api_key=api_key,
            port=port,
            base_url=base_url,
            host_url=host_url,
        )

        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "pure_llm")

        template_name = (
            "user_instruct.jinja" if prompt_variant == "shifted" else "user.jinja"
        )
        with open(os.path.join(prompt_dir, template_name), "r") as f:
            self.user_template = Template(f.read())

    @staticmethod
    def _parse_llm_action(response):
        return BaseAgent.parse_prompt_action(response)

    def _normalize_history_for_player(self, history):
        """Mark each shared-history stroke from this agent's viewpoint."""
        normalized = []
        for stroke in history or []:
            stroke_data = dict(stroke)
            player = stroke_data.get('player')
            if player is not None:
                stroke_data['is_target_turn'] = str(player) == str(self.player_name)
            else:
                stroke_data['is_target_turn'] = bool(
                    stroke_data.get('is_target_turn', True)
                )
            stroke_data['turn_role'] = (
                'target' if stroke_data['is_target_turn'] else 'opponent'
            )
            normalized.append(stroke_data)
        return normalized

    def predict(self, state, info, launch):
        """Predict the next action using only the current/within-rally context."""
        opponent_type = int(state[0])
        player_location_x = state[1][0]
        player_location_y = state[1][1]
        opponent_location_x = state[2][0]
        opponent_location_y = state[2][1]
        hit_x = state[3][0]
        hit_y = state[3][1]

        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0

        player_location_area = LANG_AREA_MAPPING.get(
            coord2area(player_location_x, player_location_y), 'Unknown')
        hit_position_area = LANG_AREA_MAPPING.get(coord2area(hit_x, hit_y), 'Unknown')
        opponent_location_area = LANG_AREA_MAPPING.get(
            coord2area(opponent_location_x, opponent_location_y), 'Unknown')

        state_dict = {
            'player': info.get('current_player_name', self.player_name),
            'is_target_turn': True,
            'turn_role': 'target',
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': player_location_x,
            'player_location_y': player_location_y,
            'player_location_area': player_location_area,
            'ball_distance': 0,
            'ball_distance_x': hit_x - player_location_x,
            'ball_distance_y': hit_y - player_location_y,
            'hit_x': hit_x,
            'hit_y': hit_y,
            'hit_position_area': hit_position_area,
            'opponent_type': opponent_type,
            'opponent_location_x': opponent_location_x,
            'opponent_location_y': opponent_location_y,
            'opponent_location_area': opponent_location_area,
            'opponent_move_x': 0,
            'opponent_move_y': 0,
        }

        self.tracer.start_trace(state_dict, info, launch)

        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.trajectory = []
        self.trajectory.append(state_dict)

        rally_history = info.get('rally_history')
        if rally_history is not None:
            state_history = self._normalize_history_for_player(rally_history)
        else:
            state_history = self.trajectory[:-1] if len(self.trajectory) > 1 else None
        context = self.verbalizer.build_prompt_context_nl(
            state_dict, [], launch=launch, state_history=state_history)
        context['rally_profile_instruction'] = info.get(
            'rally_profile_instruction', ''
        )

        user_prompt = self.user_template.render(**context)

        t0 = time.time()
        response, raw_llm_output, usage = self.llm.get_verified_response(
            user_prompt=user_prompt,
            max_trials=5,
            validator=BaseAgent.validate_prompt_action_response,
        )
        t1 = time.time()
        self.tracer.log_generation(
            "", user_prompt, response,
            latency_ms=int((t1 - t0) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
        )

        if response is None:
            print("Warning: LLM response failed, using default action")
            self.tracer.end_trace({"shot_type": "default_fallback"}, "LLM response failed")
            return self._default_action(launch)

        parsed_action, reasoning = self._parse_llm_action(response)
        if parsed_action is None:
            print("Warning: LLM response did not contain a valid action, using default action")
            print(response)
            self.tracer.end_trace({"shot_type": "default_fallback"}, "Invalid LLM action")
            return self._default_action(launch)

        shot_type = parsed_action['shot_type']
        landing_x = parsed_action['landing_x']
        landing_y = parsed_action['landing_y']
        moving_x = parsed_action['moving_x']
        moving_y = parsed_action['moving_y']

        if shot_type == 11:
            landing_x = 0.0
            landing_y = 0.0
            moving_x = 0.0
            moving_y = 0.0

        if launch and shot_type not in [1, 10]:
            shot_type = 1 if np.random.rand() < 0.5 else 10

        if self.trajectory:
            self.trajectory[-1]['player_type'] = int(shot_type)
            self.trajectory[-1]['landing_x'] = float(landing_x)
            self.trajectory[-1]['landing_y'] = float(landing_y)
            self.trajectory[-1]['moving_x'] = float(moving_x)
            self.trajectory[-1]['moving_y'] = float(moving_y)

        hit = state[-1] if len(state) > 0 else None
        action = (
            shot_type,
            hit,
            [landing_x, landing_y],
            [moving_x, moving_y],
            [0.1] * 10
        )

        self.tracer.end_trace({
            "shot_type": int(shot_type),
            "landing": [landing_x, landing_y],
            "movement": [moving_x, moving_y],
        }, reasoning)

        return action, reasoning

    def reset_history(self):
        self._current_rally = None
        self.trajectory = []

    def _default_action(self, launch):
        if launch:
            shot_type = 1
        else:
            shot_type = np.random.randint(1, 11)

        return (
            shot_type,
            None,
            (0.0, 0.0),
            (0.0, 0.0),
            [0.1] * 10
        ), "Default action due to LLM failure"
