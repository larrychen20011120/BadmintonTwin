"""
NaiveRAGVerbalizer - Converts state and retrieved cases to natural language.
V3 - Simplified format with Context/Action structure for new prompts.
"""

from collections import Counter
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


class NaiveRAGVerbalizer:
    """Verbalizer for NaiveRAG."""

    def __init__(self):
        pass

    @staticmethod
    def _get_area_name(x, y):
        area_id = coord2area(float(x), float(y))
        return LANG_AREA_MAPPING.get(area_id, f"Unknown({area_id})")

    @staticmethod
    def _shot_name(shot_type):
        return INDEX2EN_SHOT.get(int(shot_type), f"Unknown({shot_type})")

    def format_case_context_action(self, exp, is_serve=False):
        """Format a case in Context/Action structure."""
        ot = int(exp.get('opponent_type', 0))
        pt = int(exp.get('player_type', 0))
        ox, oy = exp.get('opponent_location_x', 0), exp.get('opponent_location_y', 0)
        hx, hy = exp.get('hit_x', 0), exp.get('hit_y', 0)
        px, py = exp.get('player_location_x', 0), exp.get('player_location_y', 0)
        lx, ly = exp.get('landing_x', 0), exp.get('landing_y', 0)
        mx, my = exp.get('moving_x', 0), exp.get('moving_y', 0)
        ps = int(exp.get('player_score', 0))
        os = int(exp.get('opponent_score', 0))
        
        on = self._shot_name(ot)
        pn = self._shot_name(pt)
        opp_area = self._get_area_name(ox, oy)
        ball_area = self._get_area_name(hx, hy)
        player_area = self._get_area_name(px, py)
        land_area = self._get_area_name(lx, ly)
        move_area = self._get_area_name(mx, my)
        
        if is_serve:
            context = f"I am at {player_area} serving. Opponent at {opp_area} receiving. Score {ps}-{os}."
            if pt == 11:
                action = "MISSED SHOT - fault serve."
            else:
                action = f"I serve a {pn} to {land_area}, then move to {move_area}."
        else:
            context = f"I am at {player_area}, opponent at {opp_area} played a {on} to {ball_area}."
            if pt == 11:
                action = "MISSED SHOT - could not reach the shuttle."
            else:
                action = f"I move to {ball_area} and hit a {pn} to {land_area}, then recover to {move_area}."
        
        return {
            'context': context,
            'action': action,
            'shot_type': pt,
            'landing_x': round(float(lx), 3),
            'landing_y': round(float(ly), 3),
            'moving_x': round(float(mx), 3),
            'moving_y': round(float(my), 3),
            'similarity': round(float(exp.get('similarity', 0)), 3)
        }

    def build_cases_list(self, experiences, is_serve=False):
        """Build a list of formatted cases."""
        cases = []
        for i, exp in enumerate(experiences, 1):
            formatted = self.format_case_context_action(exp, is_serve=is_serve)
            formatted['index'] = i
            cases.append(formatted)
        return cases

    def build_trajectory_text(self, state_history, current_state):
        """Build turn-based trajectory text for current rally."""
        lines = []
        
        for i, sd in enumerate(state_history, 1):
            br = int(sd.get('ball_round', i))
            ot = int(sd.get('opponent_type', 0))
            ox, oy = sd.get('opponent_location_x', 0), sd.get('opponent_location_y', 0)
            hx, hy = sd.get('hit_x', 0), sd.get('hit_y', 0)
            px, py = sd.get('player_location_x', 0), sd.get('player_location_y', 0)
            
            on = self._shot_name(ot)
            opp_area = self._get_area_name(ox, oy)
            ball_area = self._get_area_name(hx, hy)
            player_area = self._get_area_name(px, py)
            
            if br == 1:
                lines.append(f"Shot 1: I am at {player_area} serving.")
                lines.append(f"        I serve and move to prepare for the return.")
            else:
                lines.append(f"Shot {br}: I am at {player_area}, opponent at {opp_area} played a {on} to {ball_area}.")
                lines.append(f"        I respond and recover.")
        
        return "\n".join(lines)

    def build_prompt_context_nl(self, state_dict, experiences, launch=False, state_history=None):
        """Build full prompt context dict for Jinja2."""
        is_serve = launch or int(state_dict.get('ball_round', 1)) == 1
        cases = self.build_cases_list(experiences, is_serve=is_serve)
        
        trajectory_text = ""
        if state_history and len(state_history) > 0:
            trajectory_text = self.build_trajectory_text(state_history, state_dict)
        
        # Add semantic names to state_dict for template use
        px = state_dict.get('player_location_x', 0)
        py = state_dict.get('player_location_y', 0)
        ox = state_dict.get('opponent_location_x', 0)
        oy = state_dict.get('opponent_location_y', 0)
        hx = state_dict.get('hit_x', 0)
        hy = state_dict.get('hit_y', 0)
        ot = state_dict.get('opponent_type', 0)
        
        context = {
            'cases': cases,
            'trajectory_text': trajectory_text,
            'launch': launch,
            'top_k': len(experiences) if experiences else 0,
            'player_location_area': self._get_area_name(px, py),
            'opponent_location_area': self._get_area_name(ox, oy),
            'hit_position_area': self._get_area_name(hx, hy),
            'opponent_type_name': self._shot_name(ot),
            **state_dict
        }
        
        return context


def create_naive_rag_verbalizer():
    return NaiveRAGVerbalizer()
