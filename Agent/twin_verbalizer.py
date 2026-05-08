"""
TwinVerbalizer - Converts state and retrieved trajectory segments to natural language.
V3 - Turn-based segment format for new prompts.
"""

from collections import Counter
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


class TwinVerbalizer:
    """Verbalizer for Twin with turn-based segment format."""

    def __init__(self):
        pass

    @staticmethod
    def _get_area_name(x, y):
        area_id = coord2area(float(x), float(y))
        return LANG_AREA_MAPPING.get(area_id, f"Unknown({area_id})")

    @staticmethod
    def _shot_name(shot_type):
        return INDEX2EN_SHOT.get(int(shot_type), f"Unknown({shot_type})")

    @staticmethod
    def _scale_label(scale):
        if scale <= 3:
            return "short-scale"
        elif scale <= 7:
            return "medium-scale"
        else:
            return "long-scale"

    def format_segment_turn_based(self, seg, idx):
        """Format a trajectory segment as turn-based sequence.
        
        Segment structure:
        - steps: list of steps, each containing state and action info
        - scale: length of the segment
        - similarity: distance metric
        
        Turn-based format:
        - For each step: describe opponent's shot, then player's response
        - Only the LAST step's player action shows numerical values
        """
        steps = seg.get('steps', [])
        scale = seg.get('scale', len(steps))
        dist = seg.get('similarity', 0)
        
        lines = []
        shot_num = 1
        
        for step_idx, step in enumerate(steps):
            is_last_step = (step_idx == len(steps) - 1)
            
            # Extract data from step
            ot = int(step.get('opponent_type', 0))
            on = self._shot_name(ot)
            ox, oy = step.get('opponent_location_x', 0), step.get('opponent_location_y', 0)
            hx, hy = step.get('hit_x', 0), step.get('hit_y', 0)
            px, py = step.get('player_location_x', 0), step.get('player_location_y', 0)
            pt = int(step.get('player_type', 0))
            pn = self._shot_name(pt)
            lx, ly = step.get('landing_x', 0), step.get('landing_y', 0)
            mx, my = step.get('moving_x', 0), step.get('moving_y', 0)
            ps = int(step.get('player_score', 0))
            os = int(step.get('opponent_score', 0))
            
            opp_area = self._get_area_name(ox, oy)
            ball_area = self._get_area_name(hx, hy)
            player_area = self._get_area_name(px, py)
            land_area = self._get_area_name(lx, ly)
            move_area = self._get_area_name(mx, my)
            
            is_serve = (shot_num == 1)
            
            if is_serve:
                # Shot 1: Player's serve
                context = f"I am at {player_area} serving. Score {ps}-{os}."
                
                if pt == 11:
                    action = "MISSED SHOT - fault serve."
                else:
                    action = f"I serve a {pn} to {land_area}, then move to {move_area}."
                
                lines.append(f"Shot {shot_num}: {context}")
                lines.append(f"        {action}")
                
                if is_last_step:
                    lines.append(f"        Action: (shot_type={pt}, landing_x={round(float(lx), 3)}, landing_y={round(float(ly), 3)}, moving_x={round(float(mx), 3)}, moving_y={round(float(my), 3)})")
                
                shot_num += 1
            else:
                # Opponent's shot
                context = f"Opponent is at {opp_area}."
                action = f"Opponent hits a {on} to {ball_area}."
                
                lines.append(f"Shot {shot_num}: {context}")
                lines.append(f"        {action}")
                shot_num += 1
                
                # Player's response
                context = f"I am at {player_area}, opponent at {opp_area} played a {on} to {ball_area}."
                
                if pt == 11:
                    action = "MISSED SHOT - could not reach the shuttle."
                else:
                    action = f"I move to {ball_area} and hit a {pn} to {land_area}, then recover to {move_area}."
                
                lines.append(f"Shot {shot_num}: {context}")
                lines.append(f"        {action}")
                
                if is_last_step:
                    # Last shot - add numerical action
                    lines.append(f"        Action: (shot_type={pt}, landing_x={round(float(lx), 3)}, landing_y={round(float(ly), 3)}, moving_x={round(float(mx), 3)}, moving_y={round(float(my), 3)})")
                
                shot_num += 1
        
        num_shots = shot_num - 1
        description = "\n".join(lines)
        scale_label = self._scale_label(scale)
        
        return {
            'index': idx,
            'scale': scale,
            'scale_label': scale_label,
            'distance': round(float(dist), 3),
            'num_shots': num_shots,
            'description': description
        }

    def build_segments_list(self, segments):
        """Build a list of formatted segments in turn-based format."""
        formatted_segments = []
        for i, seg in enumerate(segments, 1):
            formatted = self.format_segment_turn_based(seg, i)
            formatted_segments.append(formatted)
        return formatted_segments

    def build_trajectory_text(self, state_history, current_state):
        """Build turn-based trajectory text for current rally."""
        lines = []
        
        for i, sd in enumerate(state_history, 1):
            br = int(sd.get('ball_round', i))
            ot = int(sd.get('opponent_type', 0))
            on = self._shot_name(ot)
            ox, oy = sd.get('opponent_location_x', 0), sd.get('opponent_location_y', 0)
            hx, hy = sd.get('hit_x', 0), sd.get('hit_y', 0)
            px, py = sd.get('player_location_x', 0), sd.get('player_location_y', 0)
            
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

    def build_prompt_context_nl(self, state_dict, segments, launch=False, state_history=None):
        """Build full prompt context dict (V3 - turn-based format) for Jinja2."""
        formatted_segments = self.build_segments_list(segments)
        
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
            'segments': formatted_segments,
            'trajectory_text': trajectory_text,
            'launch': launch,
            'top_k': len(segments) if segments else 0,
            'player_location_area': self._get_area_name(px, py),
            'opponent_location_area': self._get_area_name(ox, oy),
            'hit_position_area': self._get_area_name(hx, hy),
            'opponent_type_name': self._shot_name(ot),
            **state_dict
        }
        
        return context


def create_twin_verbalizer():
    return TwinVerbalizer()
