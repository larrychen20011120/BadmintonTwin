"""
TimeRAGVerbalizer - Converts state and retrieved time windows to natural language.
V3 - Turn-based window format for new prompts.
"""

from collections import Counter
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


class TimeRAGVerbalizer:
    """Verbalizer for TimeRAG with turn-based window format."""

    def __init__(self):
        pass

    @staticmethod
    def _get_area_name(x, y):
        area_id = coord2area(float(x), float(y))
        return LANG_AREA_MAPPING.get(area_id, f"Unknown({area_id})")

    @staticmethod
    def _shot_name(shot_type):
        return INDEX2EN_SHOT.get(int(shot_type), f"Unknown({shot_type})")

    def format_window_turn_based(self, win, idx):
        """Format a time window as turn-based sequence.
        
        Window structure:
        - window_states: list of states (opponent's shots leading up to player's action)
        - action: player's response to the last state
        
        Turn-based format:
        - For each window_state: describe opponent's shot (even shots: 2, 4, 6...)
        - Then describe player's action (odd shots: 1, 3, 5...)
        - Only the LAST player action shows numerical values
        """
        ws_list = win.get('window_states', [])
        act = win.get('action', {})
        dist = win.get('distance', win.get('similarity', 0))
        
        lines = []
        shot_num = 1
        
        for si, ws in enumerate(ws_list):
            if ws.get('padding'):
                continue
                
            # Opponent's context and shot (from window_states)
            ot = int(ws.get('opponent_type', 0))
            on = self._shot_name(ot)
            ox, oy = ws.get('opponent_location_x', 0), ws.get('opponent_location_y', 0)
            hx, hy = ws.get('hit_x', 0), ws.get('hit_y', 0)
            px, py = ws.get('player_location_x', 0), ws.get('player_location_y', 0)
            ps = int(ws.get('player_score', 0))
            os = int(ws.get('opponent_score', 0))
            
            opp_area = self._get_area_name(ox, oy)
            ball_area = self._get_area_name(hx, hy)
            player_area = self._get_area_name(px, py)
            
            is_serve = (shot_num == 1)
            is_last_shot = (si == len(ws_list) - 1)
            
            if is_serve:
                # Shot 1: Player's serve
                context = f"I am at {player_area} serving. Score {ps}-{os}."
                pt = int(act.get('player_type', 0))
                pn = self._shot_name(pt)
                lx, ly = act.get('landing_x', 0), act.get('landing_y', 0)
                mx, my = act.get('moving_x', 0), act.get('moving_y', 0)
                land_area = self._get_area_name(lx, ly)
                move_area = self._get_area_name(mx, my)
                
                if pt == 11:
                    action = "MISSED SHOT - fault serve."
                else:
                    action = f"I serve a {pn} to {land_area}, then move to {move_area}."
                
                lines.append(f"Shot {shot_num}: {context}")
                lines.append(f"        {action}")
                
                # Last shot - add numerical action
                lines.append(f"        Action: (shot_type={pt}, landing_x={round(float(lx), 3)}, landing_y={round(float(ly), 3)}, moving_x={round(float(mx), 3)}, moving_y={round(float(my), 3)})")
                shot_num += 1
            else:
                # Opponent's shot (even shots: 2, 4, 6...)
                context = f"Opponent is at {opp_area}."
                action = f"Opponent hits a {on} to {ball_area}."
                
                lines.append(f"Shot {shot_num}: {context}")
                lines.append(f"        {action}")
                shot_num += 1
                
                # Player's response (odd shots: 3, 5, 7...)
                if is_last_shot:
                    # This is the final action with numerical values
                    pt = int(act.get('player_type', 0))
                    pn = self._shot_name(pt)
                    lx, ly = act.get('landing_x', 0), act.get('landing_y', 0)
                    mx, my = act.get('moving_x', 0), act.get('moving_y', 0)
                    land_area = self._get_area_name(lx, ly)
                    move_area = self._get_area_name(mx, my)
                    
                    context = f"I am at {player_area}, opponent at {opp_area} played a {on} to {ball_area}."
                    
                    if pt == 11:
                        action = "MISSED SHOT - could not reach the shuttle."
                    else:
                        action = f"I move to {ball_area} and hit a {pn} to {land_area}, then recover to {move_area}."
                    
                    lines.append(f"Shot {shot_num}: {context}")
                    lines.append(f"        {action}")
                    lines.append(f"        Action: (shot_type={pt}, landing_x={round(float(lx), 3)}, landing_y={round(float(ly), 3)}, moving_x={round(float(mx), 3)}, moving_y={round(float(my), 3)})")
                    shot_num += 1
                else:
                    # Intermediate player shot - no numerical values
                    context = f"I am at {player_area}, opponent played a {on} to {ball_area}."
                    action = "I respond and recover."
                    lines.append(f"Shot {shot_num}: {context}")
                    lines.append(f"        {action}")
                    shot_num += 1
        
        num_shots = shot_num - 1
        description = "\n".join(lines)
        
        return {
            'index': idx,
            'distance': round(float(dist), 3),
            'num_shots': num_shots,
            'description': description
        }

    def build_windows_list(self, windows):
        """Build a list of formatted windows in turn-based format."""
        formatted_windows = []
        for i, win in enumerate(windows, 1):
            formatted = self.format_window_turn_based(win, i)
            formatted_windows.append(formatted)
        return formatted_windows

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

    def build_prompt_context_nl(self, state_dict, windows, launch=False, window_size=3, state_history=None):
        """Build full prompt context dict (V3 - turn-based format) for Jinja2."""
        formatted_windows = self.build_windows_list(windows)
        
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
            'windows': formatted_windows,
            'trajectory_text': trajectory_text,
            'launch': launch,
            'top_k': len(windows) if windows else 0,
            'window_size': window_size,
            'player_location_area': self._get_area_name(px, py),
            'opponent_location_area': self._get_area_name(ox, oy),
            'hit_position_area': self._get_area_name(hx, hy),
            'opponent_type_name': self._shot_name(ot),
            **state_dict
        }
        
        return context


def create_timerag_verbalizer():
    return TimeRAGVerbalizer()
