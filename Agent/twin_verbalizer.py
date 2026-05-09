"""
TwinVerbalizer - Converts state and retrieved trajectory segments to natural language.
V6 - Natural turn-based format with incoming ball descriptions.

New format:
Shot {N}:
It is {turn}. I am at {player_area}, opponent at {opp_area}.
{Incoming ball description}.
{Action description}.
"""

from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


class TwinVerbalizer:
    """Verbalizer for Twin with natural turn-based segment format."""

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
    def _build_context_header(state_dict):
        """Build score/context header from a state snapshot."""
        ps = int(state_dict.get('player_score', 0))
        os = int(state_dict.get('opponent_score', 0))
        rally = int(state_dict.get('rally', 0))
        br = int(state_dict.get('ball_round', 1))

        score_diff = ps - os
        if score_diff > 0:
            score_text = f"I lead by {score_diff} points."
        elif score_diff < 0:
            score_text = f"I trail by {abs(score_diff)} points."
        else:
            score_text = "The score is tied."

        shots_played = max(br - 1, 0)
        if shots_played == 0:
            context_text = "This is the first shot of the rally."
        else:
            context_text = f"{shots_played} shots have been played in this rally."

        return [
            f"Current score: {ps}-{os}. {score_text}",
            f"Match Context: Rally {rally}, currently at Shot {br}. {context_text}"
        ]

    @staticmethod
    def _scale_label(scale):
        if scale <= 3:
            return "short-scale"
        elif scale <= 7:
            return "medium-scale"
        else:
            return "long-scale"

    def format_segment_turn_based(self, seg, idx):
        """Format a trajectory segment as natural turn-based sequence.
        
        New format per shot:
        Shot {N}:
        It is {turn}. I am at {player_area}, opponent at {opp_area}.
        {Incoming ball description}.
        {Action description}.
        """
        steps = seg.get('steps', [])
        scale = seg.get('scale', len(steps))
        dist = seg.get('similarity', 0)
        
        if not steps:
            return {
                'index': idx,
                'scale': scale,
                'scale_label': self._scale_label(scale),
                'distance': round(float(dist), 3),
                'num_shots': 0,
                'description': "No steps available."
            }
        
        # Get rally and score from first step
        first_step = steps[0]
        rally = int(first_step.get('rally', 0))
        first_ball_round = int(first_step.get('ball_round', 1))
        ps = int(first_step.get('player_score', 0))
        os = int(first_step.get('opponent_score', 0))
        score_diff = ps - os
        
        if score_diff > 0:
            score_text = f"I lead by {score_diff} points."
        elif score_diff < 0:
            score_text = f"I trail by {abs(score_diff)} points."
        else:
            score_text = "The score is tied."
        
        # Build header - detailed format with score first
        shots_played = first_ball_round - 1
        if shots_played == 0:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {first_ball_round}. This is the first shot of the rally."
        else:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {first_ball_round}. {shots_played} shots have been played in this rally."
        
        lines = [header, ""]
        
        last_shot_data = None
        last_opponent_shot_name = None  # Track opponent's shot type for next player's incoming ball
        
        
        # Process each step and derive opponent shots between them
        for step_idx, step in enumerate(steps):
            is_last_step = (step_idx == len(steps) - 1)
            ball_round = int(step.get('ball_round', first_ball_round + step_idx * 2))
            
            # Extract current step data
            pt = int(step.get('player_type', 0))
            pn = self._shot_name(pt)
            px, py = step.get('player_location_x', 0), step.get('player_location_y', 0)
            hx, hy = step.get('hit_x', 0), step.get('hit_y', 0)
            lx, ly = step.get('landing_x', 0), step.get('landing_y', 0)
            mx, my = step.get('moving_x', 0), step.get('moving_y', 0)
            ox, oy = step.get('opponent_location_x', 0), step.get('opponent_location_y', 0)
            
            player_area = self._get_area_name(px, py)
            hit_area = self._get_area_name(hx, hy)
            land_area = self._get_area_name(lx, ly)
            move_area = self._get_area_name(mx, my)
            opp_area = self._get_area_name(ox, oy)
            
            # Shot header
            lines.append(f"Shot {ball_round}:")
            
            if ball_round == 1:
                # Shot 1: My serve - no incoming ball
                lines.append(f"It is my turn to serve. I am at {player_area}, opponent at {opp_area}.")
                
                if pt == 11:
                    lines.append("MISSED SHOT - fault serve.")
                elif pt > 0:
                    lines.append(f"I serve a {pn} to {land_area}, then move to {move_area}.")
                else:
                    lines.append(f"I serve to {land_area}, then move to {move_area}.")
                
                if is_last_step:
                    last_shot_data = {
                        'shot_type': pt,
                        'landing_x': round(float(lx), 3),
                        'landing_y': round(float(ly), 3),
                        'moving_x': round(float(mx), 3),
                        'moving_y': round(float(my), 3)
                    }
            else:
                # My shot (Shot 3, 5, 7...) or Shot 2 if segment starts from opponent perspective
                lines.append(f"It is my turn. I am at {player_area}, opponent at {opp_area}.")
                
                # Use the tracked opponent shot name from previous opponent turn
                if last_opponent_shot_name and last_opponent_shot_name != "Unknown":
                    lines.append(f"Opponent's {last_opponent_shot_name} is arriving at {hit_area}.")
                else:
                    # For even ball_round (e.g., Shot 2), use current step's opponent_type
                    current_ot = int(step.get('opponent_type', 0))
                    if current_ot > 0:
                        current_on = self._shot_name(current_ot)
                        lines.append(f"Opponent's {current_on} is arriving at {hit_area}.")
                    else:
                        # Fallback to checking prev_step's opponent_type
                        prev_step = steps[step_idx - 1] if step_idx > 0 else None
                        if prev_step:
                            prev_ot = int(prev_step.get('opponent_type', 0))
                            prev_on = self._shot_name(prev_ot)
                            if prev_ot > 0:
                                lines.append(f"Opponent's {prev_on} is arriving at {hit_area}.")
                            else:
                                lines.append(f"Ball is arriving at {hit_area}.")
                        else:
                            lines.append(f"Ball is arriving at {hit_area}.")
                
                if pt == 11:
                    lines.append("MISSED SHOT - could not reach the shuttle.")
                elif pt == 0:
                    lines.append(f"I move to {hit_area} and hit to {land_area}, then move to {move_area}.")
                else:
                    lines.append(f"I move to {hit_area} and hit a {pn} to {land_area}, then move to {move_area}.")
                
                if is_last_step:
                    last_shot_data = {
                        'shot_type': pt,
                        'landing_x': round(float(lx), 3),
                        'landing_y': round(float(ly), 3),
                        'moving_x': round(float(mx), 3),
                        'moving_y': round(float(my), 3)
                    }
            
            # Derive opponent's shot if not the last step
            if not is_last_step:
                next_step = steps[step_idx + 1]
                next_ball_round = ball_round + 1
                
                # Opponent uses their type from next step
                next_ot = int(next_step.get('opponent_type', 0))
                next_on = self._shot_name(next_ot)
                next_ox, next_oy = next_step.get('opponent_location_x', 0), next_step.get('opponent_location_y', 0)
                next_hx, next_hy = next_step.get('hit_x', 0), next_step.get('hit_y', 0)
                
                next_opp_area = self._get_area_name(next_ox, next_oy)
                next_hit_area = self._get_area_name(next_hx, next_hy)
                
                # Track opponent's shot name for next player's incoming ball
                last_opponent_shot_name = next_on if next_ot > 0 else None
                
                # Opponent's shot
                lines.append(f"Shot {next_ball_round}:")
                lines.append(f"It is opponent's turn. I am at {move_area}, opponent at {opp_area}.")
                lines.append(f"My {pn} is arriving at {land_area}.")
                
                if next_ot == 11:
                    lines.append("Opponent MISSED SHOT - could not reach the shuttle.")
                elif next_ot > 0:
                    lines.append(f"Opponent moves to {land_area} and hits a {next_on} to {next_hit_area}, then moves to {next_opp_area}.")
                else:
                    lines.append(f"Opponent moves to {land_area} and hits to {next_hit_area}, then moves to {next_opp_area}.")
        
        # Add JSON action for the last shot only
        if last_shot_data:
            lines.append(f"Action: {{\"shot_type\": {last_shot_data['shot_type']}, \"landing_x\": {last_shot_data['landing_x']}, \"landing_y\": {last_shot_data['landing_y']}, \"moving_x\": {last_shot_data['moving_x']}, \"moving_y\": {last_shot_data['moving_y']}}}")
        
        num_shots = len(steps) * 2 - 1  # My shots + derived opponent shots
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

    def build_cases_list(self, segments):
        """Build a list of formatted segments in turn-based format."""
        formatted_cases = []
        for i, seg in enumerate(segments, 1):
            formatted = self.format_segment_turn_based(seg, i)
            formatted_cases.append(formatted)
        return formatted_cases

    def build_trajectory_text(self, state_history, current_state):
        """Build natural turn-based trajectory text for current rally.
        
        State history contains my shots only. Need to derive opponent shots.
        """
        if not state_history:
            return "\n".join(self._build_context_header(current_state))
        
        lines = list(self._build_context_header(current_state))
        lines.append("")
        last_opponent_shot_name = None  # Track opponent's shot type for next player's incoming ball
        
        # Check if history starts with even ball_round (opponent perspective - missing Shot 1)
        first_br = int(state_history[0].get('ball_round', 1))
        if first_br == 2:
            # Add synthetic Shot 1 (opponent's serve) based on first state's opponent_type
            first_sd = state_history[0]
            ot = int(first_sd.get('opponent_type', 0))
            on = self._shot_name(ot)
            ox, oy = first_sd.get('opponent_location_x', 0), first_sd.get('opponent_location_y', 0)
            hx, hy = first_sd.get('hit_x', 0), first_sd.get('hit_y', 0)
            opp_area = self._get_area_name(ox, oy)
            hit_area = self._get_area_name(hx, hy)
            
            lines.append("Shot 1:")
            lines.append(f"It is opponent's turn to serve. I am at {opp_area}, opponent at {opp_area}.")
            if ot > 0:
                lines.append(f"Opponent serves a {on} to {hit_area}.")
            else:
                lines.append(f"Opponent serves to {hit_area}.")
        
        for i, sd in enumerate(state_history):
            br = int(sd.get('ball_round', i * 2 + 1))
            pt = int(sd.get('player_type', 0))
            pn = self._shot_name(pt)
            px, py = sd.get('player_location_x', 0), sd.get('player_location_y', 0)
            hx, hy = sd.get('hit_x', 0), sd.get('hit_y', 0)
            lx, ly = sd.get('landing_x', 0), sd.get('landing_y', 0)
            mx, my = sd.get('moving_x', 0), sd.get('moving_y', 0)
            ox, oy = sd.get('opponent_location_x', 0), sd.get('opponent_location_y', 0)
            
            player_area = self._get_area_name(px, py)
            hit_area = self._get_area_name(hx, hy)
            land_area = self._get_area_name(lx, ly)
            move_area = self._get_area_name(mx, my)
            opp_area = self._get_area_name(ox, oy)
            
            # Shot header
            lines.append(f"Shot {br}:")
            
            # My shot
            if br == 1:
                lines.append(f"It is my turn to serve. I am at {player_area}, opponent at {opp_area}.")
                if pt == 11:
                    lines.append("MISSED SHOT - fault serve.")
                elif pt == 0:
                    lines.append("[Next shot to be decided]")
                else:
                    lines.append(f"I serve a {pn} to {land_area}, then move to {move_area}.")
            else:
                lines.append(f"It is my turn. I am at {player_area}, opponent at {opp_area}.")
                # Use the tracked opponent shot name from previous opponent turn
                if last_opponent_shot_name and last_opponent_shot_name != "Unknown":
                    lines.append(f"Opponent's {last_opponent_shot_name} is arriving at {hit_area}.")
                else:
                    # For Shot 2 (br == 2), use current state's opponent_type for incoming ball
                    current_ot = int(sd.get('opponent_type', 0))
                    if current_ot > 0:
                        current_on = self._shot_name(current_ot)
                        lines.append(f"Opponent's {current_on} is arriving at {hit_area}.")
                    else:
                        # Fallback to checking prev_sd's opponent_type
                        prev_sd = state_history[i - 1] if i > 0 else None
                        if prev_sd:
                            prev_ot = int(prev_sd.get('opponent_type', 0))
                            prev_on = self._shot_name(prev_ot)
                            if prev_ot > 0:
                                lines.append(f"Opponent's {prev_on} is arriving at {hit_area}.")
                            else:
                                lines.append(f"Ball is arriving at {hit_area}.")
                        else:
                            lines.append(f"Ball is arriving at {hit_area}.")
                
                if pt == 11:
                    lines.append("MISSED SHOT - could not reach the shuttle.")
                elif pt == 0:
                    lines.append("[Next shot to be decided]")
                else:
                    lines.append(f"I move to {hit_area} and hit a {pn} to {land_area}, then move to {move_area}.")
            
            # Derive opponent's next shot
            if i + 1 < len(state_history):
                next_sd = state_history[i + 1]
                next_br = br + 1
                next_ot = int(next_sd.get('opponent_type', 0))
                next_on = self._shot_name(next_ot)
                next_ox, next_oy = next_sd.get('opponent_location_x', 0), next_sd.get('opponent_location_y', 0)
                next_hx, next_hy = next_sd.get('hit_x', 0), next_sd.get('hit_y', 0)
                
                next_opp_area = self._get_area_name(next_ox, next_oy)
                next_hit_area = self._get_area_name(next_hx, next_hy)
                
                # Track opponent's shot name for next player's incoming ball
                last_opponent_shot_name = next_on if next_ot > 0 else None
                
                lines.append(f"Shot {next_br}:")
                lines.append(f"It is opponent's turn. I am at {move_area}, opponent at {opp_area}.")
                lines.append(f"My {pn} is arriving at {land_area}.")
                lines.append(f"Opponent moves to {land_area} and hits a {next_on} to {next_hit_area}, then moves to {next_opp_area}.")
        
        return "\n".join(lines)

    def build_prompt_context_nl(self, state_dict, segments, launch=False, state_history=None):
        """Build full prompt context dict for Jinja2."""
        formatted_cases = self.build_cases_list(segments)
        
        # Build trajectory text that includes current state
        trajectory_text = ""
        if state_history and len(state_history) > 0:
            full_history = state_history + [state_dict]
            trajectory_text = self.build_trajectory_text(full_history, state_dict)
        else:
            trajectory_text = self.build_trajectory_text([state_dict], state_dict)
        
        # Add semantic names to state_dict for template use
        px = state_dict.get('player_location_x', 0)
        py = state_dict.get('player_location_y', 0)
        ox = state_dict.get('opponent_location_x', 0)
        oy = state_dict.get('opponent_location_y', 0)
        hx = state_dict.get('hit_x', 0)
        hy = state_dict.get('hit_y', 0)
        ot = state_dict.get('opponent_type', 0)
        
        context = {
            'cases': formatted_cases,
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
