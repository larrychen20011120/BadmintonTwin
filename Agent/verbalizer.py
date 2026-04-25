"""
BadmintonVerbalizer - Converts state and match information to natural language strings.
"""

from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


class BadmintonVerbalizer:
    """
    Verbalizer for badminton game states and experiences.
    Converts structured data to natural language strings.
    """
    
    def __init__(self):
        """Initialize verbalizer."""
        pass
    
    def _get_area_name(self, x, y):
        """Get the area name for given coordinates using the standard mapping."""
        area_id = coord2area(float(x), float(y))
        return LANG_AREA_MAPPING.get(area_id, f"Unknown(area_id={area_id})")
    
    def _format_coords(self, x, y):
        """Format coordinates as a string."""
        return f"(x={x:.3f}, y={y:.3f})"
    
    def verbalize_state(self, state_dict):
        """
        Convert state dictionary to natural language description.
        
        Args:
            state_dict: Dictionary containing state information
        
        Returns:
            Semantic string description of the state
        """
        rally = int(state_dict.get('rally', 0))
        ball_round = int(state_dict.get('ball_round', 1))
        player_score = int(state_dict.get('player_score', 0))
        opponent_score = int(state_dict.get('opponent_score', 0))
        
        player_x = state_dict.get('player_location_x', 0)
        player_y = state_dict.get('player_location_y', 0)
        hit_x = state_dict.get('hit_x', 0)
        hit_y = state_dict.get('hit_y', 0)
        
        opponent_type = state_dict.get('opponent_type', 0)
        opponent_x = state_dict.get('opponent_location_x', 0)
        opponent_y = state_dict.get('opponent_location_y', 0)
        
        ball_dist = state_dict.get('ball_distance', 0)
        ball_dist_x = state_dict.get('ball_distance_x', 0)
        ball_dist_y = state_dict.get('ball_distance_y', 0)
        
        # Get area names using the standard mapping
        player_area = self._get_area_name(player_x, player_y)
        hit_area = self._get_area_name(hit_x, hit_y)
        opponent_area = self._get_area_name(opponent_x, opponent_y)
        opponent_type_name = INDEX2EN_SHOT.get(int(opponent_type), f"Unknown({opponent_type})")
        
        # Build narrative description with actual values
        parts = []
        
        # Rally context
        if ball_round == 1:
            parts.append(f"Rally {rally} begins with the serve.")
        else:
            parts.append(f"This is shot {ball_round} of rally {rally}.")
        
        # Score description
        score_diff = player_score - opponent_score
        if score_diff == 0:
            parts.append(f"The score is tied at {player_score}-{opponent_score}.")
        elif score_diff > 0:
            parts.append(f"You are leading {player_score}-{opponent_score}.")
        else:
            parts.append(f"You are trailing {player_score}-{opponent_score}.")
        
        # Player position with actual coordinates
        parts.append(f"You are currently at {player_area} {self._format_coords(player_x, player_y)}.")
        
        # Ball/Shuttle position
        if ball_round == 1:
            parts.append(f"You are serving from {hit_area} {self._format_coords(hit_x, hit_y)}.")
        else:
            parts.append(f"The shuttle is at {hit_area} {self._format_coords(hit_x, hit_y)}.")
            parts.append(f"The ball is at distance {ball_dist:.3f} with dx={ball_dist_x:.3f}, dy={ball_dist_y:.3f} relative to your position.")
        
        # Opponent action and position
        if ball_round > 1:
            parts.append(f"Your opponent just played a {opponent_type_name} from {opponent_area} {self._format_coords(opponent_x, opponent_y)}.")
        
        return " ".join(parts)
    
    def verbalize_experience(self, exp, exp_idx=1):
        """
        Convert experience dictionary to natural language description.
        
        Args:
            exp: Experience dictionary
            exp_idx: Experience index for labeling
        
        Returns:
            Semantic string description of the experience
        """
        rally = exp.get('rally', 0)
        ball_round = exp.get('ball_round', 1)
        player_score = int(exp.get('player_score', 0))
        opponent_score = int(exp.get('opponent_score', 0))
        similarity = exp.get('similarity', 0)
        
        player_x = exp.get('player_location_x', 0)
        player_y = exp.get('player_location_y', 0)
        hit_x = exp.get('hit_x', 0)
        hit_y = exp.get('hit_y', 0)
        
        opponent_type = exp.get('opponent_type', 0)
        opponent_x = exp.get('opponent_location_x', 0)
        opponent_y = exp.get('opponent_location_y', 0)
        
        player_type = exp.get('player_type', 0)
        landing_x = exp.get('landing_x', 0)
        landing_y = exp.get('landing_y', 0)
        
        moving_x = exp.get('moving_x', 0)
        moving_y = exp.get('moving_y', 0)
        
        # Get area names using standard mapping
        player_area = self._get_area_name(player_x, player_y)
        hit_area = self._get_area_name(hit_x, hit_y)
        opponent_area = self._get_area_name(opponent_x, opponent_y)
        landing_area = self._get_area_name(landing_x, landing_y)
        movement_area = self._get_area_name(moving_x, moving_y)
        
        opponent_type_name = INDEX2EN_SHOT.get(int(opponent_type), f"Unknown({opponent_type})")
        player_type_name = INDEX2EN_SHOT.get(int(player_type), f"Unknown({player_type})")
        
        # Build narrative
        parts = []
        
        # Header with similarity
        parts.append(f"[Case {exp_idx}, Similarity: {similarity:.2f}]")
        
        # Context
        parts.append(f"During rally {rally}, shot {ball_round} with score {player_score}-{opponent_score}:")
        
        # Player situation
        parts.append(f"The player was at {player_area} {self._format_coords(player_x, player_y)},")
        parts.append(f"receiving the shuttle at {hit_area} {self._format_coords(hit_x, hit_y)}.")
        
        # Opponent action
        if ball_round > 1:
            parts.append(f"The opponent had just played {opponent_type_name} from {opponent_area} {self._format_coords(opponent_x, opponent_y)}.")
        
        # Player's response
        parts.append(f"The player responded with {player_type_name}, landing the shuttle at {landing_area} {self._format_coords(landing_x, landing_y)},")
        parts.append(f"then moved to {movement_area} {self._format_coords(moving_x, moving_y)}.")
        
        return " ".join(parts)
    
    def verbalize_experiences(self, experiences):
        """
        Convert list of experiences to natural language description.
        
        Args:
            experiences: List of experience dictionaries
        
        Returns:
            String description of all experiences
        """
        if not experiences:
            return "No similar historical cases were found."
        
        header = f"Found {len(experiences)} similar historical cases:\n\n"
        
        verbalized_exps = []
        for i, exp in enumerate(experiences, 1):
            verbalized_exps.append(self.verbalize_experience(exp, i))
        
        separator = "\n\n"
        return header + separator.join(verbalized_exps)
    
    def build_prompt_context(self, state_dict, experiences, launch=False):
        """
        Build full prompt context with verbalized state and experiences.
        
        Args:
            state_dict: Dictionary containing state information
            experiences: List of experience dictionaries
            launch: Whether this is a serve shot
        
        Returns:
            Dictionary with verbalized strings
        """
        state_str = self.verbalize_state(state_dict)
        cases_str = self.verbalize_experiences(experiences)
        
        # Add serve note if applicable
        if launch:
            state_str += " This is a serve situation."
        
        return {
            'state_str': state_str,
            'cases_str': cases_str,
            'launch': launch,
            'top_k': len(experiences),
            # Keep original data for compatibility
            **state_dict,
            'experiences': experiences
        }


def create_verbalizer():
    """
    Factory function to create a BadmintonVerbalizer instance.
    
    Returns:
        BadmintonVerbalizer instance
    """
    return BadmintonVerbalizer()


# Convenience functions for direct use
def verbalize_state(state_dict):
    """
    Convert state dictionary to natural language string.
    
    Args:
        state_dict: Dictionary containing state information
    
    Returns:
        String description of the state
    """
    verbalizer = create_verbalizer()
    return verbalizer.verbalize_state(state_dict)


def verbalize_experiences(experiences):
    """
    Convert list of experiences to natural language string.
    
    Args:
        experiences: List of experience dictionaries
    
    Returns:
        String description of all experiences
    """
    verbalizer = create_verbalizer()
    return verbalizer.verbalize_experiences(experiences)
