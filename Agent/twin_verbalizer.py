"""
TwinVerbalizer - Converts multi-scale trajectory segments to natural language strings.
Specifically designed for the BadmintonTwin I-MSDP framework.
"""

from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.helper import coord2area


class TwinVerbalizer:
    """
    Verbalizer for BadmintonTwin's multi-scale trajectory retrieval.
    Converts trajectory segments (sequences of states and actions) to natural language.
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

    def _get_shot_type_name(self, shot_type):
        """Get shot type name from index."""
        return INDEX2EN_SHOT.get(int(shot_type), f"Unknown({shot_type})")

    def verbalize_trajectory_segment(self, segment, segment_idx=1):
        """
        Convert a trajectory segment (multi-step) to natural language description.

        Args:
            segment: Dict containing:
                - 'scale': int, length of the trajectory segment
                - 'similarity': float, normalized distance score (lower is better)
                - 'total_distance': float, cumulative distance
                - 'steps': list of dicts, each containing state and action info
            segment_idx: Index for labeling

        Returns:
            Semantic string description of the trajectory segment
        """
        scale = segment.get('scale', 1)
        similarity = segment.get('similarity', 0)
        total_distance = segment.get('total_distance', 0)
        steps = segment.get('steps', [])

        parts = []

        # Header with scale and similarity
        parts.append(
            f"[Segment {segment_idx}, Scale: {scale}, "
            f"Normalized Distance: {similarity:.4f}, Total Distance: {total_distance:.4f}]"
        )

        # Describe the trajectory pattern
        if scale == 1:
            parts.append("This is a single-shot match:")
        else:
            parts.append(f"This is a {scale}-shot tactical sequence:")

        # Describe each step in the sequence
        for step_idx, step in enumerate(steps):
            step_parts = []

            # Determine if this is the current (last) shot or a historical one
            is_current = (step_idx == len(steps) - 1)
            time_label = "Current shot" if is_current else f"Shot {step_idx + 1}"

            # Player position
            player_x = step.get('player_location_x', 0)
            player_y = step.get('player_location_y', 0)
            player_area = self._get_area_name(player_x, player_y)

            # Hit position
            hit_x = step.get('hit_x', 0)
            hit_y = step.get('hit_y', 0)
            hit_area = self._get_area_name(hit_x, hit_y)

            # Opponent info
            opponent_type = step.get('opponent_type', 0)
            opponent_type_name = self._get_shot_type_name(opponent_type)
            opponent_x = step.get('opponent_location_x', 0)
            opponent_y = step.get('opponent_location_y', 0)
            opponent_area = self._get_area_name(opponent_x, opponent_y)

            # Build step description
            if step_idx == 0:
                # First shot in sequence
                if is_current:
                    step_parts.append(
                        f"{time_label}: Player at {player_area} {self._format_coords(player_x, player_y)}, "
                        f"shuttle at {hit_area} {self._format_coords(hit_x, hit_y)}. "
                        f"Opponent played {opponent_type_name} from {opponent_area} {self._format_coords(opponent_x, opponent_y)}."
                    )
                else:
                    step_parts.append(
                        f"{time_label}: Player at {player_area} {self._format_coords(player_x, player_y)}, "
                        f"shuttle at {hit_area} {self._format_coords(hit_x, hit_y)}. "
                        f"Opponent played {opponent_type_name} from {opponent_area} {self._format_coords(opponent_x, opponent_y)}."
                    )
            else:
                step_parts.append(
                    f"{time_label}: Player at {player_area} {self._format_coords(player_x, player_y)}, "
                    f"shuttle at {hit_area} {self._format_coords(hit_x, hit_y)}. "
                    f"Opponent played {opponent_type_name} from {opponent_area} {self._format_coords(opponent_x, opponent_y)}."
                )

            # Player's response (action) - only for non-current shots (historical)
            if not is_current:
                player_type = step.get('player_type', 0)
                player_type_name = self._get_shot_type_name(player_type)
                landing_x = step.get('landing_x', 0)
                landing_y = step.get('landing_y', 0)
                landing_area = self._get_area_name(landing_x, landing_y)
                moving_x = step.get('moving_x', 0)
                moving_y = step.get('moving_y', 0)
                movement_area = self._get_area_name(moving_x, moving_y)

                step_parts.append(
                    f"Player responded with {player_type_name}, "
                    f"landing at {landing_area} {self._format_coords(landing_x, landing_y)}, "
                    f"then moved to {movement_area} {self._format_coords(moving_x, moving_y)}."
                )
            else:
                # For current shot, show what action was taken in the historical match
                if 'player_type' in step:
                    player_type = step.get('player_type', 0)
                    player_type_name = self._get_shot_type_name(player_type)
                    landing_x = step.get('landing_x', 0)
                    landing_y = step.get('landing_y', 0)
                    landing_area = self._get_area_name(landing_x, landing_y)
                    moving_x = step.get('moving_x', 0)
                    moving_y = step.get('moving_y', 0)
                    movement_area = self._get_area_name(moving_x, moving_y)

                    step_parts.append(
                        f"In the historical match, player responded with {player_type_name}, "
                        f"landing at {landing_area} {self._format_coords(landing_x, landing_y)}, "
                        f"then moved to {movement_area} {self._format_coords(moving_x, moving_y)}."
                    )

            parts.append(" ".join(step_parts))

        # Add tactical summary
        if scale > 1:
            parts.append("Tactical pattern: Observe how the player adapted their strategy across multiple shots.")

        return "\n".join(parts)

    def verbalize_trajectory_segments(self, segments):
        """
        Convert list of trajectory segments to natural language description.

        Args:
            segments: List of trajectory segment dicts

        Returns:
            String description of all trajectory segments
        """
        if not segments:
            return "No similar historical trajectory segments were found."

        header = f"Found {len(segments)} similar historical trajectory segments (multi-scale):\n\n"

        verbalized_segments = []
        for i, seg in enumerate(segments, 1):
            verbalized_segments.append(self.verbalize_trajectory_segment(seg, i))

        separator = "\n\n---\n\n"
        return header + separator.join(verbalized_segments)

    def verbalize_state(self, state_dict):
        """
        Convert state dictionary to natural language description.
        (Reused from standard verbalizer for consistency)

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

        player_area = self._get_area_name(player_x, player_y)
        hit_area = self._get_area_name(hit_x, hit_y)
        opponent_area = self._get_area_name(opponent_x, opponent_y)
        opponent_type_name = self._get_shot_type_name(opponent_type)

        parts = []

        if ball_round == 1:
            parts.append(f"Rally {rally} begins with the serve.")
        else:
            parts.append(f"This is shot {ball_round} of rally {rally}.")

        score_diff = player_score - opponent_score
        if score_diff == 0:
            parts.append(f"The score is tied at {player_score}-{opponent_score}.")
        elif score_diff > 0:
            parts.append(f"You are leading {player_score}-{opponent_score}.")
        else:
            parts.append(f"You are trailing {player_score}-{opponent_score}.")

        parts.append(
            f"You are currently at {player_area} {self._format_coords(player_x, player_y)}."
        )

        if ball_round == 1:
            parts.append(
                f"You are serving from {hit_area} {self._format_coords(hit_x, hit_y)}."
            )
        else:
            parts.append(
                f"The shuttle is at {hit_area} {self._format_coords(hit_x, hit_y)}."
            )
            parts.append(
                f"The ball is at distance {ball_dist:.3f} with "
                f"dx={ball_dist_x:.3f}, dy={ball_dist_y:.3f} relative to your position."
            )

        if ball_round > 1:
            parts.append(
                f"Your opponent just played a {opponent_type_name} from "
                f"{opponent_area} {self._format_coords(opponent_x, opponent_y)}."
            )

        return " ".join(parts)

    def build_prompt_context(self, state_dict, segments, launch=False):
        """
        Build full prompt context with verbalized state and trajectory segments.

        Args:
            state_dict: Dictionary containing state information
            segments: List of trajectory segment dicts
            launch: Whether this is a serve shot

        Returns:
            Dictionary with verbalized strings
        """
        state_str = self.verbalize_state(state_dict)
        segments_str = self.verbalize_trajectory_segments(segments)

        if launch:
            state_str += " This is a serve situation."

        return {
            'state_str': state_str,
            'segments_str': segments_str,
            'launch': launch,
            'top_k': len(segments),
            # Keep original data for compatibility
            **state_dict,
            'segments': segments
        }


def create_twin_verbalizer():
    """
    Factory function to create a TwinVerbalizer instance.

    Returns:
        TwinVerbalizer instance
    """
    return TwinVerbalizer()
