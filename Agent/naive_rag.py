import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from jinja2 import Template

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.llm import LLM
from Utils.helper import state_col, action_col, coord2area
from Utils.badminton_config import LANG_AREA_MAPPING, INDEX2EN_SHOT
from Utils.langfuse_tracing import BadmintonTracer


class NaiveRAGVerbalizer:
    """Verbalizer for NaiveRAG with natural turn-based format."""

    def __init__(self):
        pass

    @staticmethod
    def _get_area_name(x, y):
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            return "Unknown"
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

    def _build_stage1_description(self, exp):
        """Describe the decision context and outcome without placement details."""
        ball_round = int(exp.get('ball_round', 1))
        player_area = self._get_area_name(
            exp.get('player_location_x'), exp.get('player_location_y')
        )
        opponent_area = self._get_area_name(
            exp.get('opponent_location_x'), exp.get('opponent_location_y')
        )
        hit_area = self._get_area_name(exp.get('hit_x'), exp.get('hit_y'))
        opponent_type = int(exp.get('opponent_type', 0))
        shot_type = int(exp.get('player_type', 0))
        lines = list(self._build_context_header(exp))
        lines.append("")
        if ball_round == 1:
            lines.append(
                f"Serve situation: player at {player_area}, opponent at {opponent_area}."
            )
        else:
            incoming = self._shot_name(opponent_type) if opponent_type else "Unknown"
            lines.append(
                f"Player at {player_area}, opponent at {opponent_area}; "
                f"the opponent's {incoming} arrives at {hit_area}."
            )
        if shot_type == 11:
            lines.append("Outcome: Missed; no legal shot was completed.")
        else:
            lines.append(f"Outcome: Returned with {self._shot_name(shot_type)}.")
        return "\n".join(lines)

    def _build_stage2_description(self, exp):
        """Describe the complete historical action for placement imitation."""
        pt = int(exp.get('player_type', 0))
        pn = self._shot_name(pt)
        px, py = exp.get('player_location_x', 0), exp.get('player_location_y', 0)
        hx, hy = exp.get('hit_x', 0), exp.get('hit_y', 0)
        lx, ly = exp.get('landing_x', 0), exp.get('landing_y', 0)
        mx, my = exp.get('moving_x', 0), exp.get('moving_y', 0)
        ox, oy = exp.get('opponent_location_x', 0), exp.get('opponent_location_y', 0)
        ps = int(exp.get('player_score', 0))
        os = int(exp.get('opponent_score', 0))
        rally = int(exp.get('rally', 0))
        ball_round = int(exp.get('ball_round', 1))
        ot = int(exp.get('opponent_type', 0))
        on = self._shot_name(ot)

        player_area = self._get_area_name(px, py)
        hit_area = self._get_area_name(hx, hy)
        land_area = self._get_area_name(lx, ly)
        move_area = self._get_area_name(mx, my)
        opp_area = self._get_area_name(ox, oy)

        score_diff = ps - os
        if score_diff > 0:
            score_text = f"I lead by {score_diff} points."
        elif score_diff < 0:
            score_text = f"I trail by {abs(score_diff)} points."
        else:
            score_text = "The score is tied."

        shots_played = ball_round - 1
        if shots_played == 0:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {ball_round}. This is the first shot of the rally."
        else:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {ball_round}. {shots_played} shots have been played in this rally."

        lines = [header, ""]
        if ball_round == 1:
            lines.append(f"It is my turn to serve. I am at {player_area}, opponent at {opp_area}.")
            if pt == 11:
                lines.append("MISSED SHOT - fault serve.")
            elif pt > 0:
                lines.append(f"I serve a {pn} to {land_area}, then move to {move_area}.")
            else:
                lines.append(f"I serve to {land_area}, then move to {move_area}.")
        else:
            lines.append(f"It is my turn. I am at {player_area}, opponent at {opp_area}.")
            if ot > 0:
                lines.append(f"Opponent's {on} is arriving at {hit_area}.")
            else:
                lines.append(f"Ball is arriving at {hit_area}.")

            if pt == 11:
                lines.append("MISSED SHOT - could not reach the shuttle.")
            elif pt > 0:
                lines.append(f"I move to {hit_area} and hit a {pn} to {land_area}, then move to {move_area}.")
            else:
                lines.append(f"I move to {hit_area} and hit to {land_area}, then move to {move_area}.")

        if pt > 0:
            prompt_action = BaseAgent.build_prompt_action(pt, lx, ly, mx, my)
            lines.append(BaseAgent.format_prompt_action(prompt_action))

        return "\n".join(lines)

    def build_case_verbalization(self, exp):
        """Build stage-specific rank-independent text for one database row."""
        return {
            'num_shots': 1,
            'stage1_description': self._build_stage1_description(exp),
            'stage2_description': self._build_stage2_description(exp),
        }

    def format_stage2_case(self, exp, idx, is_serve=False):
        """Attach runtime retrieval data to a cached Stage 2 case."""
        verbalization = exp.get('case_verbalization')
        if verbalization is None:
            verbalization = self.build_case_verbalization(exp)
        return {
            'index': idx,
            'num_shots': verbalization['num_shots'],
            'stage2_description': verbalization['stage2_description'],
        }

    def build_stage2_cases(self, experiences, is_serve=False):
        """Build complete-action cases for Stage 2 placement generation."""
        cases = []
        for i, exp in enumerate(experiences, 1):
            cases.append(self.format_stage2_case(exp, i, is_serve=is_serve))
        return cases

    def format_stage1_case(self, exp, idx):
        """Render reachability and shot evidence without placement coordinates."""
        verbalization = exp.get('case_verbalization')
        if verbalization is None:
            verbalization = self.build_case_verbalization(exp)
        return {
            'index': idx,
            'case_id': int(exp['record_id']),
            'stage1_description': verbalization['stage1_description'],
        }

    def build_stage1_cases(self, experiences):
        return [
            self.format_stage1_case(exp, idx)
            for idx, exp in enumerate(experiences, 1)
        ]

    def _format_stroke_lines(self, sd, is_current=False):
        """Format one completed or current stroke from the active agent's viewpoint."""
        br = int(sd.get('ball_round', 0))
        is_target_turn = bool(sd.get('is_target_turn', True))
        pt = int(sd.get('player_type', 0))
        pn = self._shot_name(pt)
        ot = int(sd.get('opponent_type', 0))
        on = self._shot_name(ot)

        px, py = sd.get('player_location_x', 0), sd.get('player_location_y', 0)
        hx, hy = sd.get('hit_x', 0), sd.get('hit_y', 0)
        lx, ly = sd.get('landing_x', 0), sd.get('landing_y', 0)
        mx, my = sd.get('moving_x', 0), sd.get('moving_y', 0)
        ox, oy = sd.get('opponent_location_x', 0), sd.get('opponent_location_y', 0)

        hitter_area = self._get_area_name(px, py)
        receiver_area = self._get_area_name(ox, oy)
        hit_area = self._get_area_name(hx, hy)
        land_area = self._get_area_name(lx, ly)
        move_area = self._get_area_name(mx, my)

        lines = [f"Shot {br}:"]
        if is_target_turn:
            lines.append(f"It is my turn. I am at {hitter_area}, opponent at {receiver_area}.")
            if br == 1:
                if pt == 0 and is_current:
                    lines.append("[Next shot to be decided]")
                elif pt == 11:
                    lines.append("MISSED SHOT - fault serve.")
                elif pt > 0:
                    lines.append(f"I serve a {pn} to {land_area}, then move to {move_area}.")
                else:
                    lines.append(f"I serve to {land_area}, then move to {move_area}.")
            else:
                if ot > 0:
                    lines.append(f"Opponent's {on} is arriving at {hit_area}.")
                else:
                    lines.append(f"Ball is arriving at {hit_area}.")

                if pt == 0 and is_current:
                    lines.append("[Next shot to be decided]")
                elif pt == 11:
                    lines.append("MISSED SHOT - could not reach the shuttle.")
                elif pt > 0:
                    lines.append(f"I move to {hit_area} and hit a {pn} to {land_area}, then move to {move_area}.")
                else:
                    lines.append(f"I move to {hit_area} and hit to {land_area}, then move to {move_area}.")
        else:
            lines.append(f"It is opponent's turn. I am at {receiver_area}, opponent at {hitter_area}.")
            if br == 1:
                if pt == 11:
                    lines.append("Opponent MISSED SHOT - fault serve.")
                elif pt > 0:
                    lines.append(f"Opponent serves a {pn} to {land_area}, then moves to {move_area}.")
                else:
                    lines.append(f"Opponent serves to {land_area}, then moves to {move_area}.")
            else:
                if ot > 0:
                    lines.append(f"My {on} is arriving at {hit_area}.")
                else:
                    lines.append(f"Ball is arriving at {hit_area}.")

                if pt == 11:
                    lines.append("Opponent MISSED SHOT - could not reach the shuttle.")
                elif pt > 0:
                    lines.append(f"Opponent moves to {hit_area} and hits a {pn} to {land_area}, then moves to {move_area}.")
                else:
                    lines.append(f"Opponent moves to {hit_area} and hits to {land_area}, then moves to {move_area}.")

        return lines

    def _build_full_trajectory_text(self, state_history, current_state):
        lines = list(self._build_context_header(current_state))
        lines.append("")

        seen_ball_rounds = set()
        current_br = int(current_state.get('ball_round', 0))
        for sd in state_history:
            br = int(sd.get('ball_round', 0))
            if br <= 0 or br in seen_ball_rounds:
                continue
            seen_ball_rounds.add(br)
            is_current = br == current_br and int(sd.get('player_type', 0)) == 0
            lines.extend(self._format_stroke_lines(sd, is_current=is_current))

        return "\n".join(lines)

    def build_trajectory_text(self, state_history, current_state):
        """Build natural turn-based trajectory text for current rally."""
        if not state_history:
            return "\n".join(self._build_context_header(current_state))

        if any('is_target_turn' in sd for sd in state_history):
            return self._build_full_trajectory_text(state_history, current_state)

        lines = list(self._build_context_header(current_state))
        lines.append("")
        last_opponent_shot_name = None

        first_br = int(state_history[0].get('ball_round', 1))
        if first_br == 2:
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

            lines.append(f"Shot {br}:")
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
                if last_opponent_shot_name and last_opponent_shot_name != "Unknown":
                    lines.append(f"Opponent's {last_opponent_shot_name} is arriving at {hit_area}.")
                else:
                    current_ot = int(sd.get('opponent_type', 0))
                    if current_ot > 0:
                        current_on = self._shot_name(current_ot)
                        lines.append(f"Opponent's {current_on} is arriving at {hit_area}.")
                    else:
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

            if i + 1 < len(state_history):
                next_sd = state_history[i + 1]
                next_br = br + 1
                next_ot = int(next_sd.get('opponent_type', 0))
                next_on = self._shot_name(next_ot)
                next_ox, next_oy = next_sd.get('opponent_location_x', 0), next_sd.get('opponent_location_y', 0)
                next_hx, next_hy = next_sd.get('hit_x', 0), next_sd.get('hit_y', 0)

                next_opp_area = self._get_area_name(next_ox, next_oy)
                next_hit_area = self._get_area_name(next_hx, next_hy)
                last_opponent_shot_name = next_on if next_ot > 0 else None

                lines.append(f"Shot {next_br}:")
                lines.append(f"It is opponent's turn. I am at {move_area}, opponent at {opp_area}.")
                lines.append(f"My {pn} is arriving at {land_area}.")
                lines.append(f"Opponent moves to {land_area} and hits a {next_on} to {next_hit_area}, then moves to {next_opp_area}.")

        return "\n".join(lines)

    def build_prompt_context_nl(self, state_dict, experiences, launch=False, state_history=None):
        """Build full prompt context dict for Jinja2."""
        is_serve = launch or int(state_dict.get('ball_round', 1)) == 1
        cases = self.build_stage2_cases(experiences, is_serve=is_serve)

        if state_history and len(state_history) > 0:
            full_history = state_history + [state_dict]
            trajectory_text = self.build_trajectory_text(full_history, state_dict)
        else:
            trajectory_text = self.build_trajectory_text([state_dict], state_dict)

        px = state_dict.get('player_location_x', 0)
        py = state_dict.get('player_location_y', 0)
        ox = state_dict.get('opponent_location_x', 0)
        oy = state_dict.get('opponent_location_y', 0)
        hx = state_dict.get('hit_x', 0)
        hy = state_dict.get('hit_y', 0)
        ot = state_dict.get('opponent_type', 0)

        return {
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

    def build_dual_prompt_context_nl(self, state_dict, retrieval_result, launch=False, state_history=None):
        """Build prompt context with separate normal and missed-shot evidence."""
        normal_cases_raw = retrieval_result.get('normal_cases', []) if isinstance(retrieval_result, dict) else []
        miss_cases_raw = retrieval_result.get('miss_cases', []) if isinstance(retrieval_result, dict) else []
        context = self.build_prompt_context_nl(
            state_dict,
            normal_cases_raw,
            launch=launch,
            state_history=state_history,
        )
        context.update({
            'normal_cases': self.build_stage2_cases(normal_cases_raw, is_serve=launch),
            'miss_cases': self.build_stage2_cases(miss_cases_raw, is_serve=launch),
            'normal_top_k': len(normal_cases_raw),
            'miss_top_k': len(miss_cases_raw),
        })
        return context


def create_naive_rag_verbalizer():
    return NaiveRAGVerbalizer()


class NaiveRAGDatabase:
    """
    Database for NaiveRAG Agent.
    Stores training data and retrieves similar experiences based on:
    - player_location_x, player_location_y
    - hit_x, hit_y
    - opponent_type
    - opponent_location_x, opponent_location_y
    """
    schema_version = 8
    verbalization_schema_version = 3
    missed_coordinate_keys = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
    
    def __init__(self, df, player_name, player_id, build_metadata=None):
        """
        Initialize database with filtered data for a specific player.
        
        Args:
            df: DataFrame with all data
            player_name: Name of the player
            player_id: ID of the player
        """
        self.player_name = player_name
        self.player_id = player_id
        self.build_metadata = dict(build_metadata or {
            'dataset': 'rally',
            'split_sha256': 'ad-hoc',
            'profiles_sha256': None,
        })
        
        # Filter data for this player
        self.data = df[df['player'] == player_name].copy().reset_index(drop=True)
        self.data['ball_distance_x'] = (
            self.data['hit_x'].astype(float) - self.data['player_location_x'].astype(float)
        )
        self.data['ball_distance_y'] = (
            self.data['hit_y'].astype(float) - self.data['player_location_y'].astype(float)
        )
        miss_mask = self.data['player_type'].astype(int) == 11
        self.data.loc[miss_mask, list(self.missed_coordinate_keys)] = np.nan
        
        # Define continuous retrieval keys (for Euclidean distance)
        self.continuous_keys = [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y',
            'ball_distance_x', 'ball_distance_y',
        ]
        
        # Define discrete retrieval keys (for Hamming distance)
        self.discrete_keys = ['opponent_type']
        
        # Define all retrieval keys
        self.retrieval_keys = self.continuous_keys + self.discrete_keys
        
        # Define metadata keys (additional context)
        self.metadata_keys = [
            'match_id', 'set', 'rally', 'ball_round',
            'player_score', 'opponent_score', 'score_status',
            'ball_distance', 'ball_distance_x', 'ball_distance_y',
            'opponent_move_x', 'opponent_move_y'
        ]
        
        # Define action keys
        self.action_keys = ['landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y']
        
        self._build_case_verbalizations()

        # Build feature matrix for fast retrieval
        self._build_feature_matrix()
        
        print(f"Database initialized for {player_name} with {len(self.data)} entries")
    
    def _build_case_verbalizations(self):
        verbalizer = NaiveRAGVerbalizer()
        self.case_verbalizations = [
            {
                'record_id': int(row.name),
                **verbalizer.build_case_verbalization(self._build_static_experience(row)),
            }
            for _, row in self.data.iterrows()
        ]

    def _build_feature_matrix(self):
        """Build feature matrix for mixed distance computation."""
        self.data_stores = {
            'classic': self.data,
            'normal': self.data[self.data['player_type'].astype(int) != 11].copy(),
            'miss': self.data[self.data['player_type'].astype(int) == 11].copy(),
        }
        self.feature_stores = {}
        for name, store_data in self.data_stores.items():
            self.feature_stores[name] = {
                'continuous': store_data[self.continuous_keys].values.astype(np.float32),
                'discrete': store_data[self.discrete_keys].values.astype(np.float32),
            }

        self.continuous_features = self.feature_stores['classic']['continuous']
        self.discrete_features = self.feature_stores['classic']['discrete']
        total_count = len(self.data_stores['classic'])
        miss_count = len(self.data_stores['miss'])
        self.stats = {
            'total_count': int(total_count),
            'normal_count': int(len(self.data_stores['normal'])),
            'miss_count': int(miss_count),
            'miss_rate': float(miss_count / total_count) if total_count else 0.0,
        }
    
    def _compute_mixed_distance(self, query_continuous, query_discrete, store_name='classic'):
        """
        Compute mixed distance combining Euclidean distance for continuous features
        and Hamming distance for discrete features.
        
        Args:
            query_continuous: Continuous features [n_continuous] (already standardized)
            query_discrete: Discrete features [n_discrete]
        
        Returns:
            distances: Array of mixed distances for all database entries
        """
        features = self.feature_stores[store_name]
        continuous_features = features['continuous']
        discrete_features = features['discrete']
        if len(continuous_features) == 0:
            return np.array([], dtype=np.float32)

        # Compute Euclidean distance for continuous features
        continuous_diff = continuous_features - query_continuous
        continuous_dist = np.sqrt(np.sum(continuous_diff ** 2, axis=1))
        
        # Compute Hamming distance for discrete features (normalized by number of features)
        discrete_diff = (discrete_features != query_discrete).astype(np.float32)
        discrete_dist = np.mean(discrete_diff, axis=1)  # Average across discrete features
        
        # Combine distances with equal weighting (can be adjusted)
        # Both are normalized to [0, 1] range approximately
        mixed_dist = continuous_dist + discrete_dist
        
        return mixed_dist

    def _extract_query_features(self, query_state):
        if isinstance(query_state, dict):
            query_continuous = np.array([query_state[k] for k in self.continuous_keys], dtype=np.float32)
            query_discrete = np.array([query_state[k] for k in self.discrete_keys], dtype=np.float32)
        else:
            n_continuous = len(self.continuous_keys)
            query_continuous = query_state[:n_continuous].astype(np.float32)
            query_discrete = query_state[n_continuous:].astype(np.float32)
        return query_continuous, query_discrete

    def _build_static_experience(self, row):
        exp = {'record_id': int(row.name)}
        is_miss = int(float(row.get('player_type', 0))) == 11

        # Add retrieval keys
        for key in self.retrieval_keys:
            exp[key] = float(row[key])

        # Add metadata
        for key in self.metadata_keys:
            if key in row:
                exp[key] = float(row[key])

        # Add score_status as integer
        if 'player_score' in row and 'opponent_score' in row:
            exp['score_status'] = int(float(row['player_score']) - float(row['opponent_score']))

        # Add semantic area names for all positions
        if 'player_location_x' in row and 'player_location_y' in row:
            player_area_id = coord2area(float(row['player_location_x']), float(row['player_location_y']))
            exp['player_location_area'] = LANG_AREA_MAPPING.get(player_area_id, 'Unknown')

        if 'hit_x' in row and 'hit_y' in row:
            hit_area_id = coord2area(float(row['hit_x']), float(row['hit_y']))
            exp['hit_position_area'] = LANG_AREA_MAPPING.get(hit_area_id, 'Unknown')

        if 'opponent_location_x' in row and 'opponent_location_y' in row:
            opponent_area_id = coord2area(float(row['opponent_location_x']), float(row['opponent_location_y']))
            exp['opponent_location_area'] = LANG_AREA_MAPPING.get(opponent_area_id, 'Unknown')

        # Add actions
        for key in self.action_keys:
            if key in row:
                if is_miss and key in self.missed_coordinate_keys:
                    exp[key] = None
                else:
                    exp[key] = float(row[key])

        # Add landing area semantic name
        if not is_miss and 'landing_x' in row and 'landing_y' in row:
            landing_area_id = coord2area(float(row['landing_x']), float(row['landing_y']))
            exp['landing_area'] = LANG_AREA_MAPPING.get(landing_area_id, 'Unknown')
        else:
            exp['landing_area'] = None if is_miss else 'Unknown'

        # Add movement area semantic name
        if not is_miss and 'moving_x' in row and 'moving_y' in row:
            movement_area_id = coord2area(float(row['moving_x']), float(row['moving_y']))
            exp['moving_area'] = LANG_AREA_MAPPING.get(movement_area_id, 'Unknown')
        else:
            exp['moving_area'] = None if is_miss else 'Unknown'

        # Add shot type name
        if 'opponent_type' in row:
            exp['opponent_type_name'] = INDEX2EN_SHOT.get(int(row['opponent_type']), f"Unknown({int(row['opponent_type'])})")

        return exp

    def _build_experience(self, row, similarity, distance):
        exp = self._build_static_experience(row)
        record_id = exp['record_id']
        exp['case_verbalization'] = dict(self.case_verbalizations[record_id])
        exp['similarity'] = float(similarity)
        exp['distance'] = float(distance)
        return exp

    def _retrieve_from_store(self, query_state, top_k=5, store_name='classic'):
        """
        Retrieve top-k similar experiences using mixed distance.

        Args:
            query_state: Dict or array with retrieval keys
            top_k: Number of experiences to retrieve

        Returns:
            List of dicts containing experience data
        """
        # Extract query features
        query_continuous, query_discrete = self._extract_query_features(query_state)

        # Compute mixed distances (lower is better)
        distances = self._compute_mixed_distance(query_continuous, query_discrete, store_name=store_name)
        if len(distances) == 0:
            return []

        # Convert distances to similarities (higher is better)
        # Using negative exponential: similarity = exp(-distance)
        similarities = np.exp(-distances)

        # Get top-k indices (smallest distances = largest similarities)
        top_indices = np.argsort(distances)[:top_k]

        # Build experience list
        store_data = self.data_stores[store_name]
        experiences = []
        for idx in top_indices:
            row = store_data.iloc[idx]
            exp = self._build_experience(row, similarities[idx], distances[idx])
            experiences.append(exp)

        return experiences

    def retrieve(self, query_state, top_k=5):
        return self._retrieve_from_store(query_state, top_k=top_k, store_name='classic')

    def retrieve_dual(self, query_state, normal_top_k=8, miss_top_k=5):
        normal_cases = self._retrieve_from_store(query_state, top_k=normal_top_k, store_name='normal')
        miss_cases = self._retrieve_from_store(query_state, top_k=miss_top_k, store_name='miss')
        nearest_normal = normal_cases[0] if normal_cases else {}
        nearest_miss = miss_cases[0] if miss_cases else {}
        return {
            'normal_cases': normal_cases,
            'miss_cases': miss_cases,
            'summary': {
                **self.stats,
                'normal_top_k': int(normal_top_k),
                'miss_top_k': int(miss_top_k),
                'retrieved_normal_count': int(len(normal_cases)),
                'retrieved_miss_count': int(len(miss_cases)),
                'nearest_normal_distance': nearest_normal.get('distance'),
                'nearest_miss_distance': nearest_miss.get('distance'),
                'nearest_normal_similarity': nearest_normal.get('similarity'),
                'nearest_miss_similarity': nearest_miss.get('similarity'),
            }
        }

    def compute_distances(self, query_state):
        """Compute one canonical distance vector for all player records."""
        query_continuous, query_discrete = self._extract_query_features(query_state)
        return self._compute_mixed_distance(
            query_continuous,
            query_discrete,
            store_name='classic',
        )

    def _retrieve_from_record_ids(self, distances, record_ids, top_k):
        record_ids = np.asarray(list(record_ids), dtype=int)
        if len(record_ids) == 0 or len(distances) == 0 or top_k <= 0:
            return []
        order = np.lexsort((record_ids, distances[record_ids]))
        selected = record_ids[order[:top_k]]
        return [
            self._build_experience(
                self.data.iloc[int(record_id)],
                np.exp(-distances[int(record_id)]),
                distances[int(record_id)],
            )
            for record_id in selected
        ]

    @staticmethod
    def _legal_return_types(launch):
        return {1, 10} if launch else set(range(2, 10))

    def available_shot_types(self, launch=False):
        legal = self._legal_return_types(launch)
        present = set(self.data['player_type'].astype(int))
        return tuple(sorted(legal & present))

    def retrieve_stage1(self, query_state, normal_top_k=10, miss_top_k=5, launch=False):
        """Retrieve legal returned and missed evidence with independent quotas."""
        distances = self.compute_distances(query_state)
        shot_types = self.data['player_type'].astype(int).to_numpy()
        legal = self._legal_return_types(launch)
        normal_ids = np.flatnonzero(np.isin(shot_types, list(legal)))
        miss_ids = np.flatnonzero(shot_types == 11)
        return {
            'normal_cases': self._retrieve_from_record_ids(
                distances, normal_ids, normal_top_k
            ),
            'miss_cases': self._retrieve_from_record_ids(
                distances, miss_ids, miss_top_k
            ),
            'distances': distances,
        }

    def retrieve_by_shot_type(self, query_state, shot_type, top_k=10, distances=None):
        """Filter by the selected action type before ranking candidates."""
        shot_type = int(shot_type)
        if shot_type == 11:
            raise ValueError('Missed shot does not have placement evidence')
        if distances is None:
            distances = self.compute_distances(query_state)
        shot_types = self.data['player_type'].astype(int).to_numpy()
        coordinates = self.data[
            ['landing_x', 'landing_y', 'moving_x', 'moving_y']
        ].to_numpy(dtype=float)
        valid_placement = np.isfinite(coordinates).all(axis=1)
        valid_placement &= (coordinates >= -1.0).all(axis=1)
        valid_placement &= (coordinates <= 1.0).all(axis=1)
        record_ids = np.flatnonzero((shot_types == shot_type) & valid_placement)
        return self._retrieve_from_record_ids(distances, record_ids, top_k)

    def validate(self):
        if self.schema_version != type(self).schema_version:
            raise ValueError(f"Unsupported NaiveRAG schema version: {self.schema_version}")
        if self.verbalization_schema_version != type(self).verbalization_schema_version:
            raise ValueError(
                f"Unsupported NaiveRAG verbalization schema version: "
                f"{self.verbalization_schema_version}"
            )
        if self.build_metadata.get('dataset') != 'rally':
            raise ValueError("NaiveRAG database was not built from the rally dataset")
        if not self.build_metadata.get('split_sha256'):
            raise ValueError("NaiveRAG database is missing rally split provenance")
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            raise ValueError(f"NaiveRAG database for {self.player_name} contains no rows")
        expected_dx = self.data['hit_x'].astype(float) - self.data['player_location_x'].astype(float)
        expected_dy = self.data['hit_y'].astype(float) - self.data['player_location_y'].astype(float)
        if not np.allclose(self.data['ball_distance_x'].astype(float), expected_dx):
            raise ValueError("NaiveRAG ball_distance_x does not match hit_x - player_location_x")
        if not np.allclose(self.data['ball_distance_y'].astype(float), expected_dy):
            raise ValueError("NaiveRAG ball_distance_y does not match hit_y - player_location_y")
        if list(self.data.index) != list(range(len(self.data))):
            raise ValueError("NaiveRAG record IDs are not aligned with the database rows")
        if len(self.case_verbalizations) != len(self.data):
            raise ValueError("NaiveRAG case verbalization count does not match row count")
        for record_id, verbalization in enumerate(self.case_verbalizations):
            if not isinstance(verbalization, dict):
                raise ValueError(f"NaiveRAG verbalization {record_id} is not an object")
            if verbalization.get('record_id') != record_id:
                raise ValueError(f"NaiveRAG verbalization {record_id} is assigned to the wrong row")
            if verbalization.get('num_shots') != 1:
                raise ValueError(f"NaiveRAG verbalization {record_id} has invalid num_shots")
            for stage in ('stage1', 'stage2'):
                description = verbalization.get(f'{stage}_description')
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(
                        f"NaiveRAG verbalization {record_id} has no {stage} description"
                    )
        for store_name in ('classic', 'normal', 'miss'):
            if store_name not in self.data_stores or store_name not in self.feature_stores:
                raise ValueError(f"NaiveRAG database is missing the {store_name} store")
            store = self.data_stores[store_name]
            features = self.feature_stores[store_name]
            if len(store) != len(features['continuous']) or len(store) != len(features['discrete']):
                raise ValueError(f"NaiveRAG {store_name} store is not aligned with its features")
            expected_continuous = store[self.continuous_keys].values.astype(np.float32)
            expected_discrete = store[self.discrete_keys].values.astype(np.float32)
            if not np.array_equal(features['continuous'], expected_continuous):
                raise ValueError(f"NaiveRAG {store_name} continuous features are assigned to the wrong rows")
            if not np.array_equal(features['discrete'], expected_discrete):
                raise ValueError(f"NaiveRAG {store_name} discrete features are assigned to the wrong rows")

        expected_normal = self.data[self.data['player_type'].astype(int) != 11]
        expected_miss = self.data[self.data['player_type'].astype(int) == 11]
        if not self.data_stores['classic'].equals(self.data):
            raise ValueError("NaiveRAG classic store does not match the canonical data")
        if not self.data_stores['normal'].equals(expected_normal):
            raise ValueError("NaiveRAG normal store does not match player_type")
        if not self.data_stores['miss'].equals(expected_miss):
            raise ValueError("NaiveRAG miss store does not match player_type")
        if not expected_miss[list(self.missed_coordinate_keys)].isna().all().all():
            raise ValueError("NaiveRAG missed-shot coordinates must all be null")
        total_count = len(self.data)
        normal_count = len(expected_normal)
        miss_count = len(expected_miss)
        expected_stats = {
            'total_count': total_count,
            'normal_count': normal_count,
            'miss_count': miss_count,
            'miss_rate': float(miss_count / total_count),
        }
        if self.stats != expected_stats:
            raise ValueError("NaiveRAG stats do not match the data stores")
        return self
    
    def save(self, filepath):
        """Save database to file."""
        self.validate()
        payload = {
                'schema_version': self.schema_version,
                'verbalization_schema_version': self.verbalization_schema_version,
                'player_name': self.player_name,
                'player_id': self.player_id,
                'build_metadata': self.build_metadata,
                'data': self.data,
                'case_verbalizations': self.case_verbalizations,
                'data_stores': self.data_stores,
                'feature_stores': self.feature_stores,
                'stats': self.stats,
                'continuous_features': self.continuous_features,
                'discrete_features': self.discrete_features,
                'continuous_keys': self.continuous_keys,
                'discrete_keys': self.discrete_keys,
                'retrieval_keys': self.retrieval_keys,
                'metadata_keys': self.metadata_keys,
                'action_keys': self.action_keys
        }
        temporary_path = f'{filepath}.tmp-{os.getpid()}'
        try:
            with open(temporary_path, 'wb') as f:
                pickle.dump(payload, f)
            os.replace(temporary_path, filepath)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        print(f"Database saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load database from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        schema_version = data.get('schema_version', 1)
        verbalization_schema_version = data.get('verbalization_schema_version')
        if schema_version != cls.schema_version or verbalization_schema_version != cls.verbalization_schema_version:
            raise ValueError(
                "This NaiveRAG database is not compatible with the rally dataset. "
                "Rebuild it with Scripts/prepare_all_data.py --naive-rag."
            )
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.schema_version = schema_version
        instance.verbalization_schema_version = verbalization_schema_version
        instance.player_name = data['player_name']
        instance.player_id = data['player_id']
        instance.build_metadata = data.get('build_metadata', {})
        instance.data = data['data']
        instance.case_verbalizations = data['case_verbalizations']
        instance.continuous_keys = data['continuous_keys']
        instance.discrete_keys = data['discrete_keys']
        instance.retrieval_keys = data['retrieval_keys']
        instance.metadata_keys = data['metadata_keys']
        instance.action_keys = data['action_keys']
        instance.data_stores = data['data_stores']
        instance.feature_stores = data['feature_stores']
        instance.stats = data['stats']
        instance.continuous_features = instance.feature_stores['classic']['continuous']
        instance.discrete_features = instance.feature_stores['classic']['discrete']
        instance.validate()
        
        print(f"Database loaded from {filepath}")
        return instance


class NaiveRAGAgent(BaseAgent):
    """
    NaiveRAG Agent that retrieves similar experiences and uses LLM to generate actions.
    Supports verbalization to convert state and experiences to natural language.
    """
    
    def __init__(self, player_name, database, model_name=None,
                 api_key=None, host=None, port=None, base_url=None,
                 host_url=None, normal_top_k=10, miss_top_k=5,
                 conditioned_top_k=10, enable_tracing=True, prompt_variant="norm",
                 case_grounding=True, action_architecture="two_stage_v1",
                 save_prompt_traces=False):
        """
        Initialize NaiveRAG Agent.
        
        Args:
            player_name: Name of the player
            database: NaiveRAGDatabase instance
            model_name: Name of the LLM model
            api_key: API key for LLM
            port: Port for local LLM server
            top_k: Number of similar experiences to retrieve
            enable_tracing: Whether to enable Langfuse tracing
        """
        super().__init__(player_name)
        
        self.database = database
        self.normal_top_k = int(normal_top_k)
        self.miss_top_k = int(miss_top_k)
        self.conditioned_top_k = int(conditioned_top_k)
        if min(self.normal_top_k, self.miss_top_k, self.conditioned_top_k) < 1:
            raise ValueError('RAG retrieval top-k values must all be positive')
        if action_architecture not in {'two_stage_v1', 'single_stage_v1'}:
            raise ValueError(f'Unsupported action architecture: {action_architecture}')
        if action_architecture == 'single_stage_v1' and case_grounding:
            raise ValueError('Single-stage NaiveRAG supports direct output only')
        self.retrieval_architecture = action_architecture
        self.prompt_variant = prompt_variant
        if not isinstance(case_grounding, bool):
            raise TypeError('case_grounding must be a bool')
        self.case_grounding = case_grounding
        if action_architecture == 'single_stage_v1':
            self.action_schema, self.action_schema_name = (
                self.get_rag_single_stage_response_schema()
            )
        else:
            self.stage1_schema, self.stage1_schema_name = self.get_rag_response_schema(
                1, case_grounding
            )
            self.stage2_schema, self.stage2_schema_name = self.get_rag_response_schema(
                2, case_grounding
            )
        tracer_name = (
            'naive_rag_single_stage'
            if action_architecture == 'single_stage_v1' else 'naive_rag'
        )
        self.tracer = BadmintonTracer(
            tracer_name,
            player_name,
            enable_tracing,
            capture_local=save_prompt_traces,
        )
        
        # Trajectory tracking for current rally
        self.trajectory = []  # list of state_dicts for the current rally
        self._current_rally = None
        self._last_case_ids = []
        self.last_decision_trace = None
        
        # Initialize verbalizer
        self.verbalizer = create_naive_rag_verbalizer()
        
        # Initialize LLM
        self.llm = LLM(
            model_name=model_name,
            host=host,
            api_key=api_key,
            port=port,
            base_url=base_url,
            host_url=host_url,
        )
        
        # Load prompts
        prompt_root = os.path.join(project_path, "Prompts", "New")
        template_suffix = 'instruct' if prompt_variant == 'shifted' else 'norm'
        if prompt_variant not in {'norm', 'shifted'}:
            raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            self.system_prompt = f.read()

        if action_architecture == 'single_stage_v1':
            action_filename = (
                'action_instruct.jinja'
                if template_suffix == 'instruct' else 'action.jinja'
            )
            action_path = os.path.join(
                prompt_root, 'naive_rag_single_stage', action_filename
            )
            with open(action_path, 'r', encoding='utf-8') as f:
                self.action_template = Template(f.read())
        else:
            prompt_dir = os.path.join(prompt_root, "naive_rag")
            with open(os.path.join(prompt_dir, f'stage1_{template_suffix}.jinja'), "r") as f:
                self.stage1_template = Template(f.read())
            with open(os.path.join(prompt_dir, f'stage2_{template_suffix}.jinja'), "r") as f:
                self.stage2_template = Template(f.read())

    @staticmethod
    def _parse_llm_action(response):
        return BaseAgent.parse_prompt_action(response)

    def _normalize_history_for_player(self, history):
        """Mark each shared-history stroke from this agent's own viewpoint."""
        normalized = []
        for stroke in history or []:
            stroke_data = dict(stroke)
            player = stroke_data.get('player')
            if player is not None:
                stroke_data['is_target_turn'] = str(player) == str(self.player_name)
            else:
                stroke_data['is_target_turn'] = bool(stroke_data.get('is_target_turn', True))
            stroke_data['turn_role'] = 'target' if stroke_data['is_target_turn'] else 'opponent'
            normalized.append(stroke_data)
        return normalized

    def _case_references(self, cases, stage, store, conditioned_shot_type=None):
        references = []
        for rank, case in enumerate(cases, 1):
            reference = {
                'database_player': self.player_name,
                'stage': int(stage),
                'store': store,
                'case_id': int(case['record_id']),
                'rank': rank,
                'distance': float(case['distance']),
            }
            if conditioned_shot_type is not None:
                reference['conditioned_shot_type'] = int(conditioned_shot_type)
            references.append(reference)
        return references

    @staticmethod
    def _stage1_assessment_references(case_assessments, normal_refs, miss_refs):
        references = {'success': normal_refs, 'miss': miss_refs}
        assessment_references = []
        for annotation in case_assessments:
            reference = dict(
                references[annotation['source']][annotation['retrieval_rank'] - 1]
            )
            reference.update({
                'source': annotation['source'],
                'impact': annotation['impact'],
                'factors': list(annotation['factors']),
            })
            assessment_references.append(reference)
        return assessment_references

    @staticmethod
    def _stage2_assessment_references(case_assessments, stage2_refs):
        assessment_references = []
        for annotation in case_assessments:
            reference = dict(stage2_refs[annotation['retrieval_rank'] - 1])
            reference['impact'] = annotation['impact']
            reference['factors'] = list(annotation['factors'])
            assessment_references.append(reference)
        return assessment_references

    @staticmethod
    def _nearest_case(cases):
        if not cases:
            return None
        return min(cases, key=lambda case: (float(case['distance']), int(case['record_id'])))

    def _stage1_fallback(self, retrieval, launch):
        case_assessments = []
        if self.case_grounding:
            case_assessments = [
                {'source': 'success', 'retrieval_rank': rank,
                 'impact': 'irrelevant', 'factors': ['assessment unavailable']}
                for rank in range(1, len(retrieval['normal_cases']) + 1)
            ] + [
                {'source': 'miss', 'retrieval_rank': rank,
                 'impact': 'irrelevant', 'factors': ['assessment unavailable']}
                for rank in range(1, len(retrieval['miss_cases']) + 1)
            ]
        ranked_candidates = [
            (case, 'success', rank)
            for rank, case in enumerate(retrieval['normal_cases'], 1)
        ] + [
            (case, 'miss', rank)
            for rank, case in enumerate(retrieval['miss_cases'], 1)
        ]
        nearest_entry = min(
            ranked_candidates,
            key=lambda item: (float(item[0]['distance']), int(item[0]['record_id'])),
            default=None,
        )
        nearest = nearest_entry[0] if nearest_entry is not None else None
        if nearest is not None:
            shot_type = int(float(nearest['player_type']))
            outcome = 'missed' if shot_type == 11 else 'returned'
            return {
                'case_assessments': case_assessments,
                'outcome': outcome,
                'shot_type': shot_type,
            }, 'fallback_nearest_case'
        return {
            'case_assessments': case_assessments,
            'outcome': 'returned',
            'shot_type': 1 if launch else 2,
        }, 'fallback_default'

    def _stage2_fallback(self, conditioned_cases):
        case_assessments = (
            [
                {'retrieval_rank': rank, 'impact': 'irrelevant',
                 'factors': ['assessment unavailable']}
                for rank in range(1, len(conditioned_cases) + 1)
            ]
            if self.case_grounding else []
        )
        nearest = self._nearest_case(conditioned_cases)
        if nearest is None:
            return {
                'case_assessments': case_assessments,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            }, 'fallback_neutral'
        return {
            'case_assessments': case_assessments,
            'landing_x': float(nearest['landing_x']),
            'landing_y': float(nearest['landing_y']),
            'moving_x': float(nearest['moving_x']),
            'moving_y': float(nearest['moving_y']),
        }, 'fallback_nearest_conditioned_case'

    def _single_stage_fallback(self, retrieval, launch):
        nearest = self._nearest_case(
            retrieval['normal_cases'] + retrieval['miss_cases']
        )
        if nearest is None:
            return {
                'outcome': 'returned',
                'shot_type': 1 if launch else 2,
                'landing_x': 0.0,
                'landing_y': 0.0,
                'moving_x': 0.0,
                'moving_y': 0.0,
            }, 'fallback_default'
        shot_type = int(float(nearest['player_type']))
        if shot_type == 11:
            return {
                'outcome': 'missed',
                'shot_type': 11,
                'landing_x': None,
                'landing_y': None,
                'moving_x': None,
                'moving_y': None,
            }, 'fallback_nearest_case'
        return {
            'outcome': 'returned',
            'shot_type': shot_type,
            'landing_x': float(nearest['landing_x']),
            'landing_y': float(nearest['landing_y']),
            'moving_x': float(nearest['moving_x']),
            'moving_y': float(nearest['moving_y']),
        }, 'fallback_nearest_case'

    def _predict_single_stage(self, state, state_dict, retrieval, context, launch,
                              rally_profile_instruction=''):
        normal_cases = self.verbalizer.build_stage2_cases(
            retrieval['normal_cases'], is_serve=launch
        )
        miss_cases = self.verbalizer.build_stage2_cases(
            retrieval['miss_cases'], is_serve=launch
        )
        prompt = self.action_template.render(
            trajectory_text=context['trajectory_text'],
            normal_cases=normal_cases,
            miss_cases=miss_cases,
            rally_profile_instruction=rally_profile_instruction,
        )
        available_types = self.database.available_shot_types(launch=launch)
        started_at = time.time()
        response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
            self.llm,
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            schema=self.action_schema,
            schema_name=self.action_schema_name,
            max_trials=2,
            validator=lambda candidate: BaseAgent.validate_rag_single_stage_response(
                candidate, launch=launch, available_types=available_types
            ),
        )
        self.tracer.log_generation(
            self.system_prompt,
            prompt,
            response,
            latency_ms=int((time.time() - started_at) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
            stage='action',
        )
        parsed = BaseAgent.parse_rag_single_stage_response(
            response, launch=launch, available_types=available_types
        )
        if parsed is None:
            parsed, action_source = self._single_stage_fallback(retrieval, launch)
        else:
            action_source = 'llm'

        shot_type = int(parsed['shot_type'])
        if shot_type == 11:
            landing_x = landing_y = moving_x = moving_y = 0.0
        else:
            landing_x = float(parsed['landing_x'])
            landing_y = float(parsed['landing_y'])
            moving_x = float(parsed['moving_x'])
            moving_y = float(parsed['moving_y'])
        if self.trajectory:
            self.trajectory[-1].update({
                'player_type': shot_type,
                'landing_x': landing_x,
                'landing_y': landing_y,
                'moving_x': moving_x,
                'moving_y': moving_y,
            })

        action = (
            shot_type,
            state[-1] if len(state) > 0 else None,
            [landing_x, landing_y],
            [moving_x, moving_y],
            [0.1] * 10,
        )
        normal_refs = self._case_references(retrieval['normal_cases'], 1, 'normal')
        miss_refs = self._case_references(retrieval['miss_cases'], 1, 'miss')
        case_ids = normal_refs + miss_refs
        self._last_case_ids = case_ids
        self.last_decision_trace = {
            'schema_version': 6,
            'architecture': 'single_stage_v1',
            'case_grounding': False,
            'outcome': parsed['outcome'],
            'retrieval': {
                'normal_case_ids': normal_refs,
                'miss_case_ids': miss_refs,
            },
            'case_ids': case_ids,
            'case_assessments': [],
            'action': {
                'shot_type': shot_type,
                'landing_x': None if shot_type == 11 else landing_x,
                'landing_y': None if shot_type == 11 else landing_y,
                'moving_x': None if shot_type == 11 else moving_x,
                'moving_y': None if shot_type == 11 else moving_y,
            },
            'action_source': action_source,
        }
        self.tracer.end_trace({
            'shot_type': shot_type,
            'landing': [landing_x, landing_y],
            'movement': [moving_x, moving_y],
        }, '')
        return action, ''
    
    def predict(self, state, info, launch):
        """
        Predict action using NaiveRAG approach.
        
        Args:
            state: Environment state [opponent_shot_type, player_pos, opponent_pos, landing_pos]
            info: Additional info dict
            launch: Whether this is a serve
        
        Returns:
            action tuple and reasoning
        """
        
        query_state = {
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'ball_distance_x': state[3][0] - state[1][0],
            'ball_distance_y': state[3][1] - state[1][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1]
        }
        
        # Build state dictionary with semantic area names
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        player_location_x = query_state['player_location_x']
        player_location_y = query_state['player_location_y']
        hit_x = query_state['hit_x']
        hit_y = query_state['hit_y']
        opponent_location_x = query_state['opponent_location_x']
        opponent_location_y = query_state['opponent_location_y']

        # Compute semantic area names
        player_location_area = LANG_AREA_MAPPING.get(coord2area(player_location_x, player_location_y), 'Unknown')
        hit_position_area = LANG_AREA_MAPPING.get(coord2area(hit_x, hit_y), 'Unknown')
        opponent_location_area = LANG_AREA_MAPPING.get(coord2area(opponent_location_x, opponent_location_y), 'Unknown')

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
            'ball_distance': float(np.sqrt(
                query_state['ball_distance_x'] ** 2 + query_state['ball_distance_y'] ** 2
            )),
            'ball_distance_x': query_state['ball_distance_x'],
            'ball_distance_y': query_state['ball_distance_y'],
            'hit_x': hit_x,
            'hit_y': hit_y,
            'hit_position_area': hit_position_area,
            'opponent_type': query_state['opponent_type'],
            'opponent_location_x': opponent_location_x,
            'opponent_location_y': opponent_location_y,
            'opponent_location_area': opponent_location_area,
            'opponent_move_x': 0,  # Not directly available in state
            'opponent_move_y': 0,
        }

        self.tracer.start_trace(state_dict, info, launch)

        # Track rally trajectory
        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.trajectory = []
        self.trajectory.append(state_dict)

        # Stage 1: retrieve successful and missed evidence independently.
        t0 = time.time()
        retrieval_result = self.database.retrieve_stage1(
            query_state,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            launch=launch,
        )
        experiences = retrieval_result['normal_cases'] + retrieval_result['miss_cases']
        t1 = time.time()
        retrieval_stage = (
            'single_stage'
            if self.retrieval_architecture == 'single_stage_v1' else 'stage1'
        )
        self.tracer.log_retrieval(
            state_dict,
            experiences,
            'naive_rag',
            int((t1 - t0) * 1000),
            stage=retrieval_stage,
        )
        
        # Point retrieval compares the current decision state with point cases.
        context = self.verbalizer.build_prompt_context_nl(
            state_dict, [], launch=launch, state_history=None
        )
        available_types = self.database.available_shot_types(launch=launch)
        if self.retrieval_architecture == 'single_stage_v1':
            return self._predict_single_stage(
                state,
                state_dict,
                retrieval_result,
                context,
                launch,
                info.get('rally_profile_instruction', ''),
            )
        stage1_context = {
            **context,
            'normal_cases': self.verbalizer.build_stage1_cases(
                retrieval_result['normal_cases']
            ),
            'miss_cases': self.verbalizer.build_stage1_cases(
                retrieval_result['miss_cases']
            ),
            'rally_profile_instruction': info.get('rally_profile_instruction', ''),
            'case_grounding': self.case_grounding,
        }
        stage1_prompt = self.stage1_template.render(**stage1_context)
        stage1_schema = (
            BaseAgent.constrain_case_assessment_count(
                self.stage1_schema,
                len(retrieval_result['normal_cases']) + len(retrieval_result['miss_cases']),
            )
            if self.case_grounding else self.stage1_schema
        )
        t0 = time.time()
        stage1_response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
            self.llm,
            system_prompt=self.system_prompt,
            user_prompt=stage1_prompt,
            schema=stage1_schema,
            schema_name=self.stage1_schema_name,
            max_trials=2,
            validator=lambda response: BaseAgent.validate_rag_stage1_response(
                response,
                launch=launch,
                available_types=available_types,
                case_grounding=self.case_grounding,
                normal_cases=retrieval_result['normal_cases'],
                miss_cases=retrieval_result['miss_cases'],
            ),
        )
        t1 = time.time()
        self.tracer.log_generation(
            self.system_prompt, stage1_prompt, stage1_response,
            latency_ms=int((t1 - t0) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
            stage='stage1',
        )
        stage1 = BaseAgent.parse_rag_stage1_response(
            stage1_response,
            launch=launch,
            available_types=available_types,
            case_grounding=self.case_grounding,
            normal_cases=retrieval_result['normal_cases'],
            miss_cases=retrieval_result['miss_cases'],
        )
        if stage1 is None:
            stage1, stage1_source = self._stage1_fallback(retrieval_result, launch)
        else:
            stage1_source = 'llm'

        stage1_normal_refs = self._case_references(
            retrieval_result['normal_cases'], 1, 'normal'
        )
        stage1_miss_refs = self._case_references(
            retrieval_result['miss_cases'], 1, 'miss'
        )
        stage1_assessment_refs = self._stage1_assessment_references(
            stage1['case_assessments'], stage1_normal_refs, stage1_miss_refs
        )
        shot_type = int(stage1['shot_type'])
        stage2_refs = []
        stage2_assessment_refs = []
        if shot_type == 11:
            landing_x = landing_y = moving_x = moving_y = 0.0
            stage2 = {
                'status': 'skipped_missed',
                'conditioned_shot_type': None,
                'case_ids': [],
                'case_assessments': [],
                'source': 'skipped',
            }
        else:
            t0 = time.time()
            conditioned_cases = self.database.retrieve_by_shot_type(
                query_state,
                shot_type,
                top_k=self.conditioned_top_k,
                distances=retrieval_result['distances'],
            )
            if any(int(float(case['player_type'])) != shot_type for case in conditioned_cases):
                raise ValueError('Shot-conditioned retrieval returned a different shot type')
            t1 = time.time()
            self.tracer.log_retrieval(
                state_dict,
                conditioned_cases,
                'naive_rag',
                int((t1 - t0) * 1000),
                stage='stage2',
            )
            stage2_context = self.verbalizer.build_prompt_context_nl(
                state_dict,
                conditioned_cases,
                launch=launch,
                state_history=None,
            )
            stage2_context.update({
                'conditioned_cases': stage2_context['cases'],
                'selected_shot_type': INDEX2EN_SHOT[shot_type],
                'rally_profile_instruction': info.get('rally_profile_instruction', ''),
                'case_grounding': self.case_grounding,
            })
            stage2_prompt = self.stage2_template.render(**stage2_context)
            stage2_schema = (
                BaseAgent.constrain_case_assessment_count(
                    self.stage2_schema, len(conditioned_cases)
                )
                if self.case_grounding else self.stage2_schema
            )
            t0 = time.time()
            stage2_response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
                self.llm,
                system_prompt=self.system_prompt,
                user_prompt=stage2_prompt,
                schema=stage2_schema,
                schema_name=self.stage2_schema_name,
                max_trials=2,
                validator=lambda response: BaseAgent.validate_rag_stage2_response(
                    response,
                    case_grounding=self.case_grounding,
                    conditioned_cases=conditioned_cases,
                ),
            )
            t1 = time.time()
            self.tracer.log_generation(
                self.system_prompt,
                stage2_prompt,
                stage2_response,
                latency_ms=int((t1 - t0) * 1000),
                raw_llm_output=raw_llm_output,
                usage=usage,
                stage='stage2',
            )
            placement = BaseAgent.parse_rag_stage2_response(
                stage2_response,
                case_grounding=self.case_grounding,
                conditioned_cases=conditioned_cases,
            )
            if placement is None:
                placement, stage2_source = self._stage2_fallback(conditioned_cases)
            else:
                stage2_source = 'llm'
            landing_x = float(placement['landing_x'])
            landing_y = float(placement['landing_y'])
            moving_x = float(placement['moving_x'])
            moving_y = float(placement['moving_y'])
            stage2_refs = self._case_references(
                conditioned_cases,
                2,
                'shot_conditioned',
                conditioned_shot_type=shot_type,
            )
            stage2_assessment_refs = self._stage2_assessment_references(
                placement['case_assessments'], stage2_refs
            )
            stage2 = {
                'status': 'completed',
                'conditioned_shot_type': shot_type,
                'case_ids': stage2_refs,
                'case_assessments': stage2_assessment_refs,
                'source': stage2_source,
            }
        reasoning = ''
        
        # Update the last trajectory state with the actual player_type and landing/moving
        # This ensures the next iteration will show the correct shot info in history
        if self.trajectory:
            self.trajectory[-1]['player_type'] = int(shot_type)
            self.trajectory[-1]['landing_x'] = float(landing_x)
            self.trajectory[-1]['landing_y'] = float(landing_y)
            self.trajectory[-1]['moving_x'] = float(moving_x)
            self.trajectory[-1]['moving_y'] = float(moving_y)
        
        # Build action (normalized coordinates)
        hit = state[-1] if len(state) > 0 else None
        action = (
            shot_type,
            hit,
            [landing_x, landing_y],
            [moving_x, moving_y],
            [0.1] * 10  # Uniform distribution for compatibility
        )
        
        self.tracer.end_trace({
            "shot_type": int(shot_type),
            "landing": [landing_x, landing_y],
            "movement": [moving_x, moving_y],
        }, reasoning)
        all_case_ids = stage1_normal_refs + stage1_miss_refs + stage2_refs
        all_case_assessments = stage1_assessment_refs + stage2_assessment_refs
        self._last_case_ids = all_case_ids
        self.last_decision_trace = {
            'schema_version': 6,
            'architecture': self.retrieval_architecture,
            'case_grounding': self.case_grounding,
            'stage1': {
                'normal_case_ids': stage1_normal_refs,
                'miss_case_ids': stage1_miss_refs,
                'case_assessments': stage1_assessment_refs,
                'outcome': stage1['outcome'],
                'shot_type': None if shot_type == 11 else shot_type,
                'source': stage1_source,
            },
            'stage2': stage2,
            'case_ids': all_case_ids,
            'case_assessments': all_case_assessments,
            'action': {
                'shot_type': int(shot_type),
                'landing_x': None if shot_type == 11 else float(landing_x),
                'landing_y': None if shot_type == 11 else float(landing_y),
                'moving_x': None if shot_type == 11 else float(moving_x),
                'moving_y': None if shot_type == 11 else float(moving_y),
            },
            'action_source': (
                stage1_source if shot_type == 11
                else f'{stage1_source}+{stage2["source"]}'
            ),
        }
        
        return action, reasoning
    
    def reset_history(self):
        """Reset rally trajectory tracking."""
        self._current_rally = None
        self.trajectory = []
        self._last_case_ids = []
        self.last_decision_trace = None
    
    def verbalize_current_state(self, state, info, launch=False):
        """
        Get verbalized state string and retrieved cases string for a given state.
        
        This is useful for debugging and analysis to see how the verbalizer
        converts the current state and retrieved experiences to natural language.
        
        Args:
            state: Environment state [opponent_shot_type, player_pos, opponent_pos, landing_pos]
            info: Additional info dict
            launch: Whether this is a serve
        
        Returns:
            Tuple of (state_str, cases_str, experiences)
            - state_str: Verbalized state as string
            - cases_str: Verbalized retrieved experiences as string
            - experiences: List of raw experience dictionaries
        """
        
        # Build query state for retrieval
        query_state = {
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'ball_distance_x': state[3][0] - state[1][0],
            'ball_distance_y': state[3][1] - state[1][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1]
        }
        
        # Retrieve similar experiences
        experiences = self.database.retrieve(
            query_state, top_k=self.conditioned_top_k
        )

        # Build state dictionary with semantic area names
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        player_location_x = query_state['player_location_x']
        player_location_y = query_state['player_location_y']
        hit_x = query_state['hit_x']
        hit_y = query_state['hit_y']
        opponent_location_x = query_state['opponent_location_x']
        opponent_location_y = query_state['opponent_location_y']

        # Compute semantic area names
        player_location_area = LANG_AREA_MAPPING.get(coord2area(player_location_x, player_location_y), 'Unknown')
        hit_position_area = LANG_AREA_MAPPING.get(coord2area(hit_x, hit_y), 'Unknown')
        opponent_location_area = LANG_AREA_MAPPING.get(coord2area(opponent_location_x, opponent_location_y), 'Unknown')

        state_dict = {
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
            'opponent_type': query_state['opponent_type'],
            'opponent_location_x': opponent_location_x,
            'opponent_location_y': opponent_location_y,
            'opponent_location_area': opponent_location_area,
            'opponent_move_x': 0,
            'opponent_move_y': 0,
        }
        
        # Use verbalizer if available
        if self.verbalizer is not None:
            state_str = self.verbalizer.verbalize_state(state_dict)
            cases_str = self.verbalizer.verbalize_experiences(experiences)
        else:
            # Fallback to simple string representation
            state_str = str(state_dict)
            cases_str = str(experiences)
        
        return state_str, cases_str, experiences
