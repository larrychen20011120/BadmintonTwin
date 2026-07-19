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


class TimeRAGVerbalizer:
    """Verbalizer for TimeRAG with natural turn-based window format."""

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
        st = int(shot_type)
        if st == 0:
            return "Unknown"
        return INDEX2EN_SHOT.get(st, f"Unknown({st})")

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

    def _format_stroke_lines(self, sd, is_current=False):
        """Format one real ball-round stroke from the target player's viewpoint."""
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

    def build_window_verbalization(self, win):
        """Build the rank-independent text stored with one database window."""
        ws_list = win.get('window_states', [])
        act = win.get('action', {})

        seen_ball_rounds = set()
        valid_ws = []
        for ws in ws_list:
            if ws.get('padding'):
                continue
            br = int(ws.get('ball_round', 0))
            if br > 0 and br not in seen_ball_rounds:
                seen_ball_rounds.add(br)
                valid_ws.append(ws)

        if not valid_ws:
            return {
                'num_shots': 0,
                'description': "No valid window states."
            }

        first_ws = valid_ws[0]
        rally = int(first_ws.get('rally', 0))
        first_ball_round = int(first_ws.get('ball_round', 1))
        ps = int(first_ws.get('player_score', 0))
        os = int(first_ws.get('opponent_score', 0))
        score_diff = ps - os
        if score_diff > 0:
            score_text = f"I lead by {score_diff} points."
        elif score_diff < 0:
            score_text = f"I trail by {abs(score_diff)} points."
        else:
            score_text = "The score is tied."

        shots_played = first_ball_round - 1
        if shots_played == 0:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {first_ball_round}. This is the first shot of the rally."
        else:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {first_ball_round}. {shots_played} shots have been played in this rally."

        lines = [header, ""]
        for ws in valid_ws:
            lines.extend(self._format_stroke_lines(ws))

        shot_type = int(act.get('player_type', 0))
        if shot_type > 0:
            prompt_action = BaseAgent.build_prompt_action(
                shot_type,
                act.get('landing_x'),
                act.get('landing_y'),
                act.get('moving_x'),
                act.get('moving_y'),
            )
            lines.append(BaseAgent.format_prompt_action(prompt_action))

        return {
            'num_shots': len(valid_ws),
            'description': "\n".join(lines)
        }

    def format_window_turn_based(self, win, idx):
        """Attach the runtime retrieval rank to a precomputed window."""
        verbalization = win.get('case_verbalization')
        if verbalization is None:
            verbalization = self.build_window_verbalization(win)
        return {
            'index': idx,
            'num_shots': verbalization['num_shots'],
            'description': verbalization['description'],
        }

    def build_cases_list(self, windows):
        """Build a list of formatted windows in turn-based format."""
        return [self.format_window_turn_based(win, i) for i, win in enumerate(windows, 1)]

    def build_trajectory_text(self, state_history, current_state):
        """Build natural turn-based trajectory text for current rally."""
        if not state_history:
            return "\n".join(self._build_context_header(current_state))

        lines = list(self._build_context_header(current_state))
        lines.append("")

        seen_ball_rounds = set()
        valid_history = []
        for sd in state_history:
            br = int(sd.get('ball_round', 0))
            if br > 0 and br not in seen_ball_rounds:
                seen_ball_rounds.add(br)
                valid_history.append(sd)

        current_br = int(current_state.get('ball_round', 0))
        for sd in valid_history:
            is_current = int(sd.get('ball_round', 0)) == current_br and int(sd.get('player_type', 0)) == 0
            lines.extend(self._format_stroke_lines(sd, is_current=is_current))

        return "\n".join(lines)

    def build_prompt_context_nl(self, state_dict, windows, launch=False, window_size=3, state_history=None):
        """Build full prompt context dict for Jinja2."""
        formatted_cases = self.build_cases_list(windows)

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
            'cases': formatted_cases,
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

    def build_dual_prompt_context_nl(self, state_dict, retrieval_result, launch=False, window_size=3, state_history=None):
        normal_windows = retrieval_result.get('normal_cases', []) if isinstance(retrieval_result, dict) else []
        miss_windows = retrieval_result.get('miss_cases', []) if isinstance(retrieval_result, dict) else []
        context = self.build_prompt_context_nl(
            state_dict,
            normal_windows,
            launch=launch,
            window_size=window_size,
            state_history=state_history,
        )
        context.update({
            'normal_cases': context['cases'],
            'miss_cases': self.build_cases_list(miss_windows),
            'normal_top_k': len(normal_windows),
            'miss_top_k': len(miss_windows),
        })
        return context


def create_timerag_verbalizer():
    return TimeRAGVerbalizer()


class TimeRAGDatabase:
    """
    Database for TimeRAG Agent.
    Stores training data as time windows and retrieves similar sequences using DTW distance.
    
    Each window contains consecutive ball-round states from both players with
    action_t as the target player's decision to predict.
    """
    schema_version = 7
    verbalization_schema_version = 2
    missed_coordinate_keys = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
    
    def __init__(self, df, player_name, player_id, window_size=3):
        """
        Initialize database with windowed data for a specific player.
        
        Args:
            df: DataFrame with all data
            player_name: Name of the player
            player_id: ID of the player
            window_size: Number of consecutive states in each window (default: 3)
        """
        self.player_name = player_name
        self.player_id = player_id
        self.window_size = window_size
        
        # Keep the full rally data so each window can include both players.
        self.data = df.copy()
        miss_mask = self.data['player_type'].astype(int) == 11
        self.data.loc[miss_mask, list(self.missed_coordinate_keys)] = np.nan
        
        # Define continuous features for each state (for Euclidean distance)
        self.continuous_keys = [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y'
        ]
        
        # Define discrete features for each state (for Hamming distance)
        self.discrete_keys = ['opponent_type']
        
        # Define action keys (target for prediction)
        self.action_keys = ['landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y']
        
        # Define metadata keys
        self.metadata_keys = [
            'rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
            'ball_distance', 'ball_distance_x', 'ball_distance_y',
            'opponent_move_x', 'opponent_move_y'
        ]
        
        # Build windows
        self.windows = []
        self.windows_continuous = []  # [n_windows, window_size, n_continuous]
        self.windows_discrete = []    # [n_windows, window_size, n_discrete]
        self.windows_actions = []     # [n_windows, n_action_features]
        self.windows_metadata = []    # List of dicts
        
        self._build_windows()
        self._build_window_verbalizations()
        
        print(f"TimeRAG Database initialized for {player_name} with {len(self.windows)} windows (window_size={window_size})")

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _state_value(self, state, key, default=0.0):
        if state is None:
            return default
        if hasattr(state, 'get'):
            return state.get(key, default)
        return default

    def _is_target_turn(self, state):
        if state is None:
            return 0.0

        player = self._state_value(state, 'player', None)
        if player is not None and not pd.isna(player):
            return 1.0 if str(player) == str(self.player_name) else 0.0

        return 1.0 if bool(self._state_value(state, 'is_target_turn', True)) else 0.0

    def build_feature_window(self, window_states):
        """Convert a padded state window into retrieval feature arrays."""
        window_continuous = np.zeros((self.window_size, len(self.continuous_keys)), dtype=np.float32)
        window_discrete = np.zeros((self.window_size, len(self.discrete_keys)), dtype=np.float32)

        for w, state in enumerate(window_states[-self.window_size:]):
            if state is None:
                continue

            for j, key in enumerate(self.continuous_keys):
                window_continuous[w, j] = self._safe_float(self._state_value(state, key, 0.0))

            for j, key in enumerate(self.discrete_keys):
                window_discrete[w, j] = self._safe_float(self._state_value(state, key, 0.0))

        return window_continuous, window_discrete
    
    def _build_windows(self):
        """Build total-ball-round windows from full rally data."""
        group_cols = ['match_id']
        if 'set' in self.data.columns:
            group_cols.append('set')
        group_cols.append('rally')
        grouped = self.data.groupby(group_cols)

        for group_key, rally_df in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            match_id = group_key[0]
            rally_num = group_key[-1]
            set_num = group_key[1] if 'set' in self.data.columns and len(group_key) > 2 else None
            rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
            
            # Skip rallies that are too short
            if len(rally_df) < 1:
                continue
            
            for i in range(len(rally_df)):
                current_row = rally_df.iloc[i]
                if str(current_row.get('player', '')) != str(self.player_name):
                    continue

                # Build window: [state_t-(window_size-1), ..., state_t-1, state_t]
                window_states = []
                
                for w in range(self.window_size):
                    # Calculate which shot in the rally we need (going backwards)
                    shot_idx = i - (self.window_size - 1 - w)
                    
                    if shot_idx < 0:
                        # Padding: use zeros for missing states
                        window_states.append(None)  # Marker for padding
                    else:
                        window_states.append(rally_df.iloc[shot_idx])
                
                is_miss = int(float(current_row.get('player_type', 0))) == 11
                action = {}
                for key in self.action_keys:
                    if key not in current_row:
                        continue
                    if is_miss and key in self.missed_coordinate_keys:
                        action[key] = None
                    else:
                        action[key] = float(current_row[key])
                
                # Get metadata
                metadata = {
                    'match_id': match_id,
                    'rally': rally_num,
                    'ball_round': int(current_row['ball_round']),
                    'target_player': self.player_name,
                    'player_score': float(current_row['player_score']),
                    'opponent_score': float(current_row['opponent_score']),
                    'score_status': float(current_row['player_score']) - float(current_row['opponent_score'])
                }
                if set_num is not None:
                    metadata['set'] = int(set_num)

                window_continuous, window_discrete = self.build_feature_window(window_states)
                
                self.windows.append(window_states)
                self.windows_continuous.append(window_continuous)
                self.windows_discrete.append(window_discrete)
                self.windows_actions.append(action)
                self.windows_metadata.append(metadata)
        
        # Convert to numpy arrays for faster computation
        if len(self.windows_continuous) > 0:
            self.windows_continuous = np.array(self.windows_continuous)  # [n_windows, window_size, n_continuous]
            self.windows_discrete = np.array(self.windows_discrete)      # [n_windows, window_size, n_discrete]

        self._build_window_stores()

    def _build_window_stores(self):
        self.window_stores = {
            'classic': list(range(len(self.windows_actions))),
            'normal': [],
            'miss': [],
        }
        for idx, action in enumerate(self.windows_actions):
            shot_type = int(float(action.get('player_type', 0)))
            if shot_type == 11:
                self.window_stores['miss'].append(idx)
            else:
                self.window_stores['normal'].append(idx)

        total_count = len(self.window_stores['classic'])
        miss_count = len(self.window_stores['miss'])
        self.stats = {
            'total_count': int(total_count),
            'normal_count': int(len(self.window_stores['normal'])),
            'miss_count': int(miss_count),
            'miss_rate': float(miss_count / total_count) if total_count else 0.0,
        }
    
    def _compute_point_distance(self, query_continuous, query_discrete, 
                                db_continuous, db_discrete):
        """
        Compute point-to-point distance for a single time step.
        
        D = Euclidean(continuous) + Hamming(discrete)
        
        Args:
            query_continuous: [n_continuous]
            query_discrete: [n_discrete]
            db_continuous: [n_continuous]
            db_discrete: [n_continuous]
        
        Returns:
            distance: scalar
        """
        # Euclidean distance for continuous features
        continuous_dist = np.sqrt(np.sum((query_continuous - db_continuous) ** 2))
        
        # Hamming distance for discrete features (normalized)
        discrete_dist = np.mean(query_discrete != db_discrete)
        
        return continuous_dist + discrete_dist
    
    def _compute_window_dtw_distance(self, query_continuous, query_discrete):
        """
        Compute DTW distance between query window and all database windows.
        
        Args:
            query_continuous: [window_size, n_continuous]
            query_discrete: [window_size, n_discrete]
        
        Returns:
            distances: [n_windows] array of DTW distances
        """
        n_windows = len(self.windows)
        distances = np.zeros(n_windows)
        
        for w in range(n_windows):
            # Compute DTW distance for this window
            db_continuous = self.windows_continuous[w]  # [window_size, n_continuous]
            db_discrete = self.windows_discrete[w]      # [window_size, n_discrete]
            
            # Use simple DTW with fixed window size (no warping needed for same-length sequences)
            # But we compute accumulated distance
            dtw_dist = 0.0
            for t in range(self.window_size):
                point_dist = self._compute_point_distance(
                    query_continuous[t], query_discrete[t],
                    db_continuous[t], db_discrete[t]
                )
                dtw_dist += point_dist
            
            distances[w] = dtw_dist
        
        return distances

    def _build_static_window_data(self, window_id):
        window_states = self.windows[window_id]
        action = dict(self.windows_actions[window_id])
        window_data = {
            'window_id': int(window_id),
            'window_states': [],
            'action': action,
            'metadata': dict(self.windows_metadata[window_id]),
        }

        for state in window_states:
            if state is None:
                window_data['window_states'].append({'padding': True})
                continue

            state_data = {}
            for key in self.continuous_keys + self.metadata_keys + self.action_keys:
                if key in state:
                    state_data[key] = float(state[key])

            state_data['player'] = str(state.get('player', '')) if hasattr(state, 'get') else ''
            state_data['is_target_turn'] = bool(self._is_target_turn(state))
            state_data['turn_role'] = 'target' if state_data['is_target_turn'] else 'opponent'

            if 'opponent_type' in state:
                state_data['opponent_type'] = float(state['opponent_type'])
            if 'player_score' in state and 'opponent_score' in state:
                state_data['score_status'] = int(float(state['player_score']) - float(state['opponent_score']))
            if 'player_location_x' in state and 'player_location_y' in state:
                area_id = coord2area(float(state['player_location_x']), float(state['player_location_y']))
                state_data['player_location_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
            if 'hit_x' in state and 'hit_y' in state:
                area_id = coord2area(float(state['hit_x']), float(state['hit_y']))
                state_data['hit_position_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
            if 'opponent_location_x' in state and 'opponent_location_y' in state:
                area_id = coord2area(float(state['opponent_location_x']), float(state['opponent_location_y']))
                state_data['opponent_location_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
            if 'landing_x' in state and 'landing_y' in state:
                area_id = coord2area(float(state['landing_x']), float(state['landing_y']))
                state_data['landing_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
            if 'moving_x' in state and 'moving_y' in state:
                area_id = coord2area(float(state['moving_x']), float(state['moving_y']))
                state_data['movement_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
            if 'opponent_type' in state:
                shot_type = int(state['opponent_type'])
                state_data['opponent_type_name'] = INDEX2EN_SHOT.get(shot_type, f"Unknown({shot_type})")
            if 'player_type' in state:
                shot_type = int(state['player_type'])
                state_data['player_type_name'] = INDEX2EN_SHOT.get(shot_type, f"Unknown({shot_type})")

            window_data['window_states'].append(state_data)

        is_miss = int(float(action.get('player_type', 0))) == 11
        if not is_miss and 'landing_x' in action and 'landing_y' in action:
            area_id = coord2area(float(action['landing_x']), float(action['landing_y']))
            action['landing_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
        else:
            action['landing_area'] = None
        if not is_miss and 'moving_x' in action and 'moving_y' in action:
            area_id = coord2area(float(action['moving_x']), float(action['moving_y']))
            action['moving_area'] = LANG_AREA_MAPPING.get(area_id, 'Unknown')
        else:
            action['moving_area'] = None
        if 'player_type' in action:
            shot_type = int(action['player_type'])
            action['player_type_name'] = INDEX2EN_SHOT.get(shot_type, f"Unknown({shot_type})")
        return window_data

    def _build_window_verbalizations(self):
        verbalizer = TimeRAGVerbalizer()
        self.window_verbalizations = [
            {
                'window_id': window_id,
                **verbalizer.build_window_verbalization(self._build_static_window_data(window_id)),
            }
            for window_id in range(len(self.windows))
        ]
    
    def _retrieve_from_indices(self, query_window_continuous, query_window_discrete, indices, top_k=5):
        """
        Retrieve top-k similar windows using DTW distance.

        Args:
            query_window_continuous: [window_size, n_continuous] continuous features
            query_window_discrete: [window_size, n_discrete] discrete features
            top_k: Number of windows to retrieve

        Returns:
            List of dicts containing window data and actions
        """
        # Ensure query is numpy array
        query_continuous = np.array(query_window_continuous, dtype=np.float32)
        query_discrete = np.array(query_window_discrete, dtype=np.float32)

        distances = self._compute_window_dtw_distance(query_continuous, query_discrete)
        if len(indices) == 0 or len(distances) == 0:
            return []

        # Get top-k indices (smallest distances)
        candidate_indices = np.array(indices, dtype=int)
        top_indices = candidate_indices[np.argsort(distances[candidate_indices])[:top_k]]

        # Build result
        results = []
        for idx in top_indices:
            window_data = self._build_static_window_data(int(idx))
            window_data['case_verbalization'] = dict(self.window_verbalizations[int(idx)])
            window_data['distance'] = float(distances[idx])
            window_data['similarity'] = float(np.exp(-distances[idx]))
            results.append(window_data)

        return results

    def retrieve(self, query_window_continuous, query_window_discrete, top_k=5):
        """Retrieve top-k similar windows using the classic all-window store."""
        return self._retrieve_from_indices(
            query_window_continuous,
            query_window_discrete,
            self.window_stores.get('classic', list(range(len(self.windows_actions)))),
            top_k=top_k,
        )

    def retrieve_dual(self, query_window_continuous, query_window_discrete, normal_top_k=8, miss_top_k=5):
        normal_windows = self._retrieve_from_indices(
            query_window_continuous,
            query_window_discrete,
            self.window_stores.get('normal', []),
            top_k=normal_top_k,
        )
        miss_windows = self._retrieve_from_indices(
            query_window_continuous,
            query_window_discrete,
            self.window_stores.get('miss', []),
            top_k=miss_top_k,
        )
        nearest_normal = normal_windows[0] if normal_windows else {}
        nearest_miss = miss_windows[0] if miss_windows else {}
        return {
            'normal_cases': normal_windows,
            'miss_cases': miss_windows,
            'summary': {
                **self.stats,
                'normal_top_k': int(normal_top_k),
                'miss_top_k': int(miss_top_k),
                'retrieved_normal_count': int(len(normal_windows)),
                'retrieved_miss_count': int(len(miss_windows)),
                'nearest_normal_distance': nearest_normal.get('distance'),
                'nearest_miss_distance': nearest_miss.get('distance'),
                'nearest_normal_similarity': nearest_normal.get('similarity'),
                'nearest_miss_similarity': nearest_miss.get('similarity'),
            }
        }

    def validate(self):
        if self.schema_version != type(self).schema_version:
            raise ValueError(f"Unsupported TimeRAG schema version: {self.schema_version}")
        if self.verbalization_schema_version != type(self).verbalization_schema_version:
            raise ValueError(
                f"Unsupported TimeRAG verbalization schema version: "
                f"{self.verbalization_schema_version}"
            )
        if not isinstance(self.window_size, int) or self.window_size < 1:
            raise ValueError("TimeRAG window_size must be a positive integer")

        count = len(self.windows)
        if count == 0:
            raise ValueError(f"TimeRAG database for {self.player_name} contains no windows")
        aligned_lengths = {
            'continuous': len(self.windows_continuous),
            'discrete': len(self.windows_discrete),
            'actions': len(self.windows_actions),
            'metadata': len(self.windows_metadata),
            'verbalizations': len(self.window_verbalizations),
        }
        mismatched = {name: length for name, length in aligned_lengths.items() if length != count}
        if mismatched:
            raise ValueError(f"TimeRAG window data is not aligned: expected {count}, got {mismatched}")
        expected_continuous_shape = (count, self.window_size, len(self.continuous_keys))
        expected_discrete_shape = (count, self.window_size, len(self.discrete_keys))
        if tuple(np.asarray(self.windows_continuous).shape) != expected_continuous_shape:
            raise ValueError("TimeRAG continuous feature shape does not match its schema")
        if tuple(np.asarray(self.windows_discrete).shape) != expected_discrete_shape:
            raise ValueError("TimeRAG discrete feature shape does not match its schema")
        if any(len(window) != self.window_size for window in self.windows):
            raise ValueError("TimeRAG raw window length does not match window_size")

        for window_id, verbalization in enumerate(self.window_verbalizations):
            if not isinstance(verbalization, dict):
                raise ValueError(f"TimeRAG verbalization {window_id} is not an object")
            if verbalization.get('window_id') != window_id:
                raise ValueError(f"TimeRAG verbalization {window_id} is assigned to the wrong window")
            num_shots = verbalization.get('num_shots')
            if not isinstance(num_shots, int) or not 0 <= num_shots <= self.window_size:
                raise ValueError(f"TimeRAG verbalization {window_id} has invalid num_shots")
            if not isinstance(verbalization.get('description'), str) or not verbalization['description'].strip():
                raise ValueError(f"TimeRAG verbalization {window_id} has no description")

        classic = self.window_stores.get('classic')
        normal = self.window_stores.get('normal')
        miss = self.window_stores.get('miss')
        if not all(isinstance(store, list) for store in (classic, normal, miss)):
            raise ValueError("TimeRAG window stores must be lists")
        if classic != list(range(count)):
            raise ValueError("TimeRAG classic store does not cover every window")
        if len(normal) != len(set(normal)) or len(miss) != len(set(miss)):
            raise ValueError("TimeRAG window stores contain duplicate indexes")
        if set(normal) & set(miss) or set(normal) | set(miss) != set(classic):
            raise ValueError("TimeRAG normal and miss stores do not partition the windows")
        expected_miss = [
            window_id
            for window_id, action in enumerate(self.windows_actions)
            if int(float(action.get('player_type', 0))) == 11
        ]
        expected_miss_set = set(expected_miss)
        expected_normal = [window_id for window_id in classic if window_id not in expected_miss_set]
        if normal != expected_normal or miss != expected_miss:
            raise ValueError("TimeRAG normal and miss stores do not match player_type")
        for window_id in expected_miss:
            action = self.windows_actions[window_id]
            if any(action.get(key) is not None for key in self.missed_coordinate_keys):
                raise ValueError("TimeRAG missed-shot action coordinates must all be null")
        for window in self.windows:
            for state in window:
                if state is None or int(float(state.get('player_type', 0))) != 11:
                    continue
                if any(not pd.isna(state.get(key)) for key in self.missed_coordinate_keys):
                    raise ValueError("TimeRAG missed-shot window coordinates must all be null")

        rebuilt_features = [self.build_feature_window(window) for window in self.windows]
        rebuilt_continuous = np.asarray([features[0] for features in rebuilt_features], dtype=np.float32)
        rebuilt_discrete = np.asarray([features[1] for features in rebuilt_features], dtype=np.float32)
        if not np.array_equal(np.asarray(self.windows_continuous), rebuilt_continuous):
            raise ValueError("TimeRAG continuous features are assigned to the wrong windows")
        if not np.array_equal(np.asarray(self.windows_discrete), rebuilt_discrete):
            raise ValueError("TimeRAG discrete features are assigned to the wrong windows")
        expected_stats = {
            'total_count': count,
            'normal_count': len(expected_normal),
            'miss_count': len(expected_miss),
            'miss_rate': float(len(expected_miss) / count),
        }
        if self.stats != expected_stats:
            raise ValueError("TimeRAG stats do not match the window stores")
        return self
    
    def save(self, filepath):
        """Save database to file."""
        self.validate()
        payload = {
                'schema_version': self.schema_version,
                'verbalization_schema_version': self.verbalization_schema_version,
                'player_name': self.player_name,
                'player_id': self.player_id,
                'window_size': self.window_size,
                'windows': self.windows,
                'windows_continuous': self.windows_continuous,
                'windows_discrete': self.windows_discrete,
                'windows_actions': self.windows_actions,
                'windows_metadata': self.windows_metadata,
                'window_verbalizations': self.window_verbalizations,
                'window_stores': self.window_stores,
                'stats': self.stats,
                'continuous_keys': self.continuous_keys,
                'discrete_keys': self.discrete_keys,
                'action_keys': self.action_keys,
                'metadata_keys': self.metadata_keys
        }
        temporary_path = f'{filepath}.tmp-{os.getpid()}'
        try:
            with open(temporary_path, 'wb') as f:
                pickle.dump(payload, f)
            os.replace(temporary_path, filepath)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        print(f"TimeRAG Database saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load database from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        schema_version = data.get('schema_version', 1)
        verbalization_schema_version = data.get('verbalization_schema_version')
        if schema_version != cls.schema_version or verbalization_schema_version != cls.verbalization_schema_version:
            raise ValueError(
                "This TimeRAG database does not contain compatible offline verbalizations. "
                "Rebuild it with Scripts/prepare_all_data.py --timerag."
            )
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.schema_version = schema_version
        instance.verbalization_schema_version = verbalization_schema_version
        instance.player_name = data['player_name']
        instance.player_id = data['player_id']
        instance.window_size = data['window_size']
        instance.windows = data['windows']
        instance.windows_continuous = data['windows_continuous']
        instance.windows_discrete = data['windows_discrete']
        instance.windows_actions = data['windows_actions']
        instance.windows_metadata = data['windows_metadata']
        instance.window_verbalizations = data['window_verbalizations']
        instance.window_stores = data['window_stores']
        instance.stats = data['stats']
        instance.continuous_keys = data['continuous_keys']
        instance.discrete_keys = data['discrete_keys']
        instance.action_keys = data['action_keys']
        instance.metadata_keys = data['metadata_keys']
        instance.validate()
        
        print(f"TimeRAG Database loaded from {filepath}")
        return instance


class TimeRAGAgent(BaseAgent):
    """
    TimeRAG Agent that retrieves similar time windows and uses LLM to generate actions.
    Maintains a sliding window of recent states for retrieval.
    """
    
    def __init__(self, player_name, database, model_name=None,
                 api_key=None, host=None, port=None, base_url=None,
                 host_url=None, top_k=5, normal_top_k=None, miss_top_k=None,
                 retrieval_mode="classic", enable_tracing=True, prompt_variant="norm"):
        """
        Initialize TimeRAG Agent.
        
        Args:
            player_name: Name of the player
            database: TimeRAGDatabase instance
            model_name: Name of the LLM model
            api_key: API key for LLM
            port: Port for local LLM server
            top_k: Number of similar windows to retrieve
            enable_tracing: Whether to enable Langfuse tracing
        """
        super().__init__(player_name)
        
        self.database = database
        self.window_size = database.window_size
        self.top_k = top_k
        self.normal_top_k = normal_top_k if normal_top_k is not None else top_k
        self.miss_top_k = miss_top_k if miss_top_k is not None else max(1, min(5, top_k))
        self.retrieval_mode = retrieval_mode
        self.prompt_variant = prompt_variant
        self.tracer = BadmintonTracer("timerag", player_name, enable_tracing)
        
        # Trajectory tracking for current rally (full state_dicts)
        self.trajectory = []
        self._current_rally = None
        
        # Initialize verbalizer
        self.verbalizer = create_timerag_verbalizer()
        
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
        prompt_dir = os.path.join(prompt_root, "timerag")
        template_names = {
            ("classic", "norm"): "norm.jinja",
            ("classic", "shifted"): "shift.jinja",
            ("dual_db", "norm"): "norm.jinja",
            ("dual_db", "shifted"): "shift.jinja",
        }
        template_key = (retrieval_mode, prompt_variant)
        if template_key not in template_names:
            raise ValueError(
                f"Unsupported retrieval_mode/prompt_variant: {retrieval_mode}/{prompt_variant}"
            )
        user_template_name = template_names[template_key]

        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            self.system_prompt = f.read()
        
        # Load user template
        with open(os.path.join(prompt_dir, user_template_name), "r") as f:
            self.user_template = Template(f.read())
        
        # State history buffer for building windows
        self.state_history = []

    @staticmethod
    def _parse_llm_action(response):
        return BaseAgent.parse_prompt_action(response)
    
    def _normalize_history_for_player(self, history):
        """Mark each historical stroke as target/opponent for this agent."""
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

    def _build_state_dict_from_env(self, state, info):
        """Build a stroke dict from the current rollout state."""
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        player_location_x = state[1][0]
        player_location_y = state[1][1]
        hit_x = state[3][0] if len(state) > 3 else 0
        hit_y = state[3][1] if len(state) > 3 else 0
        opponent_location_x = state[2][0]
        opponent_location_y = state[2][1]
        current_player = info.get('current_player_name', self.player_name)

        player_location_area = LANG_AREA_MAPPING.get(coord2area(player_location_x, player_location_y), 'Unknown')
        hit_position_area = LANG_AREA_MAPPING.get(coord2area(hit_x, hit_y), 'Unknown')
        opponent_location_area = LANG_AREA_MAPPING.get(coord2area(opponent_location_x, opponent_location_y), 'Unknown')

        is_target_turn = str(current_player) == str(self.player_name)
        return {
            'player': current_player,
            'is_target_turn': is_target_turn,
            'turn_role': 'target' if is_target_turn else 'opponent',
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': player_location_x,
            'player_location_y': player_location_y,
            'player_location_area': player_location_area,
            'ball_distance': float(np.sqrt((hit_x - player_location_x) ** 2 + (hit_y - player_location_y) ** 2)),
            'ball_distance_x': hit_x - player_location_x,
            'ball_distance_y': hit_y - player_location_y,
            'hit_x': hit_x,
            'hit_y': hit_y,
            'hit_position_area': hit_position_area,
            'opponent_type': int(state[0]),
            'opponent_type_name': INDEX2EN_SHOT.get(int(state[0]), f"Unknown({int(state[0])})"),
            'opponent_location_x': opponent_location_x,
            'opponent_location_y': opponent_location_y,
            'opponent_location_area': opponent_location_area,
            'opponent_move_x': 0,
            'opponent_move_y': 0,
        }

    def _build_query_window(self, state_dict, rally_history=None):
        """
        Build query window from consecutive total ball rounds.

        If rollout provides full rally history, use both players' previous strokes.
        Otherwise fall back to this agent's local history for direct callers.
        """
        if rally_history is not None:
            query_states = self._normalize_history_for_player(rally_history) + [state_dict]
        else:
            self.state_history.append(dict(state_dict))
            if len(self.state_history) > self.window_size:
                self.state_history.pop(0)
            query_states = self.state_history

        query_states = query_states[-self.window_size:]
        if len(query_states) < self.window_size:
            query_states = [None] * (self.window_size - len(query_states)) + query_states

        return self.database.build_feature_window(query_states)

    def reset_history(self):
        """Reset state history and trajectory tracking."""
        self.state_history = []
        self.trajectory = []
        self._current_rally = None

    def predict(self, state, info, launch):
        """
        Predict action using TimeRAG approach.

        Args:
            state: Environment state
            info: Additional info dict
            launch: Whether this is a serve

        Returns:
            action tuple and reasoning
        """
        state_dict = self._build_state_dict_from_env(state, info)

        # Build query window
        rally_history = info.get('rally_history')
        query_continuous, query_discrete = self._build_query_window(state_dict, rally_history)

        # Retrieve similar windows
        t0 = time.time()
        if self.retrieval_mode == 'dual_db':
            retrieval_result = self.database.retrieve_dual(
                query_continuous,
                query_discrete,
                normal_top_k=self.normal_top_k,
                miss_top_k=self.miss_top_k,
            )
            retrieved_windows = retrieval_result['normal_cases'] + retrieval_result['miss_cases']
        else:
            retrieval_result = self.database.retrieve(query_continuous, query_discrete, top_k=self.top_k)
            retrieved_windows = retrieval_result
        t1 = time.time()

        self.tracer.start_trace(state_dict, info, launch)
        self.tracer.log_retrieval(state_dict, retrieved_windows, "timerag", int((t1 - t0) * 1000))
        
        # Track rally trajectory
        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.trajectory = []
        self.trajectory.append(state_dict)
        
        # Build context for LLM using verbalizer
        if rally_history is not None:
            state_history = self._normalize_history_for_player(rally_history)
        else:
            state_history = self.trajectory[:-1] if len(self.trajectory) > 1 else None
        if self.retrieval_mode == 'dual_db':
            context = self.verbalizer.build_dual_prompt_context_nl(state_dict, retrieval_result, launch=launch, window_size=self.window_size, state_history=state_history)
        else:
            context = self.verbalizer.build_prompt_context_nl(state_dict, retrieved_windows, launch=launch, window_size=self.window_size, state_history=state_history)
        context['style_shift_instruction'] = info.get('style_shift_instruction', '')
        
        # Render user prompt
        user_prompt = self.user_template.render(**context)
        
        # Get LLM response
        t0 = time.time()
        response, raw_llm_output, usage = self.llm.get_verified_response(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_trials=5,
            validator=BaseAgent.validate_prompt_action_response,
        )
        t1 = time.time()
        self.tracer.log_generation(
            self.system_prompt, user_prompt, response,
            latency_ms=int((t1 - t0) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
        )
        
        if response is None:
            print("Warning: LLM response failed, using default action")
            self.tracer.end_trace({"shot_type": "default_fallback"}, "LLM response failed")
            return self._default_action(launch)

        # Parse response
        parsed_action, reasoning = self._parse_llm_action(response)
        if parsed_action is None:
            print("Warning: LLM response did not contain a valid action, using default action")
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
        
        # Apply serve mask if needed
        if launch and shot_type not in [1, 10]:
            shot_type = 1 if np.random.rand() < 0.5 else 10
        
        # Update the last trajectory state with the actual player_type and landing/moving
        # This ensures the next iteration will show the correct shot info in history
        if self.trajectory:
            self.trajectory[-1]['player_type'] = int(shot_type)
            self.trajectory[-1]['landing_x'] = float(landing_x)
            self.trajectory[-1]['landing_y'] = float(landing_y)
            self.trajectory[-1]['moving_x'] = float(moving_x)
            self.trajectory[-1]['moving_y'] = float(moving_y)
        
        # Build action
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
    
    def _default_action(self, launch):
        """Default action when LLM fails."""
        if launch:
            shot_type = 1
        else:
            shot_type = np.random.randint(1, 12)
        
        return (
            shot_type,
            None,
            (0.0, 0.0),
            (0.0, 0.0),
            [0.1] * 10
        ), "Default action due to LLM failure"
