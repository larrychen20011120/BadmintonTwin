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
        """Build stage-specific rank-independent text for one database window."""
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
                'stage1_description': "No valid window states.",
                'stage2_description': "No valid window states.",
            }

        decision_ws = valid_ws[-1]
        rally = int(decision_ws.get('rally', 0))
        decision_ball_round = int(decision_ws.get('ball_round', 1))
        ps = int(decision_ws.get('player_score', 0))
        os = int(decision_ws.get('opponent_score', 0))
        score_diff = ps - os
        if score_diff > 0:
            score_text = f"I lead by {score_diff} points."
        elif score_diff < 0:
            score_text = f"I trail by {abs(score_diff)} points."
        else:
            score_text = "The score is tied."

        shots_played = decision_ball_round - 1
        if shots_played == 0:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {decision_ball_round}. This is the first shot of the rally."
        else:
            header = f"Current score: {ps}-{os}. {score_text}\nMatch Context: Rally {rally}, currently at Shot {decision_ball_round}. {shots_played} shots have been played in this rally."

        stage2_lines = [header, ""]
        for ws in valid_ws:
            stage2_lines.extend(self._format_stroke_lines(ws))

        shot_type = int(act.get('player_type', 0))
        if shot_type > 0:
            prompt_action = BaseAgent.build_prompt_action(
                shot_type,
                act.get('landing_x'),
                act.get('landing_y'),
                act.get('moving_x'),
                act.get('moving_y'),
            )
            stage2_lines.append(BaseAgent.format_prompt_action(prompt_action))

        stage1_lines = [header, ""]
        for state in valid_ws[:-1]:
            stage1_lines.extend(self._format_stroke_lines(state))
        pending = dict(decision_ws)
        pending['player_type'] = 0
        stage1_lines.extend(self._format_stroke_lines(pending, is_current=True))
        if shot_type == 11:
            stage1_lines.append("Outcome: Missed; no legal shot was completed.")
        else:
            stage1_lines.append(f"Outcome: Returned with {self._shot_name(shot_type)}.")

        return {
            'num_shots': len(valid_ws),
            'stage1_description': "\n".join(stage1_lines),
            'stage2_description': "\n".join(stage2_lines),
        }

    def format_stage2_case(self, win, idx):
        """Attach runtime retrieval data to a cached Stage 2 window."""
        verbalization = win.get('case_verbalization')
        if verbalization is None:
            verbalization = self.build_window_verbalization(win)
        return {
            'index': idx,
            'num_shots': verbalization['num_shots'],
            'stage2_description': verbalization['stage2_description'],
        }

    def build_stage2_cases(self, windows):
        """Build complete-action windows for Stage 2 placement generation."""
        return [self.format_stage2_case(win, i) for i, win in enumerate(windows, 1)]

    def format_stage1_case(self, window, idx):
        """Render a temporal case while hiding the final placement decision."""
        verbalization = window.get('case_verbalization')
        if verbalization is None:
            verbalization = self.build_window_verbalization(window)
        return {
            'index': idx,
            'case_id': int(window['window_id']),
            'stage1_description': verbalization['stage1_description'],
        }

    def build_stage1_cases(self, windows):
        return [
            self.format_stage1_case(window, idx)
            for idx, window in enumerate(windows, 1)
        ]

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
        formatted_cases = self.build_stage2_cases(windows)

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
            'miss_cases': self.build_stage2_cases(miss_windows),
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
    schema_version = 10
    verbalization_schema_version = 3
    missed_coordinate_keys = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
    
    def __init__(self, df, player_name, player_id, window_size=3, build_metadata=None):
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
        self.build_metadata = dict(build_metadata or {
            'dataset': 'rally',
            'split_sha256': 'ad-hoc',
            'profiles_sha256': None,
            'window_size': window_size,
        })
        
        # Keep the full rally data so each window can include both players.
        self.data = df.copy()
        self.data['ball_distance_x'] = (
            self.data['hit_x'].astype(float) - self.data['player_location_x'].astype(float)
        )
        self.data['ball_distance_y'] = (
            self.data['hit_y'].astype(float) - self.data['player_location_y'].astype(float)
        )
        miss_mask = self.data['player_type'].astype(int) == 11
        self.data.loc[miss_mask, list(self.missed_coordinate_keys)] = np.nan
        
        # Define continuous features for each state (for Euclidean distance)
        self.continuous_keys = [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y',
            'ball_distance_x', 'ball_distance_y',
        ]
        
        # Define discrete features for each state (for Hamming distance)
        self.discrete_keys = ['opponent_type']
        
        # Define action keys (target for prediction)
        self.action_keys = ['landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y']
        
        # Define metadata keys
        self.metadata_keys = [
            'match_id', 'set', 'rally', 'ball_round',
            'player_score', 'opponent_score', 'score_status',
            'ball_distance', 'ball_distance_x', 'ball_distance_y',
            'opponent_move_x', 'opponent_move_y'
        ]
        
        # Build windows
        self.windows = []
        self.windows_continuous = []  # [n_windows, window_size, n_continuous]
        self.windows_discrete = []    # [n_windows, window_size, n_discrete]
        self.windows_valid_masks = [] # [n_windows, window_size]
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

    def build_feature_window(self, window_states, include_mask=False):
        """Convert a padded state window into retrieval feature arrays."""
        window_continuous = np.zeros((self.window_size, len(self.continuous_keys)), dtype=np.float32)
        window_discrete = np.zeros((self.window_size, len(self.discrete_keys)), dtype=np.float32)
        valid_mask = np.zeros(self.window_size, dtype=bool)

        for w, state in enumerate(window_states[-self.window_size:]):
            if state is None or bool(state.get('padding', False)):
                continue
            valid_mask[w] = True

            for j, key in enumerate(self.continuous_keys):
                window_continuous[w, j] = self._safe_float(self._state_value(state, key, 0.0))

            for j, key in enumerate(self.discrete_keys):
                window_discrete[w, j] = self._safe_float(self._state_value(state, key, 0.0))

        if include_mask:
            return window_continuous, window_discrete, valid_mask
        return window_continuous, window_discrete
    
    def _build_windows(self):
        """Build total-ball-round windows from full rally data."""
        missing = sorted({'match_id', 'set', 'rally'} - set(self.data.columns))
        if missing:
            raise ValueError(f"TimeRAG data is missing rally key columns: {missing}")
        group_cols = ['match_id', 'set', 'rally']
        grouped = self.data.groupby(group_cols)

        for group_key, rally_df in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            match_id = group_key[0]
            rally_num = group_key[-1]
            set_num = group_key[1]
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
                metadata['set'] = int(set_num)

                window_continuous, window_discrete = self.build_feature_window(window_states)
                
                self.windows.append(window_states)
                self.windows_continuous.append(window_continuous)
                self.windows_discrete.append(window_discrete)
                self.windows_valid_masks.append(
                    np.asarray([state is not None for state in window_states], dtype=bool)
                )
                self.windows_actions.append(action)
                self.windows_metadata.append(metadata)
        
        # Convert to numpy arrays for faster computation
        if len(self.windows_continuous) > 0:
            self.windows_continuous = np.array(self.windows_continuous)  # [n_windows, window_size, n_continuous]
            self.windows_discrete = np.array(self.windows_discrete)      # [n_windows, window_size, n_discrete]
            self.windows_valid_masks = np.array(self.windows_valid_masks, dtype=bool)

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
    
    def _compute_window_dtw_distance(self, query_continuous, query_discrete, query_valid_mask=None):
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
        if query_valid_mask is None:
            query_valid_mask = np.ones(self.window_size, dtype=bool)
        else:
            query_valid_mask = np.asarray(query_valid_mask, dtype=bool)
        
        for w in range(n_windows):
            # Compute DTW distance for this window
            db_continuous = self.windows_continuous[w]  # [window_size, n_continuous]
            db_discrete = self.windows_discrete[w]      # [window_size, n_discrete]
            
            # Use simple DTW with fixed window size (no warping needed for same-length sequences)
            # But we compute accumulated distance
            dtw_dist = 0.0
            for t in range(self.window_size):
                query_valid = bool(query_valid_mask[t])
                database_valid = bool(self.windows_valid_masks[w][t])
                if not query_valid and not database_valid:
                    continue
                if query_valid != database_valid:
                    dtw_dist += 1.0
                    continue
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
    
    def _retrieve_from_indices(self, query_window_continuous, query_window_discrete,
                               indices, top_k=5, query_valid_mask=None):
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

        distances = self._compute_window_dtw_distance(
            query_continuous,
            query_discrete,
            query_valid_mask=query_valid_mask,
        )
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

    def retrieve(self, query_window_continuous, query_window_discrete, top_k=5,
                 query_valid_mask=None):
        """Retrieve top-k similar windows using the classic all-window store."""
        return self._retrieve_from_indices(
            query_window_continuous,
            query_window_discrete,
            self.window_stores.get('classic', list(range(len(self.windows_actions)))),
            top_k=top_k,
            query_valid_mask=query_valid_mask,
        )

    def retrieve_dual(self, query_window_continuous, query_window_discrete,
                      normal_top_k=8, miss_top_k=5, query_valid_mask=None):
        normal_windows = self._retrieve_from_indices(
            query_window_continuous,
            query_window_discrete,
            self.window_stores.get('normal', []),
            top_k=normal_top_k,
            query_valid_mask=query_valid_mask,
        )
        miss_windows = self._retrieve_from_indices(
            query_window_continuous,
            query_window_discrete,
            self.window_stores.get('miss', []),
            top_k=miss_top_k,
            query_valid_mask=query_valid_mask,
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

    def compute_distances(self, query_window_continuous, query_window_discrete,
                          query_valid_mask=None):
        """Compute one canonical distance vector for all player windows."""
        query_continuous = np.asarray(query_window_continuous, dtype=np.float32)
        query_discrete = np.asarray(query_window_discrete, dtype=np.float32)
        return self._compute_window_dtw_distance(
            query_continuous, query_discrete, query_valid_mask=query_valid_mask
        )

    def _retrieve_from_distances(self, distances, indices, top_k):
        candidate_indices = np.asarray(list(indices), dtype=int)
        if len(candidate_indices) == 0 or len(distances) == 0 or top_k <= 0:
            return []
        order = np.lexsort((candidate_indices, distances[candidate_indices]))
        selected = candidate_indices[order[:top_k]]
        results = []
        for window_id in selected:
            window_data = self._build_static_window_data(int(window_id))
            window_data['case_verbalization'] = dict(
                self.window_verbalizations[int(window_id)]
            )
            window_data['distance'] = float(distances[int(window_id)])
            window_data['similarity'] = float(np.exp(-distances[int(window_id)]))
            results.append(window_data)
        return results

    @staticmethod
    def _legal_return_types(launch):
        return {1, 10} if launch else set(range(2, 10))

    def available_shot_types(self, launch=False):
        legal = self._legal_return_types(launch)
        present = {
            int(float(action.get('player_type', 0)))
            for action in self.windows_actions
        }
        return tuple(sorted(legal & present))

    def retrieve_stage1(
        self,
        query_window_continuous,
        query_window_discrete,
        normal_top_k=10,
        miss_top_k=5,
        launch=False,
        query_valid_mask=None,
    ):
        distances = self.compute_distances(
            query_window_continuous,
            query_window_discrete,
            query_valid_mask=query_valid_mask,
        )
        legal = self._legal_return_types(launch)
        normal_ids = [
            window_id
            for window_id, action in enumerate(self.windows_actions)
            if int(float(action.get('player_type', 0))) in legal
        ]
        miss_ids = self.window_stores.get('miss', [])
        return {
            'normal_cases': self._retrieve_from_distances(
                distances, normal_ids, normal_top_k
            ),
            'miss_cases': self._retrieve_from_distances(
                distances, miss_ids, miss_top_k
            ),
            'distances': distances,
        }

    def retrieve_by_shot_type(
        self,
        query_window_continuous,
        query_window_discrete,
        shot_type,
        top_k=10,
        distances=None,
        query_valid_mask=None,
    ):
        shot_type = int(shot_type)
        if shot_type == 11:
            raise ValueError('Missed shot does not have placement evidence')
        if distances is None:
            distances = self.compute_distances(
                query_window_continuous,
                query_window_discrete,
                query_valid_mask=query_valid_mask,
            )
        coordinate_keys = ('landing_x', 'landing_y', 'moving_x', 'moving_y')
        indices = [
            window_id
            for window_id, action in enumerate(self.windows_actions)
            if int(float(action.get('player_type', 0))) == shot_type
            and all(
                action.get(key) is not None
                and np.isfinite(float(action[key]))
                and -1.0 <= float(action[key]) <= 1.0
                for key in coordinate_keys
            )
        ]
        return self._retrieve_from_distances(distances, indices, top_k)

    def validate(self):
        if self.schema_version != type(self).schema_version:
            raise ValueError(f"Unsupported TimeRAG schema version: {self.schema_version}")
        if self.verbalization_schema_version != type(self).verbalization_schema_version:
            raise ValueError(
                f"Unsupported TimeRAG verbalization schema version: "
                f"{self.verbalization_schema_version}"
            )
        if self.build_metadata.get('dataset') != 'rally':
            raise ValueError("TimeRAG database was not built from the rally dataset")
        if not self.build_metadata.get('split_sha256'):
            raise ValueError("TimeRAG database is missing rally split provenance")
        if self.build_metadata.get('window_size') != self.window_size:
            raise ValueError("TimeRAG build metadata window_size does not match the database")
        if not isinstance(self.window_size, int) or self.window_size < 1:
            raise ValueError("TimeRAG window_size must be a positive integer")

        count = len(self.windows)
        if count == 0:
            raise ValueError(f"TimeRAG database for {self.player_name} contains no windows")
        aligned_lengths = {
            'continuous': len(self.windows_continuous),
            'discrete': len(self.windows_discrete),
            'valid_masks': len(self.windows_valid_masks),
            'actions': len(self.windows_actions),
            'metadata': len(self.windows_metadata),
            'verbalizations': len(self.window_verbalizations),
        }
        mismatched = {name: length for name, length in aligned_lengths.items() if length != count}
        if mismatched:
            raise ValueError(f"TimeRAG window data is not aligned: expected {count}, got {mismatched}")
        expected_continuous_shape = (count, self.window_size, len(self.continuous_keys))
        expected_discrete_shape = (count, self.window_size, len(self.discrete_keys))
        expected_mask_shape = (count, self.window_size)
        if tuple(np.asarray(self.windows_continuous).shape) != expected_continuous_shape:
            raise ValueError("TimeRAG continuous feature shape does not match its schema")
        if tuple(np.asarray(self.windows_discrete).shape) != expected_discrete_shape:
            raise ValueError("TimeRAG discrete feature shape does not match its schema")
        if tuple(np.asarray(self.windows_valid_masks).shape) != expected_mask_shape:
            raise ValueError("TimeRAG valid-mask shape does not match its schema")
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
            for stage in ('stage1', 'stage2'):
                description = verbalization.get(f'{stage}_description')
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(
                        f"TimeRAG verbalization {window_id} has no {stage} description"
                    )

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
                if state is not None:
                    expected_dx = float(state['hit_x']) - float(state['player_location_x'])
                    expected_dy = float(state['hit_y']) - float(state['player_location_y'])
                    if not np.isclose(float(state['ball_distance_x']), expected_dx):
                        raise ValueError("TimeRAG ball_distance_x does not match hit_x - player_location_x")
                    if not np.isclose(float(state['ball_distance_y']), expected_dy):
                        raise ValueError("TimeRAG ball_distance_y does not match hit_y - player_location_y")
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
        rebuilt_masks = np.asarray([
            [state is not None for state in window]
            for window in self.windows
        ], dtype=bool)
        if not np.array_equal(np.asarray(self.windows_valid_masks), rebuilt_masks):
            raise ValueError("TimeRAG valid masks are assigned to the wrong windows")
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
                'build_metadata': self.build_metadata,
                'windows': self.windows,
                'windows_continuous': self.windows_continuous,
                'windows_discrete': self.windows_discrete,
                'windows_valid_masks': self.windows_valid_masks,
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
                "This TimeRAG database is not compatible with the rally dataset. "
                "Rebuild it with Scripts/prepare_all_data.py --timerag."
            )
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.schema_version = schema_version
        instance.verbalization_schema_version = verbalization_schema_version
        instance.player_name = data['player_name']
        instance.player_id = data['player_id']
        instance.window_size = data['window_size']
        instance.build_metadata = data.get('build_metadata', {})
        instance.windows = data['windows']
        instance.windows_continuous = data['windows_continuous']
        instance.windows_discrete = data['windows_discrete']
        instance.windows_valid_masks = data['windows_valid_masks']
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
                 host_url=None, normal_top_k=10, miss_top_k=5,
                 conditioned_top_k=10, enable_tracing=True, prompt_variant="norm",
                 case_grounding=True, action_architecture="two_stage_v1",
                 save_prompt_traces=False):
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
        self.normal_top_k = int(normal_top_k)
        self.miss_top_k = int(miss_top_k)
        self.conditioned_top_k = int(conditioned_top_k)
        if min(self.normal_top_k, self.miss_top_k, self.conditioned_top_k) < 1:
            raise ValueError('RAG retrieval top-k values must all be positive')
        if action_architecture not in {'two_stage_v1', 'single_stage_v1'}:
            raise ValueError(f'Unsupported action architecture: {action_architecture}')
        if action_architecture == 'single_stage_v1' and case_grounding:
            raise ValueError('Single-stage TimeRAG supports direct output only')
        self.retrieval_architecture = action_architecture
        self.prompt_variant = prompt_variant
        if not isinstance(case_grounding, bool):
            raise TypeError('case_grounding must be a bool')
        self.case_grounding = case_grounding
        tracer_name = (
            'timerag_single_stage'
            if action_architecture == 'single_stage_v1' else 'timerag'
        )
        self.tracer = BadmintonTracer(
            tracer_name,
            player_name,
            enable_tracing,
            capture_local=save_prompt_traces,
        )
        
        # Trajectory tracking for current rally (full state_dicts)
        self.trajectory = []
        self._current_rally = None
        self._last_case_ids = []
        self.last_decision_trace = None
        
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
                prompt_root, 'timerag_single_stage', action_filename
            )
            with open(action_path, 'r', encoding='utf-8') as f:
                self.action_template = Template(f.read())
            self.action_schema, self.action_schema_name = (
                BaseAgent.get_rag_single_stage_response_schema()
            )
        else:
            prompt_dir = os.path.join(prompt_root, "timerag")
            with open(os.path.join(prompt_dir, f'stage1_{template_suffix}.jinja'), "r") as f:
                self.stage1_template = Template(f.read())
            with open(os.path.join(prompt_dir, f'stage2_{template_suffix}.jinja'), "r") as f:
                self.stage2_template = Template(f.read())
            self.stage1_schema, self.stage1_schema_name = (
                BaseAgent.get_rag_response_schema(1, self.case_grounding)
            )
            self.stage2_schema, self.stage2_schema_name = (
                BaseAgent.get_rag_response_schema(2, self.case_grounding)
            )
        
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

    def _case_references(self, cases, stage, store, conditioned_shot_type=None):
        references = []
        for rank, case in enumerate(cases, 1):
            reference = {
                'database_player': self.player_name,
                'stage': int(stage),
                'store': store,
                'case_id': int(case['window_id']),
                'rank': rank,
                'distance': float(case['distance']),
            }
            if conditioned_shot_type is not None:
                reference['conditioned_shot_type'] = int(conditioned_shot_type)
            references.append(reference)
        return references

    @staticmethod
    def _map_case_assessments(case_assessments, references_by_source):
        """Map exhaustive assessments back to complete window references."""
        mapped = []
        for annotation in case_assessments:
            if not isinstance(annotation, dict):
                continue
            source = annotation.get('source')
            references = references_by_source.get(source, [])
            rank = annotation.get('retrieval_rank')
            if isinstance(rank, bool) or not isinstance(rank, int):
                continue
            if not 1 <= rank <= len(references):
                continue
            mapped.append({
                **references[rank - 1],
                'impact': annotation['impact'],
                'factors': list(annotation['factors']),
            })
            if source is not None:
                mapped[-1]['source'] = source
        return mapped

    @staticmethod
    def _nearest_case(cases):
        if not cases:
            return None
        return min(cases, key=lambda case: (float(case['distance']), int(case['window_id'])))

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
            key=lambda item: (float(item[0]['distance']), int(item[0]['window_id'])),
            default=None,
        )
        nearest = nearest_entry[0] if nearest_entry is not None else None
        if nearest is not None:
            shot_type = int(float(nearest['action']['player_type']))
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
        action = nearest['action']
        return {
            'case_assessments': case_assessments,
            'landing_x': float(action['landing_x']),
            'landing_y': float(action['landing_y']),
            'moving_x': float(action['moving_x']),
            'moving_y': float(action['moving_y']),
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
        historical_action = nearest['action']
        shot_type = int(float(historical_action['player_type']))
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
            'landing_x': float(historical_action['landing_x']),
            'landing_y': float(historical_action['landing_y']),
            'moving_x': float(historical_action['moving_x']),
            'moving_y': float(historical_action['moving_y']),
        }, 'fallback_nearest_case'

    def _predict_single_stage(self, state, retrieval, context, launch,
                              rally_profile_instruction=''):
        prompt = self.action_template.render(
            trajectory_text=context['trajectory_text'],
            normal_cases=self.verbalizer.build_stage2_cases(
                retrieval['normal_cases']
            ),
            miss_cases=self.verbalizer.build_stage2_cases(retrieval['miss_cases']),
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

        visible_query_states = query_states[-self.window_size:]
        query_states = list(visible_query_states)
        if len(query_states) < self.window_size:
            query_states = [None] * (self.window_size - len(query_states)) + query_states

        query_continuous, query_discrete, query_valid_mask = (
            self.database.build_feature_window(query_states, include_mask=True)
        )
        return (
            query_continuous,
            query_discrete,
            query_valid_mask,
            visible_query_states,
        )

    def reset_history(self):
        """Reset state history and trajectory tracking."""
        self.state_history = []
        self.trajectory = []
        self._current_rally = None
        self._last_case_ids = []
        self.last_decision_trace = None

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
        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.state_history = []
            self.trajectory = []

        # Build query window
        rally_history = info.get('rally_history')
        (
            query_continuous,
            query_discrete,
            query_valid_mask,
            query_states,
        ) = self._build_query_window(state_dict, rally_history)

        # Stage 1: retrieve successful and missed temporal evidence independently.
        t0 = time.time()
        retrieval_result = self.database.retrieve_stage1(
            query_continuous,
            query_discrete,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            launch=launch,
            query_valid_mask=query_valid_mask,
        )
        retrieved_windows = retrieval_result['normal_cases'] + retrieval_result['miss_cases']
        t1 = time.time()

        self.tracer.start_trace(state_dict, info, launch)
        self.tracer.log_retrieval(
            state_dict,
            retrieved_windows,
            "timerag",
            int((t1 - t0) * 1000),
            stage=(
                'single_stage'
                if self.retrieval_architecture == 'single_stage_v1' else 'stage1'
            ),
        )
        
        # Track rally trajectory
        self.trajectory.append(state_dict)
        
        # Build context for LLM using verbalizer
        state_history = query_states[:-1] or None
        context = self.verbalizer.build_prompt_context_nl(
            state_dict,
            [],
            launch=launch,
            window_size=self.window_size,
            state_history=state_history,
        )
        available_types = self.database.available_shot_types(launch=launch)
        if self.retrieval_architecture == 'single_stage_v1':
            return self._predict_single_stage(
                state,
                retrieval_result,
                context,
                launch,
                info.get('rally_profile_instruction', ''),
            )
        stage1_normal_cases = [
            {**case, 'player_type': case['action']['player_type']}
            for case in retrieval_result['normal_cases']
        ]
        stage1_miss_cases = [
            {**case, 'player_type': case['action']['player_type']}
            for case in retrieval_result['miss_cases']
        ]
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
            max_trials=2,
            schema=stage1_schema,
            schema_name=self.stage1_schema_name,
            validator=lambda response: BaseAgent.validate_rag_stage1_response(
                response,
                launch=launch,
                available_types=available_types,
                case_grounding=self.case_grounding,
                normal_cases=stage1_normal_cases,
                miss_cases=stage1_miss_cases,
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
            normal_cases=stage1_normal_cases,
            miss_cases=stage1_miss_cases,
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
        stage1_assessment_cases = self._map_case_assessments(
            stage1['case_assessments'],
            {
                'success': stage1_normal_refs,
                'miss': stage1_miss_refs,
            },
        )
        shot_type = int(stage1['shot_type'])
        stage2_refs = []
        stage2_assessment_cases = []
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
                query_continuous,
                query_discrete,
                shot_type,
                top_k=self.conditioned_top_k,
                distances=retrieval_result['distances'],
                query_valid_mask=query_valid_mask,
            )
            if any(
                int(float(case['action']['player_type'])) != shot_type
                for case in conditioned_cases
            ):
                raise ValueError('Shot-conditioned retrieval returned a different shot type')
            t1 = time.time()
            self.tracer.log_retrieval(
                state_dict,
                conditioned_cases,
                'timerag',
                int((t1 - t0) * 1000),
                stage='stage2',
            )
            stage2_context = self.verbalizer.build_prompt_context_nl(
                state_dict,
                conditioned_cases,
                launch=launch,
                window_size=self.window_size,
                state_history=state_history,
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
                max_trials=2,
                schema=stage2_schema,
                schema_name=self.stage2_schema_name,
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
            stage2_assessment_cases = self._map_case_assessments(
                placement['case_assessments'],
                {None: stage2_refs},
            )
            stage2 = {
                'status': 'completed',
                'conditioned_shot_type': shot_type,
                'case_ids': stage2_refs,
                'case_assessments': stage2_assessment_cases,
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
        all_case_ids = stage1_normal_refs + stage1_miss_refs + stage2_refs
        all_case_assessments = stage1_assessment_cases + stage2_assessment_cases
        self._last_case_ids = all_case_ids
        self.last_decision_trace = {
            'schema_version': 6,
            'architecture': self.retrieval_architecture,
            'case_grounding': self.case_grounding,
            'stage1': {
                'normal_case_ids': stage1_normal_refs,
                'miss_case_ids': stage1_miss_refs,
                'case_assessments': stage1_assessment_cases,
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
