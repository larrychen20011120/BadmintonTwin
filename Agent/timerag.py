import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from jinja2 import Template

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Agent.timerag_verbalizer import TimeRAGVerbalizer, create_timerag_verbalizer
from Utils.llm import LLM
from Utils.helper import state_col, action_col, coord2area
from Utils.badminton_config import LANG_AREA_MAPPING, INDEX2EN_SHOT
from Utils.langfuse_tracing import BadmintonTracer


class TimeRAGDatabase:
    """
    Database for TimeRAG Agent.
    Stores training data as time windows and retrieves similar sequences using DTW distance.
    
    Each window contains consecutive states [state_t-2, state_t-1, state_t] 
    with action_t as the target to predict.
    """
    
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
        
        # Filter data for this player
        self.data = df[df['player'] == player_name].copy()
        
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
        
        print(f"TimeRAG Database initialized for {player_name} with {len(self.windows)} windows (window_size={window_size})")
    
    def _build_windows(self):
        """Build time windows from rally data."""
        # Group by match_id and rally to process each rally separately
        # Note: 'set' column may not be available in all datasets
        grouped = self.data.groupby(['match_id', 'rally'])
        
        for (match_id, rally_num), rally_df in grouped:
            rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
            
            # Skip rallies that are too short
            if len(rally_df) < 1:
                continue
            
            # Create windows with padding for the beginning of rally
            for i in range(len(rally_df)):
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
                
                # Get action for the current state (state_t)
                current_row = rally_df.iloc[i]
                action = {key: float(current_row[key]) for key in self.action_keys if key in current_row}
                
                # Get metadata
                metadata = {
                    'match_id': match_id,
                    'rally': rally_num,
                    'ball_round': int(current_row['ball_round']),
                    'player_score': float(current_row['player_score']),
                    'opponent_score': float(current_row['opponent_score']),
                    'score_status': float(current_row['player_score']) - float(current_row['opponent_score'])
                }
                # Add 'set' column if it exists in the data
                if 'set' in current_row:
                    metadata['set'] = int(current_row['set'])
                
                # Build feature arrays for this window
                window_continuous = np.zeros((self.window_size, len(self.continuous_keys)), dtype=np.float32)
                window_discrete = np.zeros((self.window_size, len(self.discrete_keys)), dtype=np.float32)
                
                for w, state in enumerate(window_states):
                    if state is not None:
                        # Fill continuous features
                        for j, key in enumerate(self.continuous_keys):
                            window_continuous[w, j] = float(state[key])
                        # Fill discrete features
                        for j, key in enumerate(self.discrete_keys):
                            window_discrete[w, j] = float(state[key])
                    # else: padding remains as zeros
                
                self.windows.append(window_states)
                self.windows_continuous.append(window_continuous)
                self.windows_discrete.append(window_discrete)
                self.windows_actions.append(action)
                self.windows_metadata.append(metadata)
        
        # Convert to numpy arrays for faster computation
        if len(self.windows_continuous) > 0:
            self.windows_continuous = np.array(self.windows_continuous)  # [n_windows, window_size, n_continuous]
            self.windows_discrete = np.array(self.windows_discrete)      # [n_windows, window_size, n_discrete]
    
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
    
    def retrieve(self, query_window_continuous, query_window_discrete, top_k=5):
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

        # Compute DTW distances
        distances = self._compute_window_dtw_distance(query_continuous, query_discrete)

        # Convert distances to similarities (higher is better)
        # Using negative exponential: similarity = exp(-distance)
        similarities = np.exp(-distances)

        # Get top-k indices (smallest distances)
        top_indices = np.argsort(distances)[:top_k]

        # Build result
        results = []
        for idx in top_indices:
            window_states = self.windows[idx]
            action = self.windows_actions[idx]
            metadata = self.windows_metadata[idx]

            # Build window data
            window_data = {
                'window_states': [],
                'action': action,
                'metadata': metadata,
                'distance': float(distances[idx]),
                'similarity': float(similarities[idx])
            }

            # Add each state in the window
            for t, state in enumerate(window_states):
                if state is not None:
                    state_data = {}
                    for key in self.continuous_keys + self.discrete_keys + self.metadata_keys + self.action_keys:
                        if key in state:
                            state_data[key] = float(state[key])

                    # Add score_status as integer (compute if not present)
                    if 'player_score' in state and 'opponent_score' in state:
                        state_data['score_status'] = int(float(state['player_score']) - float(state['opponent_score']))

                    # Add semantic area names for all positions
                    if 'player_location_x' in state and 'player_location_y' in state:
                        player_area_id = coord2area(float(state['player_location_x']), float(state['player_location_y']))
                        state_data['player_location_area'] = LANG_AREA_MAPPING.get(player_area_id, 'Unknown')

                    if 'hit_x' in state and 'hit_y' in state:
                        hit_area_id = coord2area(float(state['hit_x']), float(state['hit_y']))
                        state_data['hit_position_area'] = LANG_AREA_MAPPING.get(hit_area_id, 'Unknown')

                    if 'opponent_location_x' in state and 'opponent_location_y' in state:
                        opponent_area_id = coord2area(float(state['opponent_location_x']), float(state['opponent_location_y']))
                        state_data['opponent_location_area'] = LANG_AREA_MAPPING.get(opponent_area_id, 'Unknown')

                    # Add area names for action positions
                    if 'landing_x' in state and 'landing_y' in state:
                        landing_area_id = coord2area(float(state['landing_x']), float(state['landing_y']))
                        state_data['landing_area'] = LANG_AREA_MAPPING.get(landing_area_id, 'Unknown')

                    if 'moving_x' in state and 'moving_y' in state:
                        movement_area_id = coord2area(float(state['moving_x']), float(state['moving_y']))
                        state_data['movement_area'] = LANG_AREA_MAPPING.get(movement_area_id, 'Unknown')

                    # Add shot type name
                    if 'opponent_type' in state:
                        state_data['opponent_type_name'] = INDEX2EN_SHOT.get(int(state['opponent_type']), f"Unknown({int(state['opponent_type'])})")

                    window_data['window_states'].append(state_data)
                else:
                    # Padding marker - still append but will be filtered in template
                    window_data['window_states'].append({'padding': True})

            # Also add semantic info to action if not present
            if 'landing_area' not in action and 'landing_x' in action and 'landing_y' in action:
                landing_area_id = coord2area(float(action['landing_x']), float(action['landing_y']))
                action['landing_area'] = LANG_AREA_MAPPING.get(landing_area_id, 'Unknown')

            if 'movement_area' not in action and 'moving_x' in action and 'moving_y' in action:
                movement_area_id = coord2area(float(action['moving_x']), float(action['moving_y']))
                action['movement_area'] = LANG_AREA_MAPPING.get(movement_area_id, 'Unknown')

            results.append(window_data)

        return results
    
    def save(self, filepath):
        """Save database to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'player_name': self.player_name,
                'player_id': self.player_id,
                'window_size': self.window_size,
                'windows': self.windows,
                'windows_continuous': self.windows_continuous,
                'windows_discrete': self.windows_discrete,
                'windows_actions': self.windows_actions,
                'windows_metadata': self.windows_metadata,
                'continuous_keys': self.continuous_keys,
                'discrete_keys': self.discrete_keys,
                'action_keys': self.action_keys,
                'metadata_keys': self.metadata_keys
            }, f)
        print(f"TimeRAG Database saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load database from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.player_name = data['player_name']
        instance.player_id = data['player_id']
        instance.window_size = data['window_size']
        instance.windows = data['windows']
        instance.windows_continuous = data['windows_continuous']
        instance.windows_discrete = data['windows_discrete']
        instance.windows_actions = data['windows_actions']
        instance.windows_metadata = data['windows_metadata']
        instance.continuous_keys = data['continuous_keys']
        instance.discrete_keys = data['discrete_keys']
        instance.action_keys = data['action_keys']
        instance.metadata_keys = data['metadata_keys']
        
        print(f"TimeRAG Database loaded from {filepath}")
        return instance


class TimeRAGAgent(BaseAgent):
    """
    TimeRAG Agent that retrieves similar time windows and uses LLM to generate actions.
    Maintains a sliding window of recent states for retrieval.
    """
    
    def __init__(self, player_name, database, model_name="Qwen/Qwen2.5-7B-Instruct",
                 api_key="EMPTY", port=16000, top_k=5, enable_tracing=True):
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
        self.tracer = BadmintonTracer("timerag", player_name, enable_tracing)
        
        # Trajectory tracking for current rally (full state_dicts)
        self.trajectory = []
        self._current_rally = None
        
        # Initialize verbalizer
        self.verbalizer = create_timerag_verbalizer()
        
        # Initialize LLM
        self.llm = LLM(model_name=model_name, api_key=api_key, port=port)
        
        # Load prompts
        prompt_dir = os.path.join(project_path, "Prompts", "Agent")
        with open(os.path.join(prompt_dir, "timerag_system.md"), "r") as f:
            self.system_prompt = f.read()
        
        # Load user template
        with open(os.path.join(prompt_dir, "timerag_user.jinja"), "r") as f:
            self.user_template = Template(f.read())
        
        # Load schema
        schema_dir = os.path.join(project_path, "Schemas", "Agent")
        with open(os.path.join(schema_dir, "timerag_schema.json"), "r") as f:
            self.schema = json.load(f)
        
        # State history buffer for building windows
        self.state_history = []
    
    def _build_query_window(self, state):
        """
        Build query window from current state and history.
        
        Args:
            state: Current environment state
        
        Returns:
            query_continuous: [window_size, n_continuous]
            query_discrete: [window_size, n_discrete]
        """
        # Extract features from current state
        current_continuous = np.array([
            state[1][0], state[1][1],  # player_location_x, player_location_y
            state[3][0], state[3][1],  # hit_x, hit_y
            state[2][0], state[2][1],  # opponent_location_x, opponent_location_y
        ], dtype=np.float32)
        
        current_discrete = np.array([
            float(state[0])  # opponent_type
        ], dtype=np.float32)
        
        # Add to history
        self.state_history.append({
            'continuous': current_continuous,
            'discrete': current_discrete
        })
        
        # Keep only window_size most recent states
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
        
        # Build window with padding if needed
        query_continuous = np.zeros((self.window_size, len(current_continuous)), dtype=np.float32)
        query_discrete = np.zeros((self.window_size, len(current_discrete)), dtype=np.float32)
        
        # Fill from history (padding at beginning if history is short)
        n_history = len(self.state_history)
        for i in range(n_history):
            query_continuous[self.window_size - n_history + i] = self.state_history[i]['continuous']
            query_discrete[self.window_size - n_history + i] = self.state_history[i]['discrete']
        
        return query_continuous, query_discrete
    
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
        # Build query window
        query_continuous, query_discrete = self._build_query_window(state)
        
        # Retrieve similar windows
        t0 = time.time()
        retrieved_windows = self.database.retrieve(query_continuous, query_discrete, top_k=self.top_k)
        t1 = time.time()
        
        # Build state dictionary for current state
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        player_location_x = state[1][0]
        player_location_y = state[1][1]
        hit_x = state[3][0] if len(state) > 3 else 0
        hit_y = state[3][1] if len(state) > 3 else 0
        opponent_location_x = state[2][0]
        opponent_location_y = state[2][1]

        # Compute semantic area names for current state
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
            'opponent_type': int(state[0]),
            'opponent_location_x': opponent_location_x,
            'opponent_location_y': opponent_location_y,
            'opponent_location_area': opponent_location_area,
            'opponent_move_x': 0,
            'opponent_move_y': 0,
        }

        self.tracer.start_trace(state_dict, info, launch)
        self.tracer.log_retrieval(state_dict, retrieved_windows, "timerag", int((t1 - t0) * 1000))
        
        # Track rally trajectory
        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.trajectory = []
        self.trajectory.append(state_dict)
        
        # Build context for LLM using verbalizer
        state_history = self.trajectory[:-1] if len(self.trajectory) > 1 else None
        context = self.verbalizer.build_prompt_context_nl(state_dict, retrieved_windows, launch=launch, window_size=self.window_size, state_history=state_history)
        
        # Render user prompt
        user_prompt = self.user_template.render(**context)
        
        # Get LLM response
        t0 = time.time()
        response, raw_llm_output, usage = self.llm.get_verified_response(
            schema=self.schema,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_trials=5
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
        shot_type = response['shot_type']
        landing_x = response.get('landing_x', 0.0)
        landing_y = response.get('landing_y', 0.0)
        moving_x = response.get('moving_x', 0.0)
        moving_y = response.get('moving_y', 0.0)
        reasoning = response.get('reasoning', '')
        
        if shot_type == 11:
            landing_x = 0.0
            landing_y = 0.0
            moving_x = 0.0
            moving_y = 0.0
        
        # Apply serve mask if needed
        if launch and shot_type not in [1, 10]:
            shot_type = 1 if np.random.rand() < 0.5 else 10
        
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
