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
from Agent.naive_rag_verbalizer import NaiveRAGVerbalizer, create_naive_rag_verbalizer
from Utils.llm import LLM
from Utils.helper import state_col, action_col, coord2area
from Utils.badminton_config import LANG_AREA_MAPPING, INDEX2EN_SHOT
from Utils.langfuse_tracing import BadmintonTracer


class NaiveRAGDatabase:
    """
    Database for NaiveRAG Agent.
    Stores training data and retrieves similar experiences based on:
    - player_location_x, player_location_y
    - hit_x, hit_y
    - opponent_type
    - opponent_location_x, opponent_location_y
    """
    
    def __init__(self, df, player_name, player_id):
        """
        Initialize database with filtered data for a specific player.
        
        Args:
            df: DataFrame with all data
            player_name: Name of the player
            player_id: ID of the player
        """
        self.player_name = player_name
        self.player_id = player_id
        
        # Filter data for this player
        self.data = df[df['player'] == player_name].copy()
        
        # Define continuous retrieval keys (for Euclidean distance)
        self.continuous_keys = [
            'player_location_x', 'player_location_y',
            'hit_x', 'hit_y',
            'opponent_location_x', 'opponent_location_y'
        ]
        
        # Define discrete retrieval keys (for Hamming distance)
        self.discrete_keys = ['opponent_type']
        
        # Define all retrieval keys
        self.retrieval_keys = self.continuous_keys + self.discrete_keys
        
        # Define metadata keys (additional context)
        self.metadata_keys = [
            'rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
            'ball_distance', 'ball_distance_x', 'ball_distance_y',
            'opponent_move_x', 'opponent_move_y'
        ]
        
        # Define action keys
        self.action_keys = ['landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y']
        
        # Build feature matrix for fast retrieval
        self._build_feature_matrix()
        
        print(f"Database initialized for {player_name} with {len(self.data)} entries")
    
    def _build_feature_matrix(self):
        """Build feature matrix for mixed distance computation."""
        # Build continuous features matrix (already standardized in data)
        self.continuous_features = self.data[self.continuous_keys].values.astype(np.float32)
        
        # Build discrete features matrix  
        self.discrete_features = self.data[self.discrete_keys].values.astype(np.float32)
    
    def _compute_mixed_distance(self, query_continuous, query_discrete):
        """
        Compute mixed distance combining Euclidean distance for continuous features
        and Hamming distance for discrete features.
        
        Args:
            query_continuous: Continuous features [n_continuous] (already standardized)
            query_discrete: Discrete features [n_discrete]
        
        Returns:
            distances: Array of mixed distances for all database entries
        """
        # Compute Euclidean distance for continuous features
        continuous_diff = self.continuous_features - query_continuous
        continuous_dist = np.sqrt(np.sum(continuous_diff ** 2, axis=1))
        
        # Compute Hamming distance for discrete features (normalized by number of features)
        discrete_diff = (self.discrete_features != query_discrete).astype(np.float32)
        discrete_dist = np.mean(discrete_diff, axis=1)  # Average across discrete features
        
        # Combine distances with equal weighting (can be adjusted)
        # Both are normalized to [0, 1] range approximately
        mixed_dist = continuous_dist + discrete_dist
        
        return mixed_dist
    
    def retrieve(self, query_state, top_k=5):
        """
        Retrieve top-k similar experiences using mixed distance.

        Args:
            query_state: Dict or array with retrieval keys
            top_k: Number of experiences to retrieve

        Returns:
            List of dicts containing experience data
        """
        # Extract query features
        if isinstance(query_state, dict):
            query_continuous = np.array([query_state[k] for k in self.continuous_keys], dtype=np.float32)
            query_discrete = np.array([query_state[k] for k in self.discrete_keys], dtype=np.float32)
        else:
            # Assume query_state is array in order [continuous..., discrete...]
            n_continuous = len(self.continuous_keys)
            query_continuous = query_state[:n_continuous].astype(np.float32)
            query_discrete = query_state[n_continuous:].astype(np.float32)

        # Compute mixed distances (lower is better)
        distances = self._compute_mixed_distance(query_continuous, query_discrete)

        # Convert distances to similarities (higher is better)
        # Using negative exponential: similarity = exp(-distance)
        similarities = np.exp(-distances)

        # Get top-k indices (smallest distances = largest similarities)
        top_indices = np.argsort(distances)[:top_k]

        # Build experience list
        experiences = []
        for idx in top_indices:
            row = self.data.iloc[idx]
            exp = {}

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
                    exp[key] = float(row[key])

            # Add landing area semantic name
            if 'landing_x' in row and 'landing_y' in row:
                landing_area_id = coord2area(float(row['landing_x']), float(row['landing_y']))
                exp['landing_area'] = LANG_AREA_MAPPING.get(landing_area_id, 'Unknown')
            else:
                exp['landing_area'] = 'Unknown'

            # Add movement area semantic name
            if 'moving_x' in row and 'moving_y' in row:
                movement_area_id = coord2area(float(row['moving_x']), float(row['moving_y']))
                exp['movement_area'] = LANG_AREA_MAPPING.get(movement_area_id, 'Unknown')
            else:
                exp['movement_area'] = 'Unknown'

            # Add shot type name
            if 'opponent_type' in row:
                exp['opponent_type_name'] = INDEX2EN_SHOT.get(int(row['opponent_type']), f"Unknown({int(row['opponent_type'])})")

            # Add similarity score
            exp['similarity'] = float(similarities[idx])

            experiences.append(exp)

        return experiences
    
    def save(self, filepath):
        """Save database to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'player_name': self.player_name,
                'player_id': self.player_id,
                'data': self.data,
                'continuous_features': self.continuous_features,
                'discrete_features': self.discrete_features,
                'continuous_keys': self.continuous_keys,
                'discrete_keys': self.discrete_keys,
                'retrieval_keys': self.retrieval_keys,
                'metadata_keys': self.metadata_keys,
                'action_keys': self.action_keys
            }, f)
        print(f"Database saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load database from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.player_name = data['player_name']
        instance.player_id = data['player_id']
        instance.data = data['data']
        instance.continuous_features = data['continuous_features']
        instance.discrete_features = data['discrete_features']
        instance.continuous_keys = data['continuous_keys']
        instance.discrete_keys = data['discrete_keys']
        instance.retrieval_keys = data['retrieval_keys']
        instance.metadata_keys = data['metadata_keys']
        instance.action_keys = data['action_keys']
        
        print(f"Database loaded from {filepath}")
        return instance


class NaiveRAGAgent(BaseAgent):
    """
    NaiveRAG Agent that retrieves similar experiences and uses LLM to generate actions.
    Supports verbalization to convert state and experiences to natural language.
    """
    
    def __init__(self, player_name, database, model_name="Qwen/Qwen2.5-7B-Instruct",
                 api_key="EMPTY", port=16000, base_url=None, top_k=5, enable_tracing=True,
                 prompt_variant="norm"):
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
        self.top_k = top_k
        self.prompt_variant = prompt_variant
        self.tracer = BadmintonTracer("naive_rag", player_name, enable_tracing)
        
        # Trajectory tracking for current rally
        self.trajectory = []  # list of state_dicts for the current rally
        self._current_rally = None
        
        # Initialize verbalizer
        self.verbalizer = create_naive_rag_verbalizer()
        
        # Initialize LLM
        self.llm = LLM(model_name=model_name, api_key=api_key, port=port, base_url=base_url)
        
        # Load prompts
        prompt_dir = os.path.join(project_path, "Prompts", "Agent")
        if prompt_variant == "shifted":
            system_prompt_name = "naive_rag_shifted_system.md"
            user_template_name = "naive_rag_shifted_user.jinja"
        elif prompt_variant == "norm":
            system_prompt_name = "naive_rag_system.md"
            user_template_name = "naive_rag_user.jinja"
        else:
            raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")

        with open(os.path.join(prompt_dir, system_prompt_name), "r") as f:
            self.system_prompt = f.read()
        
        # Load user template
        with open(os.path.join(prompt_dir, user_template_name), "r") as f:
            self.user_template = Template(f.read())
        
        # Load schema
        schema_dir = os.path.join(project_path, "Schemas", "Agent")
        with open(os.path.join(schema_dir, "naive_rag_schema.json"), "r") as f:
            self.schema = json.load(f)
    
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
            'player_location_y': state[2][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
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
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': player_location_x,
            'player_location_y': player_location_y,
            'player_location_area': player_location_area,
            'ball_distance': 0,  # Will be computed if needed
            'ball_distance_x': hit_x - player_location_x,
            'ball_distance_y': hit_y - player_location_y,
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

        # Retrieve similar experiences
        t0 = time.time()
        experiences = self.database.retrieve(query_state, top_k=self.top_k)
        t1 = time.time()
        self.tracer.log_retrieval(state_dict, experiences, "naive_rag", int((t1 - t0) * 1000))
        
        # Build context for LLM using verbalizer
        state_history = self.trajectory[:-1] if len(self.trajectory) > 1 else None
        context = self.verbalizer.build_prompt_context_nl(state_dict, experiences, launch=launch, state_history=state_history)
        context['style_shift_instruction'] = info.get('style_shift_instruction', '')

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

        # For missed shot (type 11), force coordinates to 0.0
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
        
        return action, reasoning
    
    def reset_history(self):
        """Reset rally trajectory tracking."""
        self._current_rally = None
        self.trajectory = []
    
    def _default_action(self, launch):
        """Default action when LLM fails."""
        if launch:
            shot_type = 1
        else:
            shot_type = np.random.randint(1, 11)
        
        return (
            shot_type,
            None,
            (0.0, 0.0),
            (0.0, 0.0),
            [0.1] * 10
        ), "Default action due to LLM failure"
    
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
            'player_location_y': state[2][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1]
        }
        
        # Retrieve similar experiences
        experiences = self.database.retrieve(query_state, top_k=self.top_k)

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
