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
from Agent.twin_verbalizer import TwinVerbalizer, create_twin_verbalizer
from Utils.llm import LLM
from Utils.helper import coord2area
from Utils.badminton_config import LANG_AREA_MAPPING, INDEX2EN_SHOT
from Utils.langfuse_tracing import BadmintonTracer


class TwinDatabase:
    """
    Database for BadmintonTwin Agent using Incremental Multi-Scale Dynamic Programming (I-MSDP).

    Stores historical rally trajectories and retrieves similar multi-scale trajectory segments
    using an incremental DP approach that reuses similarity states from the previous step.
    """

    # Continuous feature keys (Euclidean distance)
    continuous_feature_keys = [
        'player_location_x', 'player_location_y',
        'hit_x', 'hit_y',
        'opponent_location_x', 'opponent_location_y'
    ]

    # Discrete feature keys (Hamming distance)
    discrete_feature_keys = ['opponent_type']

    # Combined keys (kept for backward compatibility in segment building / serialization)
    state_feature_keys = continuous_feature_keys + discrete_feature_keys

    # Action keys
    action_keys = ['landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y']

    # Metadata keys for additional context
    metadata_keys = [
        'rally', 'ball_round', 'player_score', 'opponent_score',
        'ball_distance', 'ball_distance_x', 'ball_distance_y',
        'opponent_move_x', 'opponent_move_y'
    ]

    def __init__(self, df, player_name, player_id, max_scale=10):
        """
        Initialize TwinDatabase.

        Args:
            df: DataFrame with all data
            player_name: Name of the player
            player_id: ID of the player
            max_scale: Maximum suffix length k (L in the paper)
        """
        self.player_name = player_name
        self.player_id = player_id
        self.max_scale = max_scale

        # Filter data for this player
        self.data = df[df['player'] == player_name].copy()

        # Build trajectories (grouped by rally)
        self._build_trajectory_index()

        print(f"TwinDatabase initialized for {player_name} with {self.num_trajectories} trajectories, "
              f"max_scale={max_scale}, max_length={self.max_traj_length}")

    def _build_trajectory_index(self):
        """
        Build trajectory index from player data.
        Each trajectory corresponds to one rally (sequence of shots by this player).
        """
        self.trajectories = []
        self.trajectory_rally_ids = []

        # Group by available columns
        group_cols = ['rally']
        if 'set' in self.data.columns:
            group_cols = ['set', 'rally']
        if 'match_id' in self.data.columns:
            group_cols = ['match_id'] + group_cols

        grouped = self.data.groupby(group_cols)

        for rally_id, rally_df in grouped:
            rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
            if len(rally_df) == 0:
                continue

            # Extract state features
            states_cont = rally_df[self.continuous_feature_keys].values.astype(np.float32)
            states_disc = rally_df[self.discrete_feature_keys].values.astype(np.float32)
            states_full = rally_df[self.state_feature_keys].values.astype(np.float32)

            # Extract actions
            actions = rally_df[self.action_keys].values.astype(np.float32)

            # Extract metadata
            metadata = []
            for _, row in rally_df.iterrows():
                meta = {k: row[k] for k in self.metadata_keys if k in row}
                # Add score_status
                if 'player_score' in row and 'opponent_score' in row:
                    meta['score_status'] = int(float(row['player_score']) - float(row['opponent_score']))
                metadata.append(meta)

            self.trajectories.append({
                'states_full': states_full,
                'states_cont': states_cont,
                'states_disc': states_disc,
                'actions': actions,
                'metadata': metadata,
                'length': len(rally_df)
            })
            self.trajectory_rally_ids.append(rally_id)

        self.num_trajectories = len(self.trajectories)
        if self.num_trajectories == 0:
            self.max_traj_length = 0
            self.padded_states = np.zeros((0, 0, len(self.state_feature_keys)), dtype=np.float32)
            self.padded_continuous = np.zeros((0, 0, len(self.continuous_feature_keys)), dtype=np.float32)
            self.padded_discrete = np.zeros((0, 0, len(self.discrete_feature_keys)), dtype=np.float32)
            self.padded_actions = np.zeros((0, 0, len(self.action_keys)), dtype=np.float32)
            self.trajectory_lengths = np.array([], dtype=np.int32)
            return

        self.max_traj_length = max(traj['length'] for traj in self.trajectories)
        self.trajectory_lengths = np.array([traj['length'] for traj in self.trajectories], dtype=np.int32)

        # Build padded state and action arrays for vectorized computation
        d_cont = len(self.continuous_feature_keys)
        d_disc = len(self.discrete_feature_keys)
        d_state = len(self.state_feature_keys)
        d_action = len(self.action_keys)
        self.padded_continuous = np.full((self.num_trajectories, self.max_traj_length, d_cont), np.inf, dtype=np.float32)
        self.padded_discrete = np.full((self.num_trajectories, self.max_traj_length, d_disc), np.inf, dtype=np.float32)
        self.padded_states = np.full((self.num_trajectories, self.max_traj_length, d_state), np.inf, dtype=np.float32)
        self.padded_actions = np.full((self.num_trajectories, self.max_traj_length, d_action), np.inf, dtype=np.float32)

        for i, traj in enumerate(self.trajectories):
            length = traj['length']
            self.padded_continuous[i, :length] = traj['states_cont']
            self.padded_discrete[i, :length] = traj['states_disc']
            self.padded_states[i, :length] = traj['states_full']
            self.padded_actions[i, :length] = traj['actions']

    def _compute_base_distance(self, query_state):
        """
        Compute D_{t,1}: Mixed distance between query_state and all trajectory points.

        Euclidean distance for continuous features (positions),
        Hamming distance for discrete features (shot type).

        Args:
            query_state: Dict with keys in state_feature_keys

        Returns:
            base_dist: [N, M_max] array of mixed distances
        """
        # Euclidean distance for continuous features
        query_cont = np.array([float(query_state[k]) for k in self.continuous_feature_keys], dtype=np.float32)
        diff = self.padded_continuous - query_cont.reshape(1, 1, -1)
        cont_dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Hamming distance for discrete features
        query_disc = int(query_state[self.discrete_feature_keys[0]])
        disc_dist = (self.padded_discrete[:, :, 0] != query_disc).astype(np.float32)

        # Combined distance
        dist = cont_dist + disc_dist

        # Mask invalid positions (beyond trajectory length) with inf
        mask = np.arange(self.max_traj_length).reshape(1, -1) >= self.trajectory_lengths.reshape(-1, 1)
        dist[mask] = np.inf

        return dist

    def incremental_retrieve(self, query_state, prev_distance_tensor=None, top_k=5):
        """
        I-MSDP incremental retrieval.

        Uses the DP recurrence:
            D_{t,1}[i,j] = Delta(s_t, T^{(i)}_j)
            D_{t,k}[i,j] = D_{t,1}[i,j] + D_{t-1,k-1}[i,j-1]   for k > 1

        Args:
            query_state: Current state feature dict/array
            prev_distance_tensor: [max_scale, N, M_max] distance tensor from previous step (D_{t-1})
                                None if this is the first step of the rally.
            top_k: Number of trajectory segments to retrieve.

        Returns:
            segments: List of trajectory segment dicts for prompt construction
            curr_distance_tensor: [max_scale, N, M_max] current distance tensor (D_t)
        """
        # Step 1: Compute base case D_{t,1}
        base_dist = self._compute_base_distance(query_state)  # [N, M_max]

        # Step 2: Initialize current distance tensor
        curr_dist = np.full((self.max_scale, self.num_trajectories, self.max_traj_length), np.inf, dtype=np.float32)
        curr_dist[0] = base_dist.copy()  # k=1 maps to index 0

        # Step 3: Recursive scale expansion for k = 2 .. max_scale
        if prev_distance_tensor is not None and self.num_trajectories > 0:
            for k in range(2, self.max_scale + 1):
                # D_{t,k}[i,j] = D_{t,1}[i,j] + D_{t-1,k-1}[i,j-1]
                # For j >= 1; j=0 cannot inherit from j-1
                # We shift prev_distance_tensor[k-2] (which is D_{t-1,k-1}) by +1 in the j dimension
                shifted_prev = np.full((self.num_trajectories, self.max_traj_length), np.inf, dtype=np.float32)
                shifted_prev[:, 1:] = prev_distance_tensor[k - 2, :, :-1]

                # Only compute where both base_dist and shifted_prev are valid (not inf)
                valid_mask = np.isfinite(base_dist) & np.isfinite(shifted_prev)
                curr_dist[k - 1][valid_mask] = base_dist[valid_mask] + shifted_prev[valid_mask]

        # Step 4: Collect all valid candidates with normalized distances
        candidates = []
        for k in range(1, self.max_scale + 1):
            for i in range(self.num_trajectories):
                traj_len = self.trajectory_lengths[i]
                # A k-length suffix ending at j requires j >= k-1 and j < traj_len
                for j in range(k - 1, traj_len):
                    dist = curr_dist[k - 1, i, j]
                    if np.isfinite(dist):
                        norm_dist = dist / k
                        candidates.append({
                            'traj_idx': i,
                            'end_idx': j,
                            'scale': k,
                            'distance': float(dist),
                            'normalized_distance': float(norm_dist)
                        })

        # Step 5: Sort by normalized distance (ascending; lower is better)
        candidates.sort(key=lambda x: x['normalized_distance'])

        # Step 6: Temporal Non-Maximum Suppression (T-NMS)
        # If overlapping, prefer longer scale.
        selected = []
        for cand in candidates:
            # Find overlapping selected candidates from the same trajectory
            overlapping = []
            for idx, sel in enumerate(selected):
                if cand['traj_idx'] == sel['traj_idx']:
                    cand_start = cand['end_idx'] - cand['scale'] + 1
                    cand_end = cand['end_idx']
                    sel_start = sel['end_idx'] - sel['scale'] + 1
                    sel_end = sel['end_idx']
                    # Check if intervals intersect
                    if not (cand_end < sel_start or cand_start > sel_end):
                        overlapping.append(idx)

            if len(overlapping) == 0:
                # No overlap: add directly
                selected.append(cand)
            else:
                # Overlap exists: prefer longer scale
                max_overlap_scale = max(selected[idx]['scale'] for idx in overlapping)
                if cand['scale'] > max_overlap_scale:
                    # Remove all overlapping selected candidates and add current
                    for idx in sorted(overlapping, reverse=True):
                        selected.pop(idx)
                    selected.append(cand)
                # Else: skip current candidate (shorter or equal scale)

            if len(selected) >= top_k:
                break

        # Step 7: Build trajectory segments from selected candidates
        segments = []
        for sel in selected:
            segment = self._get_trajectory_segment(sel['traj_idx'], sel['end_idx'], sel['scale'])
            segment['scale'] = sel['scale']
            segment['similarity'] = sel['normalized_distance']
            segment['total_distance'] = sel['distance']
            segments.append(segment)

        return segments, curr_dist

    def _get_trajectory_segment(self, traj_idx, end_idx, scale):
        """
        Extract a trajectory segment [end_idx-scale+1 .. end_idx] from a trajectory.

        Args:
            traj_idx: Trajectory index
            end_idx: Ending position in the trajectory
            scale: Length of the segment

        Returns:
            segment: Dict with 'steps' list containing state/action/metadata for each step
        """
        traj = self.trajectories[traj_idx]
        start_idx = end_idx - scale + 1

        steps = []
        for pos in range(start_idx, end_idx + 1):
            step = {}

            # State features
            for key_idx, key in enumerate(self.state_feature_keys):
                step[key] = float(traj['states_full'][pos, key_idx])

            # Actions
            for key_idx, key in enumerate(self.action_keys):
                step[key] = float(traj['actions'][pos, key_idx])

            # Metadata
            meta = traj['metadata'][pos]
            for key, val in meta.items():
                if isinstance(val, (np.integer, np.floating)):
                    step[key] = float(val)
                else:
                    step[key] = val

            # Semantic area names
            player_x = step.get('player_location_x', 0)
            player_y = step.get('player_location_y', 0)
            step['player_location_area'] = LANG_AREA_MAPPING.get(coord2area(player_x, player_y), 'Unknown')

            hit_x = step.get('hit_x', 0)
            hit_y = step.get('hit_y', 0)
            step['hit_position_area'] = LANG_AREA_MAPPING.get(coord2area(hit_x, hit_y), 'Unknown')

            opp_x = step.get('opponent_location_x', 0)
            opp_y = step.get('opponent_location_y', 0)
            step['opponent_location_area'] = LANG_AREA_MAPPING.get(coord2area(opp_x, opp_y), 'Unknown')

            land_x = step.get('landing_x', 0)
            land_y = step.get('landing_y', 0)
            step['landing_area'] = LANG_AREA_MAPPING.get(coord2area(land_x, land_y), 'Unknown')

            move_x = step.get('moving_x', 0)
            move_y = step.get('moving_y', 0)
            step['movement_area'] = LANG_AREA_MAPPING.get(coord2area(move_x, move_y), 'Unknown')

            # Shot type names
            opp_type = int(step.get('opponent_type', 0))
            step['opponent_type_name'] = INDEX2EN_SHOT.get(opp_type, f"Unknown({opp_type})")

            player_type = int(step.get('player_type', 0))
            step['player_type_name'] = INDEX2EN_SHOT.get(player_type, f"Unknown({player_type})")

            steps.append(step)

        return {'steps': steps}

    def save(self, filepath):
        """Save database to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'player_name': self.player_name,
                'player_id': self.player_id,
                'max_scale': self.max_scale,
                'data': self.data,
                'trajectories': self.trajectories,
                'trajectory_rally_ids': self.trajectory_rally_ids,
                'num_trajectories': self.num_trajectories,
                'max_traj_length': self.max_traj_length,
                'trajectory_lengths': self.trajectory_lengths,
                'padded_states': self.padded_states,
                'padded_continuous': self.padded_continuous,
                'padded_discrete': self.padded_discrete,
                'padded_actions': self.padded_actions,
            }, f)
        print(f"TwinDatabase saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load database from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.player_name = data['player_name']
        instance.player_id = data['player_id']
        instance.max_scale = data['max_scale']
        instance.data = data['data']
        instance.trajectories = data['trajectories']
        instance.trajectory_rally_ids = data['trajectory_rally_ids']
        instance.num_trajectories = data['num_trajectories']
        instance.max_traj_length = data['max_traj_length']
        instance.trajectory_lengths = data['trajectory_lengths']
        instance.padded_states = data['padded_states']
        instance.padded_actions = data['padded_actions']

        if 'padded_continuous' in data and 'padded_discrete' in data:
            instance.padded_continuous = data['padded_continuous']
            instance.padded_discrete = data['padded_discrete']
        else:
            # Backward compat: extract from padded_states (old format)
            n_cont = len(cls.continuous_feature_keys)
            instance.padded_continuous = data['padded_states'][:, :, :n_cont]
            instance.padded_discrete = data['padded_states'][:, :, n_cont:]

        print(f"TwinDatabase loaded from {filepath}")
        return instance


class TwinAgent(BaseAgent):
    """
    BadmintonTwin Agent using Incremental Multi-Scale Dynamic Programming (I-MSDP).

    Retrieves multi-scale trajectory segments from historical data using an incremental
    DP approach, then uses an LLM to generate actions based on the retrieved segments.
    """

    def __init__(self, player_name, database, model_name="Qwen/Qwen2.5-7B-Instruct",
                 api_key="EMPTY", port=16000, base_url=None, top_k=5, max_scale=10,
                 enable_tracing=True, prompt_variant="norm"):
        """
        Initialize TwinAgent.

        Args:
            player_name: Name of the player
            database: TwinDatabase instance
            model_name: Name of the LLM model
            api_key: API key for LLM
            port: Port for local LLM server
            top_k: Number of trajectory segments to retrieve
            max_scale: Maximum suffix length (passed to database, should match)
            enable_tracing: Whether to enable Langfuse tracing
        """
        super().__init__(player_name)

        self.database = database
        self.top_k = top_k
        self.max_scale = max_scale
        self.prompt_variant = prompt_variant
        self.tracer = BadmintonTracer("twin", player_name, enable_tracing)

        # Initialize verbalizer
        self.verbalizer = create_twin_verbalizer()

        # Initialize LLM
        self.llm = LLM(model_name=model_name, api_key=api_key, port=port, base_url=base_url)

        # Load prompts
        prompt_dir = os.path.join(project_path, "Prompts", "Agent")
        if prompt_variant == "shifted":
            system_prompt_name = "twin_shifted_system.md"
            user_template_name = "twin_shifted_user.jinja"
        elif prompt_variant == "norm":
            system_prompt_name = "twin_system.md"
            user_template_name = "twin_user.jinja"
        else:
            raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")

        with open(os.path.join(prompt_dir, system_prompt_name), "r") as f:
            self.system_prompt = f.read()

        # Load user template
        with open(os.path.join(prompt_dir, user_template_name), "r") as f:
            self.user_template = Template(f.read())

        # Load schema
        schema_dir = os.path.join(project_path, "Schemas", "Agent")
        with open(os.path.join(schema_dir, "twin_schema.json"), "r") as f:
            self.schema = json.load(f)

        # Rally tracking for incremental retrieval
        self.current_rally = None
        self.query_history = []
        self.trajectory = []  # full state_dicts for trajectory narrative
        self.prev_dist_tensor = None

    def reset_history(self):
        """Reset rally tracking state."""
        self.current_rally = None
        self.query_history = []
        self.trajectory = []
        self.prev_dist_tensor = None

    def predict(self, state, info, launch):
        """
        Predict action using BadmintonTwin (I-MSDP) approach.

        Args:
            state: Environment state [opponent_shot_type, player_pos, opponent_pos, landing_pos]
            info: Additional info dict
            launch: Whether this is a serve

        Returns:
            action tuple and reasoning
        """
        # Detect rally change and reset incremental state if needed
        rally_num = info.get('rally', 0)
        if self.current_rally != rally_num:
            self.current_rally = rally_num
            self.query_history = []
            self.trajectory = []
            self.prev_dist_tensor = None

        # Build query state from env state (same features as NaiveRAG, but corrected mapping)
        query_state = {
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1]
        }
        self.query_history.append(query_state)

        # Incremental retrieval using I-MSDP
        t0 = time.time()
        segments, self.prev_dist_tensor = self.database.incremental_retrieve(
            query_state,
            prev_distance_tensor=self.prev_dist_tensor,
            top_k=self.top_k
        )
        t1 = time.time()

        # Build state dictionary with semantic area names
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        player_location_x = query_state['player_location_x']
        player_location_y = query_state['player_location_y']
        hit_x = query_state['hit_x']
        hit_y = query_state['hit_y']
        opponent_location_x = query_state['opponent_location_x']
        opponent_location_y = query_state['opponent_location_y']

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

        self.tracer.start_trace(state_dict, info, launch)
        self.tracer.log_retrieval(state_dict, segments, "twin", int((t1 - t0) * 1000))
        
        # Track rally trajectory
        self.trajectory.append(state_dict)

        # Build context for LLM using verbalizer
        state_history = self.trajectory[:-1] if len(self.trajectory) > 1 else None
        context = self.verbalizer.build_prompt_context_nl(state_dict, segments, launch=launch, state_history=state_history)
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
        Get verbalized state string and retrieved trajectory segments for a given state.

        Args:
            state: Environment state
            info: Additional info dict
            launch: Whether this is a serve

        Returns:
            Tuple of (state_str, segments_str, segments)
        """
        query_state = {
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1]
        }

        segments, _ = self.database.incremental_retrieve(
            query_state,
            prev_distance_tensor=None,  # One-shot retrieval for visualization
            top_k=self.top_k
        )

        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0

        state_dict = {
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': query_state['player_location_x'],
            'player_location_y': query_state['player_location_y'],
            'ball_distance': 0,
            'ball_distance_x': query_state['hit_x'] - query_state['player_location_x'],
            'ball_distance_y': query_state['hit_y'] - query_state['player_location_y'],
            'hit_x': query_state['hit_x'],
            'hit_y': query_state['hit_y'],
            'opponent_type': query_state['opponent_type'],
            'opponent_location_x': query_state['opponent_location_x'],
            'opponent_location_y': query_state['opponent_location_y'],
        }

        if self.verbalizer is not None:
            state_str = self.verbalizer.verbalize_state(state_dict)
            segments_str = self.verbalizer.verbalize_trajectory_segments(segments)
        else:
            state_str = str(state_dict)
            segments_str = str(segments)

        return state_str, segments_str, segments
