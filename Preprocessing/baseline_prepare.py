import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Utils.helper import *


def prepare_bc_data(df, split_content, player2index, output_dir):
    """
    從 train split 提取 BC 訓練資料。
    
    Args:
        df: DataFrame with merged data (player column should be mapped to IDs)
        split_content: Train/test split dict
        player2index: Player name to ID mapping
        output_dir: Directory to save output files
    
    Output:
        - train_state.pkl: numpy array of states
        - train_action.pkl: numpy array of actions
    """
    state_data, action_data = [], []
    
    # Ensure player column contains IDs (integers), not names
    if df['player'].dtype == object:
        print("Converting player names to IDs for BC data preparation...")
        df = df.copy()
        # Assume it's already mapped or map it now
        if df['player'].iloc[0] in player2index:
            # Contains names, map to IDs
            df['player'] = df['player'].map(player2index)
    
    for player_name, match_ids in split_content["train"].items():
        if player_name not in player2index:
            print(f"Warning: {player_name} not in player_index_map, skipping")
            continue
        
        player_id = player2index[player_name]
        selected_df = df[(df['player'] == player_id) & (df['match_id'].isin(match_ids))]
        
        if len(selected_df) == 0:
            print(f"Warning: No data found for {player_name}, skipping")
            continue
            
        state_data.extend(selected_df[state_col].values)
        action_data.extend(selected_df[action_col].values)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle
    state_path = os.path.join(output_dir, "train_state.pkl")
    action_path = os.path.join(output_dir, "train_action.pkl")
    
    with open(state_path, 'wb') as f:
        pickle.dump(np.array(state_data), f)
    with open(action_path, 'wb') as f:
        pickle.dump(np.array(action_data), f)
    
    print(f"BC data saved to {output_dir}")
    print(f"  - train_state.pkl: {len(state_data)} samples")
    print(f"  - train_action.pkl: {len(action_data)} samples")


def prepare_naive_rag_data(df, split_content, player2index, output_dir):
    """
    為每個 train 中的玩家構建 NaiveRAGDatabase。
    
    Args:
        df: DataFrame with merged data (player column should be player names)
        split_content: Train/test split dict
        player2index: Player name to ID mapping
        output_dir: Directory to save output files
    
    Output:
        - {player_name}_db.pkl: One database file per player
    """
    # Import here to avoid circular import issues
    from Agent.naive_rag import NaiveRAGDatabase
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all columns needed by NaiveRAGDatabase
    # These are based on state_col and action_col from Utils.helper
    required_cols = [
        'player',  # Must be player names, not IDs
        'match_id',
        # state_col columns
        'rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
        'player_location_x', 'player_location_y',
        'ball_distance', 'ball_distance_x', 'ball_distance_y', 'hit_x', 'hit_y',
        'opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y',
        # action_col columns
        'landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y'
    ]
    
    # Check if all required columns are present
    available_cols = [c for c in required_cols if c in df.columns]
    missing_cols = set(required_cols) - set(available_cols)
    if missing_cols:
        print(f"Warning: Missing columns in DataFrame: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
    
    # Ensure player column contains names (strings), not IDs
    # If df['player'] contains IDs, we need to map them back to names
    if df['player'].dtype in ['int64', 'float64'] or not df['player'].dtype == object:
        print("Warning: player column appears to contain IDs, mapping back to names...")
        index2player = {v: k for k, v in player2index.items()}
        df = df.copy()
        df['player'] = df['player'].map(index2player)
    
    # Filter to only include required columns that exist
    df_filtered = df[available_cols].copy()
    
    # Build database for each player in training set
    train_players = split_content.get('train', {})
    successful_builds = 0
    
    for player_name in tqdm(train_players.keys(), desc="Building NaiveRAG databases"):
        if player_name not in player2index:
            print(f"Warning: {player_name} not in player_index_map, skipping")
            continue
        
        player_id = player2index[player_name]
        
        # Filter data for this player (df has player names)
        player_data = df_filtered[df_filtered['player'] == player_name].copy()
        
        if len(player_data) == 0:
            print(f"Warning: No data found for {player_name}, skipping")
            continue
        
        try:
            # Create database - pass all data, NaiveRAGDatabase will filter internally
            database = NaiveRAGDatabase(df_filtered, player_name, player_id)
            
            # Save database
            safe_name = player_name.replace(" ", "_").replace(".", "")
            output_path = os.path.join(output_dir, f"{safe_name}_db.pkl")
            database.save(output_path)
            successful_builds += 1
        except Exception as e:
            print(f"Error building database for {player_name}: {e}")
            continue
    
    print(f"\nSuccessfully built {successful_builds} NaiveRAG databases")
    print(f"All databases saved to: {output_dir}")


def prepared_rallynet_data(all_data, split_content):
    """
    Prepare data for RallyNet (保留原有功能).
    
    Note: This function is kept for compatibility but may need updates
    based on specific RallyNet requirements.
    """
    all_player_trajectories_data = {}
    player_trajectories_data = []
    player_trajectory_data = []
    player_state_action_pair = []
    player_state_list = []
    player_action_list = []

    for index, row in all_data.iterrows():
        current_state = [row[col] for col in state_col]
        current_action = [row[col] for col in action_col]
        
        player_state_action_pair = current_state + current_action
        player_trajectory_data.append(player_state_action_pair)

        nan_flag = np.isnan(row['server'])
        if row['server'] == 3 or nan_flag:
            player_trajectories_data.append(player_trajectory_data.copy())
            player_trajectory_data = []  # 清空以供下一個軌跡使用

    # Note: target_players is not defined in the original code
    # This may need to be fixed based on actual requirements
    print("Warning: prepared_rallynet_data() may need updates for target_players")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline data preparation')
    parser.add_argument('--bc', action='store_true', 
                        help='Prepare BC data')
    parser.add_argument('--naive-rag', action='store_true', 
                        help='Prepare NaiveRAG databases')
    parser.add_argument('--all', action='store_true', 
                        help='Prepare all baseline data')
    parser.add_argument('--train_split', type=str,
                        default='Data/intermediate_data/train_test_split.json',
                        help='Path to train test split JSON')
    parser.add_argument('--merged_csv', type=str,
                        default='Data/intermediate_data/merged.csv',
                        help='Path to merged CSV data')
    parser.add_argument('--player_map', type=str,
                        default='Data/intermediate_data/player_index_map.json',
                        help='Path to player index mapping')
    parser.add_argument('--bc_output', type=str,
                        default='Experiments/bc/data',
                        help='Output directory for BC data')
    parser.add_argument('--naive_rag_output', type=str,
                        default='Experiments/naive_rag/data',
                        help='Output directory for NaiveRAG databases')
    
    args = parser.parse_args()
    
    # If no specific flag is set, default to --all
    if not (args.bc or args.naive_rag or args.all):
        args.all = True
    
    print("="*60)
    print("Baseline Data Preparation")
    print("="*60)
    
    # Load common data
    print("Loading data...")
    with open(args.train_split, 'r') as f:
        split_content = json.load(f)
    with open(args.player_map, 'r') as f:
        player2index = json.load(f)
    
    df = pd.read_csv(args.merged_csv)
    
    # Filter valid columns and remove NaN
    # For BC, we need state_col + action_col + match_id + player
    # For NaiveRAG, we need the same columns but player must be names (not IDs)
    base_cols = state_col + action_col + ['match_id']
    available_base_cols = [c for c in base_cols if c in df.columns]
    df = df[available_base_cols].dropna(subset=state_col + action_col).reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows from {args.merged_csv}")
    print(f"Training players: {len(split_content.get('train', {}))}")
    print()
    
    # Execute selected preprocessing
    if args.all or args.bc:
        print("-"*60)
        print("Preparing BC data...")
        print("-"*60)
        # For BC, we need player mapped to IDs
        df_bc = df.copy()
        df_bc['player'] = df_bc['player'].map(player2index)
        prepare_bc_data(df_bc, split_content, player2index, args.bc_output)
        print()
    
    if args.all or args.naive_rag:
        print("-"*60)
        print("Preparing NaiveRAG databases...")
        print("-"*60)
        # For NaiveRAG, we keep player names
        prepare_naive_rag_data(df, split_content, player2index, args.naive_rag_output)
        print()
    
    print("="*60)
    print("Data preparation complete!")
    print("="*60)
