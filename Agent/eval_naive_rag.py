import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Agent.naive_rag import NaiveRAGAgent, NaiveRAGDatabase
from Agent.verbalizer import create_verbalizer, verbalize_state, verbalize_experiences
from Utils.helper import state_col, action_col, transition, terminate, edit_distance


def build_state_tensor_from_row(row, player_id):
    """Build state tensor from dataframe row."""
    state_values = [row[col] for col in state_col]
    state_values[-1] = player_id  # state_col already includes 'player' as last element
    return torch.tensor(state_values, dtype=torch.float32)


def build_action_tensor_from_row(row):
    """Build action tensor from dataframe row."""
    action_values = [row[col] for col in action_col]
    return torch.tensor(action_values, dtype=torch.float32)


def state_to_env_format(state_18):
    """Convert 18-dim state to 8-dim env format.
    
    Env format: [opponent_shot_type, player_pos, opponent_pos, landing_pos]
    State format: [rally, ball_round, player_score, opponent_score, score_status,
                   player_location_x, player_location_y, ball_distance, ball_distance_x, 
                   ball_distance_y, hit_x, hit_y, opponent_type, opponent_location_x, 
                   opponent_location_y, opponent_move_x, opponent_move_y, player]
    """
    opponent_type = int(state_18[12].item())  # index 12: opponent_type
    player_pos = (state_18[5].item(), state_18[6].item())  # index 5,6: player_location
    opponent_pos = (state_18[13].item(), state_18[14].item())  # index 13,14: opponent_location
    landing_pos = (state_18[10].item(), state_18[11].item())  # index 10,11: hit_x, hit_y
    return [opponent_type, player_pos, opponent_pos, landing_pos]


def build_state_dict_from_tensor(state_18, info=None):
    """Build state dictionary from 18-dim state tensor for verbalization.
    
    Args:
        state_18: 18-dim state tensor
        info: Optional additional info dict with rally, round, score info
    
    Returns:
        State dictionary suitable for verbalization
    """
    state_dict = {
        'rally': info.get('rally', 0) if info else 0,
        'ball_round': info.get('round', [1])[-1] if info and info.get('round') else int(state_18[1].item()),
        'player_score': info.get('score', [0, 0])[0] if info else int(state_18[2].item()),
        'opponent_score': info.get('score', [0, 0])[1] if info else int(state_18[3].item()),
        'score_status': info.get('score', [0, 0])[0] - info.get('score', [0, 0])[1] if info else int(state_18[4].item()),
        'player_location_x': float(state_18[5].item()),
        'player_location_y': float(state_18[6].item()),
        'ball_distance': float(state_18[7].item()),
        'ball_distance_x': float(state_18[8].item()),
        'ball_distance_y': float(state_18[9].item()),
        'hit_x': float(state_18[10].item()),
        'hit_y': float(state_18[11].item()),
        'opponent_type': float(state_18[12].item()),
        'opponent_location_x': float(state_18[13].item()),
        'opponent_location_y': float(state_18[14].item()),
        'opponent_move_x': float(state_18[15].item()),
        'opponent_move_y': float(state_18[16].item()),
    }
    return state_dict


def get_verbalized_state_and_cases(state_18, database, info=None, top_k=5):
    """Get verbalized state string and retrieved cases string for a given state.
    
    This function is useful for debugging and analysis purposes to see
    how the verbalizer converts states and retrieved experiences.
    
    Args:
        state_18: 18-dim state tensor
        database: NaiveRAGDatabase instance for retrieval
        info: Optional additional info dict
        top_k: Number of similar experiences to retrieve
    
    Returns:
        Tuple of (state_str, cases_str, experiences) - verbalized strings and raw experiences
    """
    # Build state dictionary
    state_dict = build_state_dict_from_tensor(state_18, info)
    
    # Build query for retrieval (similar to NaiveRAGAgent.predict)
    query_state = {
        'player_location_x': state_dict['player_location_x'],
        'player_location_y': state_dict['player_location_y'],
        'hit_x': state_dict['hit_x'],
        'hit_y': state_dict['hit_y'],
        'opponent_type': state_dict['opponent_type'],
        'opponent_location_x': state_dict['opponent_location_x'],
        'opponent_location_y': state_dict['opponent_location_y']
    }
    
    # Retrieve similar experiences
    experiences = database.retrieve(query_state, top_k=top_k)
    
    # Create verbalizer and generate strings
    verbalizer = create_verbalizer()
    state_str = verbalizer.verbalize_state(state_dict)
    cases_str = verbalizer.verbalize_experiences(experiences)
    
    return state_str, cases_str, experiences


def evaluate_naive_rag_on_test(test_split_path, merged_csv_path, player_index_map_path,
                               database_dir, model_name="Qwen/Qwen2.5-7B-Instruct",
                               api_key="EMPTY", port=16000, top_k=5, skip_first_n=0,
                               use_verbalizer=False, verbalized_prompt=False):
    """
    Evaluate NaiveRAG Agent on test set.
    
    Args:
        skip_first_n: Number of initial shots to skip in each rally (default: 0)
    """
    
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    with open(player_index_map_path, 'r') as f:
        player2index = json.load(f)
    
    # Load and filter data
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    results = {}
    ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Outer tqdm for players
    player_items = list(test_split['test'].items())
    for target_player_name, test_match_ids in tqdm(player_items, desc="Evaluating players", position=0):
        if target_player_name not in player2index:
            continue
        
        target_player_id = player2index[target_player_name]
        
        # Load database for target player
        safe_name = target_player_name.replace(" ", "_").replace(".", "")
        db_path = os.path.join(database_dir, f"{safe_name}_db.pkl")
        
        if not os.path.exists(db_path):
            print(f"Warning: Database not found for {target_player_name} at {db_path}")
            continue
        
        try:
            target_database = NaiveRAGDatabase.load(db_path)
        except Exception as e:
            print(f"Error loading database for {target_player_name}: {e}")
            continue
        
        # Create NaiveRAG agent for target player
        target_agent = NaiveRAGAgent(
            player_name=target_player_name,
            database=target_database,
            model_name=model_name,
            api_key=api_key,
            port=port,
            top_k=top_k,
            use_verbalizer=use_verbalizer,
            verbalized_prompt=verbalized_prompt
        )
        
        # Storage for metrics
        rally_land_dtw_list = []
        rally_move_dtw_list = []
        rally_shot_ctc_list = []
        rally_edit_dist_list = []
        total_rallies = 0
        total_shots = 0
        
        # Inner tqdm for matches
        for match_id in tqdm(test_match_ids, desc=f"Matches for {target_player_name}", position=1, leave=False):
            match_df = df[(df['match_id'] == match_id)].copy()
            if len(match_df) == 0:
                continue
            
            # Get opponent name
            players_in_match = match_df['player'].unique()
            opponent_name = None
            for p in players_in_match:
                if p != target_player_name:
                    opponent_name = p
                    break
            
            if opponent_name is None or opponent_name not in player2index:
                continue
            
            opponent_player_id = player2index[opponent_name]
            
            # Load opponent database if available
            opponent_safe_name = opponent_name.replace(" ", "_").replace(".", "")
            opponent_db_path = os.path.join(database_dir, f"{opponent_safe_name}_db.pkl")
            
            if os.path.exists(opponent_db_path):
                try:
                    opponent_database = NaiveRAGDatabase.load(opponent_db_path)
                    opponent_agent = NaiveRAGAgent(
                        player_name=opponent_name,
                        database=opponent_database,
                        model_name=model_name,
                        api_key=api_key,
                        port=port,
                        top_k=top_k,
                        use_verbalizer=use_verbalizer,
                        verbalized_prompt=verbalized_prompt
                    )
                except:
                    # Fallback: opponent also uses target's database
                    opponent_agent = target_agent
            else:
                # Fallback: opponent also uses target's database
                opponent_agent = target_agent
            
            # Group by rally
            rallies = match_df.groupby(['set', 'rally'])
            
            for (set_num, rally_num), rally_df in rallies:
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                
                if len(rally_df) == 0:
                    continue
                
                # Per-rally storage
                rally_land_pred_seq = []
                rally_land_gt_seq = []
                rally_move_pred_seq = []
                rally_move_gt_seq = []
                rally_shot_logits_seq = []
                rally_shot_gt_seq = []
                rally_shot_pred_seq = []
                
                # Counter for skip_first_n
                target_player_shot_count = 0
                
                # Get first row
                first_row = rally_df.iloc[0]
                current_player_name = first_row['player']
                current_player_id = target_player_id if current_player_name == target_player_name else opponent_player_id
                
                # Initialize state
                current_state = build_state_tensor_from_row(first_row, current_player_id)

                rally_done = False
                shot_idx = 0
                
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == target_player_name)
                    current_player_id = target_player_id if is_target_turn else opponent_player_id
                    
                    # Update player_id
                    current_state[17] = current_player_id
                    
                    # Check if serve
                    launch = (current_state[1].item() == 1)
                    
                    # Convert to env format
                    env_state = state_to_env_format(current_state)
                    
                    # Get current agent
                    current_agent = target_agent if is_target_turn else opponent_agent
                    
                    # Predict
                    try:
                        action, reasoning = current_agent.predict(
                            env_state,
                            {'rally': rally_num, 'round': [shot_idx], 'score': [int(row['player_score']), int(row['opponent_score'])], 'action': [None]},
                            launch
                        )
                        
                        pred_shot_type = action[0]
                        pred_land = np.array(action[2])
                        pred_move = np.array(action[3])
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        # Fallback to random
                        pred_shot_type = np.random.randint(1, 12)
                        pred_land = np.random.rand(2) * 2 - 1
                        pred_move = np.random.rand(2) * 2 - 1
                    
                    # Record if target player
                    if is_target_turn:
                        target_player_shot_count += 1
                        if target_player_shot_count > skip_first_n:
                            gt_action = build_action_tensor_from_row(row)
                            gt_land = gt_action[0:2].numpy()
                            gt_move = gt_action[3:5].numpy()
                            gt_shot_type = int(gt_action[2].item())
                            
                            rally_land_pred_seq.append(pred_land)
                            rally_land_gt_seq.append(gt_land)
                            rally_move_pred_seq.append(pred_move)
                            rally_move_gt_seq.append(gt_move)
                            rally_shot_pred_seq.append(pred_shot_type)
                            rally_shot_gt_seq.append(gt_shot_type)

                            
                            
                            # Create logits for CTC
                            dummy_logits = torch.zeros(12)
                            dummy_logits[pred_shot_type] = 1.0
                            rally_shot_logits_seq.append(dummy_logits)
                    
                    # Build action for transition
                    action_tensor = torch.tensor([
                        pred_land[0], pred_land[1],
                        float(pred_shot_type),
                        pred_move[0], pred_move[1]
                    ], dtype=torch.float32)
                    
                    # Transition
                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                    
                    # Check termination
                    if terminate(next_state):
                        rally_done = True
                    
                    current_state = next_state
                    shot_idx += 1
                
                # Calculate per-rally metrics
                if len(rally_land_gt_seq) > 0:
                    land_pred_array = np.array(rally_land_pred_seq)
                    land_gt_array = np.array(rally_land_gt_seq)
                    land_dtw = dtw_ndim.distance(land_pred_array, land_gt_array) / len(land_gt_array)
                    rally_land_dtw_list.append(land_dtw)
                    
                    move_pred_array = np.array(rally_move_pred_seq)
                    move_gt_array = np.array(rally_move_gt_seq)
                    move_dtw = dtw_ndim.distance(move_pred_array, move_gt_array) / len(move_gt_array)
                    rally_move_dtw_list.append(move_dtw)
                    
                    # CTC loss
                    rally_shot_logits_tensor = torch.stack(rally_shot_logits_seq)
                    rally_shot_targets_tensor = torch.tensor(rally_shot_gt_seq, dtype=torch.long)
                    
                    log_probs = F.log_softmax(rally_shot_logits_tensor.unsqueeze(1), dim=-1)
                    input_length = torch.tensor([log_probs.shape[0]], dtype=torch.long)
                    target_length = torch.tensor([len(rally_shot_targets_tensor)], dtype=torch.long)
                    
                    rally_ctc = ctc_loss_fn(log_probs, rally_shot_targets_tensor, input_length, target_length)
                    rally_shot_ctc_list.append(rally_ctc.item())
                    
                    # Edit distance
                    rally_edit_dist = edit_distance(rally_shot_pred_seq, rally_shot_gt_seq)
                    rally_edit_dist_list.append(rally_edit_dist)
                    
                    total_rallies += 1
                    total_shots += len(rally_shot_gt_seq)

                    # print("Predict Shot: ", rally_shot_pred_seq)
                    # print("Answer  Shot: ", rally_shot_gt_seq)
        
        # Calculate average scores
        if total_rallies > 0:
            avg_land_dtw = np.mean(rally_land_dtw_list)
            avg_move_dtw = np.mean(rally_move_dtw_list)
            avg_shot_ctc = np.mean(rally_shot_ctc_list)
            avg_edit_dist = np.mean(rally_edit_dist_list)
            
            results[target_player_name] = {
                'land_dtw': float(avg_land_dtw),
                'move_dtw': float(avg_move_dtw),
                'shot_ctc': float(avg_shot_ctc),
                'edit_dist': float(avg_edit_dist),
                'num_rallies': total_rallies,
                'num_shots': total_shots
            }
        else:
            results[target_player_name] = {
                'land_dtw': float('inf'),
                'move_dtw': float('inf'),
                'shot_ctc': float('inf'),
                'edit_dist': float('inf'),
                'num_rallies': 0,
                'num_shots': 0
            }
    
    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*105)
    print("NaiveRAG Agent Evaluation Results on Test Set (Per-Rally Basis)")
    print("="*105)
    print(f"{'Player Name':<30} {'Land DTW':<12} {'Move DTW':<12} {'Shot CTC':<12} {'Edit Dist':<12} {'Rallies':<8} {'Shots':<8}")
    print("-"*105)
    
    total_land_dtw = []
    total_move_dtw = []
    total_shot_ctc = []
    total_edit_dist = []
    total_rallies = []
    total_shots = []
    
    for player_name, scores in sorted(results.items()):
        if scores['num_rallies'] > 0:
            print(f"{player_name:<30} {scores['land_dtw']:<12.4f} {scores['move_dtw']:<12.4f} "
                  f"{scores['shot_ctc']:<12.4f} {scores['edit_dist']:<12.4f} {scores['num_rallies']:<8} {scores['num_shots']:<8}")
            total_land_dtw.append(scores['land_dtw'])
            total_move_dtw.append(scores['move_dtw'])
            total_shot_ctc.append(scores['shot_ctc'])
            total_edit_dist.append(scores['edit_dist'])
            total_rallies.append(scores['num_rallies'])
            total_shots.append(scores['num_shots'])
    
    print("-"*105)
    if total_land_dtw:
        total_rallies_sum = sum(total_rallies)
        total_shots_sum = sum(total_shots)
        w_land = sum([s * n for s, n in zip(total_land_dtw, total_rallies)]) / total_rallies_sum
        w_move = sum([s * n for s, n in zip(total_move_dtw, total_rallies)]) / total_rallies_sum
        w_ctc = sum([s * n for s, n in zip(total_shot_ctc, total_rallies)]) / total_rallies_sum
        w_edit = sum([s * n for s, n in zip(total_edit_dist, total_rallies)]) / total_rallies_sum
        print(f"{'Weighted Average (by rallies)':<30} {w_land:<12.4f} {w_move:<12.4f} {w_ctc:<12.4f} {w_edit:<12.4f} "
              f"{total_rallies_sum:<8} {total_shots_sum:<8}")
        print(f"{'Simple Average':<30} {np.mean(total_land_dtw):<12.4f} {np.mean(total_move_dtw):<12.4f} "
              f"{np.mean(total_shot_ctc):<12.4f} {np.mean(total_edit_dist):<12.4f}")
    print("="*105)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate NaiveRAG Agent on test set')
    parser.add_argument('--test_split', type=str,
                        default='Data/intermediate_data/train_test_split.json',
                        help='Path to train test split JSON')
    parser.add_argument('--merged_csv', type=str,
                        default='Data/intermediate_data/merged.csv',
                        help='Path to merged CSV data')
    parser.add_argument('--player_map', type=str,
                        default='Data/intermediate_data/player_index_map.json',
                        help='Path to player index mapping')
    parser.add_argument('--database_dir', type=str,
                        default='Experiments/naive_rag/data/norm',
                        help='Directory containing NaiveRAG databases')
    parser.add_argument('--model_name', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='LLM model name')
    parser.add_argument('--api_key', type=str,
                        default='EMPTY',
                        help='API key for LLM')
    parser.add_argument('--port', type=int,
                        default=16000,
                        help='Port for local LLM server')
    parser.add_argument('--top_k', type=int,
                        default=5,
                        help='Number of similar experiences to retrieve')
    parser.add_argument('--skip_first_n', type=int,
                        default=0,
                        help='Number of initial shots to skip in each rally (default: 0)')
    parser.add_argument('--use_verbalizer', action='store_true',
                        help='Enable verbalizer to convert state and experiences to natural language')
    parser.add_argument('--verbalized_prompt', action='store_true',
                        help='Use verbalized prompt template (requires --use_verbalizer)')
    parser.add_argument('--output', type=str,
                        default='Experiments/naive_rag/results/naive_rag_eval_results.json',
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if args.skip_first_n > 0:
        print(f"Skipping first {args.skip_first_n} shots in each rally")
    
    if args.use_verbalizer:
        print("Verbalizer enabled")
        if args.verbalized_prompt:
            print("Using verbalized prompt template")
    
    results = evaluate_naive_rag_on_test(
        args.test_split,
        args.merged_csv,
        args.player_map,
        args.database_dir,
        args.model_name,
        args.api_key,
        args.port,
        args.top_k,
        args.skip_first_n,
        args.use_verbalizer,
        args.verbalized_prompt
    )
    
    print_results(results)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
