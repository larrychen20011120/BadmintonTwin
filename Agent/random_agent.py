import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
import pandas as pd
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.helper import *


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomAgent(BaseAgent):

    def __init__(self, player_name, seed=None):
        super().__init__(player_name)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def predict(self, state, info, launch):
        hit = state[-1]
        output = self.rng.rand(4,) * 2 - 1
        output_shot = self.rng.randint(11, size=1)[0] + 1

        uniform_dist = [0.1] * 10
        action = (output_shot, hit, tuple(output[:2]), tuple(output[2:]), uniform_dist)

        return action, ""


def build_state_tensor_from_row(row, player_id):
    """
    Build state tensor from dataframe row using state_col ordering.
    """
    state_values = [row[col] for col in state_col]
    state_values[-1] = player_id
    return torch.tensor(state_values, dtype=torch.float32)


def build_action_tensor_from_row(row):
    """
    Build action tensor from dataframe row using action_col ordering.
    """
    action_values = [row[col] for col in action_col]
    return torch.tensor(action_values, dtype=torch.float32)


def state_to_env_format(state_18):
    """
    Convert 18-dim state to 8-dim env format for RandomAgent.
    """
    opponent_type = int(state_18[12].item())
    player_pos = (state_18[5].item(), state_18[6].item())
    opponent_pos = (state_18[13].item(), state_18[14].item())
    landing_pos = (state_18[10].item(), state_18[11].item())
    return [opponent_type, player_pos, opponent_pos, landing_pos]


def evaluate_random_on_test(test_split_path, merged_csv_path, player_index_map_path, seed=None, skip_first_n=0):
    """
    Evaluate Random Agent on test set.
    """
    if seed is not None:
        set_random_seed(seed)
    
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    with open(player_index_map_path, 'r') as f:
        player2index = json.load(f)
    
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    results = {}
    ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    player_items = list(test_split['test'].items())
    for target_player_name, test_match_ids in tqdm(player_items, desc="Evaluating players", position=0):
        if target_player_name not in player2index:
            continue
            
        target_player_id = player2index[target_player_name]
        
        rally_land_dtw_list = []
        rally_move_dtw_list = []
        rally_shot_ctc_list = []
        rally_edit_dist_list = []
        total_rallies = 0
        total_shots = 0
        
        for match_id in tqdm(test_match_ids, desc=f"Matches for {target_player_name}", position=1, leave=False):
            match_df = df[(df['match_id'] == match_id)].copy()
            if len(match_df) == 0:
                continue
            
            players_in_match = match_df['player'].unique()
            opponent_name = None
            for p in players_in_match:
                if p != target_player_name:
                    opponent_name = p
                    break
            
            if opponent_name is None or opponent_name not in player2index:
                continue
                
            opponent_player_id = player2index[opponent_name]
            
            rallies = match_df.groupby(['set', 'rally'])
            
            for (set_num, rally_num), rally_df in rallies:
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                
                if len(rally_df) == 0:
                    continue
                
                rally_land_pred_seq = []
                rally_land_gt_seq = []
                rally_move_pred_seq = []
                rally_move_gt_seq = []
                rally_shot_logits_seq = []
                rally_shot_gt_seq = []
                rally_shot_pred_seq = []
                
                target_player_shot_count = 0
                
                first_row = rally_df.iloc[0]
                current_player_name = first_row['player']
                current_player_id = target_player_id if current_player_name == target_player_name else opponent_player_id
                
                current_state = build_state_tensor_from_row(first_row, current_player_id)
                
                rally_done = False
                shot_idx = 0
                
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == target_player_name)
                    current_player_id = target_player_id if is_target_turn else opponent_player_id
                    
                    current_state[17] = current_player_id
                    
                    launch = (current_state[1].item() == 1)
                    
                    env_state = state_to_env_format(current_state)
                    
                    # Create RandomAgent for current player with seed
                    random_agent = RandomAgent(ground_truth_player, seed=seed)
                    
                    action, _ = random_agent.predict(env_state, {'rally': 0, 'round': [0], 'score': [0, 0], 'action': [None]}, launch)
                    
                    pred_shot_type = action[0]
                    pred_land = np.array(action[2])
                    pred_move = np.array(action[3])
                    
                    if is_target_turn:
                        target_player_shot_count += 1
                        if target_player_shot_count >= skip_first_n:
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
                            
                            dummy_logits = torch.zeros(12)
                            dummy_logits[pred_shot_type] = 1.0
                            rally_shot_logits_seq.append(dummy_logits)
                    
                    action_tensor = torch.tensor([
                        pred_land[0], pred_land[1],
                        float(pred_shot_type),
                        pred_move[0], pred_move[1]
                    ], dtype=torch.float32)
                    
                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor, current_state, other_player_id)
                    
                    if terminate(next_state):
                        rally_done = True
                    
                    current_state = next_state
                    shot_idx += 1
                
                if len(rally_land_gt_seq) > 0:
                    land_pred_array = np.array(rally_land_pred_seq)
                    land_gt_array = np.array(rally_land_gt_seq)
                    land_dtw = normalized_dtw_ndim_distance(land_pred_array, land_gt_array)
                    rally_land_dtw_list.append(land_dtw)
                    
                    move_pred_array = np.array(rally_move_pred_seq)
                    move_gt_array = np.array(rally_move_gt_seq)
                    move_dtw = normalized_dtw_ndim_distance(move_pred_array, move_gt_array)
                    rally_move_dtw_list.append(move_dtw)
                    
                    rally_shot_logits_tensor = torch.stack(rally_shot_logits_seq)
                    rally_shot_targets_tensor = torch.tensor(rally_shot_gt_seq, dtype=torch.long)
                    
                    log_probs = F.log_softmax(rally_shot_logits_tensor.unsqueeze(1), dim=-1)
                    input_length = torch.tensor([log_probs.shape[0]], dtype=torch.long)
                    target_length = torch.tensor([len(rally_shot_targets_tensor)], dtype=torch.long)
                    
                    rally_ctc = ctc_loss_fn(log_probs, rally_shot_targets_tensor, input_length, target_length)
                    rally_shot_ctc_list.append(rally_ctc.item())
                    
                    rally_edit_dist = edit_distance(rally_shot_pred_seq, rally_shot_gt_seq)
                    rally_edit_dist_list.append(rally_edit_dist)
                    
                    total_rallies += 1
                    total_shots += len(rally_shot_gt_seq)
        
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
    """Print evaluation results"""
    print("\n" + "="*105)
    print("Random Agent Evaluation Results on Test Set (Per-Rally Basis)")
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


def run_evaluation(args):
    """Run evaluation on test set."""
    print("="*50)
    print("Random Agent Evaluation Mode")
    print("="*50)
    
    if args.seed is not None:
        print(f"Random seed set to: {args.seed}")
    
    if args.skip_first_n > 0:
        print(f"Skipping first {args.skip_first_n} shots in evaluation")
    
    results = evaluate_random_on_test(
        args.test_split,
        args.merged_csv,
        args.player_map,
        seed=args.seed,
        skip_first_n=args.skip_first_n
    )
    
    print_results(results)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    import argparse
    from Utils.runtime_report import print_runtime_box
    
    parser = argparse.ArgumentParser(description='Random Agent - Simulate or Evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: None)')
    
    # Evaluation arguments
    parser.add_argument('--test_split', type=str, 
                        default='Data/intermediate_data/train_test_split.json',
                        help='Path to train test split JSON')
    parser.add_argument('--merged_csv', type=str, 
                        default='Data/intermediate_data/merged.csv',
                        help='Path to merged CSV data')
    parser.add_argument('--player_map', type=str,
                        default='Data/intermediate_data/player_index_map.json',
                        help='Path to player index mapping')
    parser.add_argument('--output', type=str, default='random_eval_results.json',
                        help='Output JSON file for evaluation results')
    parser.add_argument('--skip_first_n', type=int, default=0,
                        help='Number of initial shots to skip in evaluation (default: 0)')
    
    args = parser.parse_args()

    print_runtime_box(
        "Runtime Parameters: Agent/random_agent.py",
        [
            ("seed", args.seed),
            ("test_split", args.test_split),
            ("merged_csv", args.merged_csv),
            ("player_map", args.player_map),
            ("output", args.output),
            ("skip_first_n", args.skip_first_n),
        ],
    )
    
    run_evaluation(args)
