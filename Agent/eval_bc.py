import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.helper import terminate, transition, state_col, action_col, list_subtraction


class BC(nn.Module):
    """Behavioral Cloning Model - Copied from bc.py"""
    
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len, player_id):
        super(BC, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.player_id = player_id
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)
        player_embeds = self.player_embedding(player_ids)
        state_features = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)
        return state_features

    def forward(self, x):
        x = x.float()
        embedded = self.embed_and_transform(x)
        out = self.fc(embedded)
        out = self.relu(out)
        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)
        return land_logit, shot_logit, move_logit


def build_state_tensor_from_row(row, player_id):
    """
    Build state tensor from dataframe row using state_col ordering.
    state_col = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
                 'player_location_x', 'player_location_y',
                 'ball_distance','ball_distance_x','ball_distance_y', 'hit_x', 'hit_y',
                 'opponent_type', 'opponent_location_x', 'opponent_location_y', 
                 'opponent_move_x', 'opponent_move_y']
    Note: 'player' column is added separately as player_id
    """
    state_values = [row[col] for col in state_col]
    state_values[-1] = player_id  # Add player_id as the last column
    return torch.tensor(state_values, dtype=torch.float32)


def build_action_tensor_from_row(row):
    """
    Build action tensor from dataframe row using action_col ordering.
    action_col = ['landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y']
    """
    action_values = [row[col] for col in action_col]
    return torch.tensor(action_values, dtype=torch.float32)


def evaluate_bc_on_test(test_split_path, merged_csv_path, player_index_map_path, 
                        model_path, device='cuda'):
    """
    Evaluate BC model on test set using state_col/action_col approach.
    Both players use BC models to simulate rallies.
    Only evaluate the target player specified in train_test_split.json.
    """
    
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    with open(player_index_map_path, 'r') as f:
        player2index = json.load(f)
    
    # Load and filter data (same as bc.py training)
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    results = {}
    
    # Initialize model
    model = BC(input_size=32, hidden_size=128, player_id_len=43, shot_type_len=12, player_id=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Outer tqdm for players
    player_items = list(test_split['test'].items())
    for target_player_name, test_match_ids in tqdm(player_items, desc="Evaluating players", position=0):
        if target_player_name not in player2index:
            continue
            
        target_player_id = player2index[target_player_name]
        
        # Storage for sequences
        target_land_pred_seq = []
        target_land_gt_seq = []
        target_move_pred_seq = []
        target_move_gt_seq = []
        target_shot_logits_seq = []
        target_shot_gt_seq = []
        
        # Inner tqdm for matches
        for match_id in test_match_ids:
            match_df = df[(df['match_id'] == match_id)].copy()
            if len(match_df) == 0:
                continue
            
            # Get the opponent name for this match
            players_in_match = match_df['player'].unique()
            opponent_name = None
            for p in players_in_match:
                if p != target_player_name:
                    opponent_name = p
                    break
            
            if opponent_name is None or opponent_name not in player2index:
                continue
                
            opponent_player_id = player2index[opponent_name]
            
            # Group by rally
            rallies = match_df.groupby(['set', 'rally'])
            
            for (set_num, rally_num), rally_df in rallies:
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                
                if len(rally_df) == 0:
                    continue
                
                # Get first row to determine starting player
                first_row = rally_df.iloc[0]
                current_player_name = first_row['player']
                current_player_id = target_player_id if current_player_name == target_player_name else opponent_player_id
                
                # Initialize state from first row using state_col
                current_state = build_state_tensor_from_row(first_row, current_player_id).to(device)
                
                rally_done = False
                shot_idx = 0
                
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == target_player_name)
                    current_player_id = target_player_id if is_target_turn else opponent_player_id
                    
                    # Update player_id in state (last element)
                    current_state[17] = current_player_id
                    
                    # Check if serve (ball_round == 1)
                    launch = (current_state[1].item() == 1)  # ball_round == 1
                    
                    # Predict using BC model
                    with torch.no_grad():
                        bc_land, bc_shot_logit, bc_move = model(current_state.unsqueeze(0))
                    
                    # Apply mask
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    bc_shot_probs = F.softmax(bc_shot_logit, dim=-1)
                    masked_probs = bc_shot_probs.masked_fill(~mask, 0.0)
                    sum_probs = masked_probs.sum(dim=-1, keepdim=True)
                    bc_shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
                    bc_shot_type = torch.argmax(bc_shot_probs, dim=-1)
                    
                    # Record if this is target player
                    if is_target_turn:
                        target_land_pred_seq.append(bc_land.squeeze(0).cpu().numpy())
                        # Ground truth action using action_col
                        gt_action = build_action_tensor_from_row(row)
                        target_land_gt_seq.append(gt_action[0:2].numpy())  # landing_x, landing_y
                        target_move_pred_seq.append(bc_move.squeeze(0).cpu().numpy())
                        target_move_gt_seq.append(gt_action[3:5].numpy())  # moving_x, moving_y
                        target_shot_logits_seq.append(bc_shot_logit.squeeze(0).cpu().numpy())
                        target_shot_gt_seq.append(int(gt_action[2].item()))  # player_type
                    
                    # Build action for transition [landing_x, landing_y, shot_type, move_x, move_y]
                    action_tensor = torch.tensor([
                        bc_land[0, 0].item(), bc_land[0, 1].item(),
                        float(bc_shot_type.item()),
                        bc_move[0, 0].item(), bc_move[0, 1].item()
                    ], dtype=torch.float32)
                    

                    # Transition to next state
                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)

                    # Check if rally terminates
                    if terminate(next_state.cpu()):
                        rally_done = True
                    
                    # Move to next shot
                    current_state = next_state.to(device)
                    shot_idx += 1
        
        # Calculate scores for this player
        if len(target_land_gt_seq) > 0:
            # DTW for land positions (2D continuous) - using dtw_ndim
            land_pred_array = np.array(target_land_pred_seq)  # [T, 2]
            land_gt_array = np.array(target_land_gt_seq)  # [T, 2]
            land_dtw = dtw_ndim.distance(land_pred_array, land_gt_array)
            
            # DTW for move positions (2D continuous) - using dtw_ndim
            move_pred_array = np.array(target_move_pred_seq)
            move_gt_array = np.array(target_move_gt_seq)
            move_dtw = dtw_ndim.distance(move_pred_array, move_gt_array)
            
            # CTC Loss for shot types
            # Stack all sequences - CTC expects log_probs of shape (T, N, C) 
            # where N is batch size (number of rallies/sequences)
            # For simplicity, we treat all shots as one long sequence
            shot_logits = torch.tensor(np.array(target_shot_logits_seq))  # [T, 12]
            shot_targets = torch.tensor(target_shot_gt_seq, dtype=torch.long)  # [T]
            
            # Log softmax for CTC (needs to be log probabilities)
            log_probs = F.log_softmax(shot_logits, dim=-1)  # [T, 12]
            
            # For CTC, we need:
            # - log_probs: (T, N, C) where T is max sequence length, N is batch size
            # - targets: (sum of target lengths) - concatenated targets
            # - input_lengths: (N,) - lengths of input sequences
            # - target_lengths: (N,) - lengths of target sequences
            
            # Here we treat the entire sequence as one batch item
            log_probs = log_probs.unsqueeze(1)  # [T, 1, 12] - (T, N, C) format
            
            input_lengths = torch.tensor([log_probs.size(0)])  # [T]
            target_lengths = torch.tensor([len(shot_targets)])  # [T]
            
            # CTC Loss with blank=0
            ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
            shot_ctc = ctc_loss_fn(log_probs, shot_targets, input_lengths, target_lengths)
            
            results[target_player_name] = {
                'land_dtw': float(land_dtw),
                'move_dtw': float(move_dtw),
                'shot_ctc': float(shot_ctc.item()),
                'num_samples': len(target_land_gt_seq)
            }
        else:
            results[target_player_name] = {
                'land_dtw': float('inf'),
                'move_dtw': float('inf'),
                'shot_ctc': float('inf'),
                'num_samples': 0
            }
    
    return results


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*80)
    print("BC Evaluation Results on Test Set")
    print("="*80)
    print(f"{'Player Name':<30} {'Land DTW':<12} {'Move DTW':<12} {'Shot CTC':<12} {'Samples':<10}")
    print("-"*80)
    
    total_land_dtw = []
    total_move_dtw = []
    total_shot_ctc = []
    total_samples = []
    
    for player_name, scores in sorted(results.items()):
        if scores['num_samples'] > 0:
            print(f"{player_name:<30} {scores['land_dtw']:<12.4f} {scores['move_dtw']:<12.4f} "
                  f"{scores['shot_ctc']:<12.4f} {scores['num_samples']:<10}")
            total_land_dtw.append(scores['land_dtw'])
            total_move_dtw.append(scores['move_dtw'])
            total_shot_ctc.append(scores['shot_ctc'])
            total_samples.append(scores['num_samples'])
    
    print("-"*80)
    if total_land_dtw:
        total_samples_sum = sum(total_samples)
        w_land = sum([s * n for s, n in zip(total_land_dtw, total_samples)]) / total_samples_sum
        w_move = sum([s * n for s, n in zip(total_move_dtw, total_samples)]) / total_samples_sum
        w_ctc = sum([s * n for s, n in zip(total_shot_ctc, total_samples)]) / total_samples_sum
        print(f"{'Weighted Average':<30} {w_land:<12.4f} {w_move:<12.4f} {w_ctc:<12.4f}")
        print(f"{'Simple Average':<30} {np.mean(total_land_dtw):<12.4f} {np.mean(total_move_dtw):<12.4f} "
              f"{np.mean(total_shot_ctc):<12.4f}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BC model on test set')
    parser.add_argument('--model_path', type=str, default='BC_weight.pth',
                        help='Path to BC model weights')
    parser.add_argument('--test_split', type=str, 
                        default='Data/offline_data/train_test_split.json',
                        help='Path to train test split JSON')
    parser.add_argument('--merged_csv', type=str, 
                        default='Data/input_data/merged.csv',
                        help='Path to merged CSV data')
    parser.add_argument('--player_map', type=str,
                        default='Data/offline_data/player_index_map.json',
                        help='Path to player index mapping')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output', type=str, default='bc_eval_results.json',
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, using CPU")
    
    results = evaluate_bc_on_test(
        args.test_split,
        args.merged_csv,
        args.player_map,
        args.model_path,
        args.device
    )
    
    print_results(results)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
