import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from torch.nn import CTCLoss
from tqdm import tqdm
from dtaidistance import dtw_ndim

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.helper import *


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


def embed_state_for_sde(state_tensor, inverse_model):
    """
    Convert 18-dim state to 32-dim embedded state using inverse_model's embed_and_transform.
    state_tensor: [..., 18] -> [..., 32]
    """
    # Extract shot_type (index 12) and player_id (index 17)
    shot_types = state_tensor[..., 12].long()
    player_ids = state_tensor[..., 17].long()
    
    # Get embeddings
    shot_embeds = inverse_model.shot_embedding(shot_types)  # [..., 8]
    player_embeds = inverse_model.player_embedding(player_ids)  # [..., 8]
    
    # Concatenate: [0:12] + shot_embeds(8) + [13:17](4) + player_embeds(8) = 32
    # state[:, :12] + shot_embeds + state[:, 13:17] + player_embeds
    state_features = torch.cat([
        state_tensor[..., :12],  # 12 dims
        shot_embeds,              # 8 dims
        state_tensor[..., 13:17], # 4 dims
        player_embeds             # 8 dims
    ], dim=-1)  # Total: 32 dims
    
    return state_features


def evaluate_rallynetv2_on_test(test_split_path, merged_csv_path, player_index_map_path,
                                 model_path, hyperparameters_path, device='cuda', skip_first_n=0):
    """
    Evaluate RallyNetV2 model on test set using per-rally simulation.
    Similar structure to BC evaluation but adapted for SDE-based model.
    """
    
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    with open(player_index_map_path, 'r') as f:
        player2index = json.load(f)
    
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    results = {}
    
    # Load hyperparameters to get model dimensions
    with open(hyperparameters_path, 'rb') as f:
        hyperparameters = pickle.load(f)
    
    # Import model class
    from Agent.rallynetv2 import LatentSDE
    
    model = LatentSDE(
        data_size=hyperparameters['data_size'],
        latent_size=hyperparameters['latent_size'],
        context_size=hyperparameters['context_size'],
        hidden_size=hyperparameters['hidden_size'],
        player_ids_len=hyperparameters['player_ids_len'],
        shot_type_len=hyperparameters['shot_type_len']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
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
                
                current_state = build_state_tensor_from_row(first_row, current_player_id).to(device)
                
                # Initialize ht tensor with zeros for the entire rally
                # ht shape: [rally_length, 1, 18] - only store state (action affects next state)
                rally_length = len(rally_df)
                ht = torch.zeros(rally_length, 1, 18, device=device)
                
                rally_done = False
                shot_idx = 0
                
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == target_player_name)
                    current_player_id = target_player_id if is_target_turn else opponent_player_id
                    
                    current_state[17] = current_player_id
                    
                    launch = (current_state[1].item() == 1)
                    
                    # Update ht at current step with current state
                    ht[shot_idx, :, :] = current_state
                    
                    with torch.no_grad():
                        # Create time steps for current sequence length
                        ts = torch.linspace(0, 71, steps=71, device=device)
                        min_val = torch.min(ts)
                        max_val = torch.max(ts)
                        normalized_tensor = (ts - min_val) / (max_val - min_val)
                        ts = normalized_tensor * 2
                        
                        # Use model.sample to get predictions with SDE sampling
                        # Pass ht[:seq_len] which has shape [seq_len, 1, 18]
                        _xs, inv_action, inv_shot_logit, mlp_action, mlp_shot_logit = model.sample(
                            ht=ht,
                            ts=ts,
                            step=shot_idx,
                            player_ids_len=hyperparameters['player_ids_len'],
                            shot_type_len=hyperparameters['shot_type_len']
                        )
                        # Extract predictions from mlp_action (shape: [5])
                        land_logit = mlp_action[0:2].unsqueeze(0)  # [1, 2]
                        shot_type = mlp_action[2].long().unsqueeze(0)  # [1]
                        move_logit = mlp_action[3:5].unsqueeze(0)  # [1, 2]
                    
                    # Apply mask for valid shot types
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    # Re-apply mask to shot_probs and resample if needed
                    shot_probs_masked = mlp_shot_logit * mask.float()
                    shot_probs_masked = shot_probs_masked / (shot_probs_masked.sum() + 1e-8)
                    shot_type_masked = torch.argmax(shot_probs_masked).unsqueeze(0)
                    
                    if is_target_turn:
                        target_player_shot_count += 1
                        if target_player_shot_count >= skip_first_n:
                            rally_land_pred_seq.append(land_logit.squeeze(0).cpu().numpy())
                            gt_action = build_action_tensor_from_row(row)
                            rally_land_gt_seq.append(gt_action[0:2].numpy())
                            rally_move_pred_seq.append(move_logit.squeeze(0).cpu().numpy())
                            rally_move_gt_seq.append(gt_action[3:5].numpy())
                            rally_shot_logits_seq.append(mlp_shot_logit.squeeze(0).cpu())
                            rally_shot_gt_seq.append(int(gt_action[2].item()))
                            rally_shot_pred_seq.append(int(shot_type_masked.item()))
                            
                    
                    # Build action tensor for transition
                    action_tensor = torch.tensor([
                        land_logit[0, 0].item(), land_logit[0, 1].item(),
                        float(shot_type_masked.item()),
                        move_logit[0, 0].item(), move_logit[0, 1].item()
                    ], dtype=torch.float32)
                    
                    # Update state (action affects next state through transition)
                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)

                    # print(action_tensor)

                    if terminate(next_state.cpu()):
                        rally_done = True
                    
                    current_state = next_state.to(device)
                    
                    shot_idx += 1
                
                # Calculate rally-level metrics
                if len(rally_land_gt_seq) > 0:
                    land_pred_array = np.array(rally_land_pred_seq)
                    land_gt_array = np.array(rally_land_gt_seq)
                    land_dtw = dtw_ndim.distance(land_pred_array, land_gt_array)
                    rally_land_dtw_list.append(land_dtw)
                    
                    move_pred_array = np.array(rally_move_pred_seq)
                    move_gt_array = np.array(rally_move_gt_seq)
                    move_dtw = dtw_ndim.distance(move_pred_array, move_gt_array)
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
        
        # Store results for this player
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
    """Print evaluation results in formatted table"""
    print("\n" + "="*105)
    print("RallyNetV2 Evaluation Results on Test Set (Per-Rally Basis)")
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


def eval_model(args):
    """Evaluate RallyNetV2 model"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.skip_first_n > 0:
        print(f"Skipping first {args.skip_first_n} shots in evaluation")
    
    results = evaluate_rallynetv2_on_test(
        args.test_split,
        args.merged_csv,
        args.player_map,
        args.model_path,
        args.hyperparameters,
        device,
        args.skip_first_n
    )
    
    print_results(results)
    
    # Save results
    os.makedirs(os.path.dirname(args.eval_output), exist_ok=True)
    with open(args.eval_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.eval_output}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RallyNetV2 Model Evaluation')
    
    # Model paths
    parser.add_argument('--model_path', type=str, 
                        default='Experiments/rallynetv2/weight/gen_e_30000.trc',
                        help='Path to RallyNetV2 model weights (.trc file)')
    parser.add_argument('--hyperparameters', type=str,
                        default='Experiments/rallynetv2/hyperparameters.pkl',
                        help='Path to hyperparameters pickle file')
    
    # Data paths
    parser.add_argument('--test_split', type=str,
                        default='Data/intermediate_data/train_test_split.json',
                        help='Path to train test split JSON')
    parser.add_argument('--merged_csv', type=str,
                        default='Data/intermediate_data/merged.csv',
                        help='Path to merged CSV data')
    parser.add_argument('--player_map', type=str,
                        default='Data/intermediate_data/player_index_map.json',
                        help='Path to player index mapping')
    
    # Evaluation settings
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--eval_output', type=str, 
                        default='Experiments/rallynetv2/results/rallynetv2_eval_results.json',
                        help='Output JSON file for evaluation results')
    parser.add_argument('--skip_first_n', type=int, default=0,
                        help='Number of initial shots to skip in evaluation (default: 0)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("RallyNetV2 Model Evaluation Mode")
    print("="*50)
    eval_model(args)
