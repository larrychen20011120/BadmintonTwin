import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import CTCLoss
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.helper import *  


class BC(nn.Module):
    
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len, player_id,
                 distribution='deterministic', gmm_num=5, min_sigma=1e-3):
        super(BC, self).__init__()
        if distribution not in ('deterministic', 'gaussian', 'gmm'):
            raise ValueError('distribution must be deterministic, gaussian, or gmm')

        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.distribution = distribution
        self.gmm_num = gmm_num
        self.min_sigma = min_sigma
        
        self.player_id = player_id
        # Embeddings for categorical features within the state
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        # Gaussian predicts [mu_x, mu_y, log_sigma_x, log_sigma_y, raw_correlation].
        # GMM predicts [pi, mu, sigma] for K diagonal components.
        output_sizes = {'deterministic': 2, 'gaussian': 5, 'gmm': gmm_num * 5}
        land_output_size = output_sizes[distribution]
        move_output_size = output_sizes[distribution]

        self.predict_land_area = nn.Linear(hidden_size, land_output_size, bias=(distribution != 'deterministic'))
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, move_output_size, bias=(distribution != 'deterministic'))

    def embed_and_transform(self, state):
        # Assuming state[:, 12] is shot_type and state[:, 17] is player_id
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)       # Shape: [batch_size, 8]
        player_embeds = self.player_embedding(player_ids)   # Shape: [batch_size, 8]
        
        # Concatenate embeddings with other state features
        # Adjust indices based on your actual state structure
        state_features = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)  # Shape: [batch_size, 32]
        return state_features

    def forward(self, x):
        x = x.float()
        embedded = self.embed_and_transform(x)  # Shape: [batch_size, 32]
        out = self.fc(embedded)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)

        # Gaussian outputs a state-dependent correlated 2D distribution.
        # For gmm mode: outputs are raw mixture params [pi_logits, mu_flat, raw_sigma_flat].
        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]

        return land_logit, shot_logit, move_logit
    
    def translation(self, state, info): # Calculate the full states
        return_state_list = [0]*18
        
        # match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
        # player_state = ['player_location_x', 'player_location_y']
        # ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
        # opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y']

        # match_state
        if info['rally'] < 0:
            return_state_list[0] = 1
        else:
            return_state_list[0] = info['rally']
        
        return_state_list[1] = info['round'][-1]
        return_state_list[2] = info['score'][0] 
        return_state_list[3] = info['score'][1]
        return_state_list[4] = info['score'][0] - info['score'][1] # the difference score between the player and the opponent

        # player_state
        return_state_list[5] = state[1][0]
        return_state_list[6] = state[1][1]
        
        # ball_state
        return_state_list[8], return_state_list[9] = list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = state[0]
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        if info['action'][-1] != None:
            opponent_last_x = (info['action'][-1][3][0] / 177.5)
            opponent_last_y = (info['action'][-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        return_state_list[17] = self.player_id
    
        return torch.FloatTensor(return_state_list)
        
        

def gaussian_nll_loss(pred_params, target, reduction='mean'):
    """Compute correlated 2D Gaussian NLL using ShuttleNet's parameterization."""
    if pred_params.shape[-1] != 5:
        raise ValueError('Gaussian prediction must have 5 channels [mu_x, mu_y, log_sx, log_sy, raw_corr]')
    mu_x = pred_params[:, 0]
    mu_y = pred_params[:, 1]
    log_sigma_x = torch.clamp(pred_params[:, 2], min=-7.0, max=7.0)
    log_sigma_y = torch.clamp(pred_params[:, 3], min=-7.0, max=7.0)
    correlation = torch.tanh(pred_params[:, 4])
    sigma_x = torch.exp(log_sigma_x)
    sigma_y = torch.exp(log_sigma_y)
    dx = target[:, 0] - mu_x
    dy = target[:, 1] - mu_y
    one_minus_corr2 = torch.clamp(1 - correlation ** 2, min=1e-5)
    z = (
        (dx / sigma_x) ** 2
        + (dy / sigma_y) ** 2
        - 2 * correlation * dx * dy / (sigma_x * sigma_y)
    )
    nll = (
        torch.log(2 * torch.pi * sigma_x * sigma_y * torch.sqrt(one_minus_corr2))
        + z / (2 * one_minus_corr2)
    )
    if reduction == 'none':
        return nll
    if reduction == 'sum':
        return nll.sum()
    if reduction == 'mean':
        return nll.mean()
    raise ValueError('reduction must be none, sum, or mean')


def sample_from_gaussian(pred_params):
    """Sample from a state-dependent correlated 2D Gaussian."""
    if pred_params.shape[-1] != 5:
        raise ValueError('Gaussian prediction must have 5 channels [mu_x, mu_y, log_sx, log_sy, raw_corr]')

    mu = pred_params[:, 0:2]
    sigma_x = torch.exp(torch.clamp(pred_params[:, 2], min=-7.0, max=7.0))
    sigma_y = torch.exp(torch.clamp(pred_params[:, 3], min=-7.0, max=7.0))
    correlation = torch.tanh(pred_params[:, 4])
    eps_x = torch.randn_like(sigma_x)
    eps_y = torch.randn_like(sigma_y)
    sample_x = mu[:, 0] + sigma_x * eps_x
    sample_y = mu[:, 1] + sigma_y * (
        correlation * eps_x
        + torch.sqrt(torch.clamp(1 - correlation ** 2, min=1e-5)) * eps_y
    )
    return torch.clamp(torch.stack([sample_x, sample_y], dim=-1), -1.0, 1.0)


def split_gmm_params(pred_params, gmm_num, min_sigma=1e-3):
    """
    Split raw GMM head output into mixture logits, means, and positive stds.

    Layout follows RallyNet's compact format:
    [pi_logits(K), mu_x/mu_y(K*2), raw_sigma_x/raw_sigma_y(K*2)].
    """
    expected_dim = gmm_num * 5
    if pred_params.shape[1] != expected_dim:
        raise ValueError(f'GMM prediction must have {expected_dim} channels for {gmm_num} components')

    pi_logits = pred_params[:, :gmm_num]
    mus = pred_params[:, gmm_num:gmm_num * 3].reshape(-1, gmm_num, 2)
    raw_sigmas = pred_params[:, gmm_num * 3:gmm_num * 5].reshape(-1, gmm_num, 2)
    sigmas = F.softplus(raw_sigmas) + min_sigma
    return pi_logits, mus, sigmas


def gmm_nll_loss(pred_params, target, gmm_num, min_sigma=1e-3):
    """Compute NLL for a diagonal 2D Gaussian mixture."""
    pi_logits, mus, sigmas = split_gmm_params(pred_params, gmm_num, min_sigma)
    target = target.unsqueeze(1)

    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_component_prob = -0.5 * (
        ((target - mus) / sigmas) ** 2 + 2 * torch.log(sigmas) + np.log(2 * np.pi)
    ).sum(dim=-1)
    log_prob = torch.logsumexp(log_pi + log_component_prob, dim=-1)
    return -log_prob.mean()


def sample_from_gmm(pred_params, gmm_num, min_sigma=1e-3):
    """Sample a 2D point from a diagonal Gaussian mixture."""
    pi_logits, mus, sigmas = split_gmm_params(pred_params, gmm_num, min_sigma)
    component = Categorical(logits=pi_logits).sample()
    gather_index = component.view(-1, 1, 1).expand(-1, 1, 2)
    selected_mu = mus.gather(1, gather_index).squeeze(1)
    selected_sigma = sigmas.gather(1, gather_index).squeeze(1)
    sample = selected_mu + torch.randn_like(selected_mu) * selected_sigma
    return torch.clamp(sample, -1.0, 1.0)


class BC_Agent(BaseAgent):

    def __init__(self, player_name, model_path, player_id, distribution='deterministic', device='cuda', gmm_num=5):
        super().__init__(player_name)
        self.distribution = distribution
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = BC(
            input_size=32,
            hidden_size=128,
            player_id_len=43,
            shot_type_len=12,
            player_id=player_id,
            distribution=distribution,
            gmm_num=gmm_num
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, state, info, launch):
        # encode the state
        normalized_state = self.encode_state(state)
        bc_state = self.model.translation(normalized_state, info).to(self.device)

        with torch.no_grad():
            bc_land_pred, bc_shot_logit, bc_move_pred = self.model(bc_state.unsqueeze(0))

        bc_shot_type_probs = F.softmax(bc_shot_logit, dim=-1)

        if launch:
            mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool, device=self.device)
        else:
            mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool, device=self.device)

        masked_probs = bc_shot_type_probs.masked_fill(~mask, 0.0)
        sum_probs = masked_probs.sum(dim=-1, keepdim=True)
        bc_shot_type_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())

        bc_shot_type_dist = Categorical(bc_shot_type_probs)
        bc_shot_type = bc_shot_type_dist.sample()
        output_shot = bc_shot_type.unsqueeze(1).item()
        output_shot_dist = bc_shot_type_probs.tolist()[0]

        if self.distribution == 'gmm':
            output_land = sample_from_gmm(
                bc_land_pred,
                self.model.gmm_num,
                self.model.min_sigma
            ).tolist()[0]
            output_move = sample_from_gmm(
                bc_move_pred,
                self.model.gmm_num,
                self.model.min_sigma
            ).tolist()[0]
        elif self.distribution == 'gaussian':
            output_land = sample_from_gaussian(bc_land_pred).tolist()[0]
            output_move = sample_from_gaussian(bc_move_pred).tolist()[0]
        else:
            output_land = bc_land_pred.tolist()[0]
            output_move = bc_move_pred.tolist()[0]

        if output_shot == 11:
            return None
        else:
            prob_array = np.array(output_shot_dist)
            prob_array = prob_array[1:11]
            normalized_array = prob_array / prob_array.sum()
            normalized_tuple = tuple(normalized_array)

            action = (output_shot, state[-1], output_land, output_move, normalized_tuple)
            action = self.decode_action(action)
            return action

class BCDataset(Dataset):
    def __init__(self, states, landing, shot, move):
        self.states = states
        self.landing = landing
        self.shot = shot
        self.move = move

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.landing[idx],
            self.shot[idx],
            self.move[idx]
        )


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


def evaluate_bc_on_test(test_split_path, merged_csv_path, player_index_map_path, 
                        model_path, distribution='deterministic', device='cuda', skip_first_n=0, gmm_num=5):
    """
    Evaluate BC model on test set using state_col/action_col approach.
    """
    
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    with open(player_index_map_path, 'r') as f:
        player2index = json.load(f)
    
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    results = {}
    
    model = BC(
        input_size=32,
        hidden_size=128,
        player_id_len=43,
        shot_type_len=12,
        player_id=0,
        distribution=distribution,
        gmm_num=gmm_num
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
                
                rally_done = False
                shot_idx = 0
                
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == target_player_name)
                    current_player_id = target_player_id if is_target_turn else opponent_player_id
                    
                    current_state[17] = current_player_id
                    
                    launch = (current_state[1].item() == 1)
                    
                    with torch.no_grad():
                        bc_land_params, bc_shot_logit, bc_move_params = model(current_state.unsqueeze(0))

                    if distribution == 'gmm':
                        bc_land = sample_from_gmm(
                            bc_land_params,
                            model.gmm_num,
                            model.min_sigma
                        )
                        bc_move = sample_from_gmm(
                            bc_move_params,
                            model.gmm_num,
                            model.min_sigma
                        )
                    elif distribution == 'gaussian':
                        bc_land = sample_from_gaussian(bc_land_params)
                        bc_move = sample_from_gaussian(bc_move_params)
                    else:
                        bc_land = bc_land_params
                        bc_move = bc_move_params
                    
                    if launch:
                        mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(device)
                    else:
                        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool).to(device)
                    
                    bc_shot_probs = F.softmax(bc_shot_logit, dim=-1)
                    masked_probs = bc_shot_probs.masked_fill(~mask, 0.0)
                    sum_probs = masked_probs.sum(dim=-1, keepdim=True)
                    bc_shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
                    bc_shot_type = torch.argmax(bc_shot_probs, dim=-1)
                    
                    if is_target_turn:
                        target_player_shot_count += 1
                        if target_player_shot_count >= skip_first_n:
                            rally_land_pred_seq.append(bc_land.squeeze(0).cpu().numpy())
                            gt_action = build_action_tensor_from_row(row)
                            rally_land_gt_seq.append(gt_action[0:2].numpy())
                            rally_move_pred_seq.append(bc_move.squeeze(0).cpu().numpy())
                            rally_move_gt_seq.append(gt_action[3:5].numpy())
                            rally_shot_logits_seq.append(bc_shot_logit.squeeze(0).cpu())
                            rally_shot_gt_seq.append(int(gt_action[2].item()))
                            rally_shot_pred_seq.append(int(bc_shot_type.item()))
                    
                    action_tensor = torch.tensor([
                        bc_land[0, 0].item(), bc_land[0, 1].item(),
                        float(bc_shot_type.item()),
                        bc_move[0, 0].item(), bc_move[0, 1].item()
                    ], dtype=torch.float32)
                    
                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)

                    if terminate(next_state.cpu()):
                        rally_done = True
                    
                    current_state = next_state.to(device)
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
    print("BC Evaluation Results on Test Set (Per-Rally Basis)")
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


def train_model(args):
    """Train BC model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    distribution = getattr(args, 'distribution', 'deterministic')

    # Load data generated by the unified preparation entry point.
    print("Loading preprocessed data...")
    state_path = os.path.join(args.data_dir, "train_state.pkl")
    action_path = os.path.join(args.data_dir, "train_action.pkl")
    
    if not os.path.exists(state_path) or not os.path.exists(action_path):
        print(f"Error: Preprocessed data not found in {args.data_dir}")
        print("Please run: python Scripts/prepare_all_data.py --bc")
        return
    
    with open(state_path, 'rb') as f:
        state_np = pickle.load(f)
    with open(action_path, 'rb') as f:
        action_np = pickle.load(f)
    
    state_tensor = torch.tensor(state_np, dtype=torch.float32)
    action_tensor = torch.tensor(action_np, dtype=torch.float32)
    print(f"Loaded {len(state_tensor)} training samples from {args.data_dir}")

    landing = action_tensor[:, 0:2]
    shot_type = action_tensor[:, 2].long()
    move = action_tensor[:, 3:5]
    
    dataset = BCDataset(state_tensor, landing, shot_type, move)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = BC(
        args.input_dim,
        args.hidden_dim,
        args.player_id_len,
        args.shot_type_len,
        player_id=0,
        distribution=args.distribution,
        gmm_num=args.gmm_num
    ).to(device)

    class_weights = compute_shot_type_weights(method='uniform').to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        total_land_loss = 0
        total_move_loss = 0
        total_shot_loss = 0
        
        for state, land_gt, shot_gt, move_gt in loader:
            state = state.to(device)
            land_gt = land_gt.to(device)
            shot_gt = shot_gt.to(device)
            move_gt = move_gt.to(device)
            
            land_pred, shot_pred, move_pred = model(state)

            if distribution == 'gmm':
                loss_land = gmm_nll_loss(
                    land_pred,
                    land_gt,
                    model.gmm_num,
                    model.min_sigma
                )
                loss_move = gmm_nll_loss(
                    move_pred,
                    move_gt,
                    model.gmm_num,
                    model.min_sigma
                )
            elif distribution == 'gaussian':
                loss_land = gaussian_nll_loss(land_pred, land_gt)
                loss_move = gaussian_nll_loss(move_pred, move_gt)
            else:
                loss_land = mse_loss(land_pred, land_gt)
                loss_move = mse_loss(move_pred, move_gt)

            loss_shot = ce_loss(shot_pred, shot_gt)
            
            loss = loss_land + loss_move + loss_shot

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_land_loss += loss_land.item()
            total_move_loss += loss_move.item()
            total_shot_loss += loss_shot.item()
        
        if distribution in ('gaussian', 'gmm'):
            land_name = 'LandGMMNLL' if distribution == 'gmm' else 'LandNLL'
            move_name = 'MoveGMMNLL' if distribution == 'gmm' else 'MoveNLL'
            print(
                f"Epoch {epoch}: Total={total_loss:.4f} {land_name}={total_land_loss:.4f} "
                f"{move_name}={total_move_loss:.4f} ShotCE={total_shot_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}: Total={total_loss:.4f} LandMSE={total_land_loss:.4f} "
                f"MoveMSE={total_move_loss:.4f} ShotCE={total_shot_loss:.4f}"
            )
    
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


def eval_model(args):
    """Evaluate BC model"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    distribution = getattr(args, 'distribution', 'deterministic')
    print(f"Using device: {device}")
    
    if args.skip_first_n > 0:
        print(f"Skipping first {args.skip_first_n} shots in evaluation")
    
    results = evaluate_bc_on_test(
        args.test_split,
        args.merged_csv,
        args.player_map,
        args.model_path,
        distribution,
        device,
        args.skip_first_n,
        args.gmm_num
    )
    
    print_results(results)
    
    with open(args.eval_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.eval_output}")

if __name__ == "__main__":
    import argparse
    from Utils.runtime_report import print_runtime_box
    
    parser = argparse.ArgumentParser(description='BC Model - Train or Evaluate')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Mode: train or eval (default: train)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--output', type=str, default='BC_weight.pth',
                        help='Output path for trained model (default: BC_weight.pth)')
    parser.add_argument('--data_dir', type=str, default='Experiments/bc/data',
                        help='Directory containing preprocessed train_state.pkl and train_action.pkl')
    parser.add_argument('--input_dim', type=int, default=32,
                        help='Embedded state dimension (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension (default: 128)')
    parser.add_argument('--player_id_len', type=int, default=43,
                        help='Number of player IDs (default: 43)')
    parser.add_argument('--shot_type_len', type=int, default=12,
                        help='Number of shot types (default: 12)')
    
    # Evaluation arguments
    parser.add_argument('--model_path', type=str, default='Experiments/bc/BC_weight.pth',
                        help='Path to BC model weights for evaluation (default: BC_weight.pth)')
    parser.add_argument('--test_split', type=str, 
                        default='Data/intermediate_data/train_test_split.json',
                        help='Path to train test split JSON')
    parser.add_argument('--merged_csv', type=str, 
                        default='Data/intermediate_data/merged.csv',
                        help='Path to merged CSV data')
    parser.add_argument('--player_map', type=str,
                        default='Data/intermediate_data/player_index_map.json',
                        help='Path to player index mapping')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--eval_output', type=str, default='Experiments/bc/results/bc_eval_results.json',
                        help='Output JSON file for evaluation results')
    parser.add_argument('--distribution', type=str, choices=['deterministic', 'gaussian', 'gmm'], default='deterministic',
                        help='Action output mode: deterministic, gaussian, or gmm (default: deterministic)')
    parser.add_argument('--gmm_num', type=int, default=5,
                        help='Number of GMM components when distribution=gmm (default: 5)')
    parser.add_argument('--skip_first_n', type=int, default=0,
                        help='Number of initial shots to skip in evaluation (default: 0)')
    
    args = parser.parse_args()

    print_runtime_box(
        "Runtime Parameters: Agent/bc.py",
        [
            ("mode", args.mode),
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("learning_rate", args.lr),
            ("input_dim", args.input_dim),
            ("hidden_dim", args.hidden_dim),
            ("player_id_len", args.player_id_len),
            ("shot_type_len", args.shot_type_len),
            ("distribution", args.distribution),
            ("gmm_num", args.gmm_num),
            ("data_dir", args.data_dir),
            ("output", args.output),
            ("model_path", args.model_path),
            ("test_split", args.test_split),
            ("merged_csv", args.merged_csv),
            ("player_map", args.player_map),
            ("device", args.device),
            ("skip_first_n", args.skip_first_n),
            ("eval_output", args.eval_output),
        ],
    )
    
    if args.mode == 'train':
        print("="*50)
        print("BC Model Training Mode")
        print("="*50)
        train_model(args)
    elif args.mode == 'eval':
        print("="*50)
        print("BC Model Evaluation Mode")
        print("="*50)
        eval_model(args)
