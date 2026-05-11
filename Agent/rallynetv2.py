import logging
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal
import torchsde
import torch.nn.functional as F
from dtaidistance import dtw_ndim
from tqdm import tqdm, trange
from tqdm import trange
import gc

import pickle
import os
import sys
import json

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Utils.helper import *


os.environ['CUDA_LAUNCH_BLOCKING']='1'

SEQ_LEN = 71

with open("Data/intermediate_data/player_index_map.json", 'r') as f:
    player2index = json.load(f)


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

class MatchStateEncoder(nn.Module):
    """MLP encoder for match state (rally, ball_round, player_score, opponent_score, score_status)."""
    def __init__(self, input_size=5, hidden_size=32, output_size=16):
        super(MatchStateEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        self.max_values =  torch.tensor([42, 71, 21, 21, 21], dtype=torch.float32)
    
    def forward(self, match_state):
        # match_state: [seq_len, batch_size, 5]
        match_state = match_state / self.max_values.to(match_state.device)  # Normalize each feature by its max value
        return self.mlp(match_state)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len, gmm_num=5):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.gmm_num = gmm_num
        self.predict_land_area = nn.Linear(hidden_size, gmm_num * 5)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, gmm_num * 5)

    def forward(self, x):
        x = x.float()
        out = self.fc(x)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)
        land_logit = self.predict_land_area(out)    # Shape: [..., gmm_num * 5]
        move_logit = self.predict_move_area(out)    # Shape: [..., gmm_num * 5]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]
        return land_logit, shot_logit, move_logit


def split_gmm_params(pred_params, gmm_num, min_sigma=1e-3):
    """
    Split raw GMM output into mixture logits, 2D means, and positive stds.

    Layout follows Agent/bc.py: [pi_logits(K), mu(K*2), raw_sigma(K*2)].
    """
    expected_dim = gmm_num * 5
    if pred_params.shape[-1] != expected_dim:
        raise ValueError(f'GMM prediction must have {expected_dim} channels for {gmm_num} components')

    pred_params = pred_params.reshape(-1, expected_dim)
    pi_logits = pred_params[:, :gmm_num]
    mus = pred_params[:, gmm_num:gmm_num * 3].reshape(-1, gmm_num, 2)
    raw_sigmas = pred_params[:, gmm_num * 3:gmm_num * 5].reshape(-1, gmm_num, 2)
    sigmas = F.softplus(raw_sigmas) + min_sigma
    return pi_logits, mus, sigmas


def gmm_nll_loss(pred_params, target, gmm_num, min_sigma=1e-3, reduction='mean'):
    """Compute diagonal 2D GMM NLL, matching BC's GMM objective."""
    pi_logits, mus, sigmas = split_gmm_params(pred_params, gmm_num, min_sigma)
    target = target.reshape(-1, 2).unsqueeze(1)

    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_two_pi = torch.log(target.new_tensor(2 * torch.pi))
    log_component_prob = -0.5 * (
        ((target - mus) / sigmas) ** 2 + 2 * torch.log(sigmas) + log_two_pi
    ).sum(dim=-1)
    nll = -torch.logsumexp(log_pi + log_component_prob, dim=-1)

    if reduction == 'none':
        return nll
    if reduction == 'sum':
        return nll.sum()
    if reduction == 'mean':
        return nll.mean()
    raise ValueError('reduction must be none, sum, or mean')


def sample_from_gmm(pred_params, gmm_num, min_sigma=1e-3):
    """Sample a 2D point from a diagonal Gaussian mixture."""
    original_shape = pred_params.shape[:-1]
    pi_logits, mus, sigmas = split_gmm_params(pred_params, gmm_num, min_sigma)
    component = Categorical(logits=pi_logits).sample()
    gather_index = component.view(-1, 1, 1).expand(-1, 1, 2)
    selected_mu = mus.gather(1, gather_index).squeeze(1)
    selected_sigma = sigmas.gather(1, gather_index).squeeze(1)
    sample = selected_mu + torch.randn_like(selected_mu) * selected_sigma
    return torch.clamp(sample, -1.0, 1.0).reshape(*original_shape, 2)

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, player_ids_len, shot_type_len,
                 match_state_dim=5, condition_size=16, gmm_num=5, min_sigma=1e-3):
        super(LatentSDE, self).__init__()
        self.action_dim = 5
        self.match_state_dim = match_state_dim
        self.condition_size = condition_size
        self.gmm_num = gmm_num
        self.min_sigma = min_sigma
        self.state_dim = data_size - self.action_dim
        self.remaining_state_dim = self.state_dim - self.match_state_dim  # State without match_state
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.shot_type_len = shot_type_len
        self.player_ids_len = player_ids_len

        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_ids_len, 8)

        # ================= #
        self.action_model = MLP(latent_size + condition_size, 128, player_ids_len, shot_type_len, gmm_num=gmm_num).cuda(0)
        class_weights = compute_shot_type_weights(method='uniform').cuda(0)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0, reduction='none')
        # ================= #

        # Match State Encoder (condition encoder).
        self.match_state_encoder = MatchStateEncoder(input_size=match_state_dim, hidden_size=32, output_size=condition_size).cuda(0)

        # Encoder (encodes remaining_state + condition).
        self.encoder = Encoder(input_size=self.remaining_state_dim + condition_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        
        # Decoder (f_net: drift with condition).
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size + condition_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        
        # Prior drift (h_net: no condition, standard prior).
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        
        # Projector (latent -> remaining_state, conditioned on match_state).
        self.projector = nn.Linear(latent_size + condition_size, self.remaining_state_dim)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self._ctx = None
        self._condition = None

    def SDE_embed_and_transform(self, tensor_data):
        max_length, batch_size, _ = tensor_data.shape
        # Split state and action
        # State structure: [match_state(5), remaining_state(13), action(5)]
        # remaining_state structure: [features(5-11, 7 dims), shot_type(12), features(13-16, 4 dims), player_id(17)]
        match_state = tensor_data[:, :, :5]  # [rally, ball_round, player_score, opponent_score, score_status]
        remaining_state = tensor_data[:, :, 5:18]  # remaining 13 dims
        action = tensor_data[:, :, 18:]
        
        # Directly embed remaining_state without using dummy match_state
        # remaining_state indices mapping:
        # - indices 0-6: correspond to original indices 5-11 (7 dims of features)
        # - index 7: correspond to original index 12 (shot_type, needs embedding)
        # - indices 8-11: correspond to original indices 13-16 (4 dims of features)
        # - index 12: correspond to original index 17 (player_id, needs embedding)
        
        shot_types = remaining_state[:, :, 7].long()  # shot_type at index 7
        player_ids = remaining_state[:, :, 12].long()  # player_id at index 12
        
        shot_embeds = self.shot_embedding(shot_types)  # [seq_len, batch_size, 8]
        player_embeds = self.player_embedding(player_ids)  # [seq_len, batch_size, 8]
        
        # Concatenate: [features(0-6, 7 dims), shot_embed(8), features(8-11, 4 dims), player_embed(8)]
        remaining_state_embedded = torch.cat([
            remaining_state[:, :, :7],    # original indices 5-11
            shot_embeds,                   # shot embedding replaces index 12
            remaining_state[:, :, 8:12],   # original indices 13-16
            player_embeds                  # player embedding replaces index 17
        ], dim=2)  # 7 + 8 + 4 + 8 = 27 dims
        
        # Concatenate the match_state, embedded remaining_state, and action
        transformed_data = torch.cat((match_state, remaining_state_embedded, action), dim=2)
        return transformed_data

    def contextualize(self, ctx, condition):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).
        self._condition = condition  # [batch_size, condition_size]
    
    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        # Concatenate latent, context, and condition for the drift
        return self.f_net(torch.cat((y, ctx[i], self._condition), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, total_xs, xs, ts, masks, noise_std, adjoint=False, method="euler", online=False, train_env=None, seeds_env=None):
        global SEQ_LEN
        # data preprocess
        xs = self.SDE_embed_and_transform(xs).float()
        
        # Split match_state and remaining_state
        # xs structure after SDE_embed_and_transform: [match_state(5), remaining_state_embedded(27), action(5)]
        match_state = xs[:, :, :self.match_state_dim]  # [seq_len, batch_size, 5]
        remaining_state = xs[:, :, self.match_state_dim:self.match_state_dim + self.remaining_state_dim]  # [seq_len, batch_size, 27]
        
        # Encode match_state to get condition
        condition = self.match_state_encoder(match_state)  # [seq_len, batch_size, condition_size]
        
        # Contextualization is only needed for posterior inference.
        # Concatenate remaining_state with condition for the encoder
        encoder_input = torch.cat((remaining_state, condition), dim=-1)
        ctx = self.encoder(torch.flip(encoder_input, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        
        # Use the first timestep's condition for SDE (or could use all timesteps)
        self.contextualize((ts, ctx), condition[0])  # condition[0]: [batch_size, condition_size]

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=2/SEQ_LEN, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=2/SEQ_LEN, logqp=True, method=method)

        # Decode: concatenate latent with condition, project back to remaining_state
        # Expand condition to match zs length
        condition_expanded = condition[0].unsqueeze(0).expand(zs.size(0), -1, -1)  # [seq_len, batch_size, condition_size]
        zs_conditioned = torch.cat((zs, condition_expanded), dim=-1)
        _remaining_state = self.projector(zs_conditioned)  # Only reconstruct remaining_state, not match_state


        """ MLP training start """
        true_actions = xs[:, :, -self.action_dim:]
        true_land = true_actions[:, :, :2].view(-1, 2)
        true_move = true_actions[:, :, 3:].view(-1, 2)
        true_shot = true_actions[:, :, 2].long().view(-1)

        # Action model with condition
        land, shot, move = self.action_model(zs_conditioned)

        land_loss = gmm_nll_loss(
            land.view(-1, self.gmm_num * 5),
            true_land,
            self.gmm_num,
            self.min_sigma,
            reduction='none'
        )
        shot_loss = self.ce(shot.view(-1, self.shot_type_len), true_shot)
        move_loss = gmm_nll_loss(
            move.view(-1, self.gmm_num * 5),
            true_move,
            self.gmm_num,
            self.min_sigma,
            reduction='none'
        )

        # Apply mask to losses
        if masks is not None:
            mask_flat = masks.view(-1)
            valid_count = mask_flat.sum().clamp(min=1)
            land_loss = (land_loss * mask_flat).sum() / valid_count
            shot_loss = (shot_loss * mask_flat).sum() / valid_count
            move_loss = (move_loss * mask_flat).sum() / valid_count
        else:
            land_loss = land_loss.mean()
            shot_loss = shot_loss.mean()
            move_loss = move_loss.mean()
        """ MLP training end """

        # Only compute reconstruction loss for remaining_state (not match_state)
        xs_dist = Normal(loc=_remaining_state, scale=noise_std)
        # log_pxs = xs_dist.log_prob(remaining_state).sum(dim=(0, 2)).mean(dim=0)
        log_pxs = xs_dist.log_prob(remaining_state).mean()


        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        total_action_loss = land_loss + shot_loss + move_loss
        total_kl = logqp0 + logqp_path

        return log_pxs, total_kl, total_action_loss

    def predict_actions(self, model, state_sequence):
        actions = []
        type_porb = []

        if len(state_sequence) == 1:
            state_sequence = [state_sequence[0], state_sequence[0]]

        with torch.no_grad():
            for i in range(len(state_sequence) - 1):
                state1 = state_sequence[i].squeeze()  # 第一个state数组
                state2 = state_sequence[i + 1].squeeze()  # 第二个state数组
                merged_state = torch.cat((state1, state2), dim=0).unsqueeze(0)  # 合并两个state数组
                land_logit, shot_logit, move_logit = model(merged_state)  # 使用模型预测动作
                
                type_porb.append((shot_logit).clone())
                # Predict shot type
                shot_type_probs = F.softmax(shot_logit, dim=-1)
                shot_type_dist = Categorical(shot_type_probs)
                shot_type = shot_type_dist.sample()[0].unsqueeze(0) 

                action = torch.cat((land_logit, shot_type.unsqueeze(0), move_logit),1)
                actions.append(action)

        
        # actions.append(torch.tensor([[[0, 1, 0, 0, 0]]], dtype=torch.float32).cuda(0))
        return torch.cat(actions, dim=0), type_porb

    @torch.inference_mode()
    def sample(self, ht, ts, step, player_ids_len, shot_type_len, bm=None):
        global SEQ_LEN
        
        ht = self.SDE_embed_and_transform(ht).float()
        
        # Split match_state and remaining_state
        match_state = ht[:, :, :self.match_state_dim]
        remaining_state = ht[:, :, self.match_state_dim:self.match_state_dim + self.remaining_state_dim]
        
        # Encode match_state to get condition
        condition = self.match_state_encoder(match_state)
        
        # Encode remaining_state with condition
        encoder_input = torch.cat((remaining_state, condition), dim=-1)
        ctx = self.encoder(torch.flip(encoder_input, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx), condition[0])
        
        qz0_mean, qz0_logstd = self.qz0_net(ctx[step]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        
        # Evaluation only consumes the current action, so skip future SDE rollout.
        zs_conditioned = torch.cat((z0, condition[0]), dim=-1)
        # _remaining_state = self.projector(zs_conditioned)
        
        # Reconstruct full state for predict_actions (need to reconstruct match_state from input)
        # _xs = torch.cat([match_state, _remaining_state], dim=-1)
        
        # action, shot_logit = self.predict_actions(self.inverse_model, _xs)

        land_params, shot, move_params = self.action_model(zs_conditioned)
        land = sample_from_gmm(land_params, self.gmm_num, self.min_sigma)
        move = sample_from_gmm(move_params, self.gmm_num, self.min_sigma)
        shot_probs = torch.softmax(shot, dim=-1)  # [batch_size, shot_type_len]
        shot_sample = torch.multinomial(shot_probs, num_samples=1).float()
        mlp_action = torch.cat([land, shot_sample.float(), move], dim=-1)  # [batch_size, 5]
        return None, None, None, mlp_action.squeeze(0), shot
    
    @torch.no_grad()
    def sample_batch(self, ht_batch, ts, steps, player_ids_len, shot_type_len, bm=None):
        """
        Batch version of sample() for evaluating multiple rallies in parallel.
        
        Args:
            ht_batch: [max_seq_len, batch_size, 18] - batch of rally states
            ts: [seq_len] - time steps
            steps: [batch_size] - current step for each rally in the batch
            player_ids_len: int
            shot_type_len: int
            bm: Brownian motion (optional)
        
        Returns:
            Tuple of:
            - _xs: [max_future_len, batch_size, state_dim]
            - actions: [batch_size, 5] - first action for each rally
            - shot_logits: [batch_size, shot_type_len]
            - mlp_actions: [batch_size, 5]
            - shot_probs: [batch_size, shot_type_len]
        """
        global SEQ_LEN
        
        # ht_batch: [max_seq_len, batch_size, 18]
        ht_batch = self.SDE_embed_and_transform(ht_batch).float()
        
        # Split match_state and remaining_state
        match_state = ht_batch[:, :, :self.match_state_dim]
        remaining_state = ht_batch[:, :, self.match_state_dim:self.match_state_dim + self.remaining_state_dim]
        
        # Encode match_state to get condition
        condition = self.match_state_encoder(match_state)
        
        # ctx: [max_seq_len, batch_size, context_size]
        encoder_input = torch.cat((remaining_state, condition), dim=-1)
        ctx = self.encoder(torch.flip(encoder_input, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx), condition[0])
        
        batch_size = ht_batch.shape[1]
        device = ht_batch.device
        
        # Collect z0 for each rally at its respective step
        z0_list = []
        for b in range(batch_size):
            step = steps[b]
            qz0_mean, qz0_logstd = self.qz0_net(ctx[step, b:b+1]).chunk(chunks=2, dim=1)
            z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
            z0_list.append(z0)
        z0 = torch.cat(z0_list, dim=0)  # [batch_size, latent_size]
        
        # For simplicity, we process each rally individually for the SDE part
        # This could be optimized further with a custom SDE solver that handles different ts
        mlp_actions_list = []
        shot_probs_list = []
        
        for b in range(batch_size):
            step = steps[b]
            zs = torchsde.sdeint(self, z0[b:b+1], ts[step:], names={'drift': 'h'}, dt=2/SEQ_LEN, bm=bm)
            
            # Decode with condition
            condition_b = condition[0, b:b+1].unsqueeze(0).expand(zs.size(0), -1, -1)
            zs_conditioned = torch.cat((zs, condition_b), dim=-1)
            _remaining_state = self.projector(zs_conditioned)
            
            # Get first prediction
            land_params, shot, move_params = self.action_model(zs_conditioned[:1])  # Only take first timestep
            land = sample_from_gmm(land_params, self.gmm_num, self.min_sigma)
            move = sample_from_gmm(move_params, self.gmm_num, self.min_sigma)
            shot_probs = torch.softmax(shot, dim=-1)  # [1, 1, shot_type_len]
            shot_sample = torch.multinomial(shot_probs.view(-1, shot_type_len), num_samples=1).view(-1, 1, 1)
            mlp_action = torch.cat([land, shot_sample.float(), move], dim=-1)  # [1, 1, 5]
            
            mlp_actions_list.append(mlp_action.squeeze(0).squeeze(0))  # [5]
            shot_probs_list.append(shot_probs.squeeze(0).squeeze(0))  # [shot_type_len]
        
        mlp_actions = torch.stack(mlp_actions_list, dim=0)  # [batch_size, 5]
        shot_probs = torch.stack(shot_probs_list, dim=0)  # [batch_size, shot_type_len]
        
        # For compatibility, also return dummy values for other returns
        # These would need proper implementation if needed
        dummy_actions = mlp_actions.clone()
        dummy_shot_logits = shot_probs.clone()
        
        return None, dummy_actions, dummy_shot_logits, mlp_actions, shot_probs

def make_dataset(testing, device, split_ratio=0.8):
    global SEQ_LEN

    with open("Experiments/rallynetv2/data/norm/train_player_traj.pkl", 'rb') as f:
        player_data = pickle.load(f)
    with open("Experiments/rallynetv2/data/norm/train_opponent_traj.pkl", 'rb') as f:
        opponent_data = pickle.load(f)
    with open("Experiments/rallynetv2/data/norm/train_player_mask.pkl", 'rb') as f:
        player_masks = pickle.load(f)
    with open("Experiments/rallynetv2/data/norm/train_opponent_mask.pkl", 'rb') as f:
        opponent_masks = pickle.load(f)

    merged_data = []
    merged_masks = []
    shot_type_set = []

    for player_rally, opponent_rally, player_mask_rally, opponent_mask_rally in zip(player_data, opponent_data, player_masks, opponent_masks):
        merged_rally = []
        merged_mask_rally = []
        player_len = len(player_rally)
        opponent_len = len(opponent_rally)
        if player_len > 0:
            max_len = max(player_len, opponent_len)
            for i in range(max_len):
                if i < player_len:
                    merged_rally.append(player_rally[i])
                    merged_mask_rally.append(player_mask_rally[i])
                    shot_type_set.append(player_rally[i][12])
                if i < opponent_len:
                    merged_rally.append(opponent_rally[i])
                    merged_mask_rally.append(opponent_mask_rally[i])
                    shot_type_set.append(opponent_rally[i][12])
            merged_data.append(merged_rally)
            merged_masks.append(merged_mask_rally)

    max_length = max(len(rally) for rally in merged_data)
    xs = pad_and_convert_to_tensor(merged_data, max_length).cuda(0)
    masks = pad_and_convert_to_tensor(merged_masks, max_length).cuda(0)
    ts = torch.linspace(0, xs.shape[0], steps=xs.shape[0], device=device)

    SEQ_LEN = xs.shape[0]

    min_val = torch.min(ts)
    max_val = torch.max(ts)
    normalized_tensor = (ts - min_val) / (max_val - min_val)
    ts = normalized_tensor * 2

    # Keep masks in 3D form: [seq_len, num_rallies, 1] for easier batch extraction
    masks_3d = masks[:, :, :1]

    # Split dataset into training and testing
    total_rallies = xs.shape[1]
    split_index = int(total_rallies * split_ratio)

    return xs, masks, ts, len(player2index), len(set(shot_type_set))


def main(
        batch_size = 32, 
        latent_size= 32,
        context_size=64,
        hidden_size=64,
        lr_init=5e-4,
        t0=0.,
        t1=2.,
        lr_gamma=0.999,
        num_iters=30000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.1,
        adjoint=False,
        train_dir='./dump_weighted/',
        method="euler",
        eval = False, 
        load = False,
        load_step = 0,
        split_ratio = 0.8,
        gmm_num=5,
        min_sigma=1e-3
):
    create_folder_if_not_exists(train_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    xs, masks, ts, player_ids_len, shot_type_len = make_dataset(False, device=device, split_ratio=split_ratio)

    latent_sde = LatentSDE(
        data_size = 37, # 16 + 8 + 8 + (5)
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        player_ids_len = player_ids_len,
        shot_type_len = shot_type_len,
        gmm_num=gmm_num,
        min_sigma=min_sigma
    ).to(device)

    hyperparameters = {
        "data_size": 37,
        "latent_size": latent_size,
        "context_size": context_size,
        "hidden_size": hidden_size,
        "player_ids_len": player_ids_len,
        "shot_type_len": shot_type_len,
        "gmm_num": gmm_num,
        "min_sigma": min_sigma,
        "ts": ts,
        "xs": xs,
        # "target_players": target_players,
    }
    with open('hyperparameters.pkl', 'wb') as f:
        pickle.dump(hyperparameters, f)

    if load == True:
        latent_sde.load_state_dict(torch.load(train_dir+"gen_e_"+str(load_step) +".trc", map_location = device))

    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    index = 0

    torch.save(latent_sde.state_dict(), train_dir+'gen_e_0.trc')

    for global_step in tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        # apply mask or not
        batch_masks = masks[:, index:index+batch_size, :].reshape(-1, 1)
        # batch_masks = None
        # log_pxs, log_ratio = latent_sde(xs, xs[:,index:index+batch_size,:], ts, batch_masks, noise_std, adjoint, method, False)
        log_pxs, kl, action_loss = latent_sde(xs, xs[:,index:index+batch_size,:], ts, batch_masks, noise_std, adjoint, method, False)

        index += batch_size
        if (index <= int(xs.shape[1]) ) and (index + batch_size >= int(xs.shape[1])):
            shuffled_indices = torch.randperm(xs.size(1))
            xs = xs[:, shuffled_indices, :]
            masks = masks[:, shuffled_indices, :]
            index = 0

        loss = -log_pxs + (kl * kl_scheduler.val) + action_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        sys.stdout.flush()
        gc.collect()

        if (global_step % pause_every == 0) or global_step == 1:
            if load == True:
                torch.save(latent_sde.state_dict(), train_dir+'gen_e_{}.trc'.format(str(load_step + global_step)))
            else:
                torch.save(latent_sde.state_dict(), train_dir+'gen_e_{}.trc'.format(str(global_step)))
            
            logging.warning(
                f'Step: {global_step:06d} | '
                f'Total Loss: {loss.item():.4f}\n'
                f'  [Losses] NLL: {(-log_pxs).item():.3f}, KL: {kl.item():.3f}, Action: {action_loss.item():.3f}\n'
                f'  [Status] KL_Coeff: {kl_scheduler.val:.4f}'
            )

if __name__ == "__main__":
    main()
