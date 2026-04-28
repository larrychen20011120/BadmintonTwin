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

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out

class INV(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(INV, self).__init__()
        self.fc = nn.Linear(input_size + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        shot_types = state[:, :, 12].long()
        player_ids = state[:, :, 17].long()
        shot_embeds = self.shot_embedding(shot_types)
        player_embeds = self.player_embedding(player_ids)
        state = torch.cat((state[:, :, :12], shot_embeds, state[:, :, 13:17], player_embeds), dim=-1)
        return state

    def forward(self, x):
        # x = x.float()
        # raw_out = torch.cat((self.embed_and_transform(x[:, :18]), self.embed_and_transform(x[:, 18:])), -1)
        out = self.fc(x)
        out = self.relu(out)
        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)

        return land_logit, shot_logit, move_logit

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, x):
        x = x.float()
        out = self.fc(x)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)
        land_logit = self.predict_land_area(out)    # Shape: [batch_size, 2]
        move_logit = self.predict_move_area(out)    # Shape: [batch_size, 2]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]
        return land_logit, shot_logit, move_logit

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, player_ids_len, shot_type_len):
        super(LatentSDE, self).__init__()
        self.action_dim = 5
        self.state_dim = data_size - self.action_dim
        self.log_softmax = nn.LogSoftmax(dim = -1)

        self.shot_type_len = shot_type_len
        self.player_ids_len = player_ids_len
        
        # ================= #
        inverse_model = INV(32, 128, player_ids_len, shot_type_len).cuda(0)
        inverse_model.load_state_dict(torch.load("Experiments/rallynetv2/weight/INV_weight.pth", weights_only=True))
        inverse_model.eval()
        self.inverse_model = inverse_model
        # ================= #

        # ================= #
        self.action_model = MLP(latent_size, 128, player_ids_len, shot_type_len).cuda(0)
        class_weights = compute_shot_type_weights(method='uniform').cuda(0)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index = 0)
        self.mse = nn.MSELoss()
        # ================= #


        # Encoder.
        self.encoder = Encoder(input_size=self.state_dim, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
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
        self.projector = nn.Linear(latent_size, self.state_dim)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self._ctx = None


    def SDE_embed_and_transform(self, tensor_data):
        max_length, batch_size, _ = tensor_data.shape
        # Split state and action
        state = tensor_data[:, :, :18]
        action = tensor_data[:, :, 18:]
        state = self.inverse_model.embed_and_transform(state) 
        # Concatenate the state and action back together
        transformed_data = torch.cat((state, action), dim=2)
        return transformed_data

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).
    
    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, total_xs, xs, ts, masks, noise_std, adjoint=False, method="euler", online=False, train_env = None, seeds_env = None):
        global SEQ_LEN
        # data preprocess
        xs = self.SDE_embed_and_transform(xs).float()
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

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

        _xs = self.projector(zs)


        """ MLP training start """
        true_actions = xs[:,:,-self.action_dim:]
        true_land = true_actions[:, :, :2].view(-1, 2)
        true_move = true_actions[:, :, 3:].view(-1, 2)
        true_shot = true_actions[:, :, 2].long().view(-1)

        land, shot, move = self.action_model(zs)

        land_loss = self.mse(land.view(-1, 2), true_land)
        shot_loss = self.ce(shot.view(-1, self.shot_type_len), true_shot)
        move_loss = self.mse(move.view(-1, 2), true_move)

        # Apply mask to losses
        if masks is not None:
            mask_flat = masks.view(-1)
            valid_count = mask_flat.sum().clamp(min=1)
            land_loss = (land_loss * mask_flat).sum() / valid_count
            shot_loss = (shot_loss * mask_flat).sum() / valid_count
            move_loss = (move_loss * mask_flat).sum() / valid_count
        """ MLP training end """

        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs[:,:,:self.state_dim]).sum(dim=(0, 2)).mean(dim=0)


        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        return log_pxs, logqp0 + logqp_path + land_loss + shot_loss + move_loss

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

    @torch.no_grad()
    def sample(self, ht, ts, step, player_ids_len, shot_type_len, bm=None):
        global SEQ_LEN
        
        ht = self.SDE_embed_and_transform(ht).float()
        ctx = self.encoder(torch.flip(ht[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))
        qz0_mean, qz0_logstd = self.qz0_net(ctx[step]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torchsde.sdeint(self, z0, ts[step:], names={'drift': 'h'}, dt=2/SEQ_LEN, bm=bm)
        
        _xs = self.projector(zs)
        
        action, shot_logit = self.predict_actions(self.inverse_model, _xs) # 所以 forward 時不需要再過一次 embedding 了

        land, shot, move = self.action_model(zs)
        shot_probs = torch.softmax(shot, dim=-1)  # [61, 1, 12]
        shot_sample = torch.multinomial(shot_probs.view(-1, 12), num_samples=1).view(-1, 1, 1)  # [61, 1, 1]
        mlp_action = torch.cat([land, shot_sample.float(), move], dim=-1)  # [61, 1, 5]
        return _xs, action[0,:], shot_logit[0], mlp_action[0,:,:].squeeze(0), shot[0,:,:]
    
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
        
        # ctx: [max_seq_len, batch_size, context_size]
        ctx = self.encoder(torch.flip(ht_batch[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))
        
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
            _xs = self.projector(zs)
            
            # Get first prediction
            land, shot, move = self.action_model(zs[:1])  # Only take first timestep
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
        lr_init=1e-4,
        t0=0.,
        t1=2.,
        lr_gamma=0.999,
        num_iters=30000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.1,
        adjoint=False,
        train_dir='./dump/',
        method="euler",
        eval = False, 
        load = False,
        load_step = 0,
        split_ratio = 0.8
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
        shot_type_len = shot_type_len
    ).to(device)

    hyperparameters = {
        "data_size": 37,
        "latent_size": latent_size,
        "context_size": context_size,
        "hidden_size": hidden_size,
        "player_ids_len": player_ids_len,
        "shot_type_len": shot_type_len,
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
        # batch_masks = masks[:, index:index+batch_size, :].reshape(-1, 1)
        batch_masks = None
        log_pxs, log_ratio = latent_sde(xs, xs[:,index:index+batch_size,:], ts, batch_masks, noise_std, adjoint, method, False)

        index += batch_size
        if (index <= int(xs.shape[1]) ) and (index + batch_size >= int(xs.shape[1])):
            shuffled_indices = torch.randperm(xs.size(1))
            xs = xs[:, shuffled_indices, :]
            masks = masks[:, shuffled_indices, :]
            index = 0

        loss = - log_pxs +  log_ratio * kl_scheduler.val
        loss.backward()
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
                f'global_step: {global_step:06d}, '
                f'log_pxs: {log_pxs :.4f}, log_ratio: {log_ratio:.4f}, loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}')

if __name__ == "__main__":
    main()