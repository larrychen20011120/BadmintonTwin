import logging
import os
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torchsde
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from Agent.BC_model._model import SimpleNN as BC_move
from Agent.BC_model.s_model import SimpleNN as BC_landing
from Agent.BC_model.serve_model import SimpleNN as BC_serve 
from Agent.BC_model.shot_model import SimpleNN as BC_shot_type 

os.environ['CUDA_LAUNCH_BLOCKING']='1'

LAUNCH_BY_BC = True

def gmm_sample(params, gmm_components, max_std=1,trivial = 1):
    """
    Sample from a GMM parameterized by `params`, with standard deviation clamped to `max_std`.
    Args:
        params: Tensor of shape (batch, gmm_components * 5) containing GMM parameters (mean_x, mean_y, log_var_x, log_var_y, weights).
        gmm_components: Number of Gaussian components.
        max_std: Maximum allowed standard deviation (clipping value).
    Returns:
        samples: Tensor of shape (batch, 2) containing sampled (x, y) positions.
    """
    batch_size = params.size(0)
    params = params.view(batch_size, gmm_components, 5)

    # Extract GMM parameters
    means = params[:, :, :2]  # Mean (x, y), shape: (batch, gmm_components, 2)
    log_vars = params[:, :, 2:4]  # Log variances (x, y), shape: (batch, gmm_components, 2)
    weights = params[:, :, 4]  # Mixture weights, shape: (batch, gmm_components)

    # Compute probabilities for selecting components
    weights = F.softmax(weights, dim=1)  # Convert log weights to probabilities
    weights += trivial 

    # Sample component indices
    component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)  # Shape: (batch,)

    # Gather selected means and variances
    selected_means = torch.gather(means, 1, component_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
    selected_log_vars = torch.gather(log_vars, 1, component_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)

    # Compute standard deviations with clamping
    stds = torch.clamp(torch.exp(0.5 * selected_log_vars), max=max_std)  # Clamp std to max_std

    # Sample from the selected Gaussian
    samples = selected_means + stds * torch.randn_like(selected_means)  # Sample using reparameterization trick

    return samples

def find_first_zero_index(lst):
    try:
        index = lst.index(0)
        return index
    except ValueError:
        return -1

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
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        self.predict_shot_type = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        shot_types = state[:, :, 12].long()
        player_ids = state[:, :, 17].long()
        shot_embeds = self.shot_embedding(shot_types)
        player_embeds = self.player_embedding(player_ids)
        state = torch.cat((state[:, :, :12], shot_embeds, state[:, :, 13:17], player_embeds), dim=-1)
        return state

    def forward(self, x):
        x = x.float()
        out = self.fc(x)
        out = self.relu(out)
        shot_logit = self.predict_shot_type(out)
        return shot_logit


class RallyNet(nn.Module):
    
    sde_type = "ito"
    noise_type = "diagonal"
    
    def __init__(self, data_size, latent_size, context_size, hidden_size, target_players, player_ids_len, shot_type_len, device, id, ts,player_name = None):
        super(RallyNet, self).__init__()
        self.action_dim = 5
        self.state_dim = data_size - self.action_dim
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.shot_type_len = shot_type_len
        self.player_ids_len = player_ids_len        
        self.action_model = MLP(latent_size, 128, player_ids_len, shot_type_len).cuda(0)
        self.ce = nn.CrossEntropyLoss(ignore_index = 0)

        self.device = device

        self.shot_type_len = shot_type_len
        self.player_ids_len = player_ids_len
        self.target_players = target_players
        self.player_id = id
        self.ts = ts

        bc_landing = []
        for i in range(8):
            model = BC_landing(5).to(device)
            model.load_state_dict(torch.load(f"BadmintonEnv/Agent/weight/ALL/model_weights_gmm7{i+2}.pth", weights_only=True))
            model.eval()   
            bc_landing.append(model) 
        self.bc_landing = bc_landing

        bc_serve = []
        model = BC_serve(5).to(device)
        model.load_state_dict(torch.load(f"BadmintonEnv/Agent/weight/ALL/model_weights_gmm71.pth", weights_only=True))
        model.eval()
        bc_serve.append(model) 

        model = BC_serve(5).to(device)
        model.load_state_dict(torch.load(f"BadmintonEnv/Agent/weight/ALL/model_weights_gmm710.pth", weights_only=True))
        model.eval()
        bc_serve.append(model)

        self.bc_serve = bc_serve

        model = BC_move(5).to(device)
        model.load_state_dict(torch.load(f"BadmintonEnv/Agent/weight/ALL/model_weights_gmm_move_min.pth", weights_only=True))
        model.eval()

        self.bc_move = model

        model = BC_shot_type().to(device)
        if player_name != None:
            print('Simulated: ',player_name)
            model.load_state_dict(torch.load(f"BadmintonEnv/Agent/weight/{player_name}/model_weights_gmm_serve.pth", weights_only=True))
        else:
            model.load_state_dict(torch.load(f"BadmintonEnv/Agent/weight/ALL/model_weights_gmm_serve.pth", weights_only=True))
        model.eval()

        self.bc_shot_type = model

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
        state = tensor_data[:, :, :18]
        action = tensor_data[:, :, 18:]
        state = self.action_model.embed_and_transform(state) 
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
    
    def list_subtraction(self, p1,p2):
        point1 = p1.copy()
        point2 = p2.copy()
        v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
        return v[0], v[1]
    

    def translation4Serve(self, state, info, player): # Calculate the full states
        return_state_list = [0]*18
        
        state = list(state)
        state[1] = list(state[1])
        state[2] = list(state[2])
        state[3] = list(state[3])

        state[1][0] = state[1][0] / 177.5
        state[1][1] = - (state[1][1] + 240) / 240

        state[2][0] = - (state[2][0] / 177.5)
        state[2][1] = - (state[2][1] - 240) / 240

        state[3][0] = state[3][0] / 177.5
        state[3][1] = - (state[3][1] + 240) / 240

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
        return_state_list[8], return_state_list[9] = self.list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = 0
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        if info['action'][-1] != None:
            opponent_last_x = (info['action'][-1][3][0] / 177.5)
            opponent_last_y = (info['action'][-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = self.list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        if player != self.target_players[self.player_id]:
            if player in self.target_players:
                return_state_list[17] = self.target_players.index(player)
            else:
                return_state_list[17] = -1
        else:
            return_state_list[17] = self.player_id
    
        return torch.FloatTensor(return_state_list)
    
    def translation4RallyNet(self, state, info, step, player): # Calculate the full states
        return_state_list = [0]*18
        
        state = list(state)
        state[1] = list(state[1])
        state[2] = list(state[2])
        state[3] = list(state[3])

        state[1][0] = state[1][0] / 177.5
        state[1][1] = - (state[1][1] + 240) / 240

        state[2][0] = - (state[2][0] / 177.5)
        state[2][1] = - (state[2][1] - 240) / 240

        state[3][0] = state[3][0] / 177.5
        state[3][1] = - (state[3][1] + 240) / 240

        # match_state
        if info['rally'] < 0:
            return_state_list[0] = 1
        else:
            return_state_list[0] = info['rally']
        
        return_state_list[1] = info['round'][-1] - 1
        return_state_list[2] = info['score'][0] 
        return_state_list[3] = info['score'][1]
        return_state_list[4] = info['score'][0] - info['score'][1] # the difference score between the player and the opponent

        # player_state
        return_state_list[5] = state[1][0]
        return_state_list[6] = state[1][1]
        
        # ball_state
        return_state_list[8], return_state_list[9] = self.list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = state[0]
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        if info['action'][step-1] != None:
            opponent_last_x = (info['action'][step-1][3][0] / 177.5)
            opponent_last_y = (info['action'][step-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = self.list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        if player != self.target_players[self.player_id]:
            if player in self.target_players:
                return_state_list[17] = self.target_players.index(player)
            else:
                return_state_list[17] = -1
        else:
            return_state_list[17] = self.player_id
        
        return torch.FloatTensor(return_state_list).unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def action(self, states, info, launch):
        raw_states = states
        shape = (int(self.ts.shape[0]), 1, 18)
        ht = torch.zeros(shape).to(self.device)
        step = info['round'][-1]-1


        if len(info['player']) > 1 and info['player'][0] == info['player'][1]:
            target_states = info['state'][1:]
            target_player = info['player'][1:]
        elif len(info['player']) == 1:
            target_states = None
        else:
            raise RuntimeError("Unexpected runtime error. Program halted.")

        if target_states != None:
            for i in range(len(target_states)):
                ht[i,:,:] = self.translation4RallyNet(target_states[i], info, i+1, target_player[i])
            ht[step,:,:] = self.translation4RallyNet(states, info, step+1, self.target_players[self.player_id])
            # if step != len(target_states):
            #     raise RuntimeError("Current step not equal to len(target_states) + 1")
        else:
            states = self.translation4RallyNet(states, info, step+1, self.target_players[self.player_id])
            ht[step,:,:] = states

        ht = self.SDE_embed_and_transform(ht).float().to(self.device)
        ctx = self.encoder(torch.flip(ht[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((self.ts, ctx))
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torchsde.sdeint(self, z0, self.ts[:step+1], names={'drift': 'h'}, dt=2/int(self.ts.shape[0]), bm=None)

        shot = self.action_model(zs)
        shot_probs = torch.softmax(shot, dim=-1)[step,:,:]

        shot_type_dist = Categorical(shot_probs)
        shot_type = shot_type_dist.sample()
        output_shot = shot_type.item()

        none_act = 0
        if output_shot == 0:
            none_act = 1

        if launch == False:
            player_x = (states[1][0]/177.5)
            player_y = (states[1][1]/240+1)
            hit_x = (states[3][0]/177.5)
            hit_y = (states[3][1]/240+1)
            opponent_x = -(states[2][0]/177.5)
            opponent_y = -(states[2][1]/240-1)
            pre_type = states[0]
        else:
            player_x = (raw_states[1][0]/177.5)
            player_y = (raw_states[1][1]/240+1)
            hit_x = (raw_states[3][0]/177.5)
            hit_y = (raw_states[3][1]/240+1)
            opponent_x = -(raw_states[2][0]/177.5)
            opponent_y = -(raw_states[2][1]/240-1)
            pre_type = raw_states[0]

        shot_input = torch.tensor([player_x, hit_x, hit_y,player_y, opponent_x, opponent_y, pre_type], dtype=torch.float32)
        shot_probs = self.bc_shot_type(shot_input.to(self.device))

        shot_probs = torch.softmax(shot_probs, dim=-1)
        shot_probs = torch.clamp(shot_probs, min=0.0)
        # ================= only for serve ================= #
        
        if launch == True:
            mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool).to(self.device)
            masked_probs = shot_probs.masked_fill(~mask, 0.0)
            sum_probs = masked_probs.sum(dim=-1, keepdim=True)
            shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
        else:
            mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool).to(self.device)
            masked_probs = shot_probs.masked_fill(~mask, 0.0)
            sum_probs = masked_probs.sum(dim=-1, keepdim=True)
            shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
        
        shot_probs = shot_probs / shot_probs.sum() 
        shot_type_dist = Categorical(shot_probs)
        shot_type = shot_type_dist.sample()

        output_shot = shot_type.item()
        output_shot_dist = shot_probs.tolist()

        if launch == False:
            t2c = {1: 1, 2: 10, 3: 2, 4: 4, 5: 9, 6: 8, 7: 6, 8: 7, 9: 3, 10: 5, 11: 11}
            gmm_input_shot = output_shot#t2c.get(int(output_shot), "Unknown Category") 

            move_input = torch.tensor([player_x, player_y, hit_x, hit_y, opponent_x, opponent_y, states[0], gmm_input_shot], dtype=torch.float32)
            move_param = self.bc_move(move_input.to(self.device))
            move = gmm_sample(move_param.view(-1, 5 * 5), 5, max_std=1.0).unsqueeze(1)
            output_move = move[0,:,:].tolist()[0]

            MLP2_input = torch.tensor([player_x, player_y, hit_x, hit_y, opponent_x, opponent_y, states[0], gmm_input_shot], dtype=torch.float32)
            if gmm_input_shot >=2 and gmm_input_shot <= 9:
                land_param = self.bc_landing[gmm_input_shot-2](MLP2_input.to(self.device))
                land = gmm_sample(land_param.view(-1, 5 * 5), 5, max_std=1.0).unsqueeze(1)

                output_land = land[0,:,:].tolist()[0]
        else:
            t2c = {1: 1, 2: 10, 3: 2, 4: 4, 5: 9, 6: 8, 7: 6, 8: 7, 9: 3, 10: 5, 11: 11}
            gmm_input_shot = output_shot#t2c.get(int(output_shot), "Unknown Category")

            move_input = torch.tensor([player_x, player_y, hit_x, hit_y, opponent_x, opponent_y, 0, gmm_input_shot], dtype=torch.float32)
            move_param = self.bc_move(move_input.to(self.device))
            move = gmm_sample(move_param.view(-1, 5 * 5), 5, max_std=1.0).unsqueeze(1)
            output_move = move[0,:,:].tolist()[0]

            MLPserve_input = torch.tensor([player_x, player_y, opponent_x, opponent_y, gmm_input_shot], dtype=torch.float32)

            if gmm_input_shot == 1:
                land_param = self.bc_serve[0](MLPserve_input.to(self.device))
                land = gmm_sample(land_param.view(-1, 5 * 5), 5, max_std=1.0,trivial = 0).unsqueeze(1)
                output_land = land[0,:,:].tolist()[0]
                output_land[1] -= 0.25
                if output_land[0]*raw_states[1][0] < 0: output_land[0] *= -1
            else:
                land_param = self.bc_serve[1](MLPserve_input.to(self.device))
                land = gmm_sample(land_param.view(-1, 5 * 5), 5, max_std=1.0,trivial = 0).unsqueeze(1)
                output_land = land[0,:,:].tolist()[0]

        if none_act == 1:
            return None
        else:
            prob_array = np.array(output_shot_dist)
            prob_array = prob_array[1:11]
            normalized_array = prob_array / prob_array.sum()
            normalized_tuple = tuple(normalized_array)

            output_land[0] = -(output_land[0]*177.5)
            output_land[1] = -(output_land[1]*240) + 240
            output_move[0] = output_move[0]*177.5
            output_move[1] = (output_move[1]*240) - 240

            mlp_action = (output_shot, raw_states[-1], tuple(output_land), tuple(output_move), normalized_tuple)

            return mlp_action
        
    