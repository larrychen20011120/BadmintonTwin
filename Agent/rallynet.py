import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchsde
from torch import nn
from torch.distributions import Normal, kl_divergence

from Agent.base_agent import BaseAgent
from Utils.helper import transition


CHECKPOINT_FORMAT = "rallynet"
CHECKPOINT_VERSION = 2
CONTINUOUS_STATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
SERVE_SHOT_TYPES = (1, 10)
RALLY_SHOT_TYPES = tuple(range(2, 10))
ACTION_SHOT_TYPES = tuple(range(1, 11))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def split_gmm_params(params, component_count, min_sigma=1e-3):
    expected_size = component_count * 5
    if params.shape[-1] != expected_size:
        raise ValueError(
            f"Expected {expected_size} GMM parameters, got {params.shape[-1]}"
        )
    flat = params.reshape(-1, expected_size)
    logits = flat[:, :component_count]
    means = flat[:, component_count:component_count * 3].reshape(-1, component_count, 2)
    raw_sigmas = flat[:, component_count * 3:].reshape(-1, component_count, 2)
    sigmas = F.softplus(raw_sigmas) + min_sigma
    return logits, means, sigmas


def gmm_nll(params, targets, component_count, min_sigma=1e-3, reduction="mean"):
    logits, means, sigmas = split_gmm_params(params, component_count, min_sigma)
    targets = targets.reshape(-1, 2).unsqueeze(1)
    log_component = -0.5 * (
        ((targets - means) / sigmas) ** 2
        + 2.0 * torch.log(sigmas)
        + torch.log(targets.new_tensor(2.0 * torch.pi))
    ).sum(dim=-1)
    losses = -torch.logsumexp(F.log_softmax(logits, dim=-1) + log_component, dim=-1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def sample_gmm(
    params,
    component_count,
    min_sigma=1e-3,
    max_sample_sigma=1.0,
    sample=True,
    generator=None,
):
    original_shape = params.shape[:-1]
    logits, means, sigmas = split_gmm_params(params, component_count, min_sigma)
    if sample:
        probabilities = torch.softmax(logits, dim=-1)
        component = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    else:
        component = logits.argmax(dim=-1)
    gather_index = component[:, None, None].expand(-1, 1, 2)
    selected_mean = means.gather(1, gather_index).squeeze(1)
    if sample:
        selected_sigma = sigmas.gather(1, gather_index).squeeze(1)
        selected_sigma = selected_sigma.clamp(max=max_sample_sigma)
        noise = torch.randn(
            selected_mean.shape,
            dtype=selected_mean.dtype,
            device=selected_mean.device,
            generator=generator,
        )
        result = selected_mean + selected_sigma * noise
    else:
        result = selected_mean
    return result.clamp(-1.0, 1.0).reshape(*original_shape, 2)


class StateFeatureEncoder(nn.Module):
    def __init__(
        self,
        player_count,
        shot_type_count=12,
        embedding_dim=8,
        normalization_epsilon=1e-6,
    ):
        super().__init__()
        self.player_count = player_count
        self.shot_type_count = shot_type_count
        self.embedding_dim = embedding_dim
        self.normalization_epsilon = normalization_epsilon
        self.shot_embedding = nn.Embedding(
            shot_type_count + 1, embedding_dim, padding_idx=0
        )
        self.player_embedding = nn.Embedding(
            player_count + 1, embedding_dim, padding_idx=0
        )
        self.register_buffer("continuous_mean", torch.zeros(len(CONTINUOUS_STATE_INDICES)))
        self.register_buffer("continuous_std", torch.ones(len(CONTINUOUS_STATE_INDICES)))

    @property
    def output_size(self):
        return len(CONTINUOUS_STATE_INDICES) + self.embedding_dim * 2

    def set_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.continuous_mean.dtype)
        std = torch.as_tensor(std, dtype=self.continuous_std.dtype)
        std = torch.where(
            std < self.normalization_epsilon, torch.ones_like(std), std
        )
        if mean.shape != self.continuous_mean.shape or std.shape != self.continuous_std.shape:
            raise ValueError("Unexpected continuous-state normalization shape")
        self.continuous_mean.copy_(mean)
        self.continuous_std.copy_(std)

    def forward(self, states, valid_mask=None):
        if states.shape[-1] != 18:
            raise ValueError(f"Expected 18 state features, got {states.shape[-1]}")
        continuous = states[..., CONTINUOUS_STATE_INDICES].float()
        normalized = (continuous - self.continuous_mean) / self.continuous_std
        raw_shots = states[..., 12]
        raw_players = states[..., 17]
        active = (
            valid_mask.bool()
            if valid_mask is not None
            else torch.ones_like(raw_shots, dtype=torch.bool)
        )
        if (
            ((raw_shots != raw_shots.round()) | (raw_shots < 0) | (raw_shots >= self.shot_type_count))[active].any()
        ):
            raise ValueError("State contains an invalid opponent shot ID")
        if (
            ((raw_players != raw_players.round()) | (raw_players < 0) | (raw_players >= self.player_count))[active].any()
        ):
            raise ValueError("State contains an invalid player ID")
        shot_tokens = raw_shots.long() + 1
        player_tokens = raw_players.long() + 1
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            normalized = normalized * valid_mask.unsqueeze(-1)
            shot_tokens = torch.where(valid_mask, shot_tokens, torch.zeros_like(shot_tokens))
            player_tokens = torch.where(valid_mask, player_tokens, torch.zeros_like(player_tokens))
        return torch.cat(
            [
                normalized,
                self.shot_embedding(shot_tokens),
                self.player_embedding(player_tokens),
            ],
            dim=-1,
        )


class ShotTypeInformedActionPredictor(nn.Module):
    def __init__(
        self,
        player_count,
        shot_type_count=12,
        hidden_size=128,
        state_embedding_dim=8,
        shot_embedding_dim=8,
        gmm_components=5,
        min_sigma=1e-3,
        max_sample_sigma=1.0,
        normalization_epsilon=1e-6,
    ):
        super().__init__()
        self.shot_type_count = shot_type_count
        self.gmm_components = gmm_components
        self.min_sigma = min_sigma
        self.max_sample_sigma = max_sample_sigma
        self.state_encoder = StateFeatureEncoder(
            player_count,
            shot_type_count,
            embedding_dim=state_embedding_dim,
            normalization_epsilon=normalization_epsilon,
        )
        feature_size = self.state_encoder.output_size
        self.shot_classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, shot_type_count),
        )
        self.selected_shot_embedding = nn.Embedding(shot_type_count, shot_embedding_dim)
        self.movement_predictor = nn.Sequential(
            nn.Linear(feature_size + shot_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, gmm_components * 5),
        )
        self.landing_specialists = nn.ModuleDict({
            str(shot_type): nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, gmm_components * 5),
            )
            for shot_type in ACTION_SHOT_TYPES
        })

    def set_normalization(self, mean, std):
        self.state_encoder.set_normalization(mean, std)

    def shot_logits(self, states):
        return self.shot_classifier(self.state_encoder(states))

    def conditional_params(self, states, selected_shots):
        features = self.state_encoder(states)
        selected_shots = selected_shots.long()
        if not torch.all((selected_shots >= 1) & (selected_shots <= 10)):
            raise ValueError("Action specialists only support shot types 1 through 10")
        shot_features = self.selected_shot_embedding(selected_shots)
        movement_params = self.movement_predictor(
            torch.cat([features, shot_features], dim=-1)
        )
        landing_params = features.new_empty(
            features.shape[0], self.gmm_components * 5
        )
        for shot_type in ACTION_SHOT_TYPES:
            mask = selected_shots == shot_type
            if mask.any():
                landing_params[mask] = self.landing_specialists[str(shot_type)](
                    features[mask]
                )
        return landing_params, movement_params

    def forward(self, states, selected_shots):
        logits = self.shot_logits(states)
        landing_params, movement_params = self.conditional_params(states, selected_shots)
        return logits, landing_params, movement_params

    @torch.inference_mode()
    def predict(self, state, launch, sample=True, generator=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits = self.shot_logits(state)
        allowed = SERVE_SHOT_TYPES if launch else RALLY_SHOT_TYPES
        mask = torch.zeros(self.shot_type_count, dtype=torch.bool, device=logits.device)
        mask[list(allowed)] = True
        masked_logits = logits.masked_fill(~mask.unsqueeze(0), -torch.inf)
        probabilities = torch.softmax(masked_logits, dim=-1)
        if sample:
            selected_shot = torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(-1)
        else:
            selected_shot = probabilities.argmax(dim=-1)
        landing_params, movement_params = self.conditional_params(state, selected_shot)
        landing = sample_gmm(
            landing_params,
            self.gmm_components,
            self.min_sigma,
            self.max_sample_sigma,
            sample,
            generator,
        )
        movement = sample_gmm(
            movement_params,
            self.gmm_components,
            self.min_sigma,
            self.max_sample_sigma,
            sample,
            generator,
        )
        return {
            "shot_type": int(selected_shot.item()),
            "shot_probs": probabilities.squeeze(0),
            "landing": landing.squeeze(0),
            "moving": movement.squeeze(0),
        }


def _reverse_gru_with_mask(features, valid_mask, gru_module):
    """
    Reverse only valid prefixes, run GRU, then un-reverse and zero-pad.
    
    Args:
        features: [T, B, D]
        valid_mask: [T, B] bool, prefix-valid (True...True False...False)
        gru_module: nn.GRU
    
    Returns:
        context: [T, B, H]
    """
    if features.ndim != 3:
        raise ValueError("features must have shape [T, B, D]")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [T, B]")
    T, B, D = features.shape
    lengths = valid_mask.sum(dim=0).cpu()
    device = features.device

    # Reverse valid prefixes per sequence
    reversed_features = features.new_zeros((T, B, D))
    for b in range(B):
        l = int(lengths[b].item())
        if l > 0:
            reversed_features[:l, b] = torch.flip(features[:l, b], dims=(0,))

    # Pack and run GRU
    lengths_clamped = lengths.clamp_min(1)
    packed = nn.utils.rnn.pack_padded_sequence(
        reversed_features, lengths_clamped, enforce_sorted=False
    )
    output_packed, _ = gru_module(packed)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        output_packed, total_length=T
    )

    # Un-reverse valid prefixes
    context = features.new_zeros((T, B, output.shape[-1]))
    for b in range(B):
        l = int(lengths[b].item())
        if l > 0:
            context[:l, b] = torch.flip(output[:l, b], dims=(0,))

    # Zero out padding
    context = context * valid_mask.unsqueeze(-1).to(context.dtype)
    return context


class LatentErrorSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        player_count,
        shot_type_count=12,
        latent_size=32,
        context_size=64,
        hidden_size=64,
        state_embedding_dim=8,
        normalization_epsilon=1e-6,
        reconstruction_noise_std=0.1,
        time_horizon=2.0,
        max_sequence_length=71,
        solver_method="euler",
        solver_substeps=2,
    ):
        super().__init__()
        if latent_size < 1 or context_size < 1 or hidden_size < 1:
            raise ValueError("LatentErrorSDE dimensions must be positive")
        if reconstruction_noise_std <= 0:
            raise ValueError("reconstruction_noise_std must be positive")
        if time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        if max_sequence_length <= 1:
            raise ValueError("max_sequence_length must be greater than one")
        if solver_substeps < 1:
            raise ValueError("solver_substeps must be at least one")
        self.latent_size = latent_size
        self.context_size = context_size
        self.reconstruction_noise_std = reconstruction_noise_std
        self.time_horizon = time_horizon
        self.max_sequence_length = max_sequence_length
        self.solver_method = solver_method
        self.solver_substeps = solver_substeps
        self.state_encoder = StateFeatureEncoder(
            player_count,
            shot_type_count,
            embedding_dim=state_embedding_dim,
            normalization_epsilon=normalization_epsilon,
        )
        feature_size = self.state_encoder.output_size
        self.context_encoder = nn.GRU(feature_size, context_size)
        self.qz0_net = nn.Linear(context_size, latent_size * 2)
        # Posterior drift (context-conditioned)
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # Prior drift (latent-only)
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # State-dependent element-wise diffusion
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )
                for _ in range(latent_size)
            ]
        )
        # Projector: latent -> encoded state
        self.projector = nn.Linear(latent_size, feature_size)
        # Error head: latent + current state -> error logit
        self.error_head = nn.Sequential(
            nn.Linear(latent_size + feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # Learned prior initial distribution
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self.register_buffer("error_logit_offset", torch.zeros(()))
        self._ctx = None

    def set_normalization(self, mean, std):
        self.state_encoder.set_normalization(mean, std)

    def set_error_pos_weight(self, pos_weight):
        value = max(float(pos_weight), 1e-12)
        self.error_logit_offset.fill_(float(np.log(value)))

    def time_grid(self, length, device, dtype):
        if length < 1:
            raise ValueError("Sequence length must be positive")
        if self.max_sequence_length <= 1:
            raise ValueError("max_sequence_length must be greater than one")
        delta = self.time_horizon / (self.max_sequence_length - 1)
        return torch.arange(length, device=device, dtype=dtype) * delta

    def solver_dt(self):
        delta = self.time_horizon / (self.max_sequence_length - 1)
        return delta / self.solver_substeps

    def h(self, t, z):
        return self.h_net(z)

    def g(self, t, z):  # Diagonal diffusion, state-dependent.
        y = torch.split(z, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def f(self, t, z):
        if self._ctx is None:
            raise RuntimeError("LatentErrorSDE must be contextualized before integration")
        ts, context = self._ctx
        index = torch.searchsorted(ts, t, right=True) - 1
        index = index.clamp(0, context.shape[0] - 1)
        current_context = context[index]
        return self.f_net(torch.cat([z, current_context], dim=-1))

    def _initial_posterior(self, context, sample=True, generator=None):
        mean, logstd = self.qz0_net(context[0]).chunk(2, dim=-1)
        std = torch.exp(logstd)
        if sample:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            z0 = mean + std * noise
        else:
            z0 = mean
        return z0, mean, std

    def _integrate_posterior(self, z0, ts, context, brownian=None):
        if len(ts) == 1:
            empty_logqp = z0.new_zeros((0, z0.shape[0]))
            return z0.unsqueeze(0), empty_logqp
        self._ctx = (ts, context)
        try:
            result = torchsde.sdeint(
                self,
                z0,
                ts,
                dt=self.solver_dt(),
                method=self.solver_method,
                logqp=True,
                bm=brownian,
            )
        finally:
            self._ctx = None
        return result

    def _integrate_prior(self, z0, ts, brownian=None):
        if len(ts) == 1:
            return z0.unsqueeze(0), z0.new_zeros((0, z0.shape[0]))
        result = torchsde.sdeint(
            self,
            z0,
            ts,
            dt=self.solver_dt(),
            method=self.solver_method,
            names={"drift": "h"},
            logqp=False,
            bm=brownian,
        )
        return result, z0.new_zeros((0, z0.shape[0]))

    def forward(
        self,
        states,
        error_targets,
        valid_mask,
        beta_kl=1.0,
        reconstruction_weight=0.1,
        error_pos_weight=None,
        sample_latent=True,
        brownian_entropy=None,
        terminal_only_error=False,
    ):
        if states.ndim != 3:
            raise ValueError("states must have shape [T, B, 18]")
        if states.shape[0] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {states.shape[0]} exceeds model max_sequence_length "
                f"{self.max_sequence_length}"
            )
        valid_mask = valid_mask.bool()
        features = self.state_encoder(states, valid_mask)
        context = _reverse_gru_with_mask(features, valid_mask, self.context_encoder)
        z0, qz0_mean, qz0_std = self._initial_posterior(
            context, sample=sample_latent
        )
        ts = self.time_grid(states.shape[0], states.device, states.dtype)

        brownian = None
        if len(ts) > 1 and brownian_entropy is not None:
            brownian = torchsde.BrownianInterval(
                t0=ts[0],
                t1=ts[-1],
                size=(z0.shape[0], z0.shape[1] + 1),
                dtype=z0.dtype,
                device=z0.device,
                entropy=int(brownian_entropy),
            )
        latents, logqp = self._integrate_posterior(
            z0, ts, context, brownian=brownian
        )

        error_logits = self.error_head(
            torch.cat([latents, features], dim=-1)
        ).squeeze(-1)
        pos_weight = None
        if error_pos_weight is not None:
            pos_weight = error_logits.new_tensor(float(error_pos_weight))
        error_values = F.binary_cross_entropy_with_logits(
            error_logits,
            error_targets.float(),
            reduction="none",
            pos_weight=pos_weight,
        )
        error_mask = valid_mask
        if terminal_only_error:
            terminal_indices = valid_mask.sum(dim=0).clamp_min(1) - 1
            error_mask = torch.zeros_like(valid_mask)
            error_mask.scatter_(0, terminal_indices.unsqueeze(0), True)
            error_mask &= valid_mask
        error_loss = masked_mean(error_values, error_mask)

        # Reconstruction: projector reconstructs same-timestep encoded features
        predicted_features = self.projector(latents)
        recon_dist = Normal(
            loc=predicted_features,
            scale=states.new_tensor(float(self.reconstruction_noise_std)),
        )
        recon_log_prob = recon_dist.log_prob(features).sum(dim=-1)
        reconstruction_loss = masked_mean(-recon_log_prob, valid_mask)

        # Path KL: mask intervals where both endpoints are valid
        if states.shape[0] > 1:
            path_valid = valid_mask[:-1] & valid_mask[1:]
            path_kl = masked_mean(logqp, path_valid)
        else:
            path_kl = states.new_zeros(())

        qz0 = Normal(qz0_mean, qz0_std)
        pz0 = Normal(
            self.pz0_mean.expand_as(qz0_mean),
            torch.exp(self.pz0_logstd).expand_as(qz0_std),
        )
        initial_kl = kl_divergence(qz0, pz0).sum(dim=-1)
        active_sequences = valid_mask.any(dim=0)
        initial_kl_loss = masked_mean(initial_kl, active_sequences)
        total_kl = initial_kl_loss + path_kl
        total_loss = (
            error_loss
            + reconstruction_weight * reconstruction_loss
            + beta_kl * total_kl
        )
        return {
            "loss": total_loss,
            "error_loss": error_loss,
            "reconstruction_loss": reconstruction_loss,
            "initial_kl_loss": initial_kl_loss,
            "path_kl_loss": path_kl,
            "kl_loss": total_kl,
            "error_logits": error_logits,
        }

    @torch.inference_mode()
    def predict_error(self, observed_states, sample=True, generator=None, brownian_entropy=None):
        if observed_states.ndim == 2:
            observed_states = observed_states.unsqueeze(1)
        if observed_states.ndim != 3 or observed_states.shape[-1] != 18:
            raise ValueError("observed_states must have shape [T, B, 18]")
        if observed_states.shape[0] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {observed_states.shape[0]} exceeds model "
                f"max_sequence_length {self.max_sequence_length}"
            )
        valid_mask = torch.ones(
            observed_states.shape[:2], dtype=torch.bool, device=observed_states.device
        )
        features = self.state_encoder(observed_states, valid_mask)
        context = _reverse_gru_with_mask(features, valid_mask, self.context_encoder)
        z0, _, _ = self._initial_posterior(context, sample=sample, generator=generator)
        ts = self.time_grid(
            observed_states.shape[0], observed_states.device, observed_states.dtype
        )
        brownian = None
        if len(ts) > 1 and brownian_entropy is not None:
            brownian = torchsde.BrownianInterval(
                t0=ts[0],
                t1=ts[-1],
                size=z0.shape,
                dtype=z0.dtype,
                device=z0.device,
                entropy=int(brownian_entropy),
            )
        latents, _ = self._integrate_prior(z0, ts, brownian=brownian)
        logits = self.error_head(
            torch.cat([latents[-1], features[-1]], dim=-1)
        ).squeeze(-1)
        return torch.sigmoid(logits - self.error_logit_offset)


class RallyNet(nn.Module):
    def __init__(
        self,
        player_count,
        shot_type_count=12,
        latent_size=32,
        context_size=64,
        hidden_size=64,
        action_hidden_size=128,
        state_embedding_dim=8,
        selected_shot_embedding_dim=8,
        gmm_components=5,
        min_sigma=1e-3,
        max_sample_sigma=1.0,
        normalization_epsilon=1e-6,
        reconstruction_noise_std=0.1,
        max_sequence_length=71,
        time_horizon=2.0,
        solver_method="euler",
        solver_substeps=2,
        latent_sde_architecture="prefix_error_sde_v2",
    ):
        super().__init__()
        if player_count < 1:
            raise ValueError("player_count must be positive")
        if shot_type_count != 12:
            raise ValueError("RallyNet requires exactly 12 shot type classes")
        if action_hidden_size < 1 or state_embedding_dim < 1:
            raise ValueError("RallyNet action dimensions must be positive")
        if selected_shot_embedding_dim < 1 or gmm_components < 1:
            raise ValueError("RallyNet action dimensions must be positive")
        if min_sigma <= 0 or max_sample_sigma <= 0:
            raise ValueError("RallyNet GMM sigma values must be positive")
        if normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        self.model_config = {
            "player_count": player_count,
            "shot_type_count": shot_type_count,
            "latent_size": latent_size,
            "context_size": context_size,
            "hidden_size": hidden_size,
            "action_hidden_size": action_hidden_size,
            "state_embedding_dim": state_embedding_dim,
            "selected_shot_embedding_dim": selected_shot_embedding_dim,
            "gmm_components": gmm_components,
            "min_sigma": min_sigma,
            "max_sample_sigma": max_sample_sigma,
            "normalization_epsilon": normalization_epsilon,
            "reconstruction_noise_std": reconstruction_noise_std,
            "max_sequence_length": max_sequence_length,
            "time_horizon": time_horizon,
            "solver_method": solver_method,
            "solver_substeps": solver_substeps,
            "latent_sde_architecture": latent_sde_architecture,
        }
        self.error_model = LatentErrorSDE(
            player_count=player_count,
            shot_type_count=shot_type_count,
            latent_size=latent_size,
            context_size=context_size,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            normalization_epsilon=normalization_epsilon,
            reconstruction_noise_std=reconstruction_noise_std,
            max_sequence_length=max_sequence_length,
            time_horizon=time_horizon,
            solver_method=solver_method,
            solver_substeps=solver_substeps,
        )
        self.action_model = ShotTypeInformedActionPredictor(
            player_count=player_count,
            shot_type_count=shot_type_count,
            hidden_size=action_hidden_size,
            state_embedding_dim=state_embedding_dim,
            shot_embedding_dim=selected_shot_embedding_dim,
            gmm_components=gmm_components,
            min_sigma=min_sigma,
            max_sample_sigma=max_sample_sigma,
            normalization_epsilon=normalization_epsilon,
        )

    def set_normalization(self, mean, std):
        self.error_model.set_normalization(mean, std)
        self.action_model.set_normalization(mean, std)

    @torch.inference_mode()
    def predict(
        self,
        history_states,
        launch,
        sample=True,
        error_threshold=0.5,
        generator=None,
        brownian_entropy=None,
    ):
        if history_states.ndim == 2:
            history_states = history_states.unsqueeze(1)
        if history_states.shape[1] != 1:
            raise ValueError("RallyNet.predict currently supports one rally at a time")
        if not sample:
            brownian_entropy = 0
        error_probability = self.error_model.predict_error(
            history_states,
            sample=sample,
            generator=generator,
            brownian_entropy=brownian_entropy,
        ).squeeze(0)
        if sample:
            draw = torch.rand(
                (), device=error_probability.device, generator=generator
            )
            is_error = bool(draw < error_probability)
        else:
            is_error = bool(error_probability >= error_threshold)
        if is_error:
            probabilities = history_states.new_zeros(12)
            probabilities[11] = 1.0
            return {
                "is_error": True,
                "error_probability": float(error_probability.item()),
                "shot_type": 11,
                "shot_probs": probabilities,
                "landing": None,
                "moving": None,
            }
        action = self.action_model.predict(
            history_states[-1], launch=launch, sample=sample, generator=generator
        )
        action["is_error"] = False
        action["error_probability"] = float(error_probability.item())
        return action


def build_checkpoint(model, player_map, training=None):
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": dict(model.model_config),
        "player_map": dict(player_map),
        "model_state": model.state_dict(),
        "training": dict(training or {}),
    }


def save_checkpoint(path, model, player_map, training=None):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(build_checkpoint(model, player_map, training), path)


def load_checkpoint(path, device="cpu", action_only=False):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a {CHECKPOINT_FORMAT} checkpoint: {path}")
    version = checkpoint.get("version")
    if version == 1:
        if action_only:
            # Load action-only weights into a fresh v2 model
            model = RallyNet(**checkpoint["model_config"])
            action_keys = {k for k in model.state_dict() if k.startswith("action_model.")}
            state = checkpoint["model_state"]
            action_state = {k: v for k, v in state.items() if k in action_keys}
            if not action_state:
                raise ValueError(f"No action_model keys found in v1 checkpoint: {path}")
            incompatible = model.load_state_dict(action_state, strict=False)
            if incompatible.missing_keys and not all(
                k.startswith("error_model.") or k.startswith("model_config")
                for k in incompatible.missing_keys
            ):
                raise RuntimeError(
                    f"Unexpected missing keys during action-only load: {incompatible.missing_keys}"
                )
            model.to(device)
            model.eval()
            return model, checkpoint
        raise ValueError(
            f"Checkpoint v1 is not compatible with v2 RallyNet. "
            f"Use action_only=True to import only the action model, then retrain the SDE."
        )
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported RallyNet checkpoint version: {version}"
        )
    model = RallyNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


class RallyNetPolicy:
    def __init__(
        self,
        model_path,
        device="cuda",
        seed=42,
        sample=True,
        error_threshold=0.5,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.seed = int(seed)
        self.sample = sample
        self.error_threshold = float(error_threshold)
        if not 0.0 <= self.error_threshold <= 1.0:
            raise ValueError("error_threshold must be between 0 and 1")
        set_seed(self.seed)
        self.model, self.checkpoint = load_checkpoint(model_path, self.device)
        if not self.checkpoint.get("training", {}).get("complete", False):
            raise ValueError(
                "RallyNet policy requires a checkpoint with completed BC and SDE stages"
            )
        self.player_map = self.checkpoint["player_map"]
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.seed)
        self.prediction_count = 0

    def predict_next(self, state_history, launch):
        if isinstance(state_history, torch.Tensor):
            history = state_history.to(self.device, dtype=torch.float32)
        else:
            history = torch.stack([
                torch.as_tensor(state, dtype=torch.float32)
                for state in state_history
            ]).to(self.device)
        entropy = self.seed + self.prediction_count
        self.prediction_count += 1
        return self.model.predict(
            history,
            launch=launch,
            sample=self.sample,
            error_threshold=self.error_threshold,
            generator=self.generator,
            brownian_entropy=entropy,
        )


class RallyNetRollout:
    def __init__(self, policy):
        self.policy = policy
        self.history = []

    def reset(self):
        self.history = []

    def step(self, current_state, launch=False):
        state = torch.as_tensor(current_state, dtype=torch.float32).detach().cpu()
        self.history.append(state)
        prediction = self.policy.predict_next(self.history, launch=launch)
        if prediction["is_error"]:
            action = torch.tensor([0.0, 1.0, 11.0, 0.0, 0.0], dtype=torch.float32)
        else:
            landing = prediction["landing"].detach().cpu()
            moving = prediction["moving"].detach().cpu()
            action = torch.tensor(
                [
                    landing[0].item(),
                    landing[1].item(),
                    float(prediction["shot_type"]),
                    moving[0].item(),
                    moving[1].item(),
                ],
                dtype=torch.float32,
            )
        return action, prediction


class RallyNetAgent(BaseAgent):
    def __init__(
        self,
        player_name,
        model_path,
        player_id=None,
        device="cuda",
        seed=42,
        sample=True,
        error_threshold=0.5,
        opponent_id=None,
    ):
        super().__init__(player_name)
        self.policy = RallyNetPolicy(
            model_path,
            device=device,
            seed=seed,
            sample=sample,
            error_threshold=error_threshold,
        )
        if player_id is None:
            if player_name not in self.policy.player_map:
                raise ValueError(f"Player '{player_name}' not found in checkpoint player map")
            player_id = self.policy.player_map[player_name]
        self.player_id = int(player_id)
        self.opponent_id = None if opponent_id is None else int(opponent_id)
        self.state_history = []
        self._history_key = None
        self._last_round = None
        self._last_landing = None
        self._pending_opponent_state = None

    def reset_history(self):
        self.state_history = []
        self._history_key = None
        self._last_round = None
        self._last_landing = None
        self._pending_opponent_state = None

    def _opponent_id(self, info):
        if self.opponent_id is not None:
            return self.opponent_id
        for key in (
            "opponent_player",
            "opponent_player_name",
            "opponent_name",
            "opponent",
        ):
            player_name = info.get(key)
            if player_name in self.policy.player_map:
                return int(self.policy.player_map[player_name])
        players = info.get("player") or []
        for player_name in reversed(players):
            if player_name != self.player_name and player_name in self.policy.player_map:
                return int(self.policy.player_map[player_name])
        raise ValueError(
            "RallyNetAgent requires opponent_id or opponent player information"
        )

    def _translate_state(self, state, info):
        opponent_type, player_x, player_y, opponent_x, opponent_y, hit_x, hit_y = (
            self.encode_state(state)
        )
        rally = int(info.get("rally", 1))
        if rally < 0:
            rally = 1
        rounds = info.get("round") or [1]
        scores = info.get("score") or [0, 0]
        ball_round = int(rounds[-1])
        distance_x = hit_x - player_x
        distance_y = hit_y - player_y
        opponent_move_x = 0.0
        opponent_move_y = 0.0
        if self._last_landing is not None:
            opponent_move_x = opponent_x - self._last_landing[0]
            opponent_move_y = opponent_y - self._last_landing[1]
        return torch.tensor(
            [
                rally,
                ball_round,
                scores[0],
                scores[1],
                scores[0] - scores[1],
                player_x,
                player_y,
                (distance_x ** 2 + distance_y ** 2) ** 0.5,
                distance_x,
                distance_y,
                hit_x,
                hit_y,
                opponent_type,
                opponent_x,
                opponent_y,
                opponent_move_x,
                opponent_move_y,
                self.player_id,
            ],
            dtype=torch.float32,
        )

    def predict(self, state, info, launch):
        rounds = info.get("round") or [1]
        ball_round = int(rounds[-1])
        history_key = (info.get("match_id"), info.get("rally"))
        if (
            self._history_key != history_key
            or (self._last_round is not None and ball_round <= self._last_round)
        ):
            self.reset_history()
        previous_round = self._last_round
        self._history_key = history_key
        self._last_round = ball_round
        current_state = self._translate_state(state, info)
        if (
            self._pending_opponent_state is not None
            and previous_round is not None
            and ball_round >= previous_round + 2
        ):
            self.state_history.append(self._pending_opponent_state)
        self._pending_opponent_state = None
        self.state_history.append(current_state)
        prediction = self.policy.predict_next(self.state_history, launch=launch)
        if prediction["is_error"]:
            return None, "RallyNet predicted a missed return"

        self._last_landing = prediction["landing"].detach().cpu().tolist()
        model_action = torch.tensor(
            [
                self._last_landing[0],
                self._last_landing[1],
                float(prediction["shot_type"]),
                float(prediction["moving"][0].item()),
                float(prediction["moving"][1].item()),
            ],
            dtype=torch.float32,
        )
        self._pending_opponent_state = transition(
            model_action, current_state, self._opponent_id(info)
        )

        shot_probabilities = prediction["shot_probs"][1:11]
        shot_probabilities = shot_probabilities / shot_probabilities.sum().clamp_min(1e-12)
        action = (
            prediction["shot_type"],
            state[-1],
            prediction["landing"].detach().cpu().tolist(),
            prediction["moving"].detach().cpu().tolist(),
            tuple(shot_probabilities.detach().cpu().tolist()),
        )
        return action, ""

    def action(self, state, info, launch):
        normalized_action, reason = self.predict(state, info, launch)
        if normalized_action is None:
            return None, reason
        return self.decode_action(normalized_action, reason)


def load_player_map(path):
    with open(path, "r") as f:
        return json.load(f)
