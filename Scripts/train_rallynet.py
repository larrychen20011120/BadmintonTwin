#!/usr/bin/env python3
"""Train the two-stage RallyNet opponent model."""

import argparse
import json
import os
import pickle
import sys

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from Agent.rallynet import (  # noqa: E402
    CONTINUOUS_STATE_INDICES,
    RallyNet,
    gmm_nll,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from prepare_rallynet import compute_rallynet_dataset_fingerprint  # noqa: E402
from Utils.runtime_report import print_runtime_box  # noqa: E402


class BCRowDataset(Dataset):
    def __init__(self, rallies):
        states = []
        actions = []
        coordinate_masks = []
        for rally in rallies:
            error_mask = ~rally["error_targets"]
            if not error_mask.any():
                continue
            states.append(torch.from_numpy(rally["states"][error_mask]))
            actions.append(torch.from_numpy(rally["actions"][error_mask]))
            coordinate_masks.append(
                torch.from_numpy(rally["coordinate_mask"][error_mask])
            )
        if not states:
            raise ValueError("No non-error BC rows were found")
        self.states = torch.cat(states).float()
        self.actions = torch.cat(actions).float()
        self.coordinate_masks = torch.cat(coordinate_masks).bool()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.coordinate_masks[index]


class RallySequenceDataset(Dataset):
    def __init__(self, rallies):
        self.rallies = rallies

    def __len__(self):
        return len(self.rallies)

    def __getitem__(self, index):
        rally = self.rallies[index]
        return (
            torch.from_numpy(rally["states"]).float(),
            torch.from_numpy(rally["error_targets"]).float(),
        )


def collate_rallies(items):
    max_length = max(states.shape[0] for states, _ in items)
    batch_size = len(items)
    states = torch.zeros(max_length, batch_size, 18, dtype=torch.float32)
    errors = torch.zeros(max_length, batch_size, dtype=torch.float32)
    valid_mask = torch.zeros(max_length, batch_size, dtype=torch.bool)
    for batch_index, (rally_states, rally_errors) in enumerate(items):
        length = rally_states.shape[0]
        states[:length, batch_index] = rally_states
        errors[:length, batch_index] = rally_errors
        valid_mask[:length, batch_index] = True
    return states, errors, valid_mask


def collate_rally_prefixes(items):
    """Sample one observed prefix per rally for causal error training."""
    prefixes = []
    for states, errors in items:
        prefix_length = int(torch.randint(1, states.shape[0] + 1, ()).item())
        prefixes.append((states[:prefix_length], errors[:prefix_length]))
    return collate_rallies(prefixes)


def load_config(config_path=None):
    config_path = config_path or os.path.join(PROJECT_PATH, "Scripts", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f), os.path.abspath(config_path)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_PATH, path)


def load_training_data(data_dir):
    rallies_path = os.path.join(data_dir, "rallies.pkl")
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.isfile(rallies_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Missing RallyNet data in {data_dir}; run Scripts/prepare_all_data.py first"
        )
    with open(rallies_path, "rb") as f:
        payload = pickle.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if payload.get("schema_version") != 1 or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported RallyNet data schema")
    if payload.get("player_map") != metadata.get("player_map"):
        raise ValueError("RallyNet payload and metadata player maps do not match")
    if payload.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
        raise ValueError("RallyNet payload and metadata fingerprints do not match")
    rallies = payload.get("rallies", [])
    if not rallies:
        raise ValueError("RallyNet data contains no rallies")
    for rally in rallies:
        lengths = {
            len(rally[key])
            for key in ("states", "actions", "error_targets", "serve_mask", "coordinate_mask")
        }
        if lengths != {int(rally["length"])}:
            raise ValueError("RallyNet rally arrays and declared length do not match")
    fingerprint = compute_rallynet_dataset_fingerprint(rallies, payload["player_map"])
    if fingerprint != payload.get("dataset_fingerprint"):
        raise ValueError("RallyNet rally content does not match its fingerprint")
    return rallies, metadata


def training_coverage_issues(rallies):
    coordinate_shots = set()
    error_count = 0
    for rally in rallies:
        actions = rally["actions"]
        coordinate_mask = rally["coordinate_mask"]
        coordinate_shots.update(
            int(shot_type)
            for shot_type in actions[coordinate_mask, 2].tolist()
            if 1 <= int(shot_type) <= 10
        )
        error_count += int(rally["error_targets"].sum())
    issues = []
    missing_shots = sorted(set(range(1, 11)) - coordinate_shots)
    if missing_shots:
        issues.append(f"missing coordinate specialists {missing_shots}")
    if error_count == 0:
        issues.append("missing Type 11 labels")
    return issues


def compute_normalization(rallies, epsilon):
    states = torch.cat([
        torch.from_numpy(rally["states"]).float() for rally in rallies
    ])
    continuous = states[:, CONTINUOUS_STATE_INDICES]
    mean = continuous.mean(dim=0)
    std = continuous.std(dim=0, unbiased=False)
    std = torch.where(std < epsilon, torch.ones_like(std), std)
    return mean, std


def action_batch_loss(action_model, states, actions, coordinate_mask):
    selected_shots = actions[:, 2].long()
    shot_logits, landing_params, movement_params = action_model(states, selected_shots)
    shot_loss = nn.functional.cross_entropy(shot_logits, selected_shots)
    landing_values = gmm_nll(
        landing_params,
        actions[:, :2],
        action_model.gmm_components,
        action_model.min_sigma,
        reduction="none",
    )
    movement_values = gmm_nll(
        movement_params,
        actions[:, 3:5],
        action_model.gmm_components,
        action_model.min_sigma,
        reduction="none",
    )
    coordinate_mask = coordinate_mask.to(landing_values.dtype)
    coordinate_count = coordinate_mask.sum().clamp_min(1.0)
    landing_loss = (landing_values * coordinate_mask).sum() / coordinate_count
    movement_loss = (movement_values * coordinate_mask).sum() / coordinate_count
    return {
        "loss": shot_loss + landing_loss + movement_loss,
        "shot_loss": shot_loss,
        "landing_loss": landing_loss,
        "movement_loss": movement_loss,
    }


def train_bc_stage(
    model,
    train_rallies,
    device,
    epochs,
    batch_size,
    learning_rate,
    gradient_clip_norm,
    num_workers,
    pin_memory,
    checkpoint_path,
    player_map,
    provenance,
    training_config,
):
    train_loader = DataLoader(
        BCRowDataset(train_rallies),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.Adam(model.action_model.parameters(), lr=learning_rate)
    train_loss = None
    for epoch in range(1, epochs + 1):
        model.action_model.train()
        running_loss = 0.0
        sample_count = 0
        for states, actions, coordinate_mask in tqdm(
            train_loader, desc=f"BC epoch {epoch}/{epochs}", leave=False
        ):
            states = states.to(device)
            actions = actions.to(device)
            coordinate_mask = coordinate_mask.to(device)
            losses = action_batch_loss(
                model.action_model, states, actions, coordinate_mask
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(
                model.action_model.parameters(), gradient_clip_norm
            )
            optimizer.step()
            running_loss += float(losses["loss"].item()) * states.shape[0]
            sample_count += states.shape[0]

        train_loss = running_loss / max(sample_count, 1)
        if not torch.isfinite(torch.tensor(train_loss)):
            raise RuntimeError("BC training loss is not finite")
        print(f"BC epoch {epoch:03d}: train={train_loss:.4f}")

    save_checkpoint(
        checkpoint_path,
        model,
        player_map,
        {
            "stage": "bc",
            "bc_trained": True,
            "sde_trained": False,
            "complete": False,
            "epoch": epochs,
            "train_loss": train_loss,
            "provenance": provenance,
            "config": dict(training_config),
        },
    )
    return train_loss


def error_positive_weight(rallies, configured_weight):
    positives = sum(int(rally["error_targets"].sum()) for rally in rallies)
    total = sum(int(rally["length"]) for rally in rallies)
    if positives == 0:
        raise ValueError("No Type 11 labels available for SDE training")
    if configured_weight is not None:
        return float(configured_weight)
    return (total - positives) / positives


def train_sde_stage(
    model,
    train_rallies,
    device,
    num_iters,
    batch_size,
    learning_rate,
    lr_gamma,
    kl_weight,
    kl_anneal_steps,
    reconstruction_weight,
    pos_weight,
    gradient_clip_norm,
    num_workers,
    pin_memory,
    checkpoint_interval,
    checkpoint_path,
    player_map,
    provenance,
    training_config,
):
    for parameter in model.action_model.parameters():
        parameter.requires_grad_(False)
    train_loader = DataLoader(
        RallySequenceDataset(train_rallies),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_rally_prefixes,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.Adam(model.error_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    model.error_model.set_error_pos_weight(pos_weight)
    model.error_model.train()
    loader_iterator = iter(train_loader)
    train_error = None
    for step in tqdm(range(1, num_iters + 1), desc="RallyNet steps", leave=False):
        try:
            states, errors, valid_mask = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(train_loader)
            states, errors, valid_mask = next(loader_iterator)
        beta = kl_weight * min(1.0, step / max(kl_anneal_steps, 1))
        states = states.to(device)
        errors = errors.to(device)
        valid_mask = valid_mask.to(device)
        losses = model.error_model(
            states,
            errors,
            valid_mask,
            beta_kl=beta,
            reconstruction_weight=reconstruction_weight,
            error_pos_weight=pos_weight,
            terminal_only_error=True,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.error_model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        train_error = float(losses["error_loss"].item())
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("RallyNet training loss is not finite")
        if step % checkpoint_interval == 0 or step == num_iters:
            recon = float(losses["reconstruction_loss"].item())
            kl = float(losses["kl_loss"].item())
            print(
                f"RallyNet step {step:05d}/{num_iters}: "
                f"error={train_error:.4f} recon={recon:.4f} kl={kl:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.8g} beta={beta:.4f}"
            )
        if step % checkpoint_interval == 0 and step < num_iters:
            periodic_path = os.path.join(
                os.path.dirname(checkpoint_path), f"sde_step_{step}.pth"
            )
            save_checkpoint(
                periodic_path,
                model,
                player_map,
                {
                    "stage": "sde",
                    "bc_trained": True,
                    "sde_trained": True,
                    "complete": False,
                    "step": step,
                    "train_error_loss": train_error,
                    "error_pos_weight": pos_weight,
                    "provenance": provenance,
                    "config": dict(training_config),
                },
            )

    save_checkpoint(
        checkpoint_path,
        model,
        player_map,
        {
            "stage": "sde",
            "bc_trained": True,
            "sde_trained": True,
            "complete": True,
            "step": num_iters,
            "train_error_loss": train_error,
            "error_pos_weight": pos_weight,
            "provenance": provenance,
            "config": dict(training_config),
        },
    )
    for parameter in model.action_model.parameters():
        parameter.requires_grad_(True)
    return train_error


def train_rallynet(
    experiment="rally",
    stage="all",
    config_path=None,
    data_dir=None,
    output_dir=None,
    model_path=None,
    resume=None,
    action_checkpoint=None,
    device=None,
    overrides=None,
):
    config, resolved_config_path = load_config(config_path)
    if experiment != "rally":
        raise ValueError("RallyNet only supports the physical-rally experiment")
    training = dict(config["rallynet_training"])
    training.update(overrides or {})
    paths = config["paths"]["rallynet"]
    data_dir = resolve_path(data_dir or paths["data"])
    output_dir = resolve_path(output_dir or paths["checkpoint_dir"])
    model_path = resolve_path(model_path or paths["model"])
    os.makedirs(output_dir, exist_ok=True)

    device_name = device or training.get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    training["device"] = device_name
    seed = int(training["seed"])
    set_seed(seed)
    if stage in ("bc", "all") and int(training["bc_epochs"]) < 1:
        raise ValueError("bc_epochs must be at least 1 for the selected stage")
    if stage in ("rallynet", "all") and int(training["rallynet_num_iters"]) < 1:
        raise ValueError("rallynet_num_iters must be at least 1 for the selected stage")
    if int(training["checkpoint_interval"]) < 1:
        raise ValueError("checkpoint_interval must be at least 1")
    if not 0.0 < float(training["lr_gamma"]) <= 1.0:
        raise ValueError("lr_gamma must be between zero and one")
    positive_values = {
        "bc_batch_size": training["bc_batch_size"],
        "rallynet_batch_size": training["rallynet_batch_size"],
        "bc_learning_rate": training["bc_learning_rate"],
        "rallynet_learning_rate": training["rallynet_learning_rate"],
        "gradient_clip_norm": training["gradient_clip_norm"],
        "normalization_epsilon": training["normalization_epsilon"],
        "reconstruction_noise_std": training["reconstruction_noise_std"],
        "time_horizon": training["time_horizon"],
        "solver_substeps": training["solver_substeps"],
        "gmm_components": training["gmm_components"],
        "min_sigma": training["min_sigma"],
        "max_sample_sigma": training["max_sample_sigma"],
    }
    invalid_positive = [key for key, value in positive_values.items() if float(value) <= 0]
    if invalid_positive:
        raise ValueError(f"RallyNet training values must be positive: {invalid_positive}")
    if int(training["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative")

    train_rallies, metadata = load_training_data(data_dir)
    if metadata.get("experiment") not in (None, experiment):
        raise ValueError("Prepared RallyNet data belongs to a different experiment")
    coverage_issues = training_coverage_issues(train_rallies)
    if coverage_issues:
        raise ValueError(f"Invalid RallyNet training data: {coverage_issues}")
    longest_rally = max(int(rally["length"]) for rally in train_rallies)
    rollout_extra_steps = int(config.get("evaluation", {}).get("rollout_extra_steps", 2))
    if rollout_extra_steps < 0:
        raise ValueError("evaluation.rollout_extra_steps must be non-negative")
    required_sequence_length = longest_rally + rollout_extra_steps
    with open(resolve_path(config["paths"]["player_map"]), "r") as f:
        player_map = json.load(f)
    if metadata.get("player_map") != player_map:
        raise ValueError("Prepared RallyNet player map does not match configured player map")
    expected_ids = list(range(len(player_map)))
    if sorted(player_map.values()) != expected_ids:
        raise ValueError("RallyNet player IDs must be contiguous from zero")
    provenance = {
        "experiment": experiment,
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "split_sha256": metadata.get("split_sha256"),
        "seed": seed,
        "train_rally_keys": metadata.get("train_rally_keys", []),
        "requested_train_rally_keys": metadata.get("requested_train_rally_keys", []),
        "skipped_train_rallies": metadata.get("skipped_train_rallies", []),
        "train_match_ids": sorted({int(rally["match_id"]) for rally in train_rallies}),
        "excluded_test_match_ids": metadata.get("excluded_test_match_ids", []),
    }
    loaded_checkpoint_path = None
    if resume:
        loaded_checkpoint_path = resolve_path(resume)
        model, checkpoint = load_checkpoint(loaded_checkpoint_path, device)
        if checkpoint["player_map"] != player_map:
            raise ValueError("Resume checkpoint player map does not match current data")
        if checkpoint.get("training", {}).get("provenance") != provenance:
            raise ValueError("Warm-start checkpoint data provenance does not match")
    elif action_checkpoint and stage in ("rallynet", "all"):
        action_path = resolve_path(action_checkpoint)
        model, checkpoint = load_checkpoint(action_path, device, action_only=True)
        loaded_checkpoint_path = action_path
        if checkpoint["player_map"] != player_map:
            raise ValueError("Action checkpoint player map does not match current data")
        if checkpoint.get("training", {}).get("provenance") != provenance:
            raise ValueError("Action checkpoint data provenance does not match")
        mean, std = compute_normalization(
            train_rallies, float(training["normalization_epsilon"])
        )
        model.set_normalization(mean.to(device), std.to(device))
    else:
        model = RallyNet(
            player_count=len(player_map),
            shot_type_count=int(metadata["shot_type_count"]),
            latent_size=int(training["latent_size"]),
            context_size=int(training["context_size"]),
            hidden_size=int(training["hidden_size"]),
            action_hidden_size=int(training["action_hidden_size"]),
            state_embedding_dim=int(training["state_embedding_dim"]),
            selected_shot_embedding_dim=int(
                training["selected_shot_embedding_dim"]
            ),
            gmm_components=int(training["gmm_components"]),
            min_sigma=float(training["min_sigma"]),
            max_sample_sigma=float(training["max_sample_sigma"]),
            normalization_epsilon=float(training["normalization_epsilon"]),
            reconstruction_noise_std=float(training["reconstruction_noise_std"]),
            max_sequence_length=int(training["max_sequence_length"]),
            time_horizon=float(training["time_horizon"]),
            solver_method=training["solver_method"],
            solver_substeps=int(training["solver_substeps"]),
        ).to(device)
        mean, std = compute_normalization(
            train_rallies, float(training["normalization_epsilon"])
        )
        model.set_normalization(mean.to(device), std.to(device))

    model_sequence_length = int(model.model_config["max_sequence_length"])
    if model_sequence_length < required_sequence_length:
        raise ValueError(
            "RallyNet model max_sequence_length must cover the longest training rally "
            f"plus evaluation rollout steps ({required_sequence_length}); got "
            f"{model_sequence_length}"
        )

    if stage == "rallynet":
        if loaded_checkpoint_path is None:
            bc_checkpoint_path = os.path.join(output_dir, "bc_final.pth")
            if not os.path.isfile(bc_checkpoint_path):
                raise FileNotFoundError(
                    "RallyNet-only training requires --resume or an existing bc_final.pth"
                )
            model, checkpoint = load_checkpoint(bc_checkpoint_path, device)
            loaded_checkpoint_path = bc_checkpoint_path
            if checkpoint["player_map"] != player_map:
                raise ValueError("BC checkpoint player map does not match current data")
        checkpoint_training = checkpoint.get("training", {})
        if not checkpoint_training.get("bc_trained", False):
            raise ValueError(
                "RallyNet-only training requires a checkpoint with trained BC specialists"
            )
        if checkpoint_training.get("provenance") != provenance:
            raise ValueError("BC checkpoint data provenance does not match")

    for key, value in model.model_config.items():
        if key in training:
            training[key] = value

    sde_pos_weight = None
    if stage in ("rallynet", "all"):
        sde_pos_weight = error_positive_weight(
            train_rallies, training.get("error_pos_weight")
        )
        training["error_pos_weight"] = sde_pos_weight

    model_config = model.model_config
    runtime_rows = [
        ("config", resolved_config_path),
        ("experiment", experiment),
        ("stage", stage),
        ("device", device),
        ("seed", seed),
        ("resume", loaded_checkpoint_path or "<none>"),
        ("data_dir", data_dir),
        ("output_dir", output_dir),
        ("model_path", model_path),
        ("train_rallies", len(train_rallies)),
        ("model.latent_size", model_config["latent_size"]),
        ("model.context_size", model_config["context_size"]),
        ("model.hidden_size", model_config["hidden_size"]),
        ("model.action_hidden_size", model_config["action_hidden_size"]),
        ("model.gmm_components", model_config["gmm_components"]),
        ("model.max_sequence_length", model_config["max_sequence_length"]),
    ]
    if stage in ("bc", "all"):
        runtime_rows.extend([
            ("bc.epochs", training["bc_epochs"]),
            ("bc.batch_size", training["bc_batch_size"]),
            ("bc.learning_rate", training["bc_learning_rate"]),
        ])
    if stage in ("rallynet", "all"):
        runtime_rows.extend([
            ("rallynet.num_iters", training["rallynet_num_iters"]),
            ("rallynet.batch_size", training["rallynet_batch_size"]),
            ("rallynet.learning_rate", training["rallynet_learning_rate"]),
            ("rallynet.lr_gamma", training["lr_gamma"]),
            ("rallynet.kl_weight", training["kl_weight"]),
            ("rallynet.kl_anneal_steps", training["kl_anneal_steps"]),
            ("rallynet.reconstruction_weight", training["reconstruction_weight"]),
            ("rallynet.error_pos_weight", sde_pos_weight),
            ("rallynet.solver_method", model_config["solver_method"]),
            ("rallynet.solver_substeps", model_config["solver_substeps"]),
        ])
    print_runtime_box("Runtime Parameters: train_rallynet", runtime_rows)

    metrics = {"data_schema_version": metadata["schema_version"]}
    if stage in ("bc", "all"):
        metrics["bc_train_loss"] = train_bc_stage(
            model=model,
            train_rallies=train_rallies,
            device=device,
            epochs=int(training["bc_epochs"]),
            batch_size=int(training["bc_batch_size"]),
            learning_rate=float(training["bc_learning_rate"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            num_workers=int(training["num_workers"]),
            pin_memory=bool(training["pin_memory"]),
            checkpoint_path=os.path.join(output_dir, "bc_final.pth"),
            player_map=player_map,
            provenance=provenance,
            training_config=training,
        )

    if stage in ("rallynet", "all"):
        metrics["rallynet_train_error_loss"] = train_sde_stage(
            model=model,
            train_rallies=train_rallies,
            device=device,
            num_iters=int(training["rallynet_num_iters"]),
            batch_size=int(training["rallynet_batch_size"]),
            learning_rate=float(training["rallynet_learning_rate"]),
            lr_gamma=float(training["lr_gamma"]),
            kl_weight=float(training["kl_weight"]),
            kl_anneal_steps=int(training["kl_anneal_steps"]),
            reconstruction_weight=float(training["reconstruction_weight"]),
            pos_weight=sde_pos_weight,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            num_workers=int(training["num_workers"]),
            pin_memory=bool(training["pin_memory"]),
            checkpoint_interval=int(training["checkpoint_interval"]),
            checkpoint_path=os.path.join(output_dir, "sde_final.pth"),
            player_map=player_map,
            provenance=provenance,
            training_config=training,
        )
        metrics["error_pos_weight"] = sde_pos_weight

    if stage == "bc":
        result_path = os.path.join(output_dir, "bc_final.pth")
        print(f"BC stage checkpoint saved to {result_path}")
        return result_path

    save_checkpoint(
        model_path,
        model,
        player_map,
        {
            "stage": stage,
            "bc_trained": True,
            "sde_trained": True,
            "complete": True,
            "experiment": experiment,
            "seed": seed,
            "provenance": provenance,
            "config": dict(training),
            **metrics,
        },
    )
    print(f"RallyNet checkpoint saved to {model_path}")
    return model_path


def build_parser():
    parser = argparse.ArgumentParser(description="Train RallyNet")
    parser.add_argument("--experiment", default="rally", choices=["rally"])
    parser.add_argument("--stage", default="all", choices=["bc", "rallynet", "all"])
    parser.add_argument("--config")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path")
    parser.add_argument(
        "--resume", "--warm-start", dest="resume",
        help="Warm-start model weights; optimizer and scheduler state are not restored",
    )
    parser.add_argument(
        "--action-checkpoint",
        help="Load only action_model weights from a v1 checkpoint; SDE is initialized from scratch",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--rallynet-num-iters", type=int)
    parser.add_argument("--bc-batch-size", type=int)
    parser.add_argument("--rallynet-batch-size", type=int)
    parser.add_argument("--bc-lr", type=float)
    parser.add_argument("--rallynet-lr", type=float)
    parser.add_argument("--lr-gamma", type=float)
    parser.add_argument("--latent-size", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--action-hidden-size", type=int)
    parser.add_argument("--state-embedding-dim", type=int)
    parser.add_argument("--selected-shot-embedding-dim", type=int)
    parser.add_argument("--gmm-components", type=int)
    parser.add_argument("--min-sigma", type=float)
    parser.add_argument("--max-sample-sigma", type=float)
    parser.add_argument("--reconstruction-noise-std", type=float)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--time-horizon", type=float)
    parser.add_argument("--solver-method")
    parser.add_argument("--solver-substeps", type=int)
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--kl-anneal-steps", type=int)
    parser.add_argument("--reconstruction-weight", type=float)
    parser.add_argument("--error-pos-weight", type=float)
    return parser


def main():
    args = build_parser().parse_args()
    overrides = {
        key: value
        for key, value in {
            "seed": args.seed,
            "bc_epochs": args.bc_epochs,
            "rallynet_num_iters": args.rallynet_num_iters,
            "bc_batch_size": args.bc_batch_size,
            "rallynet_batch_size": args.rallynet_batch_size,
            "bc_learning_rate": args.bc_lr,
            "rallynet_learning_rate": args.rallynet_lr,
            "lr_gamma": args.lr_gamma,
            "latent_size": args.latent_size,
            "context_size": args.context_size,
            "hidden_size": args.hidden_size,
            "action_hidden_size": args.action_hidden_size,
            "state_embedding_dim": args.state_embedding_dim,
            "selected_shot_embedding_dim": args.selected_shot_embedding_dim,
            "gmm_components": args.gmm_components,
            "min_sigma": args.min_sigma,
            "max_sample_sigma": args.max_sample_sigma,
            "reconstruction_noise_std": args.reconstruction_noise_std,
            "max_sequence_length": args.max_sequence_length,
            "time_horizon": args.time_horizon,
            "solver_method": args.solver_method,
            "solver_substeps": args.solver_substeps,
            "kl_weight": args.kl_weight,
            "kl_anneal_steps": args.kl_anneal_steps,
            "reconstruction_weight": args.reconstruction_weight,
            "error_pos_weight": args.error_pos_weight,
        }.items()
        if value is not None
    }
    train_rallynet(
        experiment=args.experiment,
        stage=args.stage,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        resume=args.resume,
        action_checkpoint=args.action_checkpoint,
        device=args.device,
        overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
