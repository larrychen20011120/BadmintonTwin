import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jinja2 import Template
from torch.nn import CTCLoss
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Agent.naive_rag import NaiveRAGVerbalizer
from Utils.badminton_config import INDEX2EN_SHOT, LANG_AREA_MAPPING
from Utils.artifact_names import player_database_filename
from Utils.helper import action_col, coord2area, edit_distance, normalized_dtw_ndim_distance, state_col, terminate, transition
from Utils.langfuse_tracing import BadmintonTracer
from Utils.llm import LLM


DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_PREFIXES = {
    "document": "search_document: ",
    "query": "search_query: ",
}
EMBEDDING_PREFIX_CONTRACT_VERSION = 2


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_player_name(player_name):
    return str(player_name).replace(" ", "_").replace(".", "")


def get_embedding_dimension(model):
    dimension = None
    if hasattr(model, "get_embedding_dimension"):
        dimension = model.get_embedding_dimension()
    elif hasattr(model, "get_sentence_embedding_dimension"):
        dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        probe = model.encode(["embedding dimension probe"], convert_to_numpy=True, show_progress_bar=False)
        dimension = np.asarray(probe).shape[-1]
    return int(dimension)


def embedding_prefix(model_name, input_type):
    model_name = str(model_name or "").lower()
    if "nomic" in model_name:
        return EMBEDDING_PREFIXES[input_type]
    if "bge" in model_name and input_type == "query":
        return "Represent this sentence for searching relevant passages: "
    return ""


def encode_texts(model, texts, embedding_dim=None, batch_size=None, normalize_embeddings=True,
                 input_type=None, model_name=None):
    embedding_dim = int(embedding_dim or get_embedding_dimension(model))
    if not texts:
        return np.zeros((0, embedding_dim), dtype=np.float32)
    if input_type not in EMBEDDING_PREFIXES:
        raise ValueError("Twin embedding input_type must be document or query")
    prefix = embedding_prefix(model_name, input_type)
    texts = [f"{prefix}{text}" for text in texts]
    encode_kwargs = {}
    if batch_size is not None:
        encode_kwargs["batch_size"] = int(batch_size)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize_embeddings),
        show_progress_bar=False,
        **encode_kwargs,
    )
    return np.asarray(embeddings, dtype=np.float32)


def twin_memory_digest(memories):
    """Fingerprint all ordered fields that affect Twin retrieval or actions."""
    digest_keys = (
        "memory_id", "semantic_text", "player_type", "opponent_type",
        "player_location_x", "player_location_y", "hit_x", "hit_y",
        "opponent_location_x", "opponent_location_y",
        "landing_x", "landing_y", "moving_x", "moving_y",
    )
    records = [
        {key: memory.get(key) for key in digest_keys}
        for memory in memories
    ]
    payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embedding_model_key(model_name, revision=None, normalize_embeddings=True, dtype="float32",
                        provider="local", embedding_dimension=None):
    spec = {
        "provider": str(provider),
        "model_name": str(model_name),
        "revision": revision,
        "embedding_dimension": embedding_dimension,
        "normalize_embeddings": bool(normalize_embeddings),
        "dtype": str(dtype),
        "prefix_contract_version": EMBEDDING_PREFIX_CONTRACT_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    slug = re.sub(
        r"[^A-Za-z0-9._-]+", "_", f"{provider}_{model_name}"
    ).strip("_") or "embedding"
    return f"{slug}_{digest}"


def embedding_cache_paths(cache_root, player_name, model_name, revision=None,
                          normalize_embeddings=True, dtype="float32", provider="local",
                          embedding_dimension=None):
    model_dir = os.path.join(
        cache_root,
        embedding_model_key(
            model_name, revision, normalize_embeddings, dtype, provider,
            embedding_dimension,
        ),
    )
    database_stem = os.path.splitext(player_database_filename(player_name))[0]
    basename = f"{database_stem.removesuffix('_db')}_embeddings"
    return os.path.join(model_dir, f"{basename}.npy"), os.path.join(model_dir, f"{basename}.json")


class TwinEmbeddingCache:
    """Model-specific, disposable semantic embeddings for a TwinDatabase."""

    schema_version = 4

    def __init__(self, player_name, model_name, embeddings, memory_digest,
                 revision=None, normalize_embeddings=True, dtype="float32",
                 provider="local", configured_dimension=None):
        self.player_name = str(player_name)
        self.provider = str(provider)
        self.model_name = str(model_name)
        self.revision = revision
        self.configured_dimension = (
            int(configured_dimension) if configured_dimension is not None else None
        )
        self.normalize_embeddings = bool(normalize_embeddings)
        self.dtype = str(dtype)
        self.embeddings = np.asarray(embeddings)
        self.memory_digest = str(memory_digest)
        self.embedding_dim = int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    @classmethod
    def build(cls, database, model, model_name, revision=None, batch_size=None,
              normalize_embeddings=True, dtype="float32", provider="local",
              embedding_dimension=None):
        if dtype not in {"float16", "float32"}:
            raise ValueError("Twin embedding dtype must be float16 or float32")
        embedding_dim = get_embedding_dimension(model)
        if embedding_dimension is not None and embedding_dim != int(embedding_dimension):
            raise ValueError(
                f"Twin embedding dimension mismatch for {database.player_name}: model produces "
                f"{embedding_dim} dimensions, expected {embedding_dimension}."
            )
        texts = [memory.get("semantic_text", "") for memory in database.memories]
        embeddings = encode_texts(
            model,
            texts,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            input_type="document",
            model_name=model_name,
        ).astype(dtype, copy=False)
        return cls(
            database.player_name,
            model_name,
            embeddings,
            twin_memory_digest(database.memories),
            revision=revision,
            normalize_embeddings=normalize_embeddings,
            dtype=dtype,
            provider=provider,
            configured_dimension=embedding_dimension,
        )

    def validate(self, database, model_name=None, revision=None,
                 normalize_embeddings=None, dtype=None, provider=None,
                 embedding_dimension=None):
        if self.player_name != database.player_name:
            raise ValueError(
                f"Twin embedding player mismatch: cache has {self.player_name}, "
                f"memory has {database.player_name}."
            )
        if model_name is not None and (
            self.model_name != str(model_name) or self.revision != revision
        ):
            raise ValueError(
                f"Twin embedding model mismatch for {database.player_name}: cache uses "
                f"{self.model_name}@{self.revision}, expected {model_name}@{revision}."
            )
        if provider is not None and self.provider != str(provider):
            raise ValueError(
                f"Twin embedding provider mismatch for {database.player_name}: cache uses "
                f"{self.provider}, expected {provider}."
            )
        if embedding_dimension is not None and (
            self.configured_dimension != int(embedding_dimension)
            or self.embedding_dim != int(embedding_dimension)
        ):
            raise ValueError(f"Twin embedding dimension mismatch for {database.player_name}.")
        if normalize_embeddings is not None and self.normalize_embeddings != bool(normalize_embeddings):
            raise ValueError(f"Twin embedding normalization mismatch for {database.player_name}.")
        if dtype is not None and self.dtype != str(dtype):
            raise ValueError(f"Twin embedding dtype mismatch for {database.player_name}.")
        if self.embeddings.ndim != 2:
            raise ValueError(f"Twin embeddings for {database.player_name} must be a 2-D array.")
        if self.embeddings.shape[0] != len(database.memories):
            raise ValueError(
                f"Twin embedding count mismatch for {database.player_name}: "
                f"{self.embeddings.shape[0]} embeddings for {len(database.memories)} memories."
            )
        if not np.isfinite(self.embeddings).all():
            raise ValueError(f"Twin embeddings for {database.player_name} contain NaN or infinity.")
        if self.memory_digest != twin_memory_digest(database.memories):
            raise ValueError(
                f"Twin embedding cache is stale for {database.player_name}; rebuild its embeddings."
            )

    def save(self, cache_root):
        array_path, metadata_path = embedding_cache_paths(
            cache_root, self.player_name, self.model_name, self.revision,
            self.normalize_embeddings, self.dtype, self.provider,
            self.configured_dimension,
        )
        os.makedirs(os.path.dirname(array_path), exist_ok=True)
        array_fd, array_tmp = tempfile.mkstemp(dir=os.path.dirname(array_path), prefix=".tmp_")
        metadata_fd, metadata_tmp = tempfile.mkstemp(dir=os.path.dirname(metadata_path), prefix=".tmp_")
        try:
            with os.fdopen(array_fd, "wb") as f:
                np.save(f, self.embeddings, allow_pickle=False)
            array_digest = hashlib.sha256()
            with open(array_tmp, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    array_digest.update(chunk)
            metadata = {
                "schema_version": self.schema_version,
                "player_name": self.player_name,
                "provider": self.provider,
                "model_name": self.model_name,
                "revision": self.revision,
                "configured_dimension": self.configured_dimension,
                "normalize_embeddings": self.normalize_embeddings,
                "dtype": self.dtype,
                "embedding_dim": self.embedding_dim,
                "memory_count": int(self.embeddings.shape[0]),
                "memory_digest": self.memory_digest,
                "array_sha256": array_digest.hexdigest(),
                "prefix_contract_version": EMBEDDING_PREFIX_CONTRACT_VERSION,
            }
            with os.fdopen(metadata_fd, "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
            os.replace(array_tmp, array_path)
            os.replace(metadata_tmp, metadata_path)
        finally:
            for temporary_path in (array_tmp, metadata_tmp):
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
        return array_path, metadata_path

    @classmethod
    def load(cls, cache_root, database, model_name, revision=None,
             normalize_embeddings=True, dtype="float32", mmap_mode="r",
             provider="local", embedding_dimension=None):
        array_path, metadata_path = embedding_cache_paths(
            cache_root, database.player_name, model_name, revision,
            normalize_embeddings, dtype, provider, embedding_dimension,
        )
        if not os.path.isfile(array_path) or not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Missing Twin embedding cache for {database.player_name} and {model_name}. "
                "Run: python Scripts/prepare_twin_embeddings.py for the required experiment."
            )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if metadata.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported Twin embedding cache schema: {metadata.get('schema_version')}")
        embeddings = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)
        array_digest = hashlib.sha256()
        with open(array_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                array_digest.update(chunk)
        if metadata.get("array_sha256") != array_digest.hexdigest():
            raise ValueError(f"Twin embedding array checksum mismatch for {database.player_name}.")
        if metadata.get("prefix_contract_version") != EMBEDDING_PREFIX_CONTRACT_VERSION:
            raise ValueError(f"Unsupported Twin embedding prefix contract for {database.player_name}.")
        cache = cls(
            metadata["player_name"], metadata["model_name"], embeddings,
            metadata["memory_digest"], revision=metadata.get("revision"),
            normalize_embeddings=metadata.get("normalize_embeddings", True),
            dtype=metadata.get("dtype", str(embeddings.dtype)),
            provider=metadata["provider"],
            configured_dimension=metadata.get("configured_dimension"),
        )
        if str(embeddings.dtype) != cache.dtype:
            raise ValueError(f"Twin embedding array dtype mismatch for {database.player_name}.")
        if cache.embedding_dim != int(metadata.get("embedding_dim", cache.embedding_dim)):
            raise ValueError(f"Twin embedding metadata dimension mismatch for {database.player_name}.")
        if int(metadata.get("memory_count", embeddings.shape[0])) != embeddings.shape[0]:
            raise ValueError(f"Twin embedding metadata count mismatch for {database.player_name}.")
        cache.validate(
            database, model_name, revision, normalize_embeddings, dtype,
            provider, embedding_dimension,
        )
        return cache


def build_semantic_text(my_tactical_pattern, my_tempo_movement_dynamics,
                        opponent_tactical_pattern, opponent_tempo_movement_dynamics):
    return "\n".join([
        f"My tactical pattern: {my_tactical_pattern}",
        f"My tempo and movement dynamics: {my_tempo_movement_dynamics}",
        f"Opponent tactical pattern: {opponent_tactical_pattern}",
        f"Opponent tempo and movement dynamics: {opponent_tempo_movement_dynamics}",
    ])


def area_name_from_coord(x, y):
    return LANG_AREA_MAPPING.get(coord2area(safe_float(x), safe_float(y)), "Unknown")


def shot_name_from_type(shot_type):
    shot_type = safe_int(shot_type)
    if shot_type == 0:
        return "Unknown"
    if shot_type == 11:
        return "Missed shot"
    return INDEX2EN_SHOT.get(shot_type, f"Unknown({shot_type})")


class TwinDatabase:
    """Player-specific aspect-memory retrieval database for Twin."""

    schema_version = 7

    def __init__(self, df, player_name, player_id):
        self.player_name = str(player_name)
        self.player_id = player_id
        self.data = df.copy()
        if 'player_type' in self.data:
            miss_mask = self.data['player_type'].astype(int) == 11
            self.data.loc[
                miss_mask,
                ['landing_x', 'landing_y', 'moving_x', 'moving_y'],
            ] = np.nan
        self.memories = []
        self.embedding_cache = None
        self.build_metadata = {}

        self.continuous_keys = [
            "player_location_x", "player_location_y",
            "hit_x", "hit_y",
            "opponent_location_x", "opponent_location_y",
        ]
        self.discrete_keys = ["opponent_type"]
        self.action_keys = ["landing_x", "landing_y", "player_type", "moving_x", "moving_y"]
        self.metadata_keys = [
            "match_id", "rally", "ball_round", "player_score", "opponent_score", "score_status",
            "ball_distance", "ball_distance_x", "ball_distance_y", "opponent_move_x", "opponent_move_y",
        ]
        self._build_memory_stores()

    def _group_rallies(self):
        group_cols = ["match_id"]
        if "set" in self.data.columns:
            group_cols.append("set")
        group_cols.append("rally")
        grouped = []
        for key, rally_df in self.data.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)
            grouped.append((key, rally_df))
        return sorted(grouped, key=lambda item: tuple(-1 if pd.isna(v) else v for v in item[0]))

    def _row_to_stroke(self, row):
        keys = ["player", "match_id", "set"] + state_col + action_col
        stroke = {}
        for key in keys:
            if key in row:
                value = row[key]
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                stroke[key] = value
        if "player_type" in stroke:
            stroke["player_type_name"] = shot_name_from_type(stroke["player_type"])
        if "opponent_type" in stroke:
            stroke["opponent_type_name"] = shot_name_from_type(stroke["opponent_type"])
        return stroke

    def _row_to_turn_state(self, row):
        player_x = safe_float(row.get("player_location_x"))
        player_y = safe_float(row.get("player_location_y"))
        opponent_x = safe_float(row.get("opponent_location_x"))
        opponent_y = safe_float(row.get("opponent_location_y"))
        hit_x = safe_float(row.get("hit_x"))
        hit_y = safe_float(row.get("hit_y"))
        landing_x = safe_float(row.get("landing_x"))
        landing_y = safe_float(row.get("landing_y"))
        moving_x = safe_float(row.get("moving_x"))
        moving_y = safe_float(row.get("moving_y"))
        player_type = safe_int(row.get("player_type"))
        opponent_type = safe_int(row.get("opponent_type"))

        return {
            "ball_round": safe_int(row.get("ball_round"), 1),
            "player": str(row.get("player", "")),
            "is_target_turn": str(row.get("player", "")) == self.player_name,
            "player_area": area_name_from_coord(player_x, player_y),
            "opponent_area": area_name_from_coord(opponent_x, opponent_y),
            "hit_area": area_name_from_coord(hit_x, hit_y),
            "landing_area": area_name_from_coord(landing_x, landing_y),
            "moving_area": area_name_from_coord(moving_x, moving_y),
            "player_shot": shot_name_from_type(player_type),
            "opponent_shot": shot_name_from_type(opponent_type) if opponent_type > 0 else "Unknown",
            "player_type": player_type,
            "opponent_type": opponent_type,
        }

    def build_rally_turn_based_text(self, rally_df):
        """Build semantic turn-based text for a completed rally without raw coordinates."""
        rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)
        lines = []
        for _, row in rally_df.iterrows():
            state = self._row_to_turn_state(row)
            br = state["ball_round"]
            is_target_turn = state["is_target_turn"]
            shot_name = state["player_shot"]

            lines.append(f"Shot {br}:")
            if is_target_turn:
                if br == 1:
                    lines.append(
                        f"It is my turn to serve. I am at {state['player_area']}, opponent at {state['opponent_area']}."
                    )
                    if state["player_type"] == 11:
                        lines.append("I miss the serve.")
                    else:
                        lines.append(
                            f"I serve a {shot_name} to {state['landing_area']}, then recover to {state['moving_area']}."
                        )
                else:
                    lines.append(
                        f"It is my turn. I am at {state['player_area']}, opponent at {state['opponent_area']}."
                    )
                    if state["opponent_type"] > 0:
                        lines.append(f"Opponent's {state['opponent_shot']} is arriving at {state['hit_area']}.")
                    else:
                        lines.append(f"The shuttle is arriving at {state['hit_area']}.")
                    if state["player_type"] == 11:
                        lines.append("I cannot reach the shuttle and miss the shot.")
                    else:
                        lines.append(
                            f"I hit a {shot_name} to {state['landing_area']}, then recover to {state['moving_area']}."
                        )
            else:
                if br == 1:
                    lines.append(
                        f"It is opponent's turn to serve. I am at {state['opponent_area']}, opponent at {state['player_area']}."
                    )
                    if state["player_type"] == 11:
                        lines.append("Opponent misses the serve.")
                    else:
                        lines.append(
                            f"Opponent serves a {shot_name} to {state['landing_area']}, then recovers to {state['moving_area']}."
                        )
                else:
                    lines.append(
                        f"It is opponent's turn. I am at {state['opponent_area']}, opponent at {state['player_area']}."
                    )
                    if state["opponent_type"] > 0:
                        lines.append(f"My {state['opponent_shot']} is arriving at {state['hit_area']}.")
                    else:
                        lines.append(f"The shuttle is arriving at {state['hit_area']}.")
                    if state["player_type"] == 11:
                        lines.append("Opponent cannot reach the shuttle and misses the shot.")
                    else:
                        lines.append(
                            f"Opponent hits a {shot_name} to {state['landing_area']}, then recovers to {state['moving_area']}."
                        )
            lines.append("")
        return "\n".join(lines).strip()

    def _score_for_target(self, row):
        if str(row.get("player", "")) == self.player_name:
            return safe_int(row.get("player_score")), safe_int(row.get("opponent_score"))
        return safe_int(row.get("opponent_score")), safe_int(row.get("player_score"))

    def _rally_winner_role(self, rally_df):
        if "getpoint_player" not in rally_df.columns:
            return None
        winners = rally_df["getpoint_player"].dropna()
        if len(winners) == 0:
            return None
        winner = str(winners.iloc[-1])
        if winner == self.player_name:
            return "target"
        return "opponent"

    def _previous_rally_roles(self, rally_df, max_recent=5):
        if len(rally_df) == 0 or "match_id" not in rally_df.columns or "rally" not in rally_df.columns:
            return []
        first = rally_df.iloc[0]
        match_id = safe_float(first.get("match_id"))
        rally_num = safe_float(first.get("rally"))
        mask = (self.data["match_id"].astype(float) == match_id) & (self.data["rally"].astype(float) < rally_num)
        if "set" in self.data.columns and "set" in rally_df.columns:
            set_num = safe_float(first.get("set"))
            mask &= self.data["set"].astype(float) == set_num
        previous = self.data[mask].copy()
        if len(previous) == 0:
            return []
        group_cols = ["rally"]
        if "set" in previous.columns:
            group_cols = ["set", "rally"]
        roles = []
        for _, prev_rally_df in previous.groupby(group_cols, dropna=False):
            role = self._rally_winner_role(prev_rally_df)
            if role is not None:
                sort_row = prev_rally_df.iloc[0]
                roles.append((safe_float(sort_row.get("rally")), role))
        roles = [role for _, role in sorted(roles, key=lambda item: item[0])]
        return roles[-max_recent:]

    @staticmethod
    def _score_stage(target_score, opponent_score):
        max_score = max(target_score, opponent_score)
        if target_score >= 20 or opponent_score >= 20:
            if target_score > opponent_score:
                return "The target player may be near game point or already at game point."
            if opponent_score > target_score:
                return "The opponent may be near game point or already at game point."
            return "This is a late-game tied situation near game point."
        if max_score <= 7:
            return "This is an early-game rally."
        if max_score <= 15:
            return "This is a mid-game rally."
        return "This is a late-game rally before game point."

    @staticmethod
    def _streak_text(previous_roles):
        if not previous_roles:
            return "No previous rally result is available for recent point momentum."
        last_role = previous_roles[-1]
        streak = 0
        for role in reversed(previous_roles):
            if role != last_role:
                break
            streak += 1
        owner = "the target player" if last_role == "target" else "the opponent"
        if streak >= 2:
            if last_role == "target":
                return (
                    f"The target player has won the previous {streak} rallies, "
                    "so recent scoring momentum is in the target player's favor."
                )
            return (
                f"The opponent has won the previous {streak} rallies, "
                "so the target player is facing recent scoring momentum against them."
            )

        target_wins = sum(1 for role in previous_roles if role == "target")
        opponent_wins = len(previous_roles) - target_wins
        if len(previous_roles) == 1:
            return f"The previous rally was won by {owner}."
        if target_wins == opponent_wins:
            return (
                f"Recent rally outcomes are balanced: the target player and the opponent each won "
                f"{target_wins} of the previous {len(previous_roles)} rallies."
            )
        if target_wins > opponent_wins:
            return (
                f"Recent rally outcomes are mixed, with the target player winning {target_wins} "
                f"of the previous {len(previous_roles)} rallies."
            )
        return (
            f"Recent rally outcomes are mixed, with the opponent winning {opponent_wins} "
            f"of the previous {len(previous_roles)} rallies."
        )

    def build_rally_meta_text(self, rally_df):
        """Build anonymized natural-language match/rally context for LLM annotation."""
        rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)
        if len(rally_df) == 0:
            return "No rally metadata is available."
        first = rally_df.iloc[0]
        target_score, opponent_score = self._score_for_target(first)
        score_diff = target_score - opponent_score
        if score_diff > 0:
            score_text = f"Before this rally, the target player leads {target_score}-{opponent_score}, a {score_diff}-point lead."
        elif score_diff < 0:
            score_text = f"Before this rally, the target player trails {target_score}-{opponent_score}, a {abs(score_diff)}-point deficit."
        else:
            score_text = f"Before this rally, the score is tied at {target_score}-{opponent_score}."

        # serve_text = "The target player is serving." if str(first.get("player", "")) == self.player_name else "The target player is receiving."
        # stage_text = self._score_stage(target_score, opponent_score)
        # previous_roles = self._previous_rally_roles(rally_df)
        # momentum_text = self._streak_text(previous_roles)
        # num_strokes = int(len(rally_df))
        # rally_length_text = f"The completed rally contains {num_strokes} strokes."

        # return "\n".join([
        #     score_text,
        #     serve_text,
        #     stage_text,
        #     momentum_text,
        #     rally_length_text,
        # ])
        return score_text

    def _rally_meta(self, group_key, rally_df, opponent):
        meta = {
            "match_id": safe_int(group_key[0]),
            "rally": safe_int(group_key[-1]),
            "target_player": self.player_name,
            "opponent": opponent,
            "num_strokes": int(len(rally_df)),
        }
        if "set" in rally_df.columns:
            meta["set"] = safe_int(rally_df.iloc[0].get("set"))
        if len(rally_df) > 0:
            first = rally_df.iloc[0]
            meta["player_score"] = safe_float(first.get("player_score"))
            meta["opponent_score"] = safe_float(first.get("opponent_score"))
        return meta

    def _fallback_annotation(self, rally_df):
        first_round = safe_int(rally_df["ball_round"].min(), 1) if len(rally_df) else 1
        last_round = safe_int(rally_df["ball_round"].max(), first_round) if len(rally_df) else first_round
        segment = [{
            "range": [first_round, last_round],
            "phase": "Full-rally fallback phase",
        }]
        return {
            "aspect_segments": {
                "my_tactical_pattern": segment,
                "my_tempo_movement_dynamics": segment,
                "opponent_tactical_pattern": segment,
                "opponent_tempo_movement_dynamics": segment,
            },
        }

    def _annotate_rally_with_llm(self, llm, system_prompt, user_template, rally_df):
        rally_turn_based_text = self.build_rally_turn_based_text(rally_df)
        rally_meta_text = self.build_rally_meta_text(rally_df)
        context = {
            "rally_meta_text": rally_meta_text,
            "rally_turn_based_text": rally_turn_based_text,
            "first_ball_round": int(rally_df["ball_round"].min()),
            "last_ball_round": int(rally_df["ball_round"].max()),
        }
        user_prompt = user_template.render(**context)
        response, _, _ = llm.get_verified_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_trials=5,
            validator=lambda value: self._validate_rally_annotation(value, rally_df),
        )
        if self._validate_rally_annotation(response, rally_df):
            return response
        return self._fallback_annotation(rally_df)

    @staticmethod
    def _validate_rally_annotation(annotation, rally_df):
        if (
            not isinstance(annotation, dict)
            or set(annotation) != {"aspect_segments"}
            or not isinstance(annotation.get("aspect_segments"), dict)
        ):
            return False
        rounds = rally_df.get("ball_round", pd.Series(dtype=float)).dropna().astype(int)
        if rounds.empty:
            return False
        first_round = int(rounds.min())
        last_round = int(rounds.max())
        aspects = (
            "my_tactical_pattern",
            "my_tempo_movement_dynamics",
            "opponent_tactical_pattern",
            "opponent_tempo_movement_dynamics",
        )
        if set(annotation["aspect_segments"]) != set(aspects):
            return False
        for aspect in aspects:
            segments = annotation["aspect_segments"].get(aspect)
            if not isinstance(segments, list) or not segments:
                return False
            expected_start = first_round
            for segment in segments:
                if not isinstance(segment, dict) or set(segment) != {"range", "phase"}:
                    return False
                interval = segment.get("range")
                if not isinstance(interval, list) or len(interval) != 2:
                    return False
                if any(isinstance(value, bool) or not isinstance(value, int) for value in interval):
                    return False
                start, end = interval
                if start != expected_start or end < start or end > last_round:
                    return False
                phase = segment.get("phase")
                if (
                    not isinstance(phase, str)
                    or not phase.strip()
                    or len(phase.split()) > 20
                ):
                    return False
                expected_start = end + 1
            if expected_start != last_round + 1:
                return False
        return True

    def annotate_rally(self, rally_df, llm=None, use_llm=True):
        """Annotate a single completed rally for inspection or memory construction."""
        rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)
        if not use_llm:
            return self._fallback_annotation(rally_df)

        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "twin")
        with open(os.path.join(prompt_dir, "memory_user.jinja"), "r") as f:
            user_template = Template(f.read())
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            system_prompt = f.read()
        llm = llm or LLM()
        return self._annotate_rally_with_llm(
            llm,
            system_prompt,
            user_template,
            rally_df,
        )

    def build_stroke_annotations(self, rally_df, annotation):
        """Attach a single-rally annotation to each target-player stroke."""
        rally_df = rally_df.sort_values("ball_round").reset_index(drop=True)
        target_rows = rally_df[rally_df["player"].astype(str) == self.player_name] if "player" in rally_df else rally_df
        return [
            self._build_memory_from_row(row, annotation)
            for _, row in target_rows.iterrows()
        ]

    @staticmethod
    def _segment_for_round(segments, ball_round):
        def _format_segment(segment):
            return str(segment.get("phase") or "Unspecified phase").strip()

        for segment in segments or []:
            rng = segment.get("range", [ball_round, ball_round])
            if len(rng) >= 2 and safe_int(rng[0]) <= ball_round <= safe_int(rng[1]):
                return _format_segment(segment)
        if segments:
            return _format_segment(segments[0])
        return "No aspect evidence available."

    def _build_memory_from_row(self, row, annotation):
        ball_round = safe_int(row.get("ball_round"), 1)
        segments = annotation.get("aspect_segments", {}) if isinstance(annotation, dict) else {}
        my_tactical_pattern = self._segment_for_round(segments.get("my_tactical_pattern"), ball_round)
        my_tempo_movement_dynamics = self._segment_for_round(segments.get("my_tempo_movement_dynamics"), ball_round)
        opponent_tactical_pattern = self._segment_for_round(segments.get("opponent_tactical_pattern"), ball_round)
        opponent_tempo_movement_dynamics = self._segment_for_round(segments.get("opponent_tempo_movement_dynamics"), ball_round)
        semantic_text = build_semantic_text(
            my_tactical_pattern,
            my_tempo_movement_dynamics,
            opponent_tactical_pattern,
            opponent_tempo_movement_dynamics,
        )

        memory = {
            "memory_id": (
                f"{safe_int(row.get('match_id'))}:{safe_int(row.get('set'))}:"
                f"{safe_int(row.get('rally'))}:{ball_round}"
            ),
            "semantic_text": semantic_text,
            "my_tactical_pattern": my_tactical_pattern,
            "my_tempo_movement_dynamics": my_tempo_movement_dynamics,
            "opponent_tactical_pattern": opponent_tactical_pattern,
            "opponent_tempo_movement_dynamics": opponent_tempo_movement_dynamics,
            "is_miss": safe_int(row.get("player_type")) == 11,
            "outcome_type": "miss" if safe_int(row.get("player_type")) == 11 else "normal",
        }
        if memory["is_miss"]:
            memory["reachability_text"] = "The target player missed this stroke, so this memory is direct evidence for failed contact or unreachable shuttle situations."
        else:
            memory["reachability_text"] = "The target player reached and returned the shuttle in this memory."
        for key in set(self.metadata_keys + self.continuous_keys + self.discrete_keys + self.action_keys + ["player", "set"]):
            if key in row:
                value = row[key]
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                if memory["is_miss"] and key in {"landing_x", "landing_y", "moving_x", "moving_y"}:
                    value = None
                memory[key] = value
        prompt_action = BaseAgent.build_prompt_action(
            memory.get("player_type"),
            memory.get("landing_x"),
            memory.get("landing_y"),
            memory.get("moving_x"),
            memory.get("moving_y"),
        )
        memory["action_verbalization"] = BaseAgent.format_prompt_action(prompt_action)
        return memory

    def _build_memory_stores(self):
        self.memory_stores = {
            "classic": list(range(len(self.memories))),
            "normal": [],
            "miss": [],
        }
        for idx, memory in enumerate(self.memories):
            is_miss = bool(memory.get("is_miss", safe_int(memory.get("player_type")) == 11))
            memory["is_miss"] = is_miss
            memory["outcome_type"] = "miss" if is_miss else "normal"
            if "reachability_text" not in memory:
                memory["reachability_text"] = (
                    "The target player missed this stroke, so this memory is direct evidence for failed contact or unreachable shuttle situations."
                    if is_miss else
                    "The target player reached and returned the shuttle in this memory."
                )
            self.memory_stores["miss" if is_miss else "normal"].append(idx)

        total_count = len(self.memory_stores["classic"])
        miss_count = len(self.memory_stores["miss"])
        self.stats = {
            "total_count": int(total_count),
            "normal_count": int(len(self.memory_stores["normal"])),
            "miss_count": int(miss_count),
            "miss_rate": float(miss_count / total_count) if total_count else 0.0,
        }

    def build_with_llm(self, llm=None, max_rallies=None):
        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "twin")
        with open(os.path.join(prompt_dir, "memory_user.jinja"), "r") as f:
            user_template = Template(f.read())
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            system_prompt = f.read()
        llm = llm or LLM()
        self.memories = []
        built_rallies = 0
        for _, rally_df in self._group_rallies():
            if max_rallies is not None and built_rallies >= max_rallies:
                break
            annotation = self._annotate_rally_with_llm(llm, system_prompt, user_template, rally_df)
            target_rows = rally_df[rally_df["player"].astype(str) == self.player_name] if "player" in rally_df else rally_df
            for _, row in target_rows.iterrows():
                self.memories.append(self._build_memory_from_row(row, annotation))
            built_rallies += 1
        self._build_memory_stores()

    def build_with_fallback(self, max_rallies=None):
        self.memories = []
        built_rallies = 0
        for _, rally_df in self._group_rallies():
            if max_rallies is not None and built_rallies >= max_rallies:
                break
            annotation = self._fallback_annotation(rally_df)
            target_rows = rally_df[rally_df["player"].astype(str) == self.player_name] if "player" in rally_df else rally_df
            for _, row in target_rows.iterrows():
                self.memories.append(self._build_memory_from_row(row, annotation))
            built_rallies += 1
        self._build_memory_stores()

    def attach_embedding_cache(self, embedding_cache):
        embedding_cache.validate(self)
        self.embedding_cache = embedding_cache

    def _feature_arrays(self, indices=None):
        selected_memories = self.memories if indices is None else [self.memories[int(idx)] for idx in indices]
        continuous = np.array([
            [safe_float(memory.get(key)) for key in self.continuous_keys]
            for memory in selected_memories
        ], dtype=np.float32)
        discrete = np.array([
            [safe_float(memory.get(key)) for key in self.discrete_keys]
            for memory in selected_memories
        ], dtype=np.float32)
        return continuous, discrete

    def _retrieve_from_indices(self, query_state, query_embedding, indices, top_k=5, lambda_weight=0.5):
        if not self.memories:
            return []
        if self.embedding_cache is None:
            raise ValueError(
                f"Twin embedding cache is not attached for {self.player_name}. "
                "Build and load the configured embedding cache first."
            )
        semantic_embeddings = self.embedding_cache.embeddings
        indices = list(indices)
        if not indices:
            return []
        continuous, discrete = self._feature_arrays(indices)
        query_continuous = np.array([safe_float(query_state.get(k)) for k in self.continuous_keys], dtype=np.float32)
        query_discrete = np.array([safe_float(query_state.get(k)) for k in self.discrete_keys], dtype=np.float32)
        continuous_dist = np.sqrt(np.sum((continuous - query_continuous) ** 2, axis=1))
        discrete_dist = np.mean(discrete != query_discrete, axis=1)
        state_similarity = np.exp(-(continuous_dist + discrete_dist))

        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if semantic_embeddings.shape[1] != query_embedding.shape[0]:
            raise ValueError(
                f"Twin embedding dimension mismatch for {self.player_name}: database has "
                f"{semantic_embeddings.shape[1]} dimensions but query has {query_embedding.shape[0]}. "
                "Rebuild the Twin embedding cache with the configured model."
            )
        selected_embeddings = semantic_embeddings[np.array(indices, dtype=int)]
        if self.embedding_cache.normalize_embeddings:
            semantic_similarity = np.matmul(selected_embeddings, query_embedding)
        else:
            embedding_norms = np.linalg.norm(selected_embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            denominator = np.maximum(embedding_norms * query_norm, 1e-12)
            semantic_similarity = np.matmul(selected_embeddings, query_embedding) / denominator
        semantic_similarity = (semantic_similarity + 1.0) / 2.0
        scores = lambda_weight * state_similarity + (1.0 - lambda_weight) * semantic_similarity
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for local_idx in top_indices:
            memory_idx = int(indices[int(local_idx)])
            memory = dict(self.memories[memory_idx])
            memory["score"] = float(scores[local_idx])
            memory["state_similarity"] = float(state_similarity[local_idx])
            memory["semantic_similarity"] = float(semantic_similarity[local_idx])
            memory["context_text"] = memory.get("semantic_text", "")
            results.append(memory)
        return results

    def retrieve(self, query_state, query_embedding, top_k=5, lambda_weight=0.5):
        return self._retrieve_from_indices(
            query_state,
            query_embedding,
            self.memory_stores.get("classic", list(range(len(self.memories)))),
            top_k=top_k,
            lambda_weight=lambda_weight,
        )

    def retrieve_dual(self, query_state, query_embedding, normal_top_k=8, miss_top_k=5,
                      lambda_weight=0.5, miss_lambda_weight=None, launch=False):
        miss_lambda_weight = miss_lambda_weight if miss_lambda_weight is not None else lambda_weight
        legal_types = {1, 10} if launch else set(range(2, 10))
        normal_indices = [
            idx for idx in self.memory_stores.get("normal", [])
            if safe_int(self.memories[idx].get("player_type")) in legal_types
        ]
        normal_memories = self._retrieve_from_indices(
            query_state,
            query_embedding,
            normal_indices,
            top_k=normal_top_k,
            lambda_weight=lambda_weight,
        )
        miss_memories = self._retrieve_from_indices(
            query_state,
            query_embedding,
            self.memory_stores.get("miss", []),
            top_k=miss_top_k,
            lambda_weight=miss_lambda_weight,
        )
        nearest_normal = normal_memories[0] if normal_memories else {}
        nearest_miss = miss_memories[0] if miss_memories else {}
        return {
            "normal_cases": normal_memories,
            "miss_cases": miss_memories,
            "summary": {
                **self.stats,
                "normal_top_k": int(normal_top_k),
                "miss_top_k": int(miss_top_k),
                "retrieved_normal_count": int(len(normal_memories)),
                "retrieved_miss_count": int(len(miss_memories)),
                "nearest_normal_score": nearest_normal.get("score"),
                "nearest_miss_score": nearest_miss.get("score"),
                "nearest_normal_state_similarity": nearest_normal.get("state_similarity"),
                "nearest_miss_state_similarity": nearest_miss.get("state_similarity"),
                "miss_lambda_weight": float(miss_lambda_weight),
            },
        }

    def retrieve_stage1(self, query_state, query_embedding, normal_top_k=10, miss_top_k=5,
                        lambda_weight=0.5, launch=False):
        return self.retrieve_dual(
            query_state,
            query_embedding,
            normal_top_k=normal_top_k,
            miss_top_k=miss_top_k,
            lambda_weight=lambda_weight,
            launch=launch,
        )

    def available_shot_types(self, launch=False):
        legal_types = {1, 10} if launch else set(range(2, 10))
        return sorted({
            safe_int(self.memories[idx].get("player_type"))
            for idx in self.memory_stores.get("normal", [])
            if safe_int(self.memories[idx].get("player_type")) in legal_types
        })

    def retrieve_by_shot_type(self, query_state, query_embedding, shot_type, top_k=10,
                              lambda_weight=0.5):
        shot_type = safe_int(shot_type)
        if shot_type == 11:
            raise ValueError("Missed shot cannot be used for Twin Stage 2 retrieval")
        coordinate_keys = ("landing_x", "landing_y", "moving_x", "moving_y")
        indices = []
        for idx in self.memory_stores.get("normal", []):
            memory = self.memories[idx]
            if safe_int(memory.get("player_type")) != shot_type:
                continue
            coordinates = [safe_float(memory.get(key), default=np.nan) for key in coordinate_keys]
            if all(np.isfinite(value) and -1.0 <= value <= 1.0 for value in coordinates):
                indices.append(idx)
        return self._retrieve_from_indices(
            query_state,
            query_embedding,
            indices,
            top_k=top_k,
            lambda_weight=lambda_weight,
        )

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(filepath)), prefix=".tmp_"
        )
        try:
            with os.fdopen(file_descriptor, "wb") as f:
                pickle.dump({
                    "schema_version": self.schema_version,
                    "player_name": self.player_name,
                    "player_id": self.player_id,
                    "memories": self.memories,
                    "memory_digest": twin_memory_digest(self.memories),
                    "build_metadata": self.build_metadata,
                    "memory_stores": self.memory_stores,
                    "stats": self.stats,
                    "continuous_keys": self.continuous_keys,
                    "discrete_keys": self.discrete_keys,
                    "action_keys": self.action_keys,
                    "metadata_keys": self.metadata_keys,
                }, f)
            os.replace(temporary_path, filepath)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        if data.get("schema_version") != cls.schema_version:
            raise ValueError(
                f"Unsupported Twin memory schema {data.get('schema_version')} in {filepath}. "
                "Rebuild canonical Twin memory with Scripts/prepare_all_data.py."
            )
        instance = cls.__new__(cls)
        instance.schema_version = data["schema_version"]
        instance.player_name = data["player_name"]
        instance.player_id = data["player_id"]
        instance.memories = data.get("memories", [])
        instance.build_metadata = data.get("build_metadata", {})
        expected_digest = data.get("memory_digest")
        actual_digest = twin_memory_digest(instance.memories)
        if expected_digest != actual_digest:
            raise ValueError(f"Twin memory digest mismatch in {filepath}.")
        instance.embedding_cache = None
        instance.memory_stores = data.get("memory_stores")
        instance.stats = data.get("stats")
        instance.continuous_keys = data.get("continuous_keys", [])
        instance.discrete_keys = data.get("discrete_keys", [])
        instance.action_keys = data.get("action_keys", [])
        instance.metadata_keys = data.get("metadata_keys", [])
        instance.data = pd.DataFrame()
        if instance.memory_stores is None or instance.stats is None:
            instance._build_memory_stores()
        return instance


def create_twin_verbalizer():
    return NaiveRAGVerbalizer()


class TwinAgent(BaseAgent):
    """Aspect-conditioned Twin retrieval with two-stage action generation."""

    def __init__(self, player_name, database, embedding_cache, model_name=None, api_key=None,
                 host=None, port=None, base_url=None, host_url=None, normal_top_k=10,
                 miss_top_k=5, conditioned_top_k=10, lambda_weight=0.5,
                 embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                 embedding_batch_size=None, embedding_revision=None,
                 normalize_embeddings=True, embedding_dtype="float32",
                 embedding_model=None, enable_tracing=True,
                 instruction_mode="no_instruct", embedding_provider="local",
                 embedding_dimension=None):
        super().__init__(player_name)
        self.database = database
        self.normal_top_k = int(normal_top_k)
        self.miss_top_k = int(miss_top_k)
        self.conditioned_top_k = int(conditioned_top_k)
        if min(self.normal_top_k, self.miss_top_k, self.conditioned_top_k) < 1:
            raise ValueError("Twin two-stage retrieval top-k values must all be positive")
        self.lambda_weight = float(lambda_weight)
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("Twin lambda_weight must be between 0 and 1")
        if instruction_mode not in {"instruct", "no_instruct"}:
            raise ValueError("instruction_mode must be instruct or no_instruct")
        self.instruction_mode = instruction_mode
        self.retrieval_architecture = "two_stage_v1"
        self.pre_stage_architecture = "twin_aspect_v1"
        self.tracer = BadmintonTracer("twin", player_name, enable_tracing)
        self.verbalizer = NaiveRAGVerbalizer()
        self.embedding_model_name = embedding_model_name
        self.embedding_provider = embedding_provider
        self.configured_embedding_dimension = embedding_dimension
        self.embedding_batch_size = embedding_batch_size
        self.embedding_revision = embedding_revision
        self.normalize_embeddings = bool(normalize_embeddings)
        self.embedding_dtype = embedding_dtype
        embedding_cache.validate(
            database,
            embedding_model_name,
            embedding_revision,
            self.normalize_embeddings,
            embedding_dtype,
            embedding_provider,
            embedding_dimension,
        )
        database.attach_embedding_cache(embedding_cache)
        if embedding_model is None:
            raise ValueError("An initialized local embedding client is required")
        self.embedding_model = embedding_model
        self.embedding_dim = get_embedding_dimension(self.embedding_model)
        if embedding_cache.embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Twin embedding dimension mismatch for {player_name}: database has "
                f"{embedding_cache.embedding_dim} dimensions but current model produces {self.embedding_dim}. "
                "Rebuild the Twin embedding cache or align config.yaml."
            )
        self.llm = LLM(model_name=model_name, api_key=api_key, host=host, port=port, base_url=base_url, host_url=host_url)
        self.trajectory = []
        self._current_rally = None
        self._last_case_ids = []
        self.last_decision_trace = None

        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "twin")
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            self.system_prompt = f.read()
        with open(os.path.join(prompt_dir, f"stage1_{instruction_mode}.jinja"), "r") as f:
            self.stage1_template = Template(f.read())
        with open(os.path.join(prompt_dir, f"stage2_{instruction_mode}.jinja"), "r") as f:
            self.stage2_template = Template(f.read())
        with open(os.path.join(prompt_dir, "aspect_user.jinja"), "r") as f:
            self.aspect_template = Template(f.read())

    def _build_state_dict_from_env(self, state, info):
        player_score = info["score"][0] if info.get("score") else 0
        opponent_score = info["score"][1] if info.get("score") else 0
        player_x, player_y = state[1][0], state[1][1]
        opponent_x, opponent_y = state[2][0], state[2][1]
        hit_x, hit_y = state[3][0], state[3][1]
        return {
            "player": info.get("current_player_name", self.player_name),
            "is_target_turn": True,
            "turn_role": "target",
            "rally": info.get("rally", 0),
            "ball_round": info["round"][-1] if info.get("round") else 1,
            "player_score": player_score,
            "opponent_score": opponent_score,
            "score_status": int(player_score - opponent_score),
            "player_location_x": player_x,
            "player_location_y": player_y,
            "player_location_area": LANG_AREA_MAPPING.get(coord2area(player_x, player_y), "Unknown"),
            "ball_distance": float(np.sqrt((hit_x - player_x) ** 2 + (hit_y - player_y) ** 2)),
            "ball_distance_x": hit_x - player_x,
            "ball_distance_y": hit_y - player_y,
            "hit_x": hit_x,
            "hit_y": hit_y,
            "hit_position_area": LANG_AREA_MAPPING.get(coord2area(hit_x, hit_y), "Unknown"),
            "opponent_type": int(state[0]),
            "opponent_type_name": INDEX2EN_SHOT.get(int(state[0]), f"Unknown({int(state[0])})"),
            "opponent_location_x": opponent_x,
            "opponent_location_y": opponent_y,
            "opponent_location_area": LANG_AREA_MAPPING.get(coord2area(opponent_x, opponent_y), "Unknown"),
            "opponent_move_x": 0,
            "opponent_move_y": 0,
        }

    def _normalize_history_for_player(self, history):
        normalized = []
        for stroke in history or []:
            stroke_data = dict(stroke)
            player = stroke_data.get("player")
            stroke_data["is_target_turn"] = (
                str(player) == str(self.player_name)
                if player is not None else bool(stroke_data.get("is_target_turn", True))
            )
            stroke_data["turn_role"] = "target" if stroke_data["is_target_turn"] else "opponent"
            normalized.append(stroke_data)
        return normalized

    @staticmethod
    def _validate_aspect_response(response):
        required = {
            "my_tactical_pattern",
            "my_tempo_movement_dynamics",
            "opponent_tactical_pattern",
            "opponent_tempo_movement_dynamics",
        }
        return (
            isinstance(response, dict)
            and set(response) == required
            and all(
                isinstance(response[key], str)
                and response[key].strip()
                and len(response[key].split()) <= 20
                for key in required
            )
        )

    def _estimate_aspects(self, state_dict, state_history, rally_profile_instruction=""):
        trajectory_text = self.verbalizer.build_trajectory_text((state_history or []) + [state_dict], state_dict)
        user_prompt = self.aspect_template.render(
            trajectory_text=trajectory_text,
            rally_profile_instruction=rally_profile_instruction,
        )
        t0 = time.time()
        response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
            self.llm,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_trials=5,
            validator=self._validate_aspect_response,
        )
        self.tracer.log_generation(
            self.system_prompt,
            user_prompt,
            response,
            latency_ms=int((time.time() - t0) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
            stage="aspect",
        )
        if self._validate_aspect_response(response):
            aspects = dict(response)
            source = "llm"
        else:
            aspects = {
                "my_tactical_pattern": "Current tactical phase unknown",
                "my_tempo_movement_dynamics": "Current movement phase unknown",
                "opponent_tactical_pattern": "Opponent tactical phase unknown",
                "opponent_tempo_movement_dynamics": "Opponent movement phase unknown",
            }
            source = "fallback"
        aspects["semantic_query_text"] = build_semantic_text(
            aspects["my_tactical_pattern"],
            aspects["my_tempo_movement_dynamics"],
            aspects["opponent_tactical_pattern"],
            aspects["opponent_tempo_movement_dynamics"],
        )
        return aspects, source

    def _format_stage1_cases(self, memories):
        return [
            {
                "case_id": memory["memory_id"],
                "stage1_description": self.verbalizer.build_case_verbalization(memory)["stage1_description"],
            }
            for memory in memories
        ]

    def _format_stage2_cases(self, memories):
        cases = []
        for memory in memories:
            description = self.verbalizer.build_case_verbalization(memory)["stage2_description"]
            semantic_text = str(memory.get("semantic_text", "")).strip()
            if semantic_text:
                description = f"{description}\nTwin semantic context:\n{semantic_text}"
            cases.append({
                "case_id": memory["memory_id"],
                "stage2_description": description,
            })
        return cases

    def _case_references(self, cases, stage, store, conditioned_shot_type=None):
        references = []
        for rank, case in enumerate(cases, 1):
            reference = {
                "database_player": self.player_name,
                "stage": int(stage),
                "store": store,
                "case_id": str(case["memory_id"]),
                "rank": rank,
                "score": float(case["score"]),
                "state_similarity": float(case["state_similarity"]),
                "semantic_similarity": float(case["semantic_similarity"]),
            }
            if conditioned_shot_type is not None:
                reference["conditioned_shot_type"] = int(conditioned_shot_type)
            references.append(reference)
        return references

    @staticmethod
    def _nearest_case(cases):
        if not cases:
            return None
        return max(cases, key=lambda case: (float(case["score"]), str(case["memory_id"])))

    def _stage1_fallback(self, retrieval, launch):
        nearest = self._nearest_case(retrieval["normal_cases"] + retrieval["miss_cases"])
        if nearest is not None:
            shot_type = safe_int(nearest.get("player_type"))
            return {
                "rationale": "Stage 1 fallback copied the nearest legal Twin memory outcome.",
                "outcome": "missed" if shot_type == 11 else "returned",
                "shot_type": shot_type,
            }, "fallback_nearest_case"
        return {
            "rationale": "Stage 1 fallback used the deterministic legal default.",
            "outcome": "returned",
            "shot_type": 1 if launch else 2,
        }, "fallback_default"

    def _stage2_fallback(self, conditioned_cases):
        nearest = self._nearest_case(conditioned_cases)
        if nearest is None:
            return {
                "rationale": "Stage 2 fallback used neutral placement because no conditioned Twin memory exists.",
                "landing_x": 0.0,
                "landing_y": 0.0,
                "moving_x": 0.0,
                "moving_y": 0.0,
            }, "fallback_neutral"
        return {
            "rationale": "Stage 2 fallback copied placement from the nearest shot-conditioned Twin memory.",
            "landing_x": float(nearest["landing_x"]),
            "landing_y": float(nearest["landing_y"]),
            "moving_x": float(nearest["moving_x"]),
            "moving_y": float(nearest["moving_y"]),
        }, "fallback_nearest_conditioned_case"

    def predict(self, state, info, launch):
        state_dict = self._build_state_dict_from_env(state, info)
        rally_num = info.get("rally", 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.trajectory = []
        shared_history = info.get("rally_history")
        if shared_history is not None:
            state_history = self._normalize_history_for_player(shared_history)
        else:
            state_history = list(self.trajectory)
        self.trajectory.append(state_dict)

        self.tracer.start_trace(state_dict, info, launch)
        profile_instruction = info.get("rally_profile_instruction", "")
        aspects, aspect_source = self._estimate_aspects(
            state_dict,
            state_history,
            profile_instruction,
        )
        query_embedding = encode_texts(
            self.embedding_model,
            [aspects["semantic_query_text"]],
            self.embedding_dim,
            self.embedding_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            input_type="query",
            model_name=self.embedding_model_name,
        )[0]

        t0 = time.time()
        retrieval_result = self.database.retrieve_stage1(
            state_dict,
            query_embedding,
            normal_top_k=self.normal_top_k,
            miss_top_k=self.miss_top_k,
            lambda_weight=self.lambda_weight,
            launch=launch,
        )
        memories = retrieval_result["normal_cases"] + retrieval_result["miss_cases"]
        t1 = time.time()
        self.tracer.log_retrieval(
            state_dict, memories, "twin", int((t1 - t0) * 1000), stage="stage1"
        )

        context = self.verbalizer.build_prompt_context_nl(
            state_dict, [], launch=launch, state_history=state_history
        )
        available_types = self.database.available_shot_types(launch=launch)
        stage1_prompt = self.stage1_template.render(
            trajectory_text=context["trajectory_text"],
            semantic_query_text=aspects["semantic_query_text"],
            normal_cases=self._format_stage1_cases(retrieval_result["normal_cases"]),
            miss_cases=self._format_stage1_cases(retrieval_result["miss_cases"]),
            rally_profile_instruction=profile_instruction,
        )
        t0 = time.time()
        stage1_response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
            self.llm,
            system_prompt=self.system_prompt,
            user_prompt=stage1_prompt,
            max_trials=5,
            validator=lambda response: BaseAgent.validate_rag_stage1_response(
                response,
                launch=launch,
                available_types=available_types,
            ),
        )
        t1 = time.time()
        self.tracer.log_generation(
            self.system_prompt, stage1_prompt, stage1_response,
            latency_ms=int((t1 - t0) * 1000),
            raw_llm_output=raw_llm_output,
            usage=usage,
            stage="stage1",
        )
        stage1 = BaseAgent.parse_rag_stage1_response(
            stage1_response,
            launch=launch,
            available_types=available_types,
        )
        if stage1 is None:
            stage1, stage1_source = self._stage1_fallback(retrieval_result, launch)
        else:
            stage1_source = "llm"

        stage1_normal_refs = self._case_references(
            retrieval_result["normal_cases"], 1, "normal"
        )
        stage1_miss_refs = self._case_references(
            retrieval_result["miss_cases"], 1, "miss"
        )
        stage2_refs = []
        shot_type = int(stage1["shot_type"])
        if shot_type == 11:
            landing_x = landing_y = moving_x = moving_y = 0.0
            stage2 = {
                "status": "skipped_missed",
                "conditioned_shot_type": None,
                "case_ids": [],
                "rationale": "",
                "source": "skipped",
            }
            reasoning = stage1["rationale"]
        else:
            t0 = time.time()
            conditioned_cases = self.database.retrieve_by_shot_type(
                state_dict,
                query_embedding,
                shot_type,
                top_k=self.conditioned_top_k,
                lambda_weight=self.lambda_weight,
            )
            if any(safe_int(case.get("player_type")) != shot_type for case in conditioned_cases):
                raise ValueError("Twin shot-conditioned retrieval returned a different shot type")
            self.tracer.log_retrieval(
                state_dict,
                conditioned_cases,
                "twin",
                int((time.time() - t0) * 1000),
                stage="stage2",
            )
            stage2_prompt = self.stage2_template.render(
                trajectory_text=context["trajectory_text"],
                semantic_query_text=aspects["semantic_query_text"],
                selected_shot_type=INDEX2EN_SHOT[shot_type],
                conditioned_cases=self._format_stage2_cases(conditioned_cases),
                rally_profile_instruction=profile_instruction,
            )
            t0 = time.time()
            stage2_response, raw_llm_output, usage = BaseAgent.get_verified_response_or_none(
                self.llm,
                system_prompt=self.system_prompt,
                user_prompt=stage2_prompt,
                max_trials=5,
                validator=BaseAgent.validate_rag_stage2_response,
            )
            self.tracer.log_generation(
                self.system_prompt,
                stage2_prompt,
                stage2_response,
                latency_ms=int((time.time() - t0) * 1000),
                raw_llm_output=raw_llm_output,
                usage=usage,
                stage="stage2",
            )
            placement = BaseAgent.parse_rag_stage2_response(stage2_response)
            if placement is None:
                placement, stage2_source = self._stage2_fallback(conditioned_cases)
            else:
                stage2_source = "llm"
            landing_x = float(placement["landing_x"])
            landing_y = float(placement["landing_y"])
            moving_x = float(placement["moving_x"])
            moving_y = float(placement["moving_y"])
            stage2_refs = self._case_references(
                conditioned_cases,
                2,
                "shot_conditioned",
                conditioned_shot_type=shot_type,
            )
            stage2 = {
                "status": "completed",
                "conditioned_shot_type": shot_type,
                "case_ids": stage2_refs,
                "rationale": placement["rationale"],
                "source": stage2_source,
            }
            reasoning = f"Stage 1: {stage1['rationale']} Stage 2: {placement['rationale']}"

        if self.trajectory:
            self.trajectory[-1].update({
                "player_type": int(shot_type),
                "landing_x": float(landing_x),
                "landing_y": float(landing_y),
                "moving_x": float(moving_x),
                "moving_y": float(moving_y),
            })
        action = (shot_type, state[-1] if len(state) > 0 else None, [landing_x, landing_y], [moving_x, moving_y], [0.1] * 10)
        self.tracer.end_trace({"shot_type": int(shot_type), "landing": [landing_x, landing_y], "movement": [moving_x, moving_y]}, reasoning)
        all_case_ids = stage1_normal_refs + stage1_miss_refs + stage2_refs
        self._last_case_ids = all_case_ids
        self.last_decision_trace = {
            "schema_version": 2,
            "architecture": self.retrieval_architecture,
            "pre_stage": {
                "architecture": self.pre_stage_architecture,
                "semantic_query_text": aspects["semantic_query_text"],
                "aspects": {
                    key: aspects.get(key, "")
                    for key in (
                        "my_tactical_pattern",
                        "my_tempo_movement_dynamics",
                        "opponent_tactical_pattern",
                        "opponent_tempo_movement_dynamics",
                    )
                },
                "source": aspect_source,
            },
            "stage1": {
                "normal_case_ids": stage1_normal_refs,
                "miss_case_ids": stage1_miss_refs,
                "outcome": stage1["outcome"],
                "shot_type": None if shot_type == 11 else shot_type,
                "rationale": stage1["rationale"],
                "source": stage1_source,
            },
            "stage2": stage2,
            "case_ids": all_case_ids,
            "rationale": reasoning,
            "action": {
                "shot_type": shot_type,
                "landing_x": None if shot_type == 11 else landing_x,
                "landing_y": None if shot_type == 11 else landing_y,
                "moving_x": None if shot_type == 11 else moving_x,
                "moving_y": None if shot_type == 11 else moving_y,
            },
            "action_source": (
                f"{aspect_source}+{stage1_source}"
                if shot_type == 11 else
                f"{aspect_source}+{stage1_source}+{stage2['source']}"
            ),
        }
        return action, reasoning

    def on_rally_end(self, rally_history=None):
        del rally_history
        self.trajectory = []
        self._current_rally = None
        self._last_case_ids = []
        self.last_decision_trace = None

    def reset_history(self):
        self.trajectory = []
        self._current_rally = None
        self._last_case_ids = []
        self.last_decision_trace = None


def build_state_tensor_from_row(row, player_id):
    state_values = [row[col] for col in state_col]
    state_values[-1] = player_id
    return torch.tensor(state_values, dtype=torch.float32)


def build_action_tensor_from_row(row):
    return torch.tensor([row[col] for col in action_col], dtype=torch.float32)


def state_to_env_format(state_18):
    opponent_type = int(state_18[12].item())
    player_pos = (state_18[5].item(), state_18[6].item())
    opponent_pos = (state_18[13].item(), state_18[14].item())
    landing_pos = (state_18[10].item(), state_18[11].item())
    return [opponent_type, player_pos, opponent_pos, landing_pos]


def load_twin_agent(player_name, player2index, database_dir, embedding_cache_dir, top_k, lambda_weight,
                    embedding_model_name, embedding_batch_size, prompt_variant,
                    embedding_revision=None, normalize_embeddings=True, embedding_dtype="float32",
                    embedding_model=None, embedding_provider="local", embedding_dimension=None):
    from Utils.artifact_names import player_database_filename

    db_path = os.path.join(database_dir, player_database_filename(player_name))
    if not os.path.exists(db_path):
        return None
    database = TwinDatabase.load(db_path)
    embedding_cache = TwinEmbeddingCache.load(
        embedding_cache_dir,
        database,
        embedding_model_name,
        revision=embedding_revision,
        normalize_embeddings=normalize_embeddings,
        dtype=embedding_dtype,
        provider=embedding_provider,
        embedding_dimension=embedding_dimension,
    )
    return TwinAgent(
        player_name=player_name,
        database=database,
        embedding_cache=embedding_cache,
        normal_top_k=top_k,
        miss_top_k=max(1, min(5, top_k)),
        conditioned_top_k=top_k,
        lambda_weight=lambda_weight,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        embedding_revision=embedding_revision,
        normalize_embeddings=normalize_embeddings,
        embedding_dtype=embedding_dtype,
        embedding_model=embedding_model,
        instruction_mode="instruct" if prompt_variant == "shifted" else "no_instruct",
        embedding_provider=embedding_provider,
        embedding_dimension=embedding_dimension,
    )


def evaluate_twin_on_test(test_split_path, merged_csv_path, player_index_map_path, database_dir,
                          embedding_cache_dir,
                          output_path=None, top_k=10, lambda_weight=0.5,
                          embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                          embedding_batch_size=None, embedding_revision=None,
                          normalize_embeddings=True, embedding_dtype="float32",
                          prompt_variant="norm", skip_first_n=0,
                          embedding_provider="local", embedding_dimension=None):
    raise RuntimeError(
        "Legacy match-level Twin evaluation is removed. Run "
        "Scripts/single_player_eval/eval_all_rallies.py --algorithm twin instead."
    )
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    with open(player_index_map_path, 'r') as f:
        player2index = json.load(f)

    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    results = {}
    from Utils.embeddings import create_embedding_client, resolve_embedding_settings

    embedding_settings = resolve_embedding_settings(provider=embedding_provider)
    if embedding_model_name != DEFAULT_EMBEDDING_MODEL:
        embedding_settings["model_name"] = embedding_model_name
    else:
        embedding_model_name = embedding_settings["model_name"]
    if embedding_dimension is not None:
        embedding_settings["dimension"] = embedding_dimension
    else:
        embedding_dimension = embedding_settings["dimension"]
    embedding_revision = embedding_settings.get("revision")
    embedding_model = create_embedding_client(embedding_settings)

    for target_player_name, test_match_ids in tqdm(list(test_split['test'].items()), desc="Evaluating Twin players"):
        if target_player_name not in player2index:
            continue
        target_player_id = player2index[target_player_name]
        agent_cache = {}
        target_agent = load_twin_agent(
            target_player_name, player2index, database_dir, embedding_cache_dir, top_k,
            lambda_weight, embedding_model_name, embedding_batch_size, prompt_variant,
            embedding_revision, normalize_embeddings, embedding_dtype, embedding_model,
            embedding_provider, embedding_dimension,
        )
        if target_agent is None:
            results[target_player_name] = {
                'land_dtw': float('inf'), 'move_dtw': float('inf'), 'shot_ctc': float('inf'),
                'edit_dist': float('inf'), 'num_rallies': 0, 'num_shots': 0,
            }
            continue
        agent_cache[target_player_name] = target_agent

        rally_land_dtw_list = []
        rally_move_dtw_list = []
        rally_shot_ctc_list = []
        rally_edit_dist_list = []
        total_rallies = 0
        total_shots = 0

        for match_id in tqdm(test_match_ids, desc=f"Matches for {target_player_name}", leave=False):
            match_df = df[df['match_id'] == match_id].copy()
            if len(match_df) == 0:
                continue
            players_in_match = [str(p) for p in match_df['player'].dropna().unique()]
            opponent_name = next((p for p in players_in_match if p != target_player_name), None)
            if opponent_name is None or opponent_name not in player2index:
                continue
            opponent_player_id = player2index[opponent_name]

            group_cols = ['set', 'rally'] if 'set' in match_df.columns else ['rally']
            for _, rally_df in match_df.groupby(group_cols):
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
                simulated_history = []

                first_row = rally_df.iloc[0]
                current_player_name = str(first_row['player'])
                current_player_id = target_player_id if current_player_name == target_player_name else opponent_player_id
                current_state = build_state_tensor_from_row(first_row, current_player_id)

                rally_done = False
                shot_idx = 0
                while not rally_done and shot_idx < len(rally_df):
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = str(row['player'])
                    is_target_turn = ground_truth_player == target_player_name
                    current_player_id = target_player_id if is_target_turn else opponent_player_id
                    current_state[17] = current_player_id
                    launch = current_state[1].item() == 1
                    env_state = state_to_env_format(current_state)

                    agent = agent_cache.get(ground_truth_player)
                    if agent is None:
                        agent = load_twin_agent(
                            ground_truth_player, player2index, database_dir, embedding_cache_dir, top_k,
                            lambda_weight, embedding_model_name, embedding_batch_size, prompt_variant,
                            embedding_revision, normalize_embeddings, embedding_dtype, embedding_model,
                            embedding_provider, embedding_dimension,
                        )
                        if agent is not None:
                            agent_cache[ground_truth_player] = agent

                    if agent is None:
                        gt_action = build_action_tensor_from_row(row)
                        pred_shot_type = int(gt_action[2].item())
                        pred_land = gt_action[0:2].numpy()
                        pred_move = gt_action[3:5].numpy()
                    else:
                        info = {
                            'rally': safe_int(row.get('rally')),
                            'round': [safe_int(current_state[1].item())],
                            'score': [safe_float(current_state[2].item()), safe_float(current_state[3].item())],
                            'action': [None],
                            'current_player_name': ground_truth_player,
                            'rally_history': simulated_history,
                        }
                        action, _ = agent.predict(env_state, info, launch)
                        pred_shot_type = int(action[0])
                        pred_land = np.array(action[2], dtype=np.float32)
                        pred_move = np.array(action[3], dtype=np.float32)

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
                            if 0 <= pred_shot_type < len(dummy_logits):
                                dummy_logits[pred_shot_type] = 1.0
                            rally_shot_logits_seq.append(dummy_logits)

                    simulated_history.append({
                        'player': ground_truth_player,
                        'is_target_turn': is_target_turn,
                        'turn_role': 'target' if is_target_turn else 'opponent',
                        'rally': safe_int(row.get('rally')),
                        'ball_round': safe_int(current_state[1].item()),
                        'player_score': safe_float(current_state[2].item()),
                        'opponent_score': safe_float(current_state[3].item()),
                        'score_status': safe_float(current_state[4].item()),
                        'player_location_x': safe_float(current_state[5].item()),
                        'player_location_y': safe_float(current_state[6].item()),
                        'hit_x': safe_float(current_state[10].item()),
                        'hit_y': safe_float(current_state[11].item()),
                        'opponent_type': safe_int(current_state[12].item()),
                        'opponent_location_x': safe_float(current_state[13].item()),
                        'opponent_location_y': safe_float(current_state[14].item()),
                        'player_type': pred_shot_type,
                        'landing_x': float(pred_land[0]),
                        'landing_y': float(pred_land[1]),
                        'moving_x': float(pred_move[0]),
                        'moving_y': float(pred_move[1]),
                    })

                    action_tensor = torch.tensor([
                        pred_land[0], pred_land[1], float(pred_shot_type), pred_move[0], pred_move[1]
                    ], dtype=torch.float32)
                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor, current_state, other_player_id)
                    if terminate(next_state):
                        rally_done = True
                    current_state = next_state
                    shot_idx += 1

                for agent in agent_cache.values():
                    agent.on_rally_end(simulated_history)

                if len(rally_land_gt_seq) > 0:
                    land_pred_array = np.array(rally_land_pred_seq)
                    land_gt_array = np.array(rally_land_gt_seq)
                    move_pred_array = np.array(rally_move_pred_seq)
                    move_gt_array = np.array(rally_move_gt_seq)
                    rally_land_dtw_list.append(normalized_dtw_ndim_distance(land_pred_array, land_gt_array))
                    rally_move_dtw_list.append(normalized_dtw_ndim_distance(move_pred_array, move_gt_array))
                    rally_shot_logits_tensor = torch.stack(rally_shot_logits_seq)
                    rally_shot_targets_tensor = torch.tensor(rally_shot_gt_seq, dtype=torch.long)
                    log_probs = F.log_softmax(rally_shot_logits_tensor.unsqueeze(1), dim=-1)
                    input_length = torch.tensor([log_probs.shape[0]], dtype=torch.long)
                    target_length = torch.tensor([len(rally_shot_targets_tensor)], dtype=torch.long)
                    rally_shot_ctc_list.append(ctc_loss_fn(log_probs, rally_shot_targets_tensor, input_length, target_length).item())
                    rally_edit_dist_list.append(edit_distance(rally_shot_pred_seq, rally_shot_gt_seq))
                    total_rallies += 1
                    total_shots += len(rally_shot_gt_seq)

        if total_rallies > 0:
            results[target_player_name] = {
                'land_dtw': float(np.mean(rally_land_dtw_list)),
                'move_dtw': float(np.mean(rally_move_dtw_list)),
                'shot_ctc': float(np.mean(rally_shot_ctc_list)),
                'edit_dist': float(np.mean(rally_edit_dist_list)),
                'num_rallies': total_rallies,
                'num_shots': total_shots,
            }
        else:
            results[target_player_name] = {
                'land_dtw': float('inf'), 'move_dtw': float('inf'), 'shot_ctc': float('inf'),
                'edit_dist': float('inf'), 'num_rallies': 0, 'num_shots': 0,
            }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return results


def print_results(results):
    print("\n" + "=" * 105)
    print("Twin Evaluation Results on Test Set (Per-Rally Basis)")
    print("=" * 105)
    print(f"{'Player Name':<30} {'Land DTW':<12} {'Move DTW':<12} {'Shot CTC':<12} {'Edit Dist':<12} {'Rallies':<8} {'Shots':<8}")
    print("-" * 105)
    total_land_dtw, total_move_dtw, total_shot_ctc, total_edit_dist, total_rallies, total_shots = [], [], [], [], [], []
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
    print("-" * 105)
    if total_land_dtw:
        total_rallies_sum = sum(total_rallies)
        total_shots_sum = sum(total_shots)
        w_land = sum(s * n for s, n in zip(total_land_dtw, total_rallies)) / total_rallies_sum
        w_move = sum(s * n for s, n in zip(total_move_dtw, total_rallies)) / total_rallies_sum
        w_ctc = sum(s * n for s, n in zip(total_shot_ctc, total_rallies)) / total_rallies_sum
        w_edit = sum(s * n for s, n in zip(total_edit_dist, total_rallies)) / total_rallies_sum
        print(f"{'Weighted Average (by rallies)':<30} {w_land:<12.4f} {w_move:<12.4f} {w_ctc:<12.4f} {w_edit:<12.4f} "
              f"{total_rallies_sum:<8} {total_shots_sum:<8}")
    print("=" * 105)
