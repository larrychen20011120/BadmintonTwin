"""Shared rally-level split and profile loading utilities."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping

import pandas as pd


RALLY_KEY_COLUMNS = ("match_id", "set", "rally")


def rally_key(record):
    """Return the normalized physical-rally key for one assignment record."""
    if not isinstance(record, Mapping):
        raise ValueError(
            "Rally assignments must be objects containing match_id, set, and rally"
        )
    missing = [column for column in RALLY_KEY_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Rally assignment is missing fields: {missing}")
    try:
        return tuple(int(record[column]) for column in RALLY_KEY_COLUMNS)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rally assignment: {record}") from exc


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _physical_assignments(split, fold):
    assignments = split.get(fold)
    if not isinstance(assignments, Mapping):
        raise ValueError(f"Rally split field {fold!r} must be a player mapping")

    players_by_key = defaultdict(set)
    for player, records in assignments.items():
        if not isinstance(records, list):
            raise ValueError(f"Rally assignments for {player!r} must be a list")
        seen = set()
        for record in records:
            key = rally_key(record)
            if key in seen:
                raise ValueError(f"Duplicate {fold} rally for {player}: {key}")
            seen.add(key)
            players_by_key[key].add(str(player))

    return players_by_key


def _profile_metadata(profiles, fold):
    metadata = {}
    for player, profile in profiles.get("players", {}).items():
        clusters = {
            int(cluster["cluster_id"]): cluster
            for cluster in profile.get("clusters", [])
        }
        for assignment in profile.get(f"{fold}_assignments", []):
            key = rally_key(assignment)
            cluster_id = int(assignment["cluster_id"])
            cluster = clusters.get(cluster_id)
            if cluster is None:
                raise ValueError(
                    f"Missing cluster {cluster_id} for {player!r} {fold} rally {key}"
                )
            metadata[(str(player), key)] = {
                "cluster_id": cluster_id,
                "label": cluster.get("label"),
                "description": cluster.get("description"),
                "instruction": cluster.get("instruction"),
                "instruction_source": cluster.get("instruction_source"),
                "instruction_variants": cluster.get("instruction_variants", {}),
                "in_train_support": assignment.get("in_train_support"),
            }
    return metadata


def load_rally_dataset(split_path, profiles_path=None):
    """Load one canonical physical-rally view for training and evaluation.

    The source split intentionally lists a physical rally under every observed
    player. This function merges those player records into one work item while
    retaining player-specific cluster and instruction metadata.
    """
    split = _load_json(split_path)
    train_players = _physical_assignments(split, "train")
    test_players = _physical_assignments(split, "test")

    overlap = set(train_players) & set(test_players)
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"Physical rallies appear in both train and test: {sample}")

    profiles = {"players": {}}
    profile_sha256 = None
    if profiles_path:
        profiles = _load_json(profiles_path)
        expected_split_digest = profiles.get("split_sha256")
        actual_split_digest = _file_digest(split_path)
        if expected_split_digest and expected_split_digest != actual_split_digest:
            raise ValueError(
                "rally_profiles.json was generated from a different rally split"
            )
        profile_sha256 = _file_digest(profiles_path)

    profile_by_fold = {
        fold: _profile_metadata(profiles, fold) for fold in ("train", "test")
    }

    def build_records(fold, players_by_key):
        records = []
        for key in sorted(players_by_key):
            player_names = sorted(players_by_key[key])
            records.append({
                "match_id": key[0],
                "set": key[1],
                "rally": key[2],
                "players": player_names,
                "player_profiles": {
                    player: profile_by_fold[fold].get((player, key))
                    for player in player_names
                },
            })
        return records

    train_records = build_records("train", train_players)
    test_records = build_records("test", test_players)
    return {
        "train": train_records,
        "test": test_records,
        "summary": {
            "physical_train_rallies": len(train_records),
            "physical_test_rallies": len(test_records),
            "player_train_assignments": sum(len(values) for values in split["train"].values()),
            "player_test_assignments": sum(len(values) for values in split["test"].values()),
        },
        "split_sha256": _file_digest(split_path),
        "profiles_sha256": profile_sha256,
    }


def filter_rally_rows(df, records):
    """Return dataframe rows belonging to the exact physical rally records."""
    missing = sorted(set(RALLY_KEY_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing rally key columns: {missing}")

    keys = [rally_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Physical rally records must be unique before filtering")
    if not keys:
        return df.iloc[0:0].copy()

    key_frame = pd.DataFrame(keys, columns=RALLY_KEY_COLUMNS)
    source = df.copy()
    source["__rally_row_order"] = range(len(source))
    for column in RALLY_KEY_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="raise").astype(int)

    selected = source.merge(key_frame, on=list(RALLY_KEY_COLUMNS), how="inner")
    return (
        selected.sort_values("__rally_row_order")
        .drop(columns="__rally_row_order")
        .reset_index(drop=True)
    )


def index_rally_records(records):
    """Index canonical records by physical key."""
    indexed = {}
    for record in records:
        key = rally_key(record)
        if key in indexed:
            raise ValueError(f"Duplicate physical rally record: {key}")
        indexed[key] = record
    return indexed
