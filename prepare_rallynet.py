"""Prepare physical-rally training data for RallyNet."""

import hashlib
import json
import os
import pickle
from collections.abc import Mapping

import numpy as np
import pandas as pd

from Preprocessing.rally_data import load_rally_dataset, rally_key
from Utils.helper import action_col, state_col


RALLY_KEY_COLUMNS = ("match_id", "set", "rally")


def compute_rallynet_dataset_fingerprint(rallies, player_map):
    """Return the stable content fingerprint used by RallyNet checkpoints."""
    digest = hashlib.sha256()
    digest.update(json.dumps(player_map, sort_keys=True).encode("utf-8"))
    for rally in rallies:
        digest.update(
            f"{rally['match_id']}:{rally['set']}:{rally['rally']}".encode("ascii")
        )
        digest.update(str(rally["length"]).encode("ascii"))
        for key in (
            "states",
            "actions",
            "error_targets",
            "serve_mask",
            "coordinate_mask",
        ):
            digest.update(rally[key].tobytes(order="C"))
    return digest.hexdigest()


def _physical_keys(assignments, fold):
    if isinstance(assignments, Mapping):
        records_by_player = assignments.items()
    elif isinstance(assignments, list):
        records_by_player = (("<physical>", assignments),)
    else:
        raise ValueError(f"Rally split field {fold!r} must contain assignment objects")

    keys = set()
    for player, records in records_by_player:
        if not isinstance(records, list):
            raise ValueError(f"Rally assignments for {player!r} must be a list")
        player_keys = set()
        for record in records:
            key = rally_key(record)
            if key in player_keys:
                raise ValueError(f"Duplicate {fold} rally for {player}: {key}")
            player_keys.add(key)
            keys.add(key)
    return keys


def _player_identifier(value, player_map, valid_ids):
    name = str(value).strip()
    if name in player_map:
        return int(player_map[name])
    try:
        identifier = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    return identifier if identifier in valid_ids else np.nan


def _rally_key_records(keys):
    return [
        {"match_id": key[0], "set": key[1], "rally": key[2]}
        for key in sorted(keys)
    ]


def prepare_rallynet_data(
    merged_csv,
    split_path,
    player_map_path,
    output_dir,
    experiment=None,
    split_content=None,
    player_map=None,
):
    """Build complete, ordered RallyNet sequences for exact train rally keys."""
    if split_content is None:
        split_content = load_rally_dataset(split_path)
    if player_map is None:
        with open(player_map_path, "r", encoding="utf-8") as handle:
            player_map = json.load(handle)

    expected_player_ids = list(range(len(player_map)))
    if sorted(player_map.values()) != expected_player_ids:
        raise ValueError("RallyNet player IDs must be contiguous from zero")
    valid_player_ids = set(expected_player_ids)

    train_keys = _physical_keys(split_content.get("train"), "train")
    test_keys = _physical_keys(split_content.get("test"), "test")
    overlap = train_keys & test_keys
    if overlap:
        raise ValueError(
            f"Physical rallies appear in both train and test: {sorted(overlap)[:5]}"
        )
    if not train_keys:
        raise ValueError("RallyNet split contains no physical train rallies")

    df = pd.read_csv(merged_csv)
    required_columns = set(state_col + action_col + list(RALLY_KEY_COLUMNS))
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing RallyNet columns: {missing_columns}")

    source = df.copy()
    source["__source_order"] = np.arange(len(source))
    valid_source_keys = np.ones(len(source), dtype=bool)
    for column in RALLY_KEY_COLUMNS:
        values = pd.to_numeric(source[column], errors="coerce")
        valid = values.notna() & np.isfinite(values) & np.equal(values, np.floor(values))
        valid_source_keys &= valid.to_numpy()
        source[column] = values
    source = source.loc[valid_source_keys].copy()
    for column in RALLY_KEY_COLUMNS:
        source[column] = source[column].astype(int)

    key_frame = pd.DataFrame(sorted(train_keys), columns=RALLY_KEY_COLUMNS)
    selected = source.merge(key_frame, on=list(RALLY_KEY_COLUMNS), how="inner")
    selected = selected.sort_values("__source_order").drop(columns="__source_order")
    present_keys = set(
        selected[list(RALLY_KEY_COLUMNS)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing_keys = sorted(train_keys - present_keys)
    if missing_keys:
        raise ValueError(f"RallyNet train rallies have no source rows: {missing_keys[:5]}")

    rallies = []
    num_errors = 0
    skipped_rallies = []
    for key in sorted(train_keys):
        match_id, set_number, rally_number = key
        rally_df = selected[
            (selected["match_id"] == match_id)
            & (selected["set"] == set_number)
            & (selected["rally"] == rally_number)
        ].copy()
        key_label = f"{match_id}/{set_number}/{rally_number}"

        ball_rounds = pd.to_numeric(rally_df["ball_round"], errors="coerce")
        if (
            ball_rounds.isna().any()
            or not np.isfinite(ball_rounds).all()
            or not np.equal(ball_rounds, np.floor(ball_rounds)).all()
        ):
            skipped_rallies.append({"key": key, "reason": "invalid ball_round values"})
            continue
        rally_df["ball_round"] = ball_rounds.astype(int)
        rally_df = rally_df.sort_values("ball_round", kind="stable").reset_index(drop=True)
        rounds = rally_df["ball_round"].to_numpy(dtype=np.int64)
        if rounds.tolist() != list(range(1, len(rally_df) + 1)):
            skipped_rallies.append({
                "key": key,
                "reason": (
                    f"ball_round must contain each value from 1 through {len(rally_df)}"
                ),
            })
            continue

        player_ids = rally_df["player"].map(
            lambda value: _player_identifier(value, player_map, valid_player_ids)
        )
        if player_ids.isna().any():
            raise ValueError(f"RallyNet rally {key_label} contains an unmapped player")
        player_ids = player_ids.astype(np.int64).to_numpy()
        if any(
            player_ids[index] == player_ids[index - 1]
            for index in range(1, len(player_ids))
        ):
            skipped_rallies.append({"key": key, "reason": "players do not alternate"})
            continue

        shot_values = pd.to_numeric(rally_df["player_type"], errors="coerce")
        if (
            shot_values.isna().any()
            or not np.isfinite(shot_values).all()
            or not np.equal(shot_values, np.floor(shot_values)).all()
        ):
            raise ValueError(f"RallyNet rally {key_label} has invalid shot IDs")
        shot_types = shot_values.astype(np.int64).to_numpy()
        if not np.isin(shot_types, np.arange(1, 12)).all():
            raise ValueError(f"RallyNet rally {key_label} has invalid shot IDs")

        serve_mask = rounds == 1
        error_targets = shot_types == 11
        if error_targets[:-1].any():
            skipped_rallies.append({"key": key, "reason": "non-terminal Type 11"})
            continue
        invalid_serves = serve_mask & ~error_targets & ~np.isin(shot_types, (1, 10))
        invalid_nonserves = ~serve_mask & ~error_targets & ~np.isin(
            shot_types, np.arange(2, 11)
        )
        if invalid_serves.any():
            skipped_rallies.append({
                "key": key,
                "reason": "successful serve must be Type 1 or 10",
            })
            continue
        if invalid_nonserves.any():
            skipped_rallies.append({
                "key": key,
                "reason": "successful non-serve must be Type 2 through 10",
            })
            continue

        state_frame = rally_df[state_col].copy()
        state_frame["player"] = player_ids
        state_frame = state_frame.apply(pd.to_numeric, errors="coerce")
        states = state_frame.to_numpy(dtype=np.float64)
        if not np.isfinite(states).all():
            skipped_rallies.append({"key": key, "reason": "invalid state values"})
            continue

        action_frame = rally_df[action_col].apply(pd.to_numeric, errors="coerce")
        actions = action_frame.to_numpy(dtype=np.float64, copy=True)
        coordinate_columns = [0, 1, 3, 4]
        successful_coordinates = actions[~error_targets][:, coordinate_columns]
        if not np.isfinite(successful_coordinates).all():
            skipped_rallies.append({
                "key": key,
                "reason": "invalid successful-shot coordinates",
            })
            continue
        actions[error_targets, 0:2] = 0.0
        actions[error_targets, 3:5] = 0.0
        actions[:, 2] = shot_types

        num_errors += int(error_targets.sum())
        rallies.append({
            "match_id": match_id,
            "set": set_number,
            "rally": rally_number,
            "length": int(len(rally_df)),
            "states": states.astype(np.float32),
            "actions": actions.astype(np.float32),
            "error_targets": error_targets.astype(np.bool_),
            "serve_mask": serve_mask.astype(np.bool_),
            "coordinate_mask": (~error_targets).astype(np.bool_),
        })

    if not rallies:
        details = "; ".join(
            f"{'/'.join(str(value) for value in item['key'])}: {item['reason']}"
            for item in skipped_rallies[:5]
        )
        raise ValueError(
            f"RallyNet preparation produced no valid train rallies; skipped {details}"
        )

    fingerprint = compute_rallynet_dataset_fingerprint(rallies, player_map)
    prepared_train_keys = {
        (rally["match_id"], rally["set"], rally["rally"]) for rally in rallies
    }
    train_rally_keys = _rally_key_records(prepared_train_keys)
    requested_train_rally_keys = _rally_key_records(train_keys)
    skipped_rally_records = [
        {
            "match_id": int(item["key"][0]),
            "set": int(item["key"][1]),
            "rally": int(item["key"][2]),
            "reason": item["reason"],
        }
        for item in skipped_rallies
    ]
    counts = {
        "matches": len({key[0] for key in prepared_train_keys}),
        "rallies": len(rallies),
        "rows": sum(rally["length"] for rally in rallies),
        "errors": num_errors,
        "requested_train_rallies": len(train_keys),
        "skipped_train_rallies": len(skipped_rallies),
        "excluded_test_rallies": len(test_keys),
    }
    split_sha256 = split_content.get("split_sha256")
    provenance = {
        "train_rally_keys": train_rally_keys,
        "requested_train_rally_keys": requested_train_rally_keys,
        "skipped_train_rallies": skipped_rally_records,
        "excluded_test_rally_count": len(test_keys),
        "player_map": player_map,
        "dataset_fingerprint": fingerprint,
        "split_sha256": split_sha256,
        "counts": counts,
    }
    train_match_ids = sorted({key[0] for key in prepared_train_keys})
    excluded_test_match_ids = sorted({key[0] for key in test_keys})
    metadata = {
        "schema_version": 1,
        "experiment": experiment,
        "player_map": player_map,
        "shot_type_count": 12,
        "dataset_fingerprint": fingerprint,
        "split_sha256": split_sha256,
        "train_rally_keys": train_rally_keys,
        "requested_train_rally_keys": requested_train_rally_keys,
        "skipped_train_rallies": skipped_rally_records,
        "excluded_test_rally_count": len(test_keys),
        "counts": counts,
        "provenance": provenance,
        "num_matches": counts["matches"],
        "num_rallies": counts["rallies"],
        "num_rows": counts["rows"],
        "num_errors": counts["errors"],
        "train_match_ids": train_match_ids,
        "excluded_test_match_ids": excluded_test_match_ids,
    }
    payload = {
        "schema_version": 1,
        "player_map": player_map,
        "dataset_fingerprint": fingerprint,
        "rallies": rallies,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "rallies.pkl"), "wb") as handle:
        pickle.dump(payload, handle)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        f"RallyNet data saved to {output_dir}: {counts['matches']} matches, "
        f"{counts['rallies']} rallies, {len(skipped_rallies)} malformed train rallies "
        f"skipped, {len(test_keys)} test rallies excluded"
    )
    return metadata
