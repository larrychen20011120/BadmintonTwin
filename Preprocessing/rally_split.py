import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Preprocessing.rally_feature import (  # noqa: E402
    RALLY_KEY_COLUMNS,
    extract_rally_features,
)


LENGTH_LABELS = ("short", "medium", "long")


def _require_columns(df, columns):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required split columns: {missing}")


def build_physical_rally_table(df, length_quantiles=(1 / 3, 2 / 3)):
    """Build one row per physical rally with all observed participants."""
    _require_columns(df, RALLY_KEY_COLUMNS + ["ball_round", "player"])
    records = []
    for key, rally_df in df.groupby(RALLY_KEY_COLUMNS, sort=True, dropna=False):
        if any(pd.isna(value) for value in key):
            continue
        rounds = pd.to_numeric(rally_df["ball_round"], errors="coerce")
        players = tuple(sorted({
            value
            for value in rally_df["player"].dropna().astype(str).str.strip()
            if value
        }))
        if not rounds.notna().any() or not players:
            continue
        records.append({
            "match_id": int(key[0]),
            "set": int(key[1]),
            "rally": int(key[2]),
            "rally_length": float(rounds.max()),
            "players": players,
        })

    rally_table = pd.DataFrame(records)
    if rally_table.empty:
        raise ValueError("No valid physical rallies were found")

    quantiles = np.asarray(length_quantiles, dtype=float)
    if len(quantiles) != 2 or not (0 < quantiles[0] < quantiles[1] < 1):
        raise ValueError("length_quantiles must contain two increasing values in (0, 1)")
    boundaries = np.quantile(rally_table["rally_length"].to_numpy(dtype=float), quantiles)
    rally_table["length_bin"] = [
        LENGTH_LABELS[int(np.digitize(length, boundaries, right=True))]
        for length in rally_table["rally_length"]
    ]
    return rally_table, [float(value) for value in boundaries]


def _rally_key(row):
    return (int(row["match_id"]), int(row["set"]), int(row["rally"]))


def _rally_record(row):
    return {
        "match_id": int(row["match_id"]),
        "set": int(row["set"]),
        "rally": int(row["rally"]),
    }


def _select_test_rallies(
    rally_table,
    candidate_indices,
    eligible_players,
    usable_keys,
    usable_totals,
    min_train_rallies,
    test_ratio,
    seed,
):
    """Select candidate rallies with an absolute-deviation MILP."""
    candidate_indices = list(candidate_indices)
    candidate_positions = {index: position for position, index in enumerate(candidate_indices)}
    player_positions = defaultdict(list)
    player_bin_positions = defaultdict(list)
    usable_bin_populations = Counter(
        (player, label) for player, _, label in usable_keys
    )
    for index, position in candidate_positions.items():
        row = rally_table.loc[index]
        key = _rally_key(row)
        label = row["length_bin"]
        for player in row["players"]:
            if (player, key, label) in usable_keys:
                player_positions[player].append(position)
                player_bin_positions[(player, label)].append(position)
    metrics = []

    for player in eligible_players:
        metrics.append((
            f"player:{player}",
            player_positions[player],
            round(test_ratio * usable_totals[player]),
        ))
        for label in LENGTH_LABELS:
            metrics.append((
                f"player_length:{player}:{label}",
                player_bin_positions[(player, label)],
                round(test_ratio * usable_bin_populations[(player, label)]),
            ))

    for label in LENGTH_LABELS:
        positions = [
            candidate_positions[index]
            for index in candidate_indices
            if rally_table.loc[index, "length_bin"] == label
        ]
        metrics.append((f"physical_length:{label}", positions, round(test_ratio * len(positions))))

    candidate_count = len(candidate_indices)
    metric_count = len(metrics)
    variable_count = candidate_count + 2 * metric_count
    rows = []
    columns = []
    values = []
    lower_constraints = []
    upper_constraints = []

    for metric_index, (_, positions, target) in enumerate(metrics):
        for position in positions:
            rows.append(metric_index)
            columns.append(position)
            values.append(1.0)
        rows.extend((metric_index, metric_index))
        columns.extend((candidate_count + 2 * metric_index, candidate_count + 2 * metric_index + 1))
        values.extend((-1.0, 1.0))
        lower_constraints.append(float(target))
        upper_constraints.append(float(target))

    for player in eligible_players:
        constraint_index = len(lower_constraints)
        for position in player_positions[player]:
            rows.append(constraint_index)
            columns.append(position)
            values.append(1.0)
        lower_constraints.append(-np.inf)
        upper_constraints.append(float(usable_totals[player] - min_train_rallies))

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(lower_constraints), variable_count),
    ).tocsr()
    objective = np.ones(variable_count, dtype=float)
    objective[:candidate_count] = 0.0
    if candidate_count:
        rng = np.random.default_rng(seed)
        ranks = np.empty(candidate_count, dtype=float)
        ranks[rng.permutation(candidate_count)] = np.arange(1, candidate_count + 1)
        # The total tie-break cost is below one, so one unit of deviation always dominates it.
        objective[:candidate_count] = ranks * (0.25 / ranks.sum())

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.full(variable_count, np.inf, dtype=float)
    upper_bounds[:candidate_count] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:candidate_count] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, lower_constraints, upper_constraints),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"Rally split optimization failed: {result.message}")

    selected = {
        index
        for index, value in zip(candidate_indices, result.x[:candidate_count])
        if value > 0.5
    }
    selected_positions = {candidate_positions[index] for index in selected}
    absolute_deviation = sum(
        abs(sum(position in selected_positions for position in positions) - target)
        for _, positions, target in metrics
    )
    return selected, {
        "status": "optimal",
        "scipy_status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "absolute_deviation": int(absolute_deviation),
        "candidate_rallies": candidate_count,
        "selected_rallies": len(selected),
    }


def build_rally_split(
    df,
    train_ratio=0.8,
    seed=42,
    length_quantiles=(1 / 3, 2 / 3),
    min_train_rallies=50,
    min_matches=2,
    min_target_nonserve_shots=1,
):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if min_train_rallies < 0:
        raise ValueError("min_train_rallies must be non-negative")
    if min_matches < 1:
        raise ValueError("min_matches must be at least 1")
    if min_target_nonserve_shots < 1:
        raise ValueError("min_target_nonserve_shots must be at least 1")

    test_ratio = 1.0 - train_ratio
    rally_table, boundaries = build_physical_rally_table(df, length_quantiles)
    physical_by_key = {_rally_key(row): row for _, row in rally_table.iterrows()}
    features = extract_rally_features(df)
    features = features[
        features["target_nonserve_shot_count"] >= min_target_nonserve_shots
    ]

    usable_keys = set()
    usable_match_ids = defaultdict(set)
    usable_bin_totals = defaultdict(Counter)
    for _, feature in features.iterrows():
        player = str(feature["player"]).strip()
        key = _rally_key(feature)
        physical = physical_by_key.get(key)
        if not player or physical is None or player not in physical["players"]:
            continue
        usable_keys.add((player, key, physical["length_bin"]))
        usable_match_ids[player].add(key[0])
        usable_bin_totals[player][physical["length_bin"]] += 1

    players = sorted({player for participants in rally_table["players"] for player in participants})
    usable_totals = Counter(player for player, _, _ in usable_keys)
    eligibility_threshold = math.ceil(min_train_rallies / train_ratio)
    eligible_players = sorted(
        player
        for player in players
        if usable_totals[player] >= eligibility_threshold
        and len(usable_match_ids[player]) >= min_matches
    )
    eligible_set = set(eligible_players)
    train_only_players = {}
    for player in players:
        if player in eligible_set:
            continue
        reasons = []
        if usable_totals[player] < eligibility_threshold:
            reasons.append("insufficient_usable_rallies")
        if len(usable_match_ids[player]) < min_matches:
            reasons.append("insufficient_usable_matches")
        train_only_players[player] = {
            "reasons": reasons,
            "usable_rally_count": int(usable_totals[player]),
            "required_usable_rally_count": int(eligibility_threshold),
            "match_count": len(usable_match_ids[player]),
            "required_match_count": int(min_matches),
        }

    candidate_indices = [
        index
        for index, row in rally_table.iterrows()
        if set(row["players"]).issubset(eligible_set)
    ]
    test_indices, optimization = _select_test_rallies(
        rally_table,
        candidate_indices,
        eligible_players,
        usable_keys,
        usable_totals,
        min_train_rallies,
        test_ratio,
        seed,
    )

    assignments = {"train": defaultdict(list), "test": defaultdict(list)}
    raw_bin_counts = {
        "train": defaultdict(Counter),
        "test": defaultdict(Counter),
    }
    usable_fold_counts = {"train": Counter(), "test": Counter()}
    usable_fold_bins = {
        "train": defaultdict(Counter),
        "test": defaultdict(Counter),
    }
    physical_length_actuals = Counter()
    for index, row in rally_table.iterrows():
        fold = "test" if index in test_indices else "train"
        key = _rally_key(row)
        if fold == "test":
            physical_length_actuals[row["length_bin"]] += 1
        for player in row["players"]:
            assignments[fold][player].append(_rally_record(row))
            raw_bin_counts[fold][player][row["length_bin"]] += 1
            if (player, key, row["length_bin"]) in usable_keys:
                usable_fold_counts[fold][player] += 1
                usable_fold_bins[fold][player][row["length_bin"]] += 1

    for fold in ("train", "test"):
        for player in players:
            assignments[fold][player] = sorted(
                assignments[fold].get(player, []),
                key=lambda item: (item["match_id"], item["set"], item["rally"]),
            )

    player_stats = {}
    for player in players:
        raw_train = len(assignments["train"][player])
        raw_test = len(assignments["test"][player])
        length_targets = {
            label: int(round(test_ratio * usable_bin_totals[player][label]))
            for label in LENGTH_LABELS
        }
        length_actuals = {
            label: int(usable_fold_bins["test"][player][label])
            for label in LENGTH_LABELS
        }
        player_stats[player] = {
            "total": raw_train + raw_test,
            "train": raw_train,
            "test": raw_test,
            "raw_physical_counts": {
                "total": raw_train + raw_test,
                "train": raw_train,
                "test": raw_test,
            },
            "usable_total": int(usable_totals[player]),
            "usable_train": int(usable_fold_counts["train"][player]),
            "usable_test": int(usable_fold_counts["test"][player]),
            "match_count": len(usable_match_ids[player]),
            "eligible": player in eligible_set,
            "eligibility": "eligible" if player in eligible_set else "train_only",
            "length_targets": length_targets,
            "length_actuals": length_actuals,
            "usable_length_counts": {
                label: {
                    "total": int(usable_bin_totals[player][label]),
                    "train": int(usable_fold_bins["train"][player][label]),
                    "test": length_actuals[label],
                }
                for label in LENGTH_LABELS
            },
            "train_length_counts": {
                label: int(raw_bin_counts["train"][player][label]) for label in LENGTH_LABELS
            },
            "test_length_counts": {
                label: int(raw_bin_counts["test"][player][label]) for label in LENGTH_LABELS
            },
        }

    candidate_length_counts = Counter(rally_table.loc[candidate_indices, "length_bin"])
    split = {
        "schema_version": 2,
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "test_ratio": float(test_ratio),
        "length_quantiles": [float(value) for value in length_quantiles],
        "length_boundaries": {
            "short_medium": boundaries[0],
            "medium_long": boundaries[1],
        },
        "eligibility_settings": {
            "min_train_rallies": int(min_train_rallies),
            "min_matches": int(min_matches),
            "min_target_nonserve_shots": int(min_target_nonserve_shots),
            "min_eligible_usable_rallies": int(eligibility_threshold),
        },
        "eligible_players": eligible_players,
        "train_only_players": train_only_players,
        "optimization": optimization,
        "physical_length_stats": {
            label: {
                "population": int(candidate_length_counts[label]),
                "target": int(round(test_ratio * candidate_length_counts[label])),
                "actual": int(physical_length_actuals[label]),
            }
            for label in LENGTH_LABELS
        },
        "train": dict(assignments["train"]),
        "test": dict(assignments["test"]),
        "player_stats": player_stats,
    }
    validate_rally_split(split)
    return split


def validate_rally_split(split):
    train = split.get("train", {})
    test = split.get("test", {})
    if set(train) != set(test):
        raise ValueError("Train and test must contain the same player keys")

    physical_folds = {}
    for fold, assignments in (("train", train), ("test", test)):
        for player, records in assignments.items():
            seen = set()
            for record in records:
                key = _rally_key(record)
                if key in seen:
                    raise ValueError(f"Duplicate {fold} rally for {player}: {key}")
                seen.add(key)
                previous_fold = physical_folds.setdefault(key, fold)
                if previous_fold != fold:
                    raise ValueError(f"Physical rally appears in both folds: {key}")

    for player in train:
        train_keys = {_rally_key(item) for item in train[player]}
        test_keys = {_rally_key(item) for item in test[player]}
        if train_keys & test_keys:
            raise ValueError(f"Player {player} has overlapping train/test rallies")

    train_only_players = set(split.get("train_only_players", {}))
    for player in train_only_players:
        if test.get(player):
            raise ValueError(f"Train-only player {player} has test rallies")

    settings = split.get("eligibility_settings")
    stats = split.get("player_stats", {})
    if settings:
        minimum = int(settings["min_train_rallies"])
        for player in split.get("eligible_players", []):
            if player not in stats:
                raise ValueError(f"Eligible player {player} has no player_stats entry")
            if int(stats[player]["usable_train"]) < minimum:
                raise ValueError(
                    f"Eligible player {player} has fewer than {minimum} usable train rallies"
                )
    return True


def _project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description="Create an eligibility-aware atomic rally split")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "Scripts", "config.yaml"))
    parser.add_argument("--merged-csv")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    paths = config.get("paths", {})
    settings = config.get("rally_split", {})
    eligibility = config.get("profile_eligibility", {})
    merged_path = _project_path(
        args.merged_csv or paths.get("merged_csv", "Data/intermediate_data/merged.csv")
    )
    output_path = _project_path(paths.get("rally_split", "Data/intermediate_data/rally_split.json"))
    train_ratio = float(settings.get("train_ratio", 0.8))
    test_ratio = float(settings.get("test_ratio", 1.0 - train_ratio))
    if not np.isclose(train_ratio + test_ratio, 1.0):
        raise ValueError("rally_split train_ratio and test_ratio must sum to 1")
    if not bool(settings.get("keep_physical_rally_atomic", True)):
        raise ValueError("Only atomic physical-rally splitting is supported")

    split = build_rally_split(
        pd.read_csv(merged_path),
        train_ratio=train_ratio,
        seed=int(args.seed if args.seed is not None else settings.get("seed", 42)),
        length_quantiles=tuple(settings.get("length_quantiles", [1 / 3, 2 / 3])),
        min_train_rallies=int(eligibility.get("min_train_rallies", 50)),
        min_matches=int(eligibility.get("min_matches", 2)),
        min_target_nonserve_shots=int(
            eligibility.get("min_target_nonserve_shots", 1)
        ),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=2)
    print(f"Rally split saved to {output_path}")
    print(f"Players: {len(split['player_stats'])}")


if __name__ == "__main__":
    main()
