import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, StrictUndefined
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Preprocessing.rally_feature import (  # noqa: E402
    CLUSTER_FEATURE_COLUMNS,
    extract_rally_features,
)
from Utils.llm import LLM  # noqa: E402


FEATURE_METADATA = {
    "attack_shot_ratio": {
        "label": "attacking-shot frequency",
        "description": (
            "The proportion of the player's valid tactical shots classified as "
            "attacking shots. A higher z-score means more frequent attacking-shot use "
            "than in the player's usual training behavior; a lower value means less "
            "frequent use."
        ),
        "higher_instruction": "use attacking shots more frequently",
        "lower_instruction": "use attacking shots less frequently",
    },
    "defense_shot_ratio": {
        "label": "defensive-reset-shot frequency",
        "description": (
            "The proportion of valid tactical shots classified as defensive or "
            "rally-resetting shots. A higher z-score means more frequent defensive-reset-"
            "shot use than usual; a lower value means less frequent use."
        ),
        "higher_instruction": "use defensive reset shots more frequently",
        "lower_instruction": "use defensive reset shots less frequently",
    },
    "side_ratio": {
        "label": "wide side-area targeting frequency",
        "description": (
            "The proportion of mapped landing targets directed toward broad lateral "
            "regions. A higher z-score means wider side-oriented targeting than usual; "
            "a lower value means more central targeting."
        ),
        "higher_instruction": "target wide side areas more frequently",
        "lower_instruction": "target central areas more frequently",
    },
    "frontcourt_ratio": {
        "label": "front-court targeting frequency",
        "description": (
            "The proportion of mapped landing targets directed to the front court. A "
            "higher z-score means more frequent front-court targeting than usual; a "
            "lower value means less frequent front-court targeting."
        ),
        "higher_instruction": "target the front court more frequently",
        "lower_instruction": "target the front court less frequently",
    },
    "backcourt_ratio": {
        "label": "back-court targeting frequency",
        "description": (
            "The proportion of mapped landing targets directed to the back court. A "
            "higher z-score means more frequent back-court targeting than usual; a lower "
            "value means less frequent back-court targeting."
        ),
        "higher_instruction": "target the back court more frequently",
        "lower_instruction": "target the back court less frequently",
    },
    "signed_side_bias": {
        "label": "right-versus-left targeting balance",
        "description": (
            "The right-versus-left balance of mapped landing targets. A positive z-score "
            "means the mode shifts further toward the right side than the player's usual "
            "behavior, while a negative value means a shift toward the left."
        ),
        "higher_instruction": "shift targeting toward the right side",
        "lower_instruction": "shift targeting toward the left side",
    },
    "front_back_switch_ratio": {
        "label": "front-back target-switching frequency",
        "description": (
            "The proportion of consecutive front/back target transitions that switch "
            "between the front and back court. A higher z-score means more frequent "
            "front-back alternation than usual; a lower value means greater consistency "
            "in target depth."
        ),
        "higher_instruction": "alternate more frequently between front- and back-court targets",
        "lower_instruction": "maintain more consistent front-back target depth",
    },
    "left_right_switch_ratio": {
        "label": "left-right target-switching frequency",
        "description": (
            "The proportion of consecutive left/right target transitions that switch "
            "between the two sides of the court. A higher z-score means more frequent "
            "left-right alternation than usual; a lower value means greater lateral "
            "consistency."
        ),
        "higher_instruction": "alternate more frequently between left- and right-side targets",
        "lower_instruction": "maintain more consistent lateral targeting",
    },
    "log_rally_length": {
        "label": "physical rally length",
        "description": (
            "The logarithmically transformed number of strokes in the complete physical "
            "rally. A higher z-score means longer rallies than the player's usual training "
            "behavior; a lower value means shorter rallies."
        ),
        "higher_instruction": "play longer rallies",
        "lower_instruction": "play shorter rallies",
    },
    "average_movement": {
        "label": "movement distance per shot",
        "description": (
            "The player's average movement distance associated with each valid tactical "
            "shot, combining movement toward the shuttle and post-shot recovery movement. "
            "A higher z-score means more movement per shot than usual; a lower value means "
            "less movement."
        ),
        "higher_instruction": "cover more movement distance per shot",
        "lower_instruction": "cover less movement distance per shot",
    },
    "landing_opponent_distance": {
        "label": "landing distance from the opponent",
        "description": (
            "The average spatial distance between the shuttle landing point and the "
            "opponent's recorded position. A higher z-score means placements farther from "
            "the opponent than usual; a lower value means placements closer to the opponent."
        ),
        "higher_instruction": "place the shuttle farther from the opponent",
        "lower_instruction": "place the shuttle closer to the opponent",
    },
}


RATIO_SPECS = {
    "attack_shot_ratio": ("attack_shot_count", "target_nonserve_shot_count"),
    "defense_shot_ratio": ("defense_shot_count", "target_nonserve_shot_count"),
    "side_ratio": ("side_landing_count", "valid_landing_count"),
    "frontcourt_ratio": ("frontcourt_landing_count", "valid_landing_count"),
    "backcourt_ratio": ("backcourt_landing_count", "valid_landing_count"),
    "front_back_switch_ratio": ("front_back_switch_count", "front_back_transition_count"),
    "left_right_switch_ratio": ("left_right_switch_count", "left_right_transition_count"),
}

CONTINUOUS_FEATURES = {
    "log_rally_length",
    "average_movement",
    "landing_opponent_distance",
}


def _clean_number(value):
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _clean_mapping(mapping):
    return {key: _clean_number(value) for key, value in mapping.items()}


def _key(record):
    return int(record["match_id"]), int(record["set"]), int(record["rally"])


def _select_keys(feature_df, records):
    keys = {_key(record) for record in records}
    if not keys:
        return feature_df.iloc[0:0].copy()
    row_keys = list(
        zip(
            feature_df["match_id"].astype(int),
            feature_df["set"].astype(int),
            feature_df["rally"].astype(int),
        )
    )
    return feature_df[[key in keys for key in row_keys]].copy()


def _aggregate_rate(df, numerator, denominator):
    denominator_total = pd.to_numeric(df[denominator], errors="coerce").fillna(0).sum()
    if denominator_total <= 0:
        return 0.0
    numerator_total = pd.to_numeric(df[numerator], errors="coerce").fillna(0).sum()
    return float(numerator_total / denominator_total)


def _fit_feature_transform(train_df, test_df, smoothing_alpha, clip_quantiles):
    baselines = {
        feature: _aggregate_rate(train_df, numerator, denominator)
        for feature, (numerator, denominator) in RATIO_SPECS.items()
    }
    landing_total = pd.to_numeric(train_df["valid_landing_count"], errors="coerce").fillna(0).sum()
    signed_total = (
        pd.to_numeric(train_df["right_landing_count"], errors="coerce").fillna(0).sum()
        - pd.to_numeric(train_df["left_landing_count"], errors="coerce").fillna(0).sum()
    )
    baselines["signed_side_bias"] = float(signed_total / landing_total) if landing_total else 0.0

    def transform(raw_df):
        transformed = pd.DataFrame(index=raw_df.index)
        for feature, (numerator, denominator) in RATIO_SPECS.items():
            numerator_values = pd.to_numeric(raw_df[numerator], errors="coerce").fillna(0)
            denominator_values = pd.to_numeric(raw_df[denominator], errors="coerce").fillna(0)
            transformed[feature] = (
                numerator_values + smoothing_alpha * baselines[feature]
            ) / (denominator_values + smoothing_alpha)

        right = pd.to_numeric(raw_df["right_landing_count"], errors="coerce").fillna(0)
        left = pd.to_numeric(raw_df["left_landing_count"], errors="coerce").fillna(0)
        landing = pd.to_numeric(raw_df["valid_landing_count"], errors="coerce").fillna(0)
        transformed["signed_side_bias"] = (
            right - left + smoothing_alpha * baselines["signed_side_bias"]
        ) / (landing + smoothing_alpha)
        transformed["log_rally_length"] = np.log1p(
            pd.to_numeric(raw_df["rally_length"], errors="coerce").clip(lower=0)
        )
        transformed["average_movement"] = pd.to_numeric(
            raw_df["average_movement"], errors="coerce"
        )
        transformed["landing_opponent_distance"] = pd.to_numeric(
            raw_df["landing_opponent_distance"], errors="coerce"
        )
        return transformed[CLUSTER_FEATURE_COLUMNS]

    train_values = transform(train_df)
    test_values = transform(test_df)
    if len(clip_quantiles) != 2:
        raise ValueError("clip_quantiles must contain exactly two values")
    lower_quantile, upper_quantile = (float(value) for value in clip_quantiles)
    if not 0 <= lower_quantile <= upper_quantile <= 1:
        raise ValueError("clip_quantiles must be increasing values in [0, 1]")
    clipping = {}
    for feature in CONTINUOUS_FEATURES:
        valid = train_values[feature].dropna()
        if valid.empty:
            lower = upper = 0.0
        else:
            lower = float(valid.quantile(lower_quantile))
            upper = float(valid.quantile(upper_quantile))
        if lower > upper:
            lower, upper = upper, lower
        clipping[feature] = {"lower": lower, "upper": upper}
        train_values[feature] = train_values[feature].clip(lower=lower, upper=upper)
        test_values[feature] = test_values[feature].clip(lower=lower, upper=upper)

    medians = train_values.median(skipna=True).fillna(0.0)
    train_values = train_values.fillna(medians)
    test_values = test_values.fillna(medians)
    means = train_values.mean()
    scales = train_values.std(ddof=0)
    zero_variance = scales[~np.isfinite(scales) | (scales <= 1e-12)].index.tolist()
    scales.loc[zero_variance] = 1.0

    train_z = (train_values - means) / scales
    test_z = (test_values - means) / scales
    metadata = {
        "smoothing_baselines": _clean_mapping(baselines),
        "imputation_medians": _clean_mapping(medians.to_dict()),
        "means": _clean_mapping(means.to_dict()),
        "stds": _clean_mapping(scales.to_dict()),
        "zero_variance_features": zero_variance,
        "clipping": {
            feature: _clean_mapping(bounds) for feature, bounds in clipping.items()
        },
    }
    return train_values, test_values, train_z, test_z, metadata


def candidate_ks(n_samples, n_unique, k_min=2, min_cluster_size=None):
    if n_samples <= 1:
        return []
    k_start = max(2, int(k_min))
    k_max = min(
        int(math.floor(math.sqrt(n_samples / 2))),
        int(n_unique),
        int(n_samples - 1),
    )
    return list(range(k_start, k_max + 1)) if k_max >= k_start else []


def select_elbow_k(inertia_by_k):
    """Select the first maximum-distance point from the normalized end-point chord."""
    points = sorted((int(k), float(inertia)) for k, inertia in inertia_by_k.items())
    if len(points) < 2 or points[0][0] != 1 or points[1][0] != 2:
        raise ValueError("elbow selection requires fitted K=1 and K=2")
    if len(points) == 2:
        return 2

    ks = np.asarray([point[0] for point in points], dtype=float)
    inertias = np.asarray([point[1] for point in points], dtype=float)
    inertia_span = inertias[0] - inertias[-1]
    if not np.isfinite(inertia_span) or abs(inertia_span) <= 1e-12:
        return 2

    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertias - inertias[-1]) / inertia_span
    chord_start = np.array([x[0], y[0]])
    chord = np.array([x[-1], y[-1]]) - chord_start
    chord_length = np.linalg.norm(chord)
    if chord_length <= 1e-12:
        return 2

    distances = np.abs(
        chord[0] * (chord_start[1] - y)
        - (chord_start[0] - x) * chord[1]
    ) / chord_length
    candidate_distances = distances[1:]
    maximum = np.max(candidate_distances)
    # np.flatnonzero preserves ascending K, giving the deterministic smaller-K tie break.
    return int(ks[1 + int(np.flatnonzero(candidate_distances == maximum)[0])])


def _stable_cluster_mapping(labels, centers):
    cluster_ids, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(cluster_ids.tolist(), counts.tolist()))
    ordered = sorted(
        cluster_ids.tolist(),
        key=lambda cluster_id: (
            -sizes[cluster_id],
            tuple(np.round(centers[cluster_id], 8).tolist()),
        ),
    )
    return {old_id: new_id for new_id, old_id in enumerate(ordered)}


def fit_player_profile(player_name, player_features, train_records, test_records, settings):
    min_cluster_size = int(settings.get("min_cluster_size", 5))
    elbow_window_radius = int(settings.get("elbow_window_radius", 1))
    n_init = int(settings.get("n_init", 50))
    if min_cluster_size < 1:
        raise ValueError("rally_clustering.min_cluster_size must be at least 1")
    if elbow_window_radius < 0:
        raise ValueError("rally_clustering.elbow_window_radius must be non-negative")
    if n_init < 1:
        raise ValueError("rally_clustering.n_init must be at least 1")

    min_shots = int(settings.get("min_target_nonserve_shots", 1))
    train_raw = _select_keys(player_features, train_records)
    test_raw = _select_keys(player_features, test_records)
    train_raw = train_raw[train_raw["target_nonserve_shot_count"] >= min_shots].copy()
    test_raw = test_raw[test_raw["target_nonserve_shot_count"] >= min_shots].copy()

    min_train = int(settings.get("min_train_rallies", 50))
    result = {
        "status": "excluded",
        "reason": None,
        "train_rally_count": int(len(train_raw)),
        "test_rally_count": int(len(test_raw)),
        "selected_k": None,
        "selected_silhouette": None,
        "inertia_by_k": {},
        "elbow_k": None,
        "elbow_window": None,
        "k_candidates": [],
        "rejected_ks": {},
        "silhouette_scores": {},
        "selection_method": "elbow_window_then_maximum_silhouette",
        "selection_tie_break": "smaller_k",
        "train_statistics": {},
        "clusters": [],
        "train_assignments": [],
        "test_assignments": [],
    }
    if len(train_raw) < min_train:
        result["reason"] = f"requires at least {min_train} train rallies"
        return result

    train_values, test_values, train_z, test_z, transform_metadata = _fit_feature_transform(
        train_raw,
        test_raw,
        float(settings.get("smoothing_alpha", 2.0)),
        settings.get("clip_quantiles", [0.01, 0.99]),
    )
    result["train_statistics"] = transform_metadata

    train_array = train_z.to_numpy(dtype=float)
    test_array = test_z.to_numpy(dtype=float)
    unique_count = len(np.unique(train_array, axis=0))
    k_values = candidate_ks(
        len(train_array),
        unique_count,
    )
    result["k_candidates"] = k_values

    if int(settings.get("k_min", 2)) != 2:
        raise ValueError("rally_clustering.k_min must be 2")
    random_state = int(settings.get("random_state", 42))
    fitted = {}
    baseline = KMeans(n_clusters=1, random_state=random_state, n_init=n_init)
    baseline.fit(train_array)
    fitted[1] = (baseline, baseline.labels_)
    result["inertia_by_k"]["1"] = float(baseline.inertia_)
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(train_array)
        fitted[k] = (model, labels)
        result["inertia_by_k"][str(k)] = float(model.inertia_)

    if not k_values:
        result["reason"] = "no valid K candidates"
        return result

    elbow_k = select_elbow_k(result["inertia_by_k"])
    result["elbow_k"] = elbow_k
    window_lower = max(2, elbow_k - elbow_window_radius)
    window_upper = min(max(k_values), elbow_k + elbow_window_radius)
    result["elbow_window"] = {
        "radius": elbow_window_radius,
        "lower": window_lower,
        "upper": window_upper,
    }
    valid_fits = {}
    match_ids = train_raw["match_id"].to_numpy()
    for k in k_values:
        model, labels = fitted[k]
        counts = np.bincount(labels, minlength=k)
        cluster_match_counts = [
            int(pd.unique(match_ids[labels == cluster_id]).size)
            for cluster_id in range(k)
        ]
        observed_cluster_count = int(np.count_nonzero(counts))
        rejection = {
            "observed_cluster_count": observed_cluster_count,
            "cluster_sizes": [int(count) for count in counts],
            "cluster_match_counts": cluster_match_counts,
            "smallest_cluster_size": int(counts.min()),
            "smallest_cluster_match_count": int(min(cluster_match_counts)),
        }
        reasons = []
        if not window_lower <= k <= window_upper:
            reasons.append("outside_elbow_window")
        if observed_cluster_count != k:
            reasons.append("collapsed_labels")
        if int(counts.min()) < min_cluster_size:
            reasons.append("undersized_cluster")
        if reasons:
            rejection["reasons"] = reasons
            result["rejected_ks"][str(k)] = rejection
            continue
        score = float(silhouette_score(train_array, labels))
        result["silhouette_scores"][str(k)] = {
            "score": score,
            "min_cluster_size": int(counts.min()),
            "cluster_sizes": [int(count) for count in counts],
            "min_cluster_match_count": int(min(cluster_match_counts)),
            "cluster_match_counts": cluster_match_counts,
        }
        valid_fits[k] = (score, model, labels)

    if not valid_fits:
        result["reason"] = (
            "all K candidates failed the elbow window or cluster validity checks"
        )
        return result

    selected_k = max(valid_fits, key=lambda k: (valid_fits[k][0], -k))
    selected_score = valid_fits[selected_k][0]

    _, model, old_train_labels = valid_fits[selected_k]
    cluster_mapping = _stable_cluster_mapping(old_train_labels, model.cluster_centers_)
    train_labels = np.array([cluster_mapping[int(label)] for label in old_train_labels], dtype=int)
    ordered_centers = np.zeros_like(model.cluster_centers_)
    for old_id, new_id in cluster_mapping.items():
        ordered_centers[new_id] = model.cluster_centers_[old_id]

    old_train_distances = model.transform(train_array)
    ordered_train_distances = np.empty_like(old_train_distances)
    for old_id, new_id in cluster_mapping.items():
        ordered_train_distances[:, new_id] = old_train_distances[:, old_id]

    clusters = []
    support_radii = {}
    for cluster_id in range(selected_k):
        member_mask = train_labels == cluster_id
        member_values = train_values.loc[member_mask]
        center_raw = member_values.mean().to_dict()
        center_raw["rally_length"] = float(train_raw.loc[member_mask, "rally_length"].mean())
        support_radius = float(
            np.quantile(ordered_train_distances[member_mask, cluster_id], 0.95)
        )
        support_radii[cluster_id] = support_radius
        member_z = train_z.loc[member_mask]
        feature_summary_z = {}
        for feature in CLUSTER_FEATURE_COLUMNS:
            values = member_z[feature].to_numpy(dtype=float)
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            if median > 0:
                direction_support = float(np.mean(values > 0))
            elif median < 0:
                direction_support = float(np.mean(values < 0))
            else:
                direction_support = 0.5
            feature_summary_z[feature] = _clean_mapping({
                "mean": float(np.mean(values)),
                "median": median,
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(q75 - q25),
                "direction_support": direction_support,
            })
        clusters.append({
            "cluster_id": cluster_id,
            "train_size": int(member_mask.sum()),
            "train_match_count": int(train_raw.loc[member_mask, "match_id"].nunique()),
            "distance_p95": round(support_radius, 6),
            "center_raw": _clean_mapping(center_raw),
            "center_z": _clean_mapping(
                dict(zip(CLUSTER_FEATURE_COLUMNS, ordered_centers[cluster_id].tolist()))
            ),
            "feature_summary_z": feature_summary_z,
            "label": None,
            "description": None,
            "instruction": None,
            "instruction_source": None,
        })

    train_assignments = []
    for row_position, (_, row) in enumerate(train_raw.iterrows()):
        cluster_id = int(train_labels[row_position])
        train_assignments.append({
            "match_id": int(row["match_id"]),
            "set": int(row["set"]),
            "rally": int(row["rally"]),
            "cluster_id": cluster_id,
            "distance": round(float(ordered_train_distances[row_position, cluster_id]), 6),
        })

    assignments = []
    if len(test_array):
        old_test_labels = model.predict(test_array)
        distances = model.transform(test_array)
        for row_position, (_, row) in enumerate(test_raw.iterrows()):
            ordered_distances = np.empty(selected_k, dtype=float)
            for old_id, new_id in cluster_mapping.items():
                ordered_distances[new_id] = distances[row_position, old_id]
            nearest_order = np.argsort(ordered_distances)
            assigned_cluster = cluster_mapping[int(old_test_labels[row_position])]
            assignments.append({
                "match_id": int(row["match_id"]),
                "set": int(row["set"]),
                "rally": int(row["rally"]),
                "cluster_id": int(assigned_cluster),
                "nearest_distance": round(float(ordered_distances[assigned_cluster]), 6),
                "in_train_support": bool(
                    ordered_distances[assigned_cluster] <= support_radii[assigned_cluster]
                ),
                "second_nearest_cluster_id": int(nearest_order[1]),
                "second_nearest_distance": round(float(ordered_distances[nearest_order[1]]), 6),
            })

    result.update({
        "status": "clustered",
        "reason": None,
        "selected_k": int(selected_k),
        "selected_silhouette": round(float(selected_score), 6),
        "clusters": clusters,
        "train_assignments": train_assignments,
        "test_assignments": assignments,
    })
    return result


def centroid_feature_payload(center_z):
    """Return the complete, ordered centroid profile supplied to the LLM."""
    if not isinstance(center_z, dict):
        raise ValueError("cluster center_z must be an object")
    if set(center_z) != set(CLUSTER_FEATURE_COLUMNS):
        raise ValueError("cluster center_z must contain every configured feature exactly once")

    payload = []
    for feature in CLUSTER_FEATURE_COLUMNS:
        value = center_z[feature]
        if value is None or not np.isfinite(value):
            raise ValueError(f"cluster center_z contains an invalid value for {feature}")
        metadata = FEATURE_METADATA[feature]
        payload.append({
            "feature": feature,
            "label": metadata["label"],
            "description": metadata["description"],
            "z_score": round(float(value), 3),
        })
    return payload


def _valid_instruction_response(data):
    if not isinstance(data, dict) or set(data) != {"label", "description", "instruction"}:
        return False
    return all(isinstance(data[key], str) and data[key].strip() for key in data)


def _fallback_instruction(centroid_features, min_abs_z=0.5, max_tendencies=5):
    selected = sorted(
        (
            item for item in centroid_features
            if abs(float(item["z_score"])) >= float(min_abs_z)
        ),
        key=lambda item: (-abs(float(item["z_score"])), item["feature"]),
    )[: int(max_tendencies)]
    if not selected:
        return {
            "label": "Balanced tactical profile",
            "description": "A mode without a large deviation from the player's usual training behavior.",
            "instruction": "Imitate the player's usual balanced tactical behavior.",
        }
    phrases = [
        FEATURE_METADATA[item["feature"]][
            "higher_instruction" if float(item["z_score"]) > 0 else "lower_instruction"
        ]
        for item in selected
    ]
    instruction = "Imitate this tactical mode: " + "; ".join(phrases) + "."
    return {
        "label": "Distinct tactical mode",
        "description": "This mode is characterized by " + ", ".join(phrases) + ".",
        "instruction": instruction,
    }


def generate_cluster_instruction(
    cluster,
    llm,
    user_template,
    fallback_min_abs_z=0.5,
    fallback_max_tendencies=5,
):
    centroid_features = centroid_feature_payload(cluster["center_z"])
    if llm is None:
        return _fallback_instruction(
            centroid_features,
            fallback_min_abs_z,
            fallback_max_tendencies,
        ), "deterministic"

    user_prompt = user_template.render(
        centroid_features=json.dumps(centroid_features, ensure_ascii=False, indent=2),
    )
    response = llm.get_verified_response(
        schema=None,
        user_prompt=user_prompt,
        validator=_valid_instruction_response,
    )
    data = response[0] if isinstance(response, tuple) else response
    if _valid_instruction_response(data):
        return {key: data[key].strip() for key in data}, "llm"
    return _fallback_instruction(
        centroid_features,
        fallback_min_abs_z,
        fallback_max_tendencies,
    ), "deterministic_fallback"


def generate_instruction_variant(
    cluster,
    variant,
    llm,
    user_template,
    fallback_min_abs_z=0.5,
    fallback_max_tendencies=5,
):
    if variant not in {"opposite", "rewrite"}:
        raise ValueError(f"Unsupported instruction variant: {variant}")

    original = {
        key: cluster.get(key)
        for key in ("label", "description", "instruction")
    }
    if not _valid_instruction_response(original):
        raise ValueError("Cluster must have a valid instruction before variants are generated")

    if variant == "opposite":
        opposite_features = centroid_feature_payload({
            feature: -float(cluster["center_z"][feature])
            for feature in CLUSTER_FEATURE_COLUMNS
        })
        fallback = _fallback_instruction(
            opposite_features,
            fallback_min_abs_z,
            fallback_max_tendencies,
        )
        fallback_source = "deterministic_fallback"
    else:
        fallback = original
        fallback_source = "original_fallback"

    if llm is None:
        source = "deterministic" if variant == "opposite" else "original"
        return {key: fallback[key].strip() for key in fallback}, source

    user_prompt = user_template.render(
        variant=variant,
        source_profile=json.dumps(original, ensure_ascii=False, indent=2),
    )
    response = llm.get_verified_response(
        schema=None,
        user_prompt=user_prompt,
        validator=_valid_instruction_response,
    )
    data = response[0] if isinstance(response, tuple) else response
    if _valid_instruction_response(data):
        return {key: data[key].strip() for key in data}, "llm"
    return {key: fallback[key].strip() for key in fallback}, fallback_source


def add_instruction_variants(
    player_name,
    profile,
    llm,
    user_template,
    random_seed=42,
    fallback_min_abs_z=0.5,
    fallback_max_tendencies=5,
):
    clusters = sorted(profile.get("clusters", []), key=lambda item: int(item["cluster_id"]))
    if len(clusters) < 2:
        raise ValueError(f"Player {player_name!r} needs at least two clusters for random_other")

    for cluster in clusters:
        variants = {}
        for variant in ("opposite", "rewrite"):
            instruction, source = generate_instruction_variant(
                cluster,
                variant,
                llm,
                user_template,
                fallback_min_abs_z,
                fallback_max_tendencies,
            )
            variants[variant] = {**instruction, "source": source}

        candidates = [
            candidate for candidate in clusters
            if int(candidate["cluster_id"]) != int(cluster["cluster_id"])
        ]
        seed_payload = f"{random_seed}:{player_name}:{cluster['cluster_id']}".encode("utf-8")
        candidate_index = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:8], "big"
        ) % len(candidates)
        target = candidates[candidate_index]
        variants["random_other"] = {
            "label": target["label"],
            "description": target["description"],
            "instruction": target["instruction"],
            "source": "same_player_cluster",
            "target_cluster_id": int(target["cluster_id"]),
            "seed": int(random_seed),
        }
        cluster["instruction_variants"] = variants


def _project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Fit train-only player rally clusters")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "Scripts", "config.yaml"))
    parser.add_argument("--merged-csv")
    parser.add_argument("--skip-llm", action="store_true", help="Use deterministic instructions")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    paths = config.get("paths", {})
    settings = {
        **config.get("rally_clustering", {}),
        **config.get("profile_eligibility", {}),
    }
    for obsolete_setting in ("min_test_rallies", "silhouette_tie_tolerance"):
        settings.pop(obsolete_setting, None)
    configured_features = settings.get("features", CLUSTER_FEATURE_COLUMNS)
    if configured_features != CLUSTER_FEATURE_COLUMNS:
        raise ValueError(
            "rally_clustering.features must match the implemented feature order: "
            f"{CLUSTER_FEATURE_COLUMNS}"
        )
    merged_path = _project_path(
        args.merged_csv or paths.get("merged_csv", "Data/intermediate_data/merged.csv")
    )
    split_path = _project_path(paths.get("rally_split", "Data/intermediate_data/rally_split.json"))
    output_path = _project_path(paths.get("rally_profiles", "Data/intermediate_data/rally_profiles.json"))

    with open(split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    feature_df = extract_rally_features(pd.read_csv(merged_path))

    prompt_root = os.path.join(PROJECT_ROOT, "Prompts", "Preprocessing")
    with open(os.path.join(prompt_root, "rally_profile_user.jinja"), "r", encoding="utf-8") as handle:
        user_template = Environment(undefined=StrictUndefined).from_string(handle.read())
    with open(
        os.path.join(prompt_root, "rally_instruction_variant_user.jinja"),
        "r",
        encoding="utf-8",
    ) as handle:
        variant_template = Environment(undefined=StrictUndefined).from_string(handle.read())

    players = sorted(split["eligible_players"])
    profiles = {}
    for player_name in players:
        player_features = feature_df[feature_df["player"] == player_name].copy()
        profile = fit_player_profile(
            player_name,
            player_features,
            split.get("train", {}).get(player_name, []),
            split.get("test", {}).get(player_name, []),
            settings,
        )
        profiles[player_name] = profile
        print(
            f"{player_name}: {profile['status']} "
            f"(train={profile['train_rally_count']}, test={profile['test_rally_count']}, "
            f"k={profile['selected_k']})"
        )

    has_clusters = any(profile["status"] == "clustered" for profile in profiles.values())
    llm = None
    llm_metadata = {"enabled": bool(has_clusters and not args.skip_llm)}
    if has_clusters and not args.skip_llm:
        llm = LLM(generation_params={"temperature": 0.0, "top_p": 1.0, "max_tokens": 512})
        llm_metadata.update({
            "provider": llm.provider,
            "model_name": llm.model_name,
            "generation_params": llm.generation_params,
        })

    for profile in profiles.values():
        if profile["status"] != "clustered":
            continue
        for cluster in profile["clusters"]:
            instruction, source = generate_cluster_instruction(
                cluster,
                llm,
                user_template,
                float(settings.get("fallback_min_abs_z", 0.5)),
                int(settings.get("fallback_max_tendencies", 5)),
            )
            cluster.update(instruction)
            cluster["instruction_source"] = source

    for player_name, profile in profiles.items():
        if profile["status"] != "clustered":
            continue
        add_instruction_variants(
            player_name,
            profile,
            llm,
            variant_template,
            int(settings.get("instruction_variant_seed", 42)),
            float(settings.get("fallback_min_abs_z", 0.5)),
            int(settings.get("fallback_max_tendencies", 5)),
        )

    artifact = {
        "schema_version": 4,
        "feature_version": 3,
        "split_sha256": _file_digest(split_path),
        "feature_names": CLUSTER_FEATURE_COLUMNS,
        "clustering_settings": settings,
        "llm": llm_metadata,
        "players": profiles,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    print(f"Rally profiles saved to {output_path}")


if __name__ == "__main__":
    main()
