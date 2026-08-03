"""Generate static per-player rally-profile dashboards."""

import argparse
import difflib
import json
from pathlib import Path
import re
import sys
import textwrap
import unicodedata

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Preprocessing.rally_clustering import _fit_feature_transform, _select_keys  # noqa: E402
from Preprocessing.rally_feature import (  # noqa: E402
    CLUSTER_FEATURE_COLUMNS,
    IN_COURT_AREA_IDS,
    SERVE_SHOTS,
    _area_series,
    extract_rally_features,
)


KEY_COLUMNS = ("match_id", "set", "rally")
DEFAULT_OUTPUT_DIR = "Experiments/rally_profiles/visualizations"
FEATURE_LABELS = {
    "attack_shot_ratio": "Attack shot ratio",
    "defense_shot_ratio": "Defense shot ratio",
    "side_ratio": "Side ratio",
    "frontcourt_ratio": "Frontcourt ratio",
    "backcourt_ratio": "Backcourt ratio",
    "signed_side_bias": "Right-side bias",
    "front_back_switch_ratio": "Front/back switches",
    "left_right_switch_ratio": "Left/right switches",
    "log_rally_length": "Log rally length",
    "average_movement": "Movement per shot",
    "landing_opponent_distance": "Landing-opponent distance",
}
SHOT_LABELS = {
    "長球": "Clear",
    "推撲球": "Push",
    "殺球": "Smash",
    "接殺防守": "Smash defence",
    "平球": "Drive",
    "網前球": "Net shot",
    "挑球": "Lob",
    "切球": "Drop",
}
SHOT_ORDER = list(SHOT_LABELS.values())
READ_COLUMNS = [
    *KEY_COLUMNS,
    "ball_round",
    "player",
    "type",
    "landing_area",
    "landing_x",
    "landing_y",
    "ball_distance",
    "hit_x",
    "hit_y",
    "moving_x",
    "moving_y",
    "opponent_moving_distance",
]


def sanitize_filename(player_name):
    """Return a deterministic, portable filename stem for a player name."""
    if not isinstance(player_name, str) or not player_name.strip():
        raise ValueError("Player name must be a non-empty string")
    ascii_name = unicodedata.normalize("NFKD", player_name).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_.")
    return stem or "player"


sanitize_player_name = sanitize_filename


def _rally_key(record, context="record"):
    try:
        values = tuple(record[column] for column in KEY_COLUMNS)
        numeric = tuple(int(value) for value in values)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Malformed {context}: expected integer fields {list(KEY_COLUMNS)}"
        ) from exc
    if any(isinstance(value, bool) or float(value) != integer for value, integer in zip(values, numeric)):
        raise ValueError(f"Malformed {context}: rally key values must be integers")
    return numeric


def build_assignment_map(assignments, context="assignments"):
    """Map rally keys to saved cluster IDs, rejecting malformed duplicates."""
    if not isinstance(assignments, list):
        raise ValueError(f"Malformed {context}: expected a list")
    result = {}
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise ValueError(f"Malformed {context}[{index}]: expected an object")
        key = _rally_key(assignment, f"{context}[{index}]")
        try:
            cluster_value = assignment["cluster_id"]
            cluster_id = int(cluster_value)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Malformed {context}[{index}]: cluster_id must be a non-negative integer"
            ) from exc
        if (
            isinstance(cluster_value, bool)
            or float(cluster_value) != cluster_id
            or cluster_id < 0
        ):
            raise ValueError(
                f"Malformed {context}[{index}]: cluster_id must be a non-negative integer"
            )
        if key in result:
            raise ValueError(f"Duplicate rally key in {context}: {key}")
        result[key] = cluster_id
    return result


def map_rows_to_assignments(rows, train_assignments, test_assignments, require_complete=False):
    """Attach saved fold and cluster IDs to matching rows without fitting anything."""
    missing_columns = sorted(set(KEY_COLUMNS) - set(rows.columns))
    if missing_columns:
        raise ValueError(f"Rows are missing rally key columns: {missing_columns}")
    train_map = build_assignment_map(train_assignments, "train assignments")
    test_map = build_assignment_map(test_assignments, "test assignments")
    overlap = set(train_map) & set(test_map)
    if overlap:
        raise ValueError(f"Rally key appears in both train and test assignments: {min(overlap)}")

    keys = []
    for row_number, values in enumerate(rows.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)):
        keys.append(_rally_key(dict(zip(KEY_COLUMNS, values)), f"row {row_number}"))
    observed = set(keys)
    if require_complete:
        absent = (set(train_map) | set(test_map)) - observed
        if absent:
            raise ValueError(f"Saved assignment has no matching data row: {min(absent)}")

    folds = []
    clusters = []
    keep = []
    for key in keys:
        if key in train_map:
            keep.append(True)
            folds.append("train")
            clusters.append(train_map[key])
        elif key in test_map:
            keep.append(True)
            folds.append("test")
            clusters.append(test_map[key])
        else:
            keep.append(False)

    mapped = rows.loc[keep].copy()
    mapped["fold"] = folds
    mapped["cluster_id"] = np.asarray(clusters, dtype=int)
    return mapped


def aggregate_cluster_counts(rows, assignments, category_column):
    """Return cluster-by-category counts and row-normalized proportions."""
    if category_column not in rows.columns:
        raise ValueError(f"Rows are missing category column: {category_column}")
    mapped = map_rows_to_assignments(rows, assignments, [])
    mapped = mapped[mapped[category_column].notna()]
    counts = pd.crosstab(mapped["cluster_id"], mapped[category_column]).sort_index()
    totals = counts.sum(axis=1).replace(0, np.nan)
    proportions = counts.div(totals, axis=0).fillna(0.0)
    return counts, proportions


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path, description):
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {description} at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Malformed {description} at {path}: expected a JSON object")
    return data


def _select_players(requested, split, profiles):
    eligible = split.get("eligible_players")
    if not isinstance(eligible, list) or not all(isinstance(name, str) for name in eligible):
        raise ValueError("Malformed rally split: eligible_players must be a list of names")
    eligible_set = set(eligible)
    if not isinstance(profiles, dict):
        raise ValueError("Malformed rally profiles: players must be an object")

    names = requested or sorted(
        name
        for name in eligible_set
        if profiles.get(name, {}).get("status") == "clustered"
    )
    selected = []
    for name in names:
        if name in selected:
            continue
        if name not in eligible_set:
            matches = difflib.get_close_matches(name, sorted(eligible_set), n=3)
            hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
            raise ValueError(f"Player {name!r} is not eligible in the rally split.{hint}")
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            raise ValueError(f"Player {name!r} has no valid rally-profile artifact")
        if profile.get("status") != "clustered":
            reason = profile.get("reason") or "profile was not clustered"
            raise ValueError(f"Player {name!r} cannot be plotted: {reason}")
        selected.append(name)
    if not selected:
        raise ValueError("No clustered eligible players are available to plot")
    return selected


def _profile_keys(selected, profiles):
    keys = set()
    for name in selected:
        profile = profiles[name]
        train_map = build_assignment_map(profile.get("train_assignments"), f"{name} train assignments")
        test_map = build_assignment_map(profile.get("test_assignments"), f"{name} test assignments")
        if set(train_map) & set(test_map):
            raise ValueError(f"Player {name!r} has overlapping train/test assignments")
        keys.update(train_map)
    return keys


def _read_selected_rallies(csv_path, selected_keys):
    if not csv_path.is_file():
        raise ValueError(f"Merged CSV not found: {csv_path}; pass --merged-csv with its location")
    chunks = []
    try:
        reader = pd.read_csv(csv_path, usecols=READ_COLUMNS, chunksize=200_000)
        for chunk in reader:
            numeric = chunk.loc[:, KEY_COLUMNS].apply(pd.to_numeric, errors="coerce")
            valid = numeric.notna().all(axis=1)
            mask = pd.Series(False, index=chunk.index)
            if valid.any():
                row_keys = pd.MultiIndex.from_frame(numeric.loc[valid].astype(int))
                mask.loc[valid] = row_keys.isin(selected_keys)
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
    except ValueError as exc:
        raise ValueError(f"Merged CSV {csv_path} is missing required rally columns: {exc}") from exc
    if not chunks:
        raise ValueError("None of the saved rally assignments were found in the merged CSV")
    return pd.concat(chunks, ignore_index=True)


def _validate_profile(profile, player_name):
    try:
        selected_k = int(profile["selected_k"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed profile for {player_name!r}: invalid selected_k") from exc
    clusters = profile.get("clusters")
    if selected_k < 2 or not isinstance(clusters, list) or len(clusters) != selected_k:
        raise ValueError(
            f"Malformed profile for {player_name!r}: expected {selected_k} cluster records"
        )
    ids = {cluster.get("cluster_id") for cluster in clusters if isinstance(cluster, dict)}
    if ids != set(range(selected_k)):
        raise ValueError(f"Malformed profile for {player_name!r}: cluster IDs are incomplete")
    return selected_k


def _prepare_player_data(player_name, rally_rows, features, split, profile, settings):
    selected_k = _validate_profile(profile, player_name)
    player_features = features[features["player"] == player_name].copy()
    train_raw = _select_keys(player_features, split.get("train", {}).get(player_name, []))
    minimum_shots = int(settings.get("min_target_nonserve_shots", 1))
    train_raw = train_raw[train_raw["target_nonserve_shot_count"] >= minimum_shots].copy()
    if train_raw.empty:
        raise ValueError(f"No eligible train rallies were reconstructed for {player_name!r}")

    _, _, train_z, _, _ = _fit_feature_transform(
        train_raw,
        train_raw.iloc[0:0],
        float(settings.get("smoothing_alpha", 2.0)),
        settings.get("clip_quantiles", [0.01, 0.99]),
    )
    train_assignments = profile.get("train_assignments")
    train_labeled = map_rows_to_assignments(
        train_raw, train_assignments, [], require_complete=True
    )
    if len(train_labeled) != len(train_raw):
        raise ValueError(
            f"Saved train assignments do not exactly match reconstructed rallies for {player_name!r}"
        )
    train_labeled.loc[:, CLUSTER_FEATURE_COLUMNS] = train_z.to_numpy(dtype=float)

    player_mask = rally_rows["player"].astype("string").str.strip() == player_name
    strokes = map_rows_to_assignments(
        rally_rows.loc[player_mask].copy(),
        train_assignments,
        [],
        require_complete=True,
    )
    shot_type = strokes["type"].astype("string")
    tactical = strokes[
        (strokes["fold"] == "train")
        & shot_type.notna()
        & (shot_type != "接不到")
        & ~shot_type.isin(SERVE_SHOTS)
    ].copy()
    tactical["shot_label"] = tactical["type"].map(SHOT_LABELS).fillna(
        tactical["type"].astype(str)
    )
    tactical["landing_area_numeric"] = _area_series(
        tactical, "landing_area", "landing_x", "landing_y"
    )
    for column in ("landing_x", "landing_y"):
        tactical[column] = pd.to_numeric(tactical[column], errors="coerce")
    landing = tactical[
        tactical["landing_area_numeric"].isin(IN_COURT_AREA_IDS)
        & tactical["landing_x"].between(-1, 1)
        & tactical["landing_y"].between(-1, 1)
    ].copy()

    labels = set(train_labeled["cluster_id"])
    if not labels.issubset(set(range(selected_k))):
        raise ValueError(f"Profile for {player_name!r} assigns an unknown cluster ID")
    return train_labeled, tactical, landing


def _draw_half_court(ax):
    line_color = "#263238"
    faint = "#708090"
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=line_color, lw=1.5)
    for physical_x in (27.4, 50.0, 305.0, 327.6):
        x = physical_x / 177.5 - 1.0
        ax.plot([x, x], [-0.375, 1], color=faint, lw=0.7, alpha=0.8)
    for physical_y in (150.0, 204.0, 366.0):
        y = physical_y / 240.0 - 1.0
        ax.plot([-0.846, 0.846], [y, y], color=faint, lw=0.8)
    ax.plot([0, 0], [-0.375, 0.525], color=faint, lw=0.7)
    ax.plot([-1, 1], [1, 1], color="#8b1e3f", lw=2.2)
    ax.text(0.98, 0.96, "NET", ha="right", va="top", fontsize=7, color="#8b1e3f")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xlabel("Landing x (normalized)")
    ax.set_ylabel("Landing y (baseline to net)")
    ax.grid(False)


def coverage_ellipse(points, coverage=0.8):
    """Return a covariance ellipse covering the requested empirical fraction."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None
    if not 0 < coverage < 1 or not np.isfinite(points).all():
        raise ValueError("coverage must be in (0, 1) and points must be finite")

    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-12)
    eigenvectors = eigenvectors[:, order]
    centered = points - center
    inverse = np.linalg.pinv(covariance)
    squared_distances = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    radius = float(np.sqrt(np.quantile(squared_distances, coverage)))
    width, height = 2 * radius * np.sqrt(eigenvalues)
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    return {
        "center": center,
        "width": float(width),
        "height": float(height),
        "angle": angle,
        "coverage": float(coverage),
    }


def _profile_centers(profile):
    centers = []
    for cluster in sorted(profile["clusters"], key=lambda item: int(item["cluster_id"])):
        center = cluster.get("center_z")
        if not isinstance(center, dict):
            raise ValueError("Cluster profile is missing center_z")
        values = [center.get(feature) for feature in CLUSTER_FEATURE_COLUMNS]
        if any(value is None or not np.isfinite(value) for value in values):
            raise ValueError("Cluster center_z contains a missing or non-finite value")
        centers.append(values)
    return np.asarray(centers, dtype=float)


def _plot_embedding(
    ax,
    train_xy,
    visual_center_xy,
    train_clusters,
    colors,
    title,
    x_label,
    y_label,
    projected_model_center_xy=None,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    for cluster_id, color in enumerate(colors):
        train_mask = train_clusters == cluster_id
        group = train_xy[train_mask]
        ellipse = coverage_ellipse(group)
        if ellipse is not None:
            ax.add_patch(Ellipse(
                xy=ellipse["center"],
                width=ellipse["width"],
                height=ellipse["height"],
                angle=ellipse["angle"],
                facecolor=color,
                edgecolor=color,
                linewidth=1.3,
                alpha=0.11,
                zorder=1,
            ))
        ax.scatter(
            group[:, 0],
            group[:, 1],
            s=13,
            alpha=0.28,
            color=color,
            edgecolors="none",
            zorder=2,
        )
        if projected_model_center_xy is not None:
            ax.plot(
                [visual_center_xy[cluster_id, 0], projected_model_center_xy[cluster_id, 0]],
                [visual_center_xy[cluster_id, 1], projected_model_center_xy[cluster_id, 1]],
                color=color,
                linewidth=0.9,
                linestyle="--",
                alpha=0.75,
                zorder=3,
            )
            ax.scatter(
                projected_model_center_xy[cluster_id, 0],
                projected_model_center_xy[cluster_id, 1],
                s=65,
                color=color,
                marker="D",
                edgecolors="#20252b",
                linewidths=0.8,
                zorder=4,
            )
        ax.scatter(
            visual_center_xy[cluster_id, 0],
            visual_center_xy[cluster_id, 1],
            s=150,
            color=color,
            marker="*",
            edgecolors="#20252b",
            linewidths=0.9,
            zorder=5,
        )
        ax.annotate(
            f"C{cluster_id}",
            visual_center_xy[cluster_id],
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

    cluster_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=f"C{i}")
        for i, color in enumerate(colors)
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#68737d", label="train"),
        Line2D(
            [0], [0], marker="*", markersize=11, linestyle="none",
            color="#68737d", label="visual center",
        ),
    ]
    if projected_model_center_xy is not None:
        marker_handles.append(Line2D(
            [0], [0], marker="D", linestyle="none", color="#68737d",
            label="projected KMeans center",
        ))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(
        handles=cluster_handles + marker_handles,
        ncol=min(4, len(colors) + 3),
        fontsize=7,
        frameon=False,
        loc="best",
    )


def _plot_pca(ax, train, profile, colors):
    train_values = train.loc[:, CLUSTER_FEATURE_COLUMNS].to_numpy(dtype=float)
    model = PCA(n_components=2, svd_solver="full")
    train_xy = model.fit_transform(train_values)
    center_xy = model.transform(_profile_centers(profile))
    explained = model.explained_variance_ratio_ * 100
    _plot_embedding(
        ax,
        train_xy,
        center_xy,
        train["cluster_id"].to_numpy(dtype=int),
        colors,
        "Train-fitted PCA (linear reference)",
        f"PC1 ({explained[0]:.1f}%)",
        f"PC2 ({explained[1]:.1f}%)",
    )


def _plot_umap(ax, train, profile, colors):
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ValueError(
            "UMAP is required; install the project's analysis extra"
        ) from exc

    train_values = train.loc[:, CLUSTER_FEATURE_COLUMNS].to_numpy(dtype=float)
    model = UMAP(
        n_components=2,
        n_neighbors=min(15, len(train_values) - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        transform_seed=42,
        n_jobs=1,
    )
    train_xy = model.fit_transform(train_values)
    projected_center_xy = model.transform(_profile_centers(profile))
    train_clusters = train["cluster_id"].to_numpy(dtype=int)
    visual_center_xy = np.vstack([
        train_xy[train_clusters == cluster_id].mean(axis=0)
        for cluster_id in range(len(colors))
    ])
    _plot_embedding(
        ax,
        train_xy,
        visual_center_xy,
        train_clusters,
        colors,
        "Train-fitted UMAP (visualization only)",
        "UMAP 1",
        "UMAP 2",
        projected_model_center_xy=projected_center_xy,
    )


def _plot_heatmap(ax, train, colors, selected_k, plt):
    medians = np.vstack([
        train.loc[train["cluster_id"] == cluster_id, CLUSTER_FEATURE_COLUMNS]
        .median()
        .to_numpy(dtype=float)
        for cluster_id in range(selected_k)
    ])
    image = ax.imshow(medians, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(selected_k), [f"C{i}" for i in range(selected_k)])
    ax.set_xticks(
        range(len(CLUSTER_FEATURE_COLUMNS)),
        [FEATURE_LABELS[name] for name in CLUSTER_FEATURE_COLUMNS],
        rotation=48,
        ha="right",
        fontsize=8,
    )
    for cluster_id, color in enumerate(colors):
        ax.get_yticklabels()[cluster_id].set_color(color)
        ax.get_yticklabels()[cluster_id].set_fontweight("bold")
    ax.set_title("Robust cluster signatures (train median z)", loc="left", fontweight="bold")
    plt.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Median z (fixed -2 to 2)")


def _plot_landings(ax, landing, colors):
    _draw_half_court(ax)
    for cluster_id, color in enumerate(colors):
        group = landing[landing["cluster_id"] == cluster_id]
        if group.empty:
            continue
        if len(group) > 800:
            group = group.iloc[np.linspace(0, len(group) - 1, 800, dtype=int)]
        ax.scatter(
            group["landing_x"], group["landing_y"], s=9, alpha=0.22,
            color=color, edgecolors="none", label=f"C{cluster_id} (n={len(landing[landing['cluster_id'] == cluster_id])})",
        )
        if len(group) >= 15:
            density, x_edges, y_edges = np.histogram2d(
                group["landing_x"], group["landing_y"], bins=18, range=[[-1, 1], [-1, 1]]
            )
            positive = density[density > 0]
            if len(positive) and positive.max() > 1:
                levels = np.unique(np.quantile(positive, [0.65, 0.85]))
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                ax.contour(x_centers, y_centers, density.T, levels=levels, colors=[color], linewidths=0.8)
    ax.set_title("In-court tactical landing patterns", loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def _plot_shots(ax, tactical, colors, selected_k):
    counts = pd.crosstab(tactical["cluster_id"], tactical["shot_label"])
    counts = counts.reindex(index=range(selected_k), fill_value=0)
    columns = [name for name in SHOT_ORDER if name in counts.columns]
    columns.extend(sorted(set(counts.columns) - set(columns)))
    counts = counts.reindex(columns=columns, fill_value=0)
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    bottom = np.zeros(selected_k)
    shot_colors = plt_colormap("Set3", max(len(columns), 1))
    for index, shot_name in enumerate(columns):
        values = proportions[shot_name].to_numpy(dtype=float)
        ax.bar(range(selected_k), values, bottom=bottom, color=shot_colors[index], label=shot_name)
        bottom += values
    ax.set_xticks(range(selected_k), [f"C{i}" for i in range(selected_k)])
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of non-serve, non-missed shots")
    ax.set_title("Normalized shot-type distribution", loc="left", fontweight="bold")
    if columns:
        ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))


def plt_colormap(name, count):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name)
    return [cmap(index / max(count - 1, 1)) for index in range(count)]


def _plot_lengths(ax, train, colors, selected_k):
    values = [
        train.loc[train["cluster_id"] == cluster_id, "rally_length"].to_numpy(dtype=float)
        for cluster_id in range(selected_k)
    ]
    boxes = ax.boxplot(values, positions=range(selected_k), widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.38)
    ax.set_xticks(range(selected_k), [f"C{i}" for i in range(selected_k)])
    ax.set_ylabel("Physical rally length (strokes)")
    ax.set_title("Train rally-length distributions", loc="left", fontweight="bold")


def _plot_profile_text(ax, profile, colors):
    ax.axis("off")
    silhouette = profile.get("selected_silhouette")
    silhouette_text = f"{float(silhouette):.3f}" if silhouette is not None else "n/a"
    header = (
        f"Selected K = {profile['selected_k']}    Elbow K = {profile.get('elbow_k', 'n/a')}    "
        f"Silhouette = {silhouette_text}\n"
        f"Train n = {profile.get('train_rally_count', 'n/a')}"
    )
    ax.text(0, 1, header, transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold")
    y = 0.84
    cluster_count = len(profile["clusters"])
    slot_height = 0.79 / max(cluster_count, 1)
    font_size = 8.4 if cluster_count <= 4 else 7.6
    wrap_width = 165 if cluster_count <= 4 else 185
    for cluster in profile["clusters"]:
        cluster_id = int(cluster["cluster_id"])
        label = cluster.get("label") or "Unlabeled cluster"
        description = cluster.get("description") or "No saved description."
        instruction = cluster.get("instruction") or "No saved instruction."
        title = (
            f"C{cluster_id} | n={cluster.get('train_size', 'n/a')} | "
            f"matches={cluster.get('train_match_count', 'n/a')} | {label}"
        )
        body = f"{description} Instruction: {instruction}"
        ax.text(
            0, y, title, transform=ax.transAxes, va="top", color=colors[cluster_id],
            fontweight="bold", fontsize=font_size + 0.8,
        )
        wrapped = textwrap.fill(str(body), width=wrap_width)
        ax.text(
            0.02, y - 0.045, wrapped, transform=ax.transAxes, va="top",
            fontsize=font_size, linespacing=1.15,
        )
        y -= slot_height
    ax.set_title("Saved tactical labels and instructions", loc="left", fontweight="bold", pad=8)


def _render_dashboard(player_name, profile, prepared, output_path, dpi):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("matplotlib is required; install the project's analysis extra") from exc

    train, tactical, landing = prepared
    selected_k = int(profile["selected_k"])
    colors = plt_colormap("tab10", selected_k)
    profile_height = 0.35 + 0.1 * selected_k
    figure_height = 22 + 0.625 * selected_k
    fig = plt.figure(figsize=(19, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, profile_height])
    axes = {
        "pca": fig.add_subplot(grid[0, 0]),
        "umap": fig.add_subplot(grid[0, 1]),
        "heatmap": fig.add_subplot(grid[1, 0]),
        "landings": fig.add_subplot(grid[1, 1]),
        "shots": fig.add_subplot(grid[2, 0]),
        "lengths": fig.add_subplot(grid[2, 1]),
        "profile": fig.add_subplot(grid[3, :]),
    }
    fig.suptitle(
        f"{player_name} | Rally Profile Dashboard",
        fontsize=20,
        fontweight="bold",
    )
    try:
        _plot_pca(axes["pca"], train, profile, colors)
        _plot_umap(axes["umap"], train, profile, colors)
        _plot_heatmap(axes["heatmap"], train, colors, selected_k, plt)
        _plot_landings(axes["landings"], landing, colors)
        _plot_shots(axes["shots"], tactical, colors, selected_k)
        _plot_lengths(axes["lengths"], train, colors, selected_k)
        _plot_profile_text(axes["profile"], profile, colors)
        for name, ax in axes.items():
            if ax.axison and name != "landings":
                ax.grid(axis="y", color="#d9dee3", linewidth=0.5, alpha=0.65)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "Scripts" / "config.yaml"))
    parser.add_argument("--merged-csv", help="Override paths.merged_csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--player", action="append", help="Player to plot; repeat for multiple players")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing PNG dashboards from the output directory first",
    )
    return parser.parse_args(argv), parser


def run(args):
    config_path = _project_path(args.config)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML configuration at {config_path}: {exc}") from exc
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration must contain a paths mapping")
    for required in ("rally_split", "rally_profiles"):
        if not paths.get(required):
            raise ValueError(f"Configuration is missing paths.{required}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer")

    split = _load_json(_project_path(paths["rally_split"]), "rally split")
    artifact = _load_json(_project_path(paths["rally_profiles"]), "rally profiles")
    profiles = artifact.get("players")
    selected = _select_players(args.player, split, profiles)
    settings = artifact.get("clustering_settings")
    if not isinstance(settings, dict):
        raise ValueError("Malformed rally profiles: clustering_settings is missing")
    feature_names = artifact.get("feature_names")
    configured_features = settings.get("features", feature_names)
    if feature_names != CLUSTER_FEATURE_COLUMNS or configured_features != CLUSTER_FEATURE_COLUMNS:
        raise ValueError(
            "Rally-profile feature order does not match the implemented clustering features"
        )

    merged_value = args.merged_csv or paths.get("merged_csv")
    if not merged_value:
        raise ValueError("Provide --merged-csv or configure paths.merged_csv")
    keys = _profile_keys(selected, profiles)
    rally_rows = _read_selected_rallies(_project_path(merged_value), keys)
    features = extract_rally_features(rally_rows)
    output_dir = _project_path(args.output_dir)
    if args.clean and output_dir.is_dir():
        for existing in output_dir.glob("*.png"):
            existing.unlink()

    used_stems = set()
    outputs = []
    for player_name in selected:
        stem = sanitize_filename(player_name)
        candidate = stem
        suffix = 2
        while candidate.casefold() in used_stems:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        used_stems.add(candidate.casefold())
        prepared = _prepare_player_data(
            player_name, rally_rows, features, split, profiles[player_name], settings
        )
        output_path = output_dir / f"{candidate}.png"
        _render_dashboard(player_name, profiles[player_name], prepared, output_path, args.dpi)
        outputs.append(output_path)
        print(f"Saved {player_name}: {output_path}")
    return outputs


def main(argv=None):
    args, parser = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
