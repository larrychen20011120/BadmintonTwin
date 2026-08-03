"""Generate a publication-ready rally-profile case-study figure."""

import argparse
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import yaml
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Preprocessing.rally_feature import CLUSTER_FEATURE_COLUMNS, extract_rally_features  # noqa: E402
from Scripts.plot_rally_profiles import (  # noqa: E402
    _load_json,
    _prepare_player_data,
    _profile_centers,
    _profile_keys,
    _project_path,
    _read_selected_rallies,
    _select_players,
    coverage_ellipse,
    sanitize_filename,
)


DEFAULT_PLAYER = "NG Ka Long Angus"
DEFAULT_OUTPUT_DIR = "Experiments/rally_profiles/paper"
FONT_SCALE = 2.0
CLUSTER_COLORS = ["#2369A0", "#D64B3C", "#8A5AA8", "#159A9C", "#D99B2B"]
HEATMAP_LABELS = {
    "attack_shot_ratio": "Attack",
    "defense_shot_ratio": "Defense",
    "side_ratio": "Wide",
    "frontcourt_ratio": "Front",
    "backcourt_ratio": "Back",
    "signed_side_bias": "Side bias",
    "front_back_switch_ratio": "F/B switch",
    "left_right_switch_ratio": "L/R switch",
    "log_rally_length": "Length",
    "average_movement": "Movement",
    "landing_opponent_distance": "Opp. distance",
}


def _profile_center_matrix(profile):
    return np.asarray([
        [float(cluster["center_z"][feature]) for feature in CLUSTER_FEATURE_COLUMNS]
        for cluster in sorted(profile["clusters"], key=lambda item: int(item["cluster_id"]))
    ])


def _draw_compact_court(ax):
    line_color = "#34404A"
    service_color = "#94A0AA"
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=line_color, lw=0.8)
    for physical_x in (27.4, 50.0, 305.0, 327.6):
        x = physical_x / 177.5 - 1.0
        ax.plot([x, x], [-0.375, 1], color=service_color, lw=0.35)
    for physical_y in (150.0, 204.0, 366.0):
        y = physical_y / 240.0 - 1.0
        ax.plot([-0.846, 0.846], [y, y], color=service_color, lw=0.4)
    ax.plot([0, 0], [-0.375, 0.525], color=service_color, lw=0.35)
    ax.plot([-1, 1], [1, 1], color="#7D1732", lw=1.2)
    ax.text(
        0,
        1,
        "NET",
        ha="center",
        va="center",
        fontsize=8 * FONT_SCALE,
        color="#7D1732",
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.17",
            "facecolor": "white",
            "edgecolor": "#7D1732",
            "linewidth": 0.7,
        },
    )
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_pca(ax, train, profile, colors):
    from matplotlib.patches import Ellipse

    values = train.loc[:, CLUSTER_FEATURE_COLUMNS].to_numpy(dtype=float)
    model = PCA(n_components=2, svd_solver="full")
    projected = model.fit_transform(values)
    centers = model.transform(_profile_centers(profile))
    labels = train["cluster_id"].to_numpy(dtype=int)

    for cluster_id, color in enumerate(colors):
        group = projected[labels == cluster_id]
        ellipse = coverage_ellipse(group)
        if ellipse is not None:
            ax.add_patch(Ellipse(
                xy=ellipse["center"],
                width=ellipse["width"],
                height=ellipse["height"],
                angle=ellipse["angle"],
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.12,
                zorder=1,
            ))
        ax.scatter(group[:, 0], group[:, 1], s=7, color=color, alpha=0.27, edgecolors="none")
        ax.scatter(
            centers[cluster_id, 0],
            centers[cluster_id, 1],
            s=82,
            color=color,
            marker="*",
            edgecolors="#25313B",
            linewidths=0.55,
            zorder=4,
        )
        ax.annotate(
            f"C{cluster_id}",
            centers[cluster_id],
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=12 * FONT_SCALE,
            fontweight="bold",
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "(a) Rally clusters",
        loc="left",
        fontsize=14 * FONT_SCALE,
        fontweight="bold",
        pad=8 * FONT_SCALE,
    )
    ax.tick_params(labelsize=12.5 * FONT_SCALE)
    ax.grid(color="#DDE2E6", linewidth=0.4, alpha=0.75)


def _plot_centroid_heatmap(ax, profile, colors, figure):
    centers = _profile_center_matrix(profile)
    image = ax.imshow(centers, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(
        range(len(colors)),
        [f"C{i}" for i in range(len(colors))],
        fontsize=13 * FONT_SCALE,
    )
    ax.set_xticks(
        range(len(CLUSTER_FEATURE_COLUMNS)),
        [HEATMAP_LABELS[feature] for feature in CLUSTER_FEATURE_COLUMNS],
        rotation=48,
        ha="right",
        fontsize=11.5 * FONT_SCALE,
    )
    for cluster_id, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(colors[cluster_id])
        tick.set_fontweight("bold")
    for row in range(centers.shape[0]):
        for column in range(centers.shape[1]):
            value = centers[row, column]
            if abs(value) >= 0.5:
                ax.text(
                    column,
                    row,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=10 * FONT_SCALE,
                    color="white" if abs(value) >= 1.1 else "#20272D",
                    fontweight="bold",
                )
    ax.set_title(
        "(b) Cluster centroid deviations",
        loc="left",
        fontsize=14 * FONT_SCALE,
        fontweight="bold",
        pad=8 * FONT_SCALE,
    )
    colorbar = figure.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
    colorbar.set_label("Player-relative z", fontsize=12 * FONT_SCALE)
    colorbar.ax.tick_params(labelsize=11.5 * FONT_SCALE)


def _plot_landing_density(ax, landing, cluster_id, color):
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    group = landing[landing["cluster_id"] == cluster_id]
    density, x_edges, y_edges = np.histogram2d(
        group["landing_x"],
        group["landing_y"],
        bins=24,
        range=[[-1, 1], [-1, 1]],
    )
    density = gaussian_filter(density.T, sigma=1.0)
    if density.max() > 0:
        density /= density.max()
    colormap = LinearSegmentedColormap.from_list(
        f"cluster_{cluster_id}_density", ["#FFFFFF", color], N=256
    )
    ax.pcolormesh(
        x_edges,
        y_edges,
        density,
        cmap=colormap,
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=True,
    )
    if len(group) > 450:
        group = group.iloc[np.linspace(0, len(group) - 1, 450, dtype=int)]
    ax.scatter(
        group["landing_x"],
        group["landing_y"],
        s=5,
        color=color,
        alpha=0.13,
        edgecolors="none",
        rasterized=True,
    )
    _draw_compact_court(ax)
    ax.set_title(
        f"C{cluster_id}",
        color=color,
        fontsize=13 * FONT_SCALE,
        fontweight="bold",
        pad=12 * FONT_SCALE,
    )


def _instruction_text_layout(cluster, card_width):
    label = str(cluster.get("label") or "Unlabeled tactical mode")
    label_font_size = (12.0 if card_width >= 2.5 else 11.2) * FONT_SCALE
    label_width = max(
        round(16 / FONT_SCALE),
        round(card_width * 72 * 0.88 / (label_font_size * 0.58)),
    )
    label_text = textwrap.fill(
        label,
        width=label_width,
        break_on_hyphens=False,
    )
    instruction = str(cluster.get("instruction") or "No saved instruction.")
    if len(instruction) > 175:
        instruction_font_size = 10.0 * FONT_SCALE
    elif len(instruction) > 145:
        instruction_font_size = 10.3 * FONT_SCALE
    else:
        instruction_font_size = 10.6 * FONT_SCALE
    instruction_width = max(
        round(20 / FONT_SCALE),
        round(card_width * 72 * 0.88 / (instruction_font_size * 0.60)),
    )
    instruction_text = textwrap.fill(instruction, width=instruction_width)
    label_lines = label_text.count("\n") + 1
    instruction_lines = instruction_text.count("\n") + 1
    required_height = (
        0.34
        + label_lines * label_font_size * 1.08 / 72
        + instruction_lines * instruction_font_size * 1.16 / 72
    )
    return (
        label_text,
        label_font_size,
        label_lines,
        instruction_text,
        instruction_font_size,
        required_height,
    )


def _instruction_box_height(clusters, figure_width, selected_k):
    usable_width = figure_width * (0.985 - 0.045)
    card_width = usable_width / (selected_k + 0.12 * (selected_k - 1))
    required_heights = []
    for cluster in clusters:
        required_heights.append(
            _instruction_text_layout(cluster, card_width)[-1]
        )
    return max(max(required_heights), 1.4 * FONT_SCALE)


def _draw_justified_paragraph(
    ax, text, x, y, width, font_size, color, line_spacing, card_height
):
    from matplotlib.font_manager import FontProperties

    renderer = ax.figure.canvas.get_renderer()
    axis_width = ax.get_window_extent(renderer).width
    target_width = width * axis_width
    font_properties = FontProperties(family="DejaVu Sans", size=font_size)
    space_width = renderer.get_text_width_height_descent(
        " ", font_properties, ismath=False
    )[0]
    line_height = font_size * line_spacing / (72 * card_height)
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        line_y = y - line_index * line_height
        words = line.split()
        is_last_line = line_index == len(lines) - 1
        word_widths = [
            renderer.get_text_width_height_descent(
                word, font_properties, ismath=False
            )[0]
            for word in words
        ]
        gap_width = (
            (target_width - sum(word_widths)) / (len(words) - 1)
            if len(words) > 1
            else 0
        )
        should_justify = (
            not is_last_line
            and len(words) >= 3
            and space_width <= gap_width <= space_width * 3.5
        )
        if not should_justify:
            ax.text(
                x,
                line_y,
                line,
                transform=ax.transAxes,
                fontsize=font_size,
                color=color,
                va="top",
                clip_on=True,
            )
            continue
        word_x = x
        for word, word_width in zip(words, word_widths):
            ax.text(
                word_x,
                line_y,
                word,
                transform=ax.transAxes,
                fontsize=font_size,
                color=color,
                va="top",
                clip_on=True,
            )
            word_x += (word_width + gap_width) / axis_width


def _plot_instruction(ax, cluster, color, center_content=False):
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.015, 0.04),
        0.97,
        0.92,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor=color,
        linewidth=0.9,
        alpha=0.06,
    ))
    figure_width = ax.figure.get_figwidth()
    figure_height = ax.figure.get_figheight()
    card_width = ax.get_position().width * figure_width
    card_height = ax.get_position().height * figure_height
    (
        label_text,
        label_font_size,
        label_lines,
        instruction_text,
        instruction_font_size,
        _,
    ) = _instruction_text_layout(cluster, card_width)
    label_line_height = label_font_size * 1.08 / (72 * card_height)
    instruction_y = 0.91 - label_lines * label_line_height - 0.035
    instruction_lines = instruction_text.count("\n") + 1
    available_height = max(0.1, instruction_y - 0.1)
    required_height = instruction_lines * instruction_font_size * 1.16 / (72 * card_height)
    if required_height > available_height:
        instruction_font_size = max(
            9.6 * FONT_SCALE,
            instruction_font_size * available_height / required_height,
        )
        instruction = str(cluster.get("instruction") or "No saved instruction.")
        instruction_width = max(
            round(20 / FONT_SCALE),
            round(card_width * 72 * 0.88 / (instruction_font_size * 0.60)),
        )
        instruction_text = textwrap.fill(instruction, width=instruction_width)
    if center_content:
        instruction_lines = instruction_text.count("\n") + 1
        instruction_height = (
            instruction_lines * instruction_font_size * 1.16 / (72 * card_height)
        )
        content_height = label_lines * label_line_height + 0.035 + instruction_height
        label_y = min(0.91, 0.5 + content_height / 2)
        instruction_y = label_y - label_lines * label_line_height - 0.035
    else:
        label_y = 0.91
    ax.text(
        0.06,
        label_y,
        label_text,
        transform=ax.transAxes,
        fontsize=label_font_size,
        color=color,
        fontweight="bold",
        va="top",
        linespacing=1.08,
        clip_on=True,
    )
    _draw_justified_paragraph(
        ax,
        instruction_text,
        x=0.06,
        y=instruction_y,
        width=0.88,
        font_size=instruction_font_size,
        color="#172026",
        line_spacing=1.16,
        card_height=card_height,
    )


def _render_figure(player_name, profile, prepared, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("matplotlib is required; install the project's analysis extra") from exc

    train, _, landing = prepared
    selected_k = int(profile["selected_k"])
    if selected_k not in {3, 4, 5}:
        raise ValueError(
            f"This paper layout supports K=3, K=4, or K=5; {player_name!r} has K={selected_k}"
        )
    colors = CLUSTER_COLORS[:selected_k]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12 * FONT_SCALE,
        "axes.titlesize": 12 * FONT_SCALE,
        "axes.labelsize": 12 * FONT_SCALE,
        "axes.titlecolor": "#172026",
        "axes.labelcolor": "#34404A",
        "xtick.labelsize": 12 * FONT_SCALE,
        "ytick.labelsize": 12 * FONT_SCALE,
        "xtick.color": "#4E5962",
        "ytick.color": "#4E5962",
        "legend.fontsize": 12 * FONT_SCALE,
        "figure.titlesize": 12 * FONT_SCALE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure_width = {3: 9.5, 4: 11.0, 5: 13.2}[selected_k]
    sorted_clusters = sorted(profile["clusters"], key=lambda item: int(item["cluster_id"]))
    box_height = _instruction_box_height(sorted_clusters, figure_width, selected_k)
    card_height = 2.1 + box_height
    pca_height = {3: 2.1, 4: 2.2, 5: 2.3}[selected_k]
    heatmap_height = 1.9
    pca_label_gap = 0.5 * FONT_SCALE
    heatmap_label_gap = 1.75 * FONT_SCALE
    figure_height = (
        pca_height
        + pca_label_gap
        + heatmap_height
        + heatmap_label_gap
        + card_height
        + 0.3
    ) / 0.91
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[
            pca_height,
            pca_label_gap,
            heatmap_height,
            heatmap_label_gap,
            card_height,
        ],
        left=0.045,
        right=0.985,
        top=0.97,
        bottom=0.06,
        hspace=0.04,
    )
    pca_ax = fig.add_subplot(outer[0])
    heatmap_ax = fig.add_subplot(outer[2])
    _plot_pca(pca_ax, train, profile, colors)
    _plot_centroid_heatmap(heatmap_ax, profile, colors, fig)

    pca_spacer_ax = fig.add_subplot(outer[1])
    pca_spacer_ax.axis("off")
    spacer_ax = fig.add_subplot(outer[3])
    spacer_ax.axis("off")
    spacer_ax.text(
        0,
        0.25,
        "(c) Tactical landing patterns and instructions",
        fontsize=14 * FONT_SCALE,
        fontweight="bold",
        color="#172026",
        va="bottom",
    )
    cards = outer[4].subgridspec(1, selected_k, wspace=0.12)
    for column, cluster in enumerate(sorted_clusters):
        cluster_id = int(cluster["cluster_id"])
        card = cards[0, column].subgridspec(
            2, 1, height_ratios=[1.35, box_height / 1.45], hspace=0.08
        )
        court_ax = fig.add_subplot(card[0, 0])
        instruction_ax = fig.add_subplot(card[1, 0])
        _plot_landing_density(court_ax, landing, cluster_id, colors[cluster_id])
        _plot_instruction(instruction_ax, cluster, colors[cluster_id])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            metadata={
                "Title": f"Rally profile case study: {player_name}",
                "Author": "BadmintonTwin",
                "Subject": "Player-specific rally clusters and tactical instructions",
            },
        )
    finally:
        plt.close(fig)


def _render_side_by_side_figure(player_name, profile, prepared, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("matplotlib is required; install the project's analysis extra") from exc

    train, _, landing = prepared
    selected_k = int(profile["selected_k"])
    if selected_k not in {3, 4, 5}:
        raise ValueError(
            f"This paper layout supports K=3, K=4, or K=5; {player_name!r} has K={selected_k}"
        )
    colors = CLUSTER_COLORS[:selected_k]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12 * FONT_SCALE,
        "axes.titlesize": 12 * FONT_SCALE,
        "axes.labelsize": 12 * FONT_SCALE,
        "axes.titlecolor": "#172026",
        "axes.labelcolor": "#34404A",
        "xtick.labelsize": 12 * FONT_SCALE,
        "ytick.labelsize": 12 * FONT_SCALE,
        "xtick.color": "#4E5962",
        "ytick.color": "#4E5962",
        "legend.fontsize": 12 * FONT_SCALE,
        "figure.titlesize": 12 * FONT_SCALE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    sorted_clusters = sorted(profile["clusters"], key=lambda item: int(item["cluster_id"]))
    figure_width = 28.0
    outer_wspace = 0.13
    card_wspace = 0.08
    left_width_ratio = 1.3
    right_width_ratio = 1.15
    court_width_ratio = 0.85
    instruction_width_ratio = 2.05
    usable_width = figure_width * (0.985 - 0.03)
    outer_ratio_sum = left_width_ratio + right_width_ratio
    outer_unit_width = usable_width / (
        outer_ratio_sum + outer_wspace * outer_ratio_sum / 2
    )
    right_panel_width = outer_unit_width * right_width_ratio
    card_ratio_sum = court_width_ratio + instruction_width_ratio
    card_unit_width = right_panel_width / (
        card_ratio_sum + card_wspace * card_ratio_sum / 2
    )
    instruction_width = card_unit_width * instruction_width_ratio
    instruction_height = max(
        _instruction_text_layout(cluster, instruction_width)[-1]
        for cluster in sorted_clusters
    )
    row_height = max(2.8, instruction_height + 0.5)
    figure_height = max(10.5, (1.0 + selected_k * row_height + 0.6) / 0.92)

    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[left_width_ratio, right_width_ratio],
        left=0.03,
        right=0.985,
        top=0.97,
        bottom=0.05,
        wspace=outer_wspace,
    )

    left = outer[0, 0].subgridspec(
        4,
        1,
        height_ratios=[3.0, 0.8, 2.4, 1.8],
        hspace=0.04,
    )
    pca_ax = fig.add_subplot(left[0, 0])
    heatmap_ax = fig.add_subplot(left[2, 0])
    _plot_pca(pca_ax, train, profile, colors)
    _plot_centroid_heatmap(heatmap_ax, profile, colors, fig)
    for row in (1, 3):
        spacer_ax = fig.add_subplot(left[row, 0])
        spacer_ax.axis("off")

    right = outer[0, 1].subgridspec(
        selected_k + 1,
        1,
        height_ratios=[0.2] + [row_height] * selected_k,
        hspace=0.35,
    )
    heading_ax = fig.add_subplot(right[0, 0])
    heading_ax.axis("off")
    heading_ax.text(
        0,
        1.0,
        "(c) Tactical landing patterns and instructions",
        fontsize=14 * FONT_SCALE,
        fontweight="bold",
        color="#172026",
        va="bottom",
        linespacing=1.05,
    )
    for row, cluster in enumerate(sorted_clusters, start=1):
        cluster_id = int(cluster["cluster_id"])
        card = right[row, 0].subgridspec(
            1,
            2,
            width_ratios=[court_width_ratio, instruction_width_ratio],
            wspace=card_wspace,
        )
        court_ax = fig.add_subplot(card[0, 0])
        instruction_ax = fig.add_subplot(card[0, 1])
        _plot_landing_density(court_ax, landing, cluster_id, colors[cluster_id])
        _plot_instruction(
            instruction_ax,
            cluster,
            colors[cluster_id],
            center_content=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            metadata={
                "Title": f"Side-by-side rally profile case study: {player_name}",
                "Author": "BadmintonTwin",
                "Subject": "Player-specific rally clusters and tactical instructions",
            },
        )
    finally:
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "Scripts" / "config.yaml"))
    parser.add_argument("--merged-csv", help="Override paths.merged_csv")
    parser.add_argument("--player", default=DEFAULT_PLAYER)
    parser.add_argument("--output", help="Output PDF path")
    parser.add_argument(
        "--side-output",
        help="Output path for the additional layout with section (c) on the right",
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
    split = _load_json(_project_path(paths["rally_split"]), "rally split")
    artifact = _load_json(_project_path(paths["rally_profiles"]), "rally profiles")
    profiles = artifact.get("players")
    selected = _select_players([args.player], split, profiles)
    settings = artifact.get("clustering_settings")
    if not isinstance(settings, dict):
        raise ValueError("Malformed rally profiles: clustering_settings is missing")
    if artifact.get("feature_names") != CLUSTER_FEATURE_COLUMNS:
        raise ValueError("Rally-profile feature order does not match the plotting implementation")

    merged_value = args.merged_csv or paths.get("merged_csv")
    if not merged_value:
        raise ValueError("Provide --merged-csv or configure paths.merged_csv")
    rally_rows = _read_selected_rallies(
        _project_path(merged_value), _profile_keys(selected, profiles)
    )
    features = extract_rally_features(rally_rows)
    player_name = selected[0]
    profile = profiles[player_name]
    prepared = _prepare_player_data(
        player_name, rally_rows, features, split, profile, settings
    )
    output_path = (
        _project_path(args.output)
        if args.output
        else PROJECT_ROOT
        / DEFAULT_OUTPUT_DIR
        / f"{sanitize_filename(player_name)}_tactical_profile.pdf"
    )
    side_output_path = (
        _project_path(args.side_output)
        if args.side_output
        else output_path.with_name(f"{output_path.stem}_side_by_side.pdf")
    )
    _render_figure(player_name, profile, prepared, output_path)
    _render_side_by_side_figure(player_name, profile, prepared, side_output_path)
    print(f"Saved paper figure: {output_path}")
    print(f"Saved side-by-side paper figure: {side_output_path}")
    return output_path


def main(argv=None):
    args, parser = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
