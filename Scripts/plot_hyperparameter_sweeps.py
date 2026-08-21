"""Plot lambda and k sensitivity analyses with DTW and LCS y-axes.

The default values reproduce the Twin hyperparameter sweeps. Values can also
be supplied from the command line, for example:

    uv run --extra analysis python Scripts/plot_hyperparameter_sweeps.py \
        --k-values 3 5 7 10 13 \
        --k-land 0.71 0.65 0.63 0.68 0.70
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Experiments/hyperparameter_plots"
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlecolor": "#172026",
    "axes.labelcolor": "#34404A",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
COLORS = {
    "land": "#2369A0",
    "move": "#159A9C",
    "shot": "#D64B3C",
    "selected": "#6B6B6B",
}
MARKERS = {"land": "o", "move": "s", "shot": "^"}

DEFAULT_LAMBDA = {
    "x": [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1],
    "land": [0.7155, 0.6686, 0.6253, 0.6493, 0.6532, 0.6779, 0.6607],
    "move": [0.3330, 0.3252, 0.2888, 0.3024, 0.3012, 0.3247, 0.3378],
    "shot": [0.2449, 0.3251, 0.3656, 0.3421, 0.3293, 0.3277, 0.3373],
}

DEFAULT_K = {
    "x": [3, 5, 7, 10, 13],
    "land": [0.7155, 0.6458, 0.6253, 0.6772, 0.7032],
    "move": [0.3330, 0.3162, 0.2888, 0.3246, 0.3345],
    "shot": [0.2449, 0.3350, 0.3656, 0.3471, 0.2508],
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=DEFAULT_LAMBDA["x"])
    parser.add_argument("--lambda-land", nargs="+", type=float, default=DEFAULT_LAMBDA["land"])
    parser.add_argument("--lambda-move", nargs="+", type=float, default=DEFAULT_LAMBDA["move"])
    parser.add_argument("--lambda-shot", nargs="+", type=float, default=DEFAULT_LAMBDA["shot"])
    parser.add_argument("--k-values", nargs="+", type=float, default=DEFAULT_K["x"])
    parser.add_argument("--k-land", nargs="+", type=float, default=DEFAULT_K["land"])
    parser.add_argument("--k-move", nargs="+", type=float, default=DEFAULT_K["move"])
    parser.add_argument("--k-shot", nargs="+", type=float, default=DEFAULT_K["shot"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="One or more output formats (default: png pdf)",
    )
    return parser.parse_args()


def _validate_sweep(name, data):
    expected = len(data["x"])
    if expected < 2:
        raise ValueError(f"{name} requires at least two x values")
    for key, values in data.items():
        if len(values) != expected:
            raise ValueError(
                f"{name} {key!r} has {len(values)} values; expected {expected}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} {key!r} contains a non-finite value")
    if any(right <= left for left, right in zip(data["x"], data["x"][1:])):
        raise ValueError(f"{name} x values must be strictly increasing")


def _format_ticks(values):
    return [f"{value:g}" for value in values]


def _axis_limits(values, minimum_padding):
    value_range = max(values) - min(values)
    padding = max(value_range * 0.1, minimum_padding)
    return min(values) - padding, max(values) + padding


def _joint_optimum(data):
    """Return the x value that minimizes both DTWs and maximizes LCS."""
    land_best = min(range(len(data["x"])), key=data["land"].__getitem__)
    move_best = min(range(len(data["x"])), key=data["move"].__getitem__)
    shot_best = max(range(len(data["x"])), key=data["shot"].__getitem__)
    if land_best == move_best == shot_best:
        return land_best
    return None


def _plot_panel(axis, data, x_label, panel_title, dtw_limits, lcs_limits):
    lcs_axis = axis.twinx()
    lines = []

    for key, label in (("land", "Land DTW"), ("move", "Move DTW")):
        line, = axis.plot(
            data["x"],
            data[key],
            label=label,
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=2.8,
            markersize=8.5,
            markerfacecolor="white",
            markeredgewidth=2,
            solid_capstyle="round",
            zorder=3,
        )
        lines.append(line)

    shot_line, = lcs_axis.plot(
        data["x"],
        data["shot"],
        label="Shot LCS Ratio",
        color=COLORS["shot"],
        marker=MARKERS["shot"],
        linewidth=2.8,
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2,
        linestyle="--",
        solid_capstyle="round",
        zorder=3,
    )
    lines.append(shot_line)

    optimum = _joint_optimum(data)
    if optimum is not None:
        selected_x = data["x"][optimum]
        axis.axvline(
            selected_x,
            color=COLORS["selected"],
            linewidth=1.5,
            linestyle=":",
            alpha=0.8,
            zorder=1,
        )
        axis.annotate(
            "Selected",
            xy=(selected_x, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=COLORS["selected"],
        )

    axis.set_ylim(*dtw_limits)
    lcs_axis.set_ylim(*lcs_limits)
    axis.set_xticks(data["x"], _format_ticks(data["x"]))
    axis.set_xlabel(x_label, fontsize=18, labelpad=8)
    axis.set_ylabel(r"DTW Distance $\downarrow$", fontsize=16, labelpad=9)
    lcs_axis.set_ylabel(
        r"Shot LCS Ratio $\uparrow$",
        fontsize=16,
        color=COLORS["shot"],
        labelpad=9,
    )
    axis.set_title(panel_title, loc="left", fontsize=18, pad=12)

    axis.yaxis.set_major_locator(MaxNLocator(6))
    lcs_axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    lcs_axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(axis="y", color="#C7C7C7", linewidth=0.65, linestyle="--", alpha=0.7)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", labelsize=16, direction="out", length=5, width=1.2)
    lcs_axis.tick_params(
        axis="y",
        labelsize=16,
        colors=COLORS["shot"],
        direction="out",
        length=5,
        width=1.2,
    )
    for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    for tick_label in lcs_axis.get_yticklabels():
        tick_label.set_fontweight("bold")
    axis.spines["top"].set_visible(False)
    lcs_axis.spines["top"].set_visible(False)
    lcs_axis.spines["right"].set_color(COLORS["shot"])
    axis.margins(x=0.04)
    return lines


def _plot_sensitivity(lambda_data, k_data, output_dir, formats):
    dtw_values = (
        lambda_data["land"]
        + lambda_data["move"]
        + k_data["land"]
        + k_data["move"]
    )
    lcs_values = lambda_data["shot"] + k_data["shot"]
    dtw_limits = _axis_limits(dtw_values, minimum_padding=0.02)
    lcs_limits = _axis_limits(lcs_values, minimum_padding=0.01)

    figure, axes = plt.subplots(2, 1, figsize=(8.4, 9.6), constrained_layout=True)
    figure.patch.set_facecolor("white")
    lines = _plot_panel(
        axes[0],
        lambda_data,
        x_label=r"$\lambda$",
        panel_title=r"(a) Sensitivity to $\lambda$",
        dtw_limits=dtw_limits,
        lcs_limits=lcs_limits,
    )
    _plot_panel(
        axes[1],
        k_data,
        x_label=r"$k$",
        panel_title=r"(b) Sensitivity to $k$",
        dtw_limits=dtw_limits,
        lcs_limits=lcs_limits,
    )
    figure.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1,
        facecolor="white",
        edgecolor="#D9DEE2",
        fontsize=15,
        borderpad=0.55,
        columnspacing=1.8,
        handlelength=2.2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for file_format in dict.fromkeys(formats):
        output_path = output_dir / f"hyperparameter_sensitivity.{file_format}"
        figure.savefig(
            output_path,
            dpi=400 if file_format == "png" else None,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        saved_paths.append(output_path)
    plt.close(figure)
    return saved_paths


def main():
    args = _parse_args()
    plt.rcParams.update(PLOT_STYLE)
    lambda_data = {
        "x": args.lambda_values,
        "land": args.lambda_land,
        "move": args.lambda_move,
        "shot": args.lambda_shot,
    }
    k_data = {
        "x": args.k_values,
        "land": args.k_land,
        "move": args.k_move,
        "shot": args.k_shot,
    }
    _validate_sweep("lambda", lambda_data)
    _validate_sweep("k", k_data)

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    for path in _plot_sensitivity(lambda_data, k_data, output_dir, args.formats):
        print(path)


if __name__ == "__main__":
    main()
