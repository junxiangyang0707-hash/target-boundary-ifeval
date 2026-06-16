from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


DEFAULT_INPUT = (
    "runs/model_comparisons/all_experiment_test_metrics/"
    "all_experiment_test_metrics_detail.csv"
)
DEFAULT_OUTPUT_DIR = "runs/model_comparisons/all_experiment_test_metrics_line"

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
    "neutral": {"base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}

FAMILY_STYLES = {
    "M1 learning curve": {
        "family": COLOR_FAMILIES["orange"],
        "linestyle": "--",
        "marker": "o",
    },
    "M1 split comparison": {
        "family": COLOR_FAMILIES["gold"],
        "linestyle": ":",
        "marker": "s",
    },
    "M3 mean learning curve": {
        "family": COLOR_FAMILIES["blue"],
        "linestyle": "-",
        "marker": "o",
    },
    "M3 pooling": {
        "family": COLOR_FAMILIES["olive"],
        "linestyle": "-.",
        "marker": "o",
    },
    "M3 capacity": {
        "family": COLOR_FAMILIES["pink"],
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "o",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw the all-experiment test metrics comparison as line charts."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title.strip(), width=118, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=154, break_long_words=False)
    left = ax.get_position().x0
    fig.text(
        left,
        0.988,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        0.955,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.18,
    )


def short_label(label: str) -> str:
    replacements = {
        "M3 capacity ": "M3 cap ",
        "M1 TF-IDF ": "M1 ",
        "M1 composition ": "M1 comp ",
        "M1 group split": "M1 group",
    }
    short = label
    for old, new in replacements.items():
        short = short.replace(old, new)
    return textwrap.fill(short, width=11, break_long_words=False)


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "experiment_family",
        "experiment_label",
        "split_family",
        "test_auroc_mean",
        "test_auroc_std",
        "test_auprc_mean",
        "test_auprc_std",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    frame = frame.sort_values(
        ["test_auprc_mean", "test_auroc_mean"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["rank_by_test_auprc"] = frame.index + 1
    frame["x_label"] = frame["experiment_label"].map(short_label)
    return frame


def draw_line_panel(
    ax,
    frame: pd.DataFrame,
    *,
    value_col: str,
    std_col: str,
    title: str,
) -> None:
    for family_name, group in frame.groupby("experiment_family", sort=False):
        style = FAMILY_STYLES.get(
            family_name,
            {
                "family": COLOR_FAMILIES["neutral"],
                "linestyle": "-",
                "marker": "o",
            },
        )
        family = style["family"]
        part = group.sort_values("rank_by_test_auprc")
        ax.plot(
            part["rank_by_test_auprc"],
            part[value_col],
            color=family["mid"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            marker=style["marker"],
            markersize=5.2,
            markerfacecolor=family["base"],
            markeredgecolor=family["dark"],
            markeredgewidth=1.0,
            label=family_name,
            zorder=3,
        )

        with_std = part[part[std_col].fillna(0).gt(0)]
        if not with_std.empty:
            ax.errorbar(
                with_std["rank_by_test_auprc"],
                with_std[value_col],
                yerr=with_std[std_col],
                fmt="none",
                ecolor=family["dark"],
                elinewidth=1.0,
                capsize=3,
                alpha=0.85,
                zorder=2,
            )

    ax.set_title(title, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_xlabel("Experiment rank by test AUPRC")
    ax.set_ylabel("Metric proportion")
    ax.set_xlim(0.5, len(frame) + 0.5)
    ax.set_ylim(0.2, 0.95)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(frame["rank_by_test_auprc"], frame["x_label"], rotation=68, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", colors=TOKENS["muted"], labelsize=7.2, length=0)
    ax.tick_params(axis="y", colors=TOKENS["muted"], labelsize=8.5, length=0)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_line_chart(frame: pd.DataFrame, output_dir: Path, input_path: Path) -> dict[str, str]:
    use_chart_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_width = max(17.5, 0.52 * len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.34, wspace=0.12)

    draw_line_panel(
        axes[0],
        frame,
        value_col="test_auroc_mean",
        std_col="test_auroc_std",
        title="Test AUROC",
    )
    draw_line_panel(
        axes[1],
        frame,
        value_col="test_auprc_mean",
        std_col="test_auprc_std",
        title="Test AUPRC",
    )
    axes[1].set_ylabel("")

    legend_handles = []
    legend_labels = []
    for family_name, style in FAMILY_STYLES.items():
        family = style["family"]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=family["mid"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                markeredgewidth=1.0,
                linewidth=1.35,
                markersize=6.5,
            )
        )
        legend_labels.append(family_name)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.87),
        frameon=False,
        ncol=3,
        fontsize=8.8,
        handlelength=2.4,
    )

    add_chart_header(
        fig,
        axes[0],
        "All tracked experiment results as line charts",
        (
            "The x-axis keeps the previous chart's test-AUPRC ranking. Lines connect experiments within the same family; "
            "error bars show standard deviation across seeds when available. Values are proportions on the same 20%-95% range as the point chart."
        ),
    )

    png_path = output_dir / "all_experiment_test_metrics_line.png"
    svg_path = output_dir / "all_experiment_test_metrics_line.svg"
    data_csv = output_dir / "all_experiment_test_metrics_line_data.csv"
    data_json = output_dir / "all_experiment_test_metrics_line_data.json"
    manifest_path = output_dir / "all_experiment_test_metrics_line_manifest.json"

    frame.to_csv(data_csv, index=False, encoding="utf-8")
    write_json(frame.to_dict("records"), data_json)
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)

    manifest = {
        "source": str(input_path),
        "row_count": int(len(frame)),
        "sort": "test_auprc_mean descending, then test_auroc_mean descending",
        "outputs": {
            "png": str(png_path),
            "svg": str(svg_path),
            "csv": str(data_csv),
            "json": str(data_json),
        },
    }
    write_json(manifest, manifest_path)
    return {**manifest["outputs"], "manifest": str(manifest_path)}


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    frame = load_metrics(input_path)
    outputs = plot_line_chart(frame, output_dir, input_path)
    print(json.dumps({"row_count": int(len(frame)), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
