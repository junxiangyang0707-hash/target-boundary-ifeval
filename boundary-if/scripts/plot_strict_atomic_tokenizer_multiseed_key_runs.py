from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

DEFAULT_SUMMARY = "runs/strict_atomic_tokenizer_multiseed_key_runs/requested_multiseed_summary.csv"
DEFAULT_AGGREGATE = "runs/strict_atomic_tokenizer_multiseed_key_runs/requested_multiseed_aggregate.csv"
DEFAULT_OUTPUT_DIR = "runs/model_comparisons/strict_atomic_tokenizer_multiseed_key_runs"

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
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
}
MODEL_FAMILIES = {
    "M1 TF-IDF": COLOR_FAMILIES["orange"],
    "M3 mean": COLOR_FAMILIES["blue"],
    "M4 frozen": COLOR_FAMILIES["olive"],
}
CONFIG_ORDER = [
    ("M1 TF-IDF", "full"),
    ("M3 mean", "40k"),
    ("M3 mean", "full"),
    ("M4 frozen", "20k"),
    ("M4 frozen", "40k"),
    ("M4 frozen", "full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strict atomic-tokenizer multiseed key runs.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--aggregate", default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


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
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(title.strip(), width=88, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=130, break_long_words=False)
    fig.subplots_adjust(
        top=0.77,
        left=0.13,
        right=0.99,
        bottom=0.12,
        wspace=0.22,
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
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
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.18,
    )


def add_panel(
    ax: plt.Axes,
    details: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    metric: str,
    title: str,
    xlim: tuple[float, float],
    show_y_labels: bool,
) -> None:
    seed_offsets = {42: -0.16, 43: 0.0, 44: 0.16}
    seed_markers = {42: "o", 43: "s", 44: "^"}
    y_positions = {config: len(CONFIG_ORDER) - idx for idx, config in enumerate(CONFIG_ORDER)}
    for config in CONFIG_ORDER:
        model, curve_label = config
        family = MODEL_FAMILIES[model]
        y = y_positions[config]
        row = aggregate[(aggregate["model"] == model) & (aggregate["curve_label"] == curve_label)].iloc[0]
        mean = float(row[f"{metric}_mean"])
        std = float(row[f"{metric}_std"]) if pd.notna(row[f"{metric}_std"]) else 0.0
        ax.errorbar(
            mean,
            y,
            xerr=std,
            fmt="o",
            markersize=7,
            markerfacecolor=family["base"],
            markeredgecolor=family["dark"],
            markeredgewidth=1.0,
            color=family["dark"],
            ecolor=family["dark"],
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
        part = details[(details["model"] == model) & (details["curve_label"] == curve_label)]
        for _, detail_row in part.iterrows():
            seed = int(detail_row["train_seed"])
            ax.scatter(
                float(detail_row[metric]),
                y + seed_offsets.get(seed, 0.0),
                s=34,
                marker=seed_markers.get(seed, "o"),
                facecolors=TOKENS["panel"],
                edgecolors=family["dark"],
                linewidths=1.0,
                zorder=5,
            )
    if metric == "test_auprc":
        baseline = float(details["test_positive_rate"].dropna().iloc[0])
        ax.axvline(baseline, color=TOKENS["ink"], linestyle=":", linewidth=1.0, zorder=1)
        ax.text(
            baseline + 0.005,
            0.98,
            f"baseline {baseline:.1%}",
            ha="left",
            va="top",
            fontsize=8,
            color=TOKENS["muted"],
            transform=ax.get_xaxis_transform(),
        )
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=TOKENS["ink"], pad=10)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", colors=TOKENS["muted"], labelsize=8)
    ax.tick_params(axis="y", colors=TOKENS["muted"], labelsize=9, length=0)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.set_xlabel(title, color=TOKENS["ink"])
    ax.set_ylabel("")
    ax.set_yticks(
        [y_positions[config] for config in CONFIG_ORDER],
        [f"{model} {curve_label}" for model, curve_label in CONFIG_ORDER],
    )
    if not show_y_labels:
        ax.set_yticklabels([])


def main() -> None:
    args = parse_args()
    summary_path = resolve_path(args.summary)
    aggregate_path = resolve_path(args.aggregate)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details = pd.read_csv(summary_path)
    aggregate = pd.read_csv(aggregate_path)
    details.to_csv(output_dir / "strict_atomic_tokenizer_multiseed_plot_data.csv", index=False)
    aggregate.to_csv(output_dir / "strict_atomic_tokenizer_multiseed_aggregate.csv", index=False)

    use_chart_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.4), sharey=False)
    add_panel(
        axes[0],
        details,
        aggregate,
        metric="test_auroc",
        title="Test AUROC",
        xlim=(0.79, 0.865),
        show_y_labels=True,
    )
    add_panel(
        axes[1],
        details,
        aggregate,
        metric="test_auprc",
        title="Test AUPRC",
        xlim=(0.12, 0.53),
        show_y_labels=False,
    )
    legend_handles = []
    for label, family in MODEL_FAMILIES.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=family["dark"],
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                linewidth=0,
                markersize=7,
                label=label,
            )
        )
    for seed, marker in {42: "o", 43: "s", 44: "^"}.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color=TOKENS["muted"],
                markerfacecolor=TOKENS["panel"],
                markeredgecolor=TOKENS["muted"],
                linewidth=0,
                markersize=6,
                label=f"seed {seed}",
            )
        )
    axes[0].legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.05),
        frameon=False,
        ncol=6,
        borderaxespad=0,
        fontsize=8,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    add_chart_header(
        fig,
        axes[0],
        "Strict atomic-tokenizer multiseed test results",
        "Held-out atomic split, max2048 raw-prompt BPE tokenizer trained only on atomic train prompts. Dots are seed 42/43/44; intervals show mean +/- one standard deviation.",
    )
    sns.despine(fig=fig)
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"strict_atomic_tokenizer_multiseed_test_metrics.{suffix}",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
