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

DEFAULT_M1_SUMMARY = (
    "runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/"
    "learning_curve_summary.csv"
)
DEFAULT_M3_SUMMARY = (
    "runs/m3_learning_curve_atomic_constraint_heldout_seed42_mean_pooling_max2048/"
    "learning_curve_summary.csv"
)
DEFAULT_M3_55K_SEEDS = (
    "runs/m3_seed_sensitivity_55k_fixedsample42_max2048/"
    "seed_sensitivity_55k_sample42_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    "runs/model_comparisons/"
    "m1_vs_m3_learning_curve_atomic_constraint_heldout_seed42_max2048"
)

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
    "blue": {
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "orange": {
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot M1 vs M3 learning curves on the same atomic held-out data."
    )
    parser.add_argument("--m1-summary", default=DEFAULT_M1_SUMMARY)
    parser.add_argument("--m3-summary", default=DEFAULT_M3_SUMMARY)
    parser.add_argument("--m3-55k-seeds", default=DEFAULT_M3_55K_SEEDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    title = textwrap.fill(title.strip(), width=92, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=132, break_long_words=False)
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
        0.945,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.18,
    )


def normalize_curve_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "source_label" not in output.columns:
        output["source_label"] = output["curve_label"].astype(str)
    output["curve_label"] = output["curve_label"].astype(str)
    output["curve_label"] = output["curve_label"].replace({"80k": "80k/full", "full": "80k/full"})
    output["source_label"] = output["source_label"].astype(str)
    return output


def apply_m3_55k_seed_stats(m3: pd.DataFrame, seed_frame: pd.DataFrame) -> pd.DataFrame:
    output = m3.copy()
    stats: dict[str, tuple[float, float]] = {}
    for metric in ["val_auroc", "val_auprc", "test_auroc", "test_auprc"]:
        stats[metric] = (
            float(seed_frame[metric].mean()),
            float(seed_frame[metric].std(ddof=1)),
        )
        output[f"{metric}_std"] = 0.0

    mask = output["curve_label"].eq("55k")
    if not mask.any():
        raise ValueError("M3 summary does not contain a 55k row.")
    for metric, (mean_value, std_value) in stats.items():
        output.loc[mask, metric] = mean_value
        output.loc[mask, f"{metric}_std"] = std_value
    output.loc[mask, "seed_stat"] = "mean+-std over train seeds 42-45; sample_seed=42"
    output.loc[~mask, "seed_stat"] = "single run"
    return output


def prepare_combined(m1: pd.DataFrame, m3: pd.DataFrame) -> pd.DataFrame:
    order_labels = ["2k", "4k", "5k", "10k", "20k", "40k", "50k", "55k", "80k/full"]
    order_map = {label: index for index, label in enumerate(order_labels)}
    for frame, model_name in [(m1, "M1 TF-IDF logreg"), (m3, "M3 mean Transformer")]:
        frame["model"] = model_name
        for metric in ["val_auroc", "val_auprc", "test_auroc", "test_auprc"]:
            std_col = f"{metric}_std"
            if std_col not in frame.columns:
                frame[std_col] = 0.0
    combined = pd.concat([m1, m3], ignore_index=True)
    combined = combined[combined["curve_label"].isin(order_labels)].copy()
    combined["order"] = combined["curve_label"].map(order_map).astype(int)
    combined = combined.sort_values(["model", "order"], kind="mergesort")
    return combined


def draw_panel(
    ax,
    combined: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    order_labels: list[str],
    baseline: float | None = None,
) -> None:
    x_lookup = {label: index for index, label in enumerate(order_labels)}
    style = {
        "M1 TF-IDF logreg": {
            "family": COLOR_FAMILIES["orange"],
            "linestyle": "-",
            "marker": "o",
        },
        "M3 mean Transformer": {
            "family": COLOR_FAMILIES["blue"],
            "linestyle": "-",
            "marker": "s",
        },
    }
    for model_name, part in combined.groupby("model", sort=False):
        part = part.sort_values("order")
        family = style[model_name]["family"]
        x_values = [x_lookup[label] for label in part["curve_label"]]
        y_values = part[metric].astype(float).to_numpy()
        ax.plot(
            x_values,
            y_values,
            label=model_name,
            color=family["mid"],
            marker=style[model_name]["marker"],
            markerfacecolor=family["base"],
            markeredgecolor=family["dark"],
            linewidth=1.4,
            markersize=6,
            linestyle=style[model_name]["linestyle"],
        )
        std_col = f"{metric}_std"
        yerr = part[std_col].fillna(0).astype(float).to_numpy()
        has_error = yerr > 0
        if has_error.any():
            ax.errorbar(
                [x for x, keep in zip(x_values, has_error) if keep],
                y_values[has_error],
                yerr=yerr[has_error],
                fmt="none",
                ecolor=family["dark"],
                elinewidth=1.2,
                capsize=4,
                zorder=5,
            )
    if baseline is not None:
        ax.axhline(
            baseline,
            color=TOKENS["muted"],
            linestyle=":",
            linewidth=1.0,
            label="Positive-rate baseline",
        )
    ax.set_title(title, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Train sample size")
    ax.set_xticks(range(len(order_labels)), order_labels, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def main() -> None:
    args = parse_args()
    m1_path = resolve_path(args.m1_summary)
    m3_path = resolve_path(args.m3_summary)
    m3_55k_path = resolve_path(args.m3_55k_seeds)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m1 = normalize_curve_labels(pd.read_csv(m1_path))
    m3 = normalize_curve_labels(pd.read_csv(m3_path))
    seed_frame = pd.read_csv(m3_55k_path)
    m3 = apply_m3_55k_seed_stats(m3, seed_frame)
    combined = prepare_combined(m1, m3)

    combined_csv = output_dir / "combined_learning_curve.csv"
    combined_json = output_dir / "combined_learning_curve.json"
    combined.to_csv(combined_csv, index=False, encoding="utf-8")
    write_json(combined.to_dict("records"), combined_json)

    order_labels = ["2k", "4k", "5k", "10k", "20k", "40k", "50k", "55k", "80k/full"]
    val_baseline = float(combined["val_positive_rate"].dropna().iloc[0])
    test_baseline = float(combined["test_positive_rate"].dropna().iloc[0])

    use_chart_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    draw_panel(
        axes[0, 0],
        combined,
        metric="val_auroc",
        title="Validation AUROC",
        ylabel="AUROC",
        order_labels=order_labels,
    )
    draw_panel(
        axes[0, 1],
        combined,
        metric="val_auprc",
        title="Validation AUPRC",
        ylabel="AUPRC",
        order_labels=order_labels,
        baseline=val_baseline,
    )
    draw_panel(
        axes[1, 0],
        combined,
        metric="test_auroc",
        title="Test AUROC",
        ylabel="AUROC",
        order_labels=order_labels,
    )
    draw_panel(
        axes[1, 1],
        combined,
        metric="test_auprc",
        title="Test AUPRC",
        ylabel="AUPRC",
        order_labels=order_labels,
        baseline=test_baseline,
    )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        bbox_to_anchor=(0.08, 0.905),
        frameon=False,
        ncol=3,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.985, bottom=0.095, hspace=0.32, wspace=0.18)
    add_chart_header(
        fig,
        axes[0, 0],
        "M1 and M3 learning curves on the same atomic constraint held-out split",
        (
            "Train sampling is hash-stable and nested; validation/test are full. "
            "M3 55k is replaced by mean+-std over training seeds 42-45 with sample_seed=42."
        ),
    )
    png_path = output_dir / "m1_m3_learning_curve_comparison.png"
    svg_path = output_dir / "m1_m3_learning_curve_comparison.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)

    summary = {
        "m1_summary": str(m1_path),
        "m3_summary": str(m3_path),
        "m3_55k_seeds": str(m3_55k_path),
        "combined_csv": str(combined_csv),
        "combined_json": str(combined_json),
        "png": str(png_path),
        "svg": str(svg_path),
        "m3_55k_stats": {
            metric: {
                "mean": float(seed_frame[metric].mean()),
                "std": float(seed_frame[metric].std(ddof=1)),
            }
            for metric in ["val_auroc", "val_auprc", "test_auroc", "test_auprc"]
        },
    }
    write_json(summary, output_dir / "comparison_manifest.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
