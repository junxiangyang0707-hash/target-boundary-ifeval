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


DEFAULT_COMBINED = (
    "runs/model_comparisons/"
    "m1_vs_m3_learning_curve_atomic_constraint_heldout_seed42_max2048/"
    "combined_learning_curve.csv"
)
DEFAULT_OUTPUT_DIR = "runs/model_comparisons/learning_curve_metric_proportions"

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
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
}

MODEL_STYLES = {
    "M1 TF-IDF logreg": {
        "label": "M1 TF-IDF",
        "color": COLOR_FAMILIES["orange"]["mid"],
        "marker_face": COLOR_FAMILIES["orange"]["base"],
        "marker_edge": COLOR_FAMILIES["orange"]["dark"],
        "linestyle": "--",
    },
    "M3 mean Transformer": {
        "label": "M3 mean Transformer",
        "color": COLOR_FAMILIES["blue"]["mid"],
        "marker_face": COLOR_FAMILIES["blue"]["base"],
        "marker_edge": COLOR_FAMILIES["blue"]["dark"],
        "linestyle": "-",
    },
}

STAGE_ORDER = ["2k", "4k", "5k", "10k", "20k", "40k", "50k", "55k", "full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning-curve test AUROC/AUPRC as proportions."
    )
    parser.add_argument("--combined-learning-curve", default=DEFAULT_COMBINED)
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


def normalize_stage(value: Any) -> str:
    text = str(value)
    if text == "80k/full":
        return "full"
    return text


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


def build_plot_data(combined_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(combined_path)
    frame["learning_curve_stage"] = frame["curve_label"].map(normalize_stage)
    frame = frame[frame["learning_curve_stage"].isin(STAGE_ORDER)].copy()
    frame["stage_order"] = frame["learning_curve_stage"].map({stage: i for i, stage in enumerate(STAGE_ORDER)})
    frame["model_display"] = frame["model"].map(
        {model: style["label"] for model, style in MODEL_STYLES.items()}
    )
    frame = frame[frame["model_display"].notna()].copy()

    long_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for metric_name, mean_col, std_col in [
            ("Test AUROC", "test_auroc", "test_auroc_std"),
            ("Test AUPRC", "test_auprc", "test_auprc_std"),
        ]:
            long_rows.append(
                {
                    "model": row["model"],
                    "model_display": row["model_display"],
                    "learning_curve_stage": row["learning_curve_stage"],
                    "stage_order": int(row["stage_order"]),
                    "requested_train_rows": row["requested_train_rows"],
                    "actual_train_rows": row["actual_train_rows"],
                    "metric": metric_name,
                    "proportion": float(row[mean_col]),
                    "proportion_std": float(row.get(std_col, 0.0) or 0.0),
                    "seed_stat": "" if pd.isna(row.get("seed_stat", "")) else str(row.get("seed_stat", "")),
                    "source_output_dir": row.get("output_dir", ""),
                }
            )
    return pd.DataFrame(long_rows).sort_values(["metric", "model_display", "stage_order"])


def plot_learning_curve(data: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    use_chart_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.81, wspace=0.15)

    for ax, metric in zip(axes, ["Test AUROC", "Test AUPRC"]):
        metric_data = data[data["metric"].eq(metric)].copy()
        for model_name, style in MODEL_STYLES.items():
            model_data = metric_data[metric_data["model"].eq(model_name)].sort_values("stage_order")
            if model_data.empty:
                continue

            ax.plot(
                model_data["stage_order"],
                model_data["proportion"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                marker="o",
                markersize=5.5,
                markerfacecolor=style["marker_face"],
                markeredgecolor=style["marker_edge"],
                markeredgewidth=1.0,
                label=style["label"],
                zorder=3,
            )

            with_std = model_data[model_data["proportion_std"].fillna(0.0).gt(0)]
            if not with_std.empty:
                ax.errorbar(
                    with_std["stage_order"],
                    with_std["proportion"],
                    yerr=with_std["proportion_std"],
                    fmt="none",
                    ecolor=style["marker_edge"],
                    elinewidth=1.0,
                    capsize=3,
                    alpha=0.8,
                    zorder=2,
                )

        ax.set_title(metric, fontsize=11, fontweight="semibold", color=TOKENS["ink"], pad=10)
        ax.set_xlabel("Learning curve stage", fontsize=9.5, color=TOKENS["ink"])
        ax.set_ylabel("Proportion", fontsize=9.5, color=TOKENS["ink"])
        ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER, rotation=0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5, length=0)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.88),
        frameon=False,
        ncol=2,
        fontsize=9,
        handlelength=2.2,
    )
    add_chart_header(
        fig,
        axes[0],
        "Learning curve test metrics as proportions",
        "Atomic constraint held-out max2048 split. Y-axis uses proportion scale; error bars appear only for stages with multi-seed standard deviations.",
    )

    png_path = output_dir / "learning_curve_metric_proportions.png"
    svg_path = output_dir / "learning_curve_metric_proportions.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def main() -> None:
    args = parse_args()
    combined_path = resolve_path(args.combined_learning_curve)
    output_dir = resolve_path(args.output_dir)

    data = build_plot_data(combined_path)
    if data.empty:
        raise ValueError(f"No learning-curve rows found in {combined_path}")

    csv_path = output_dir / "learning_curve_metric_proportions_data.csv"
    json_path = output_dir / "learning_curve_metric_proportions_data.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    write_json(data.to_dict(orient="records"), json_path)
    plot_paths = plot_learning_curve(data, output_dir)

    manifest = {
        "source": str(combined_path),
        "row_count": int(len(data)),
        "models": sorted(data["model_display"].unique().tolist()),
        "metrics": ["Test AUROC", "Test AUPRC"],
        "learning_curve_stage_order": STAGE_ORDER,
        "outputs": {
            **plot_paths,
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    manifest_path = output_dir / "learning_curve_metric_proportions_manifest.json"
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
