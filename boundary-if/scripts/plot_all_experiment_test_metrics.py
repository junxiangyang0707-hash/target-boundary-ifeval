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

DEFAULT_OUTPUT_DIR = "runs/model_comparisons/all_experiment_test_metrics"
DEFAULT_M1_LC = "runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/learning_curve_summary.csv"
DEFAULT_M1_SPLIT = "runs/m1_split_comparison/m1_split_comparison_test.csv"
DEFAULT_M3_MEAN_AGG = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "figures/m3_initial_mean_multiseed_learning_curve_aggregate.csv"
)
DEFAULT_M3_SINGLE_LC = (
    "runs/m3_learning_curve_atomic_constraint_heldout_seed42_mean_pooling_max2048/"
    "learning_curve_summary.csv"
)
DEFAULT_M3_POOLING_AGG = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "figures/m3_55k_pooling_multiseed_aggregate.csv"
)
DEFAULT_M3_CAPACITY = (
    "runs/m3_capacity_seed_sweep_55k_sample42_max2048/"
    "figures/m3_capacity_seed_sweep_aggregate_with_baseline.csv"
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
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
    "neutral": {"base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}
FAMILY_COLORS = {
    "M1 learning curve": COLOR_FAMILIES["orange"],
    "M1 split comparison": COLOR_FAMILIES["gold"],
    "M3 mean learning curve": COLOR_FAMILIES["blue"],
    "M3 pooling": COLOR_FAMILIES["olive"],
    "M3 capacity": COLOR_FAMILIES["pink"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all tracked experiment test AUROC/AUPRC comparisons."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m1-learning-curve", default=DEFAULT_M1_LC)
    parser.add_argument("--m1-split-comparison", default=DEFAULT_M1_SPLIT)
    parser.add_argument("--m3-mean-aggregate", default=DEFAULT_M3_MEAN_AGG)
    parser.add_argument("--m3-single-learning-curve", default=DEFAULT_M3_SINGLE_LC)
    parser.add_argument("--m3-pooling-aggregate", default=DEFAULT_M3_POOLING_AGG)
    parser.add_argument("--m3-capacity-aggregate", default=DEFAULT_M3_CAPACITY)
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


def result_row(
    *,
    experiment_family: str,
    experiment_label: str,
    split_family: str,
    test_auroc_mean: float,
    test_auprc_mean: float,
    test_auroc_std: float = 0.0,
    test_auprc_std: float = 0.0,
    run_count: int = 1,
    seeds: str = "42",
    source_file: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "experiment_family": experiment_family,
        "experiment_label": experiment_label,
        "split_family": split_family,
        "test_auroc_mean": float(test_auroc_mean),
        "test_auroc_std": float(test_auroc_std),
        "test_auprc_mean": float(test_auprc_mean),
        "test_auprc_std": float(test_auprc_std),
        "run_count": int(run_count),
        "seeds": str(seeds),
        "source_file": source_file,
        "notes": notes,
    }


def add_m1_learning_curve(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.read_csv(path)
    keep_labels = ["2k", "4k", "5k", "10k", "20k", "40k", "50k", "55k", "80k/full"]
    for _, row in frame[frame["curve_label"].astype(str).isin(keep_labels)].iterrows():
        label = "full" if str(row["curve_label"]) == "80k/full" else str(row["curve_label"])
        rows.append(
            result_row(
                experiment_family="M1 learning curve",
                experiment_label=f"M1 TF-IDF {label}",
                split_family="atomic held-out max2048",
                test_auroc_mean=row["test_auroc"],
                test_auprc_mean=row["test_auprc"],
                source_file=str(path),
                notes="single seed/result",
            )
        )


def add_m1_split_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.read_csv(path)
    label_map = {
        "group_key_seed42": "M1 group split",
        "composition_heldout_c1": "M1 composition C1",
        "composition_heldout_c2": "M1 composition C2",
    }
    for _, row in frame.iterrows():
        split_name = str(row["dataset_split"])
        if split_name not in label_map:
            continue
        rows.append(
            result_row(
                experiment_family="M1 split comparison",
                experiment_label=label_map[split_name],
                split_family=split_name,
                test_auroc_mean=row["auroc"],
                test_auprc_mean=row["auprc"],
                run_count=1,
                seeds="42",
                source_file=str(path),
                notes=f"not directly comparable with atomic held-out max2048; baseline positive rate={row['positive_rate_baseline_auprc']:.4f}",
            )
        )


def add_m3_mean_curve(rows: list[dict[str, Any]], aggregate_path: Path, single_path: Path) -> None:
    aggregate = pd.read_csv(aggregate_path)
    keep_labels = ["2k", "4k", "5k", "10k", "20k", "40k", "full"]
    for _, row in aggregate[aggregate["curve_label"].astype(str).isin(keep_labels)].iterrows():
        label = str(row["curve_label"])
        rows.append(
            result_row(
                experiment_family="M3 mean learning curve",
                experiment_label=f"M3 mean {label}",
                split_family="atomic held-out max2048",
                test_auroc_mean=row["test_auroc_mean"],
                test_auroc_std=row["test_auroc_std"],
                test_auprc_mean=row["test_auprc_mean"],
                test_auprc_std=row["test_auprc_std"],
                run_count=row["run_count"],
                seeds=row["seeds"],
                source_file=str(aggregate_path),
                notes="mean/std over train seeds",
            )
        )
    single = pd.read_csv(single_path)
    for requested_label in ["50k"]:
        match = single[single["curve_label"].astype(str).eq(requested_label)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            result_row(
                experiment_family="M3 mean learning curve",
                experiment_label=f"M3 mean {requested_label}",
                split_family="atomic held-out max2048",
                test_auroc_mean=row["test_auroc"],
                test_auprc_mean=row["test_auprc"],
                run_count=1,
                seeds="42",
                source_file=str(single_path),
                notes="single seed42 run; included because 50k was tested separately",
            )
        )


def add_m3_pooling(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        pooling = str(row["pooling"])
        rows.append(
            result_row(
                experiment_family="M3 pooling",
                experiment_label=f"M3 {pooling} 55k",
                split_family="atomic held-out max2048",
                test_auroc_mean=row["test_auroc_mean"],
                test_auroc_std=row["test_auroc_std"],
                test_auprc_mean=row["test_auprc_mean"],
                test_auprc_std=row["test_auprc_std"],
                run_count=row["run_count"],
                seeds=row["seeds"],
                source_file=str(path),
                notes="mean/std over train seeds 42/43/44; mean row is the 55k baseline",
            )
        )


def add_m3_capacity(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.read_csv(path)
    frame = frame[~frame["config_label"].astype(str).eq("baseline_128_l4_h4_ffn512")].copy()
    for _, row in frame.iterrows():
        rows.append(
            result_row(
                experiment_family="M3 capacity",
                experiment_label=f"M3 capacity {row['display_name']}",
                split_family="atomic held-out max2048",
                test_auroc_mean=row["test_auroc_mean"],
                test_auroc_std=row["test_auroc_std"],
                test_auprc_mean=row["test_auprc_mean"],
                test_auprc_std=row["test_auprc_std"],
                run_count=row["run_count"],
                seeds=row["seeds"],
                source_file=str(path),
                notes="55k sample_seed42 capacity sweep; baseline omitted because M3 mean 55k is included",
            )
        )


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add_m1_learning_curve(rows, resolve_path(args.m1_learning_curve))
    add_m1_split_comparison(rows, resolve_path(args.m1_split_comparison))
    add_m3_mean_curve(
        rows,
        resolve_path(args.m3_mean_aggregate),
        resolve_path(args.m3_single_learning_curve),
    )
    add_m3_pooling(rows, resolve_path(args.m3_pooling_aggregate))
    add_m3_capacity(rows, resolve_path(args.m3_capacity_aggregate))
    frame = pd.DataFrame(rows)
    frame["rank_test_auprc"] = frame["test_auprc_mean"].rank(method="first", ascending=False).astype(int)
    frame = frame.sort_values(["test_auprc_mean", "test_auroc_mean"], ascending=[False, False], kind="mergesort")
    return frame


def draw_metric_panel(
    ax,
    frame: pd.DataFrame,
    *,
    metric_mean: str,
    metric_std: str,
    title: str,
    y_positions: dict[str, int],
) -> None:
    for _, row in frame.iterrows():
        family = FAMILY_COLORS.get(row["experiment_family"], COLOR_FAMILIES["neutral"])
        y = y_positions[row["experiment_label"]]
        x = float(row[metric_mean])
        std = float(row[metric_std])
        if std > 0:
            ax.errorbar(
                x,
                y,
                xerr=std,
                fmt="none",
                ecolor=family["dark"],
                elinewidth=1.1,
                capsize=3,
                zorder=2,
            )
        marker = "o"
        if row["split_family"] != "atomic held-out max2048":
            marker = "s"
        ax.scatter(
            x,
            y,
            s=42,
            marker=marker,
            facecolor=family["base"],
            edgecolor=family["dark"],
            linewidth=1.0,
            zorder=4,
        )
    ax.set_title(title, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_xlabel(title)
    ax.set_xlim(0.2, 0.95)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", colors=TOKENS["muted"], labelsize=8.5)
    ax.tick_params(axis="y", colors=TOKENS["muted"], labelsize=8.5)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_comparison(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "all_experiment_test_metrics_detail.csv"
    detail_json = output_dir / "all_experiment_test_metrics_detail.json"
    frame.to_csv(detail_csv, index=False, encoding="utf-8")
    write_json(frame.to_dict("records"), detail_json)

    labels = list(frame["experiment_label"])
    y_positions = {label: len(labels) - 1 - index for index, label in enumerate(labels)}
    use_chart_theme()
    height = max(10.0, 0.42 * len(labels) + 2.8)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, height), sharey=True)
    draw_metric_panel(
        axes[0],
        frame,
        metric_mean="test_auroc_mean",
        metric_std="test_auroc_std",
        title="Test AUROC",
        y_positions=y_positions,
    )
    draw_metric_panel(
        axes[1],
        frame,
        metric_mean="test_auprc_mean",
        metric_std="test_auprc_std",
        title="Test AUPRC",
        y_positions=y_positions,
    )
    y_ticks = [y_positions[label] for label in labels]
    axes[0].set_yticks(y_ticks, labels)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("Experiment, sorted by test AUPRC")

    legend_handles = []
    legend_labels = []
    for family_name, family in FAMILY_COLORS.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                markersize=7,
                linewidth=0,
            )
        )
        legend_labels.append(family_name)
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=TOKENS["panel"],
            markeredgecolor=TOKENS["muted"],
            markersize=7,
            linewidth=0,
        )
    )
    legend_labels.append("different split")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.92),
        frameon=False,
        ncol=3,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, left=0.28, right=0.985, bottom=0.06, wspace=0.12)
    add_chart_header(
        fig,
        axes[0],
        "All tracked experiment results on test AUROC and test AUPRC",
        (
            "Rows are sorted by test AUPRC. Circles use the atomic held-out max2048 test set; squares are earlier "
            "M1 split-comparison runs with different test distributions. Error bars show standard deviation across seeds when available."
        ),
    )
    png_path = output_dir / "all_experiment_test_auroc_auprc.png"
    svg_path = output_dir / "all_experiment_test_auroc_auprc.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {
        "detail_csv": str(detail_csv),
        "detail_json": str(detail_json),
        "png": str(png_path),
        "svg": str(svg_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    frame = build_rows(args)
    outputs = plot_comparison(frame, output_dir)
    manifest = {
        "row_count": int(len(frame)),
        "output_dir": str(output_dir),
        "outputs": outputs,
        "inputs": {
            "m1_learning_curve": str(resolve_path(args.m1_learning_curve)),
            "m1_split_comparison": str(resolve_path(args.m1_split_comparison)),
            "m3_mean_aggregate": str(resolve_path(args.m3_mean_aggregate)),
            "m3_single_learning_curve": str(resolve_path(args.m3_single_learning_curve)),
            "m3_pooling_aggregate": str(resolve_path(args.m3_pooling_aggregate)),
            "m3_capacity_aggregate": str(resolve_path(args.m3_capacity_aggregate)),
        },
    }
    write_json(manifest, output_dir / "all_experiment_test_metrics_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
