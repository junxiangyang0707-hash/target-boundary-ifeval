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

DEFAULT_MEAN_SUMMARY = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_summary.csv"
)
DEFAULT_CLS_SUMMARY = (
    "runs/m3_initial_cls_55k_multiseed_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_summary.csv"
)
DEFAULT_MEAN_55K_SUMMARY = (
    "runs/m3_seed_sensitivity_55k_fixedsample42_max2048/"
    "seed_sensitivity_55k_sample42_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "figures"
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
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "orange": {
        "open": TOKENS["panel"],
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "open": TOKENS["panel"],
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cross-seed M3 learning curves and 55k pooling comparison."
    )
    parser.add_argument("--mean-summary", default=DEFAULT_MEAN_SUMMARY)
    parser.add_argument("--cls-summary", default=DEFAULT_CLS_SUMMARY)
    parser.add_argument("--mean-55k-summary", default=DEFAULT_MEAN_55K_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="42,43,44")
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


def parse_seeds(raw_value: str) -> set[int]:
    return {int(item.strip()) for item in raw_value.split(",") if item.strip()}


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
    title = textwrap.fill(title.strip(), width=96, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=134, break_long_words=False)
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
        0.944,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.18,
    )


def aggregate_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
        "val_auprc_lift",
        "test_auprc_lift",
        "total_seconds",
    ]
    label_order = {label: index for index, label in enumerate(["2k", "4k", "5k", "10k", "20k", "40k", "55k", "full"])}
    rows: list[dict[str, Any]] = []
    for (pooling, label), part in frame.groupby(["pooling", "curve_label"], sort=False):
        row: dict[str, Any] = {
            "pooling": pooling,
            "curve_label": label,
            "order": label_order.get(str(label), 999),
            "run_count": int(len(part)),
            "seeds": ",".join(str(seed) for seed in sorted(part["train_seed"].astype(int))),
            "actual_train_rows_mean": float(part["actual_train_rows"].mean()),
            "val_positive_rate": float(part["val_positive_rate"].dropna().iloc[0]),
            "test_positive_rate": float(part["test_positive_rate"].dropna().iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pooling", "order"], kind="mergesort")


def draw_learning_panel(
    ax,
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    order_labels: list[str],
    baseline: float | None = None,
) -> None:
    family = COLOR_FAMILIES["blue"]
    x_lookup = {label: index for index, label in enumerate(order_labels)}
    aggregate = aggregate.set_index("curve_label").loc[order_labels].reset_index()
    x_values = [x_lookup[label] for label in aggregate["curve_label"]]
    y_values = aggregate[f"{metric}_mean"].astype(float)
    y_std = aggregate[f"{metric}_std"].astype(float)
    ax.plot(
        x_values,
        y_values,
        color=family["mid"],
        marker="o",
        markerfacecolor=family["base"],
        markeredgecolor=family["dark"],
        linewidth=1.4,
        markersize=6,
        label="Mean over seeds",
    )
    ax.fill_between(
        x_values,
        (y_values - y_std).clip(lower=0.0),
        (y_values + y_std).clip(upper=1.0),
        color=family["light"],
        alpha=0.42,
        linewidth=0,
        label="±1 seed std",
    )
    seed_offsets = {42: -0.13, 43: 0.0, 44: 0.13}
    for _, row in detail.iterrows():
        label = str(row["curve_label"])
        if label not in x_lookup:
            continue
        seed = int(row["train_seed"])
        ax.scatter(
            x_lookup[label] + seed_offsets.get(seed, 0.0),
            float(row[metric]),
            s=28,
            color=TOKENS["panel"],
            edgecolor=family["dark"],
            linewidth=0.9,
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
    ax.set_xticks(range(len(order_labels)), order_labels)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_learning_curve(mean_summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    order_labels = ["2k", "4k", "5k", "10k", "20k", "40k", "full"]
    mean_summary = mean_summary[mean_summary["curve_label"].astype(str).isin(order_labels)].copy()
    aggregate = aggregate_by_label(mean_summary)
    aggregate = aggregate[aggregate["curve_label"].astype(str).isin(order_labels)].copy()
    aggregate_csv = output_dir / "m3_initial_mean_multiseed_learning_curve_aggregate.csv"
    aggregate_json = output_dir / "m3_initial_mean_multiseed_learning_curve_aggregate.json"
    aggregate.to_csv(aggregate_csv, index=False, encoding="utf-8")
    write_json(aggregate.to_dict("records"), aggregate_json)

    val_baseline = float(mean_summary["val_positive_rate"].dropna().iloc[0])
    test_baseline = float(mean_summary["test_positive_rate"].dropna().iloc[0])
    use_chart_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    draw_learning_panel(
        axes[0, 0],
        mean_summary,
        aggregate,
        metric="val_auroc",
        title="Validation AUROC",
        ylabel="AUROC",
        order_labels=order_labels,
    )
    draw_learning_panel(
        axes[0, 1],
        mean_summary,
        aggregate,
        metric="val_auprc",
        title="Validation AUPRC",
        ylabel="AUPRC",
        order_labels=order_labels,
        baseline=val_baseline,
    )
    draw_learning_panel(
        axes[1, 0],
        mean_summary,
        aggregate,
        metric="test_auroc",
        title="Test AUROC",
        ylabel="AUROC",
        order_labels=order_labels,
    )
    draw_learning_panel(
        axes[1, 1],
        mean_summary,
        aggregate,
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
        "Initial M3 mean-pooling learning curve averaged over train seeds 42, 43, and 44",
        (
            "Atomic constraint held-out split; train subsets are hash-stable with sample_seed=42; "
            "validation and test are evaluated in full. Band shows ±1 standard deviation across train seeds."
        ),
    )
    png_path = output_dir / "m3_initial_mean_multiseed_learning_curve_metrics.png"
    svg_path = output_dir / "m3_initial_mean_multiseed_learning_curve_metrics.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {
        "aggregate_csv": str(aggregate_csv),
        "aggregate_json": str(aggregate_json),
        "png": str(png_path),
        "svg": str(svg_path),
    }


def load_mean_55k(path: Path, seeds: set[int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["train_seed"].astype(int).isin(seeds)].copy()
    frame["pooling"] = "mean"
    frame["curve_label"] = "55k"
    if "requested_train_sample_size" in frame.columns:
        frame["requested_train_rows"] = frame["requested_train_sample_size"]
    else:
        frame["requested_train_rows"] = 55000
    rename_map = {"actual_train_rows": "actual_train_rows"}
    frame = frame.rename(columns=rename_map)
    frame["val_positive_rate"] = 0.3121345
    frame["test_positive_rate"] = 0.14806506
    frame["val_auprc_lift"] = frame["val_auprc"] / frame["val_positive_rate"]
    frame["test_auprc_lift"] = frame["test_auprc"] / frame["test_positive_rate"]
    return frame[
        [
            "pooling",
            "curve_label",
            "requested_train_rows",
            "actual_train_rows",
            "train_seed",
            "sample_seed",
            "val_auroc",
            "val_auprc",
            "test_auroc",
            "test_auprc",
            "val_positive_rate",
            "test_positive_rate",
            "val_auprc_lift",
            "test_auprc_lift",
            "total_seconds",
        ]
    ].copy()


def draw_pooling_panel(
    ax,
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    metric: str,
    title: str,
    baseline: float | None = None,
) -> None:
    order = ["mean", "cls"]
    palette = {"mean": COLOR_FAMILIES["blue"], "cls": COLOR_FAMILIES["orange"]}
    x_lookup = {label: index for index, label in enumerate(order)}
    for pooling in order:
        row = aggregate[aggregate["pooling"].eq(pooling)]
        if row.empty:
            continue
        family = palette[pooling]
        x = x_lookup[pooling]
        y = float(row[f"{metric}_mean"].iloc[0])
        yerr = float(row[f"{metric}_std"].iloc[0])
        ax.bar(
            x,
            y,
            width=0.58,
            color=family["base"],
            edgecolor=family["dark"],
            linewidth=1.0,
        )
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=family["dark"], elinewidth=1.2, capsize=4)
    seed_offsets = {42: -0.12, 43: 0.0, 44: 0.12}
    for _, row in detail.iterrows():
        pooling = str(row["pooling"])
        if pooling not in x_lookup:
            continue
        family = palette[pooling]
        seed = int(row["train_seed"])
        ax.scatter(
            x_lookup[pooling] + seed_offsets.get(seed, 0.0),
            float(row[metric]),
            s=30,
            color=TOKENS["panel"],
            edgecolor=family["dark"],
            linewidth=1.0,
            zorder=5,
        )
    if baseline is not None:
        ax.axhline(baseline, color=TOKENS["muted"], linestyle=":", linewidth=1.0)
    ax.set_title(title, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_xticks(range(len(order)), ["Mean pooling", "CLS pooling"])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_pooling_comparison(
    *,
    cls_summary: pd.DataFrame,
    mean_55k_summary: pd.DataFrame | None,
    output_dir: Path,
) -> dict[str, str] | None:
    cls_55k = cls_summary[cls_summary["curve_label"].astype(str).eq("55k")].copy()
    if cls_55k.empty:
        return None
    detail = cls_55k.copy()
    if mean_55k_summary is not None and not mean_55k_summary.empty:
        detail = pd.concat([mean_55k_summary, detail], ignore_index=True, sort=False)
    aggregate = aggregate_by_label(detail)
    aggregate = aggregate[aggregate["curve_label"].astype(str).eq("55k")].copy()
    aggregate_csv = output_dir / "m3_55k_pooling_multiseed_aggregate.csv"
    detail_csv = output_dir / "m3_55k_pooling_multiseed_detail.csv"
    aggregate.to_csv(aggregate_csv, index=False, encoding="utf-8")
    detail.to_csv(detail_csv, index=False, encoding="utf-8")

    val_baseline = float(detail["val_positive_rate"].dropna().iloc[0])
    test_baseline = float(detail["test_positive_rate"].dropna().iloc[0])
    use_chart_theme()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.7), sharex=True)
    draw_pooling_panel(axes[0, 0], detail, aggregate, metric="val_auroc", title="Validation AUROC")
    draw_pooling_panel(
        axes[0, 1],
        detail,
        aggregate,
        metric="val_auprc",
        title="Validation AUPRC",
        baseline=val_baseline,
    )
    draw_pooling_panel(axes[1, 0], detail, aggregate, metric="test_auroc", title="Test AUROC")
    draw_pooling_panel(
        axes[1, 1],
        detail,
        aggregate,
        metric="test_auprc",
        title="Test AUPRC",
        baseline=test_baseline,
    )
    fig.subplots_adjust(top=0.80, left=0.08, right=0.985, bottom=0.09, hspace=0.32, wspace=0.18)
    add_chart_header(
        fig,
        axes[0, 0],
        "M3 55k pooling comparison over train seeds 42, 43, and 44",
        (
            "Both variants use the initial 128d x 4-layer Transformer and the same 55k hash-stable train sample; "
            "bars are seed means and whiskers are ±1 standard deviation."
        ),
    )
    png_path = output_dir / "m3_55k_pooling_multiseed_comparison.png"
    svg_path = output_dir / "m3_55k_pooling_multiseed_comparison.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {
        "aggregate_csv": str(aggregate_csv),
        "detail_csv": str(detail_csv),
        "png": str(png_path),
        "svg": str(svg_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    mean_path = resolve_path(args.mean_summary)
    mean_summary = pd.read_csv(mean_path)
    mean_summary = mean_summary[mean_summary["train_seed"].astype(int).isin(seeds)].copy()
    learning_outputs = plot_learning_curve(mean_summary, output_dir)

    cls_path = resolve_path(args.cls_summary)
    mean_55k_path = resolve_path(args.mean_55k_summary)
    pooling_outputs = None
    if cls_path.exists():
        cls_summary = pd.read_csv(cls_path)
        cls_summary = cls_summary[cls_summary["train_seed"].astype(int).isin(seeds)].copy()
        mean_55k_summary = load_mean_55k(mean_55k_path, seeds) if mean_55k_path.exists() else None
        pooling_outputs = plot_pooling_comparison(
            cls_summary=cls_summary,
            mean_55k_summary=mean_55k_summary,
            output_dir=output_dir,
        )

    manifest = {
        "mean_summary": str(mean_path),
        "cls_summary": str(cls_path) if cls_path.exists() else None,
        "mean_55k_summary": str(mean_55k_path) if mean_55k_path.exists() else None,
        "seeds": sorted(seeds),
        "learning_curve": learning_outputs,
        "pooling_comparison": pooling_outputs,
    }
    write_json(manifest, output_dir / "m3_initial_multiseed_plot_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
