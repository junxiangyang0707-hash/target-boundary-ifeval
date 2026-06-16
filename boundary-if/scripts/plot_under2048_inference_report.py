from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "gold": "#B8A037",
    "pink": "#BD569B",
    "pink_dark": "#8A3A6F",
    "olive": "#71B436",
    "neutral": "#7A828F",
    "neutral_dark": "#464C55",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clustering and vLLM inference summary charts."
    )
    parser.add_argument(
        "--clusters-file",
        default="data/reports/prompt_clusters_under2048_seed42.parquet",
    )
    parser.add_argument(
        "--cluster-summary-csv",
        default="data/reports/prompt_clusters_under2048_seed42.summary.csv",
    )
    parser.add_argument(
        "--inference-csv",
        default="data/reports/vllm_under2048_cluster_len_seed42.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/reports/under2048_cluster_inference_plots.summary.json",
    )
    parser.add_argument("--output-dir", default="data/reports")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", "monospace"]
    plt.rcParams["axes.unicode_minus"] = False


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.25, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=9)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.075,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(0.075, 0.915, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])


def fmt_int(value: float, _position: object = None) -> str:
    return f"{int(value):,}"


def plot_cluster_overview(clusters: pd.DataFrame, output_dir: Path) -> Path:
    cluster_order = (
        clusters.groupby("cluster")["prompt_tokens"]
        .median()
        .sort_values()
        .index.astype(int)
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax in axes:
        style_axis(ax)

    cluster_counts = clusters["cluster"].value_counts().reindex(cluster_order)
    axes[0].bar(
        range(len(cluster_order)),
        cluster_counts.to_numpy(),
        color=COLORS["blue"],
        edgecolor=COLORS["blue_dark"],
        linewidth=0.6,
    )
    axes[0].set_xlabel("Cluster ordered by median prompt length", color=TOKENS["ink"])
    axes[0].set_ylabel("Prompt count", color=TOKENS["ink"])
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))

    sns.boxplot(
        data=clusters,
        x="cluster",
        y="prompt_tokens",
        order=cluster_order,
        ax=axes[1],
        color=COLORS["orange"],
        fliersize=1.2,
        linewidth=0.7,
    )
    axes[1].set_xlabel("Cluster ordered by median prompt length", color=TOKENS["ink"])
    axes[1].set_ylabel("Prompt tokens", color=TOKENS["ink"])
    axes[1].set_ylim(0, 2050)
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))

    add_header(
        fig,
        "Clusters cover the under-2048 prompt pool with broad length variation",
        "Cluster size and token-length spread for 94,390 prompts under 2,048 input tokens.",
    )
    path = output_dir / "under2048_cluster_overview.png"
    fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.18, wspace=0.22)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_sample_coverage(metrics: pd.DataFrame, output_dir: Path) -> Path:
    heatmap = pd.crosstab(metrics["cluster"], metrics["length_bin"])
    length_order = ["001-128", "129-256", "257-512", "513-1024", "1025-2047"]
    heatmap = heatmap.reindex(columns=length_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap=sns.light_palette(COLORS["blue_dark"], as_cmap=True),
        annot=True,
        fmt="d",
        linewidths=0.35,
        linecolor=TOKENS["grid"],
        cbar_kws={"label": "sample count"},
    )
    ax.set_xlabel("Prompt token length bin", color=TOKENS["ink"])
    ax.set_ylabel("Cluster", color=TOKENS["ink"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=9)
    add_header(
        fig,
        "The 100-request sample spans clusters and prompt lengths",
        "Counts by cluster and length bin for the sampled prompts used in the vLLM run.",
    )
    path = output_dir / "under2048_sample_cluster_length_heatmap.png"
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.15)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_input_output(metrics: pd.DataFrame, output_dir: Path) -> Path:
    ok = metrics[metrics["ok"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax in axes:
        style_axis(ax)

    colors = np.where(ok["output_gt_threshold"], COLORS["pink"], COLORS["blue_dark"])
    axes[0].scatter(
        ok["prompt_tokens"],
        ok["output_tokens"],
        s=42,
        c=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    axes[0].axhline(2048, color=COLORS["pink_dark"], linestyle="--", linewidth=1.2)
    axes[0].axhline(4096, color=COLORS["neutral_dark"], linestyle=":", linewidth=1.1)
    axes[0].set_xlabel("Input tokens", color=TOKENS["ink"])
    axes[0].set_ylabel("Output tokens", color=TOKENS["ink"])
    axes[0].set_xlim(0, 2050)
    axes[0].set_ylim(0, 4300)
    axes[0].text(1660, 2048, "2048 cap", va="center", color=COLORS["pink_dark"], fontsize=8.5)
    axes[0].text(1660, 4096, "4096 max", va="center", color=COLORS["neutral_dark"], fontsize=8.5)

    bins = [0, 128, 256, 512, 1024, 2048, 4096]
    axes[1].hist(
        ok["output_tokens"],
        bins=bins,
        color=COLORS["orange"],
        edgecolor=COLORS["orange_dark"],
        linewidth=0.7,
    )
    axes[1].axvline(2048, color=COLORS["pink_dark"], linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Output tokens", color=TOKENS["ink"])
    axes[1].set_ylabel("Completed request count", color=TOKENS["ink"])
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))

    add_header(
        fig,
        "Most sampled outputs fit under 2,048, but not all",
        "Scatter and histogram for successful requests; pink marks exceed the proposed "
        "2,048 output cap.",
    )
    path = output_dir / "under2048_vllm_input_output_tokens.png"
    fig.subplots_adjust(left=0.075, right=0.93, top=0.82, bottom=0.15, wspace=0.24)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_sorted_totals(metrics: pd.DataFrame, output_dir: Path) -> Path:
    sorted_metrics = metrics.sort_values(
        ["total_tokens", "prompt_tokens"],
        kind="mergesort",
    ).reset_index(drop=True)
    sorted_metrics["rank"] = np.arange(1, len(sorted_metrics) + 1)
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    style_axis(ax)

    ax.bar(
        sorted_metrics["rank"],
        sorted_metrics["prompt_tokens"],
        width=0.78,
        color=COLORS["blue"],
        edgecolor=COLORS["blue_dark"],
        linewidth=0.35,
        label="input tokens",
    )
    ax.bar(
        sorted_metrics["rank"],
        sorted_metrics["output_tokens"],
        bottom=sorted_metrics["prompt_tokens"],
        width=0.78,
        color=COLORS["orange"],
        edgecolor=COLORS["orange_dark"],
        linewidth=0.35,
        label="output tokens",
    )
    over = sorted_metrics["output_gt_threshold"]
    ax.scatter(
        sorted_metrics.loc[over, "rank"],
        sorted_metrics.loc[over, "total_tokens"] + 90,
        marker="^",
        s=52,
        color=COLORS["pink"],
        edgecolor="white",
        linewidth=0.6,
        label="output > 2048",
        zorder=5,
    )
    ax.axhline(2048, color=COLORS["pink_dark"], linestyle="--", linewidth=1.1)
    ax.axhline(8192, color=COLORS["gold"], linestyle=":", linewidth=1.1)
    ax.set_xlabel("100 samples sorted by actual total tokens", color=TOKENS["ink"])
    ax.set_ylabel("Input + output tokens", color=TOKENS["ink"])
    ax.set_ylim(0, max(8500, sorted_metrics["total_tokens"].max() * 1.08))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=9)
    add_header(
        fig,
        "No context-window errors occurred under the 2,048-input sample",
        "Total tokens stay below 8,192 because every sampled prompt has input <2,048 "
        "and max_tokens=4,096.",
    )
    path = output_dir / "under2048_vllm_sorted_total_tokens.png"
    fig.subplots_adjust(left=0.075, right=0.965, top=0.82, bottom=0.15)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    setup_style()
    clusters_file = resolve_path(args.clusters_file)
    inference_csv = resolve_path(args.inference_csv)
    summary_json = resolve_path(args.summary_json)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.read_parquet(clusters_file)
    metrics = pd.read_csv(inference_csv)
    metrics["ok"] = metrics["ok"].astype(bool)
    metrics["context_window_error"] = metrics["context_window_error"].astype(bool)
    metrics["output_gt_threshold"] = metrics["output_gt_threshold"].astype(bool)

    paths = {
        "cluster_overview": str(plot_cluster_overview(clusters, output_dir)),
        "sample_coverage": str(plot_sample_coverage(metrics, output_dir)),
        "input_output": str(plot_input_output(metrics, output_dir)),
        "sorted_totals": str(plot_sorted_totals(metrics, output_dir)),
    }

    summary = {
        "clusters_file": str(clusters_file),
        "inference_csv": str(inference_csv),
        "plots": paths,
        "metrics": {
            "sample_count": int(len(metrics)),
            "context_window_error_count": int(metrics["context_window_error"].sum()),
            "output_gt_2048_count": int(metrics["output_gt_threshold"].sum()),
            "finish_reason_length_count": int((metrics["finish_reason"] == "length").sum()),
            "output_tokens_max": int(metrics.loc[metrics["ok"], "output_tokens"].max()),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
