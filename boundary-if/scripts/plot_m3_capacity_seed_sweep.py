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

from boundary_if.models.tiny_transformer import M3TinyTransformer, M3TinyTransformerConfig

DEFAULT_SWEEP_SUMMARY = (
    "runs/m3_capacity_seed_sweep_55k_sample42_max2048/"
    "capacity_seed_sweep_summary.csv"
)
DEFAULT_BASELINE_SUMMARY = (
    "runs/m3_seed_sensitivity_55k_fixedsample42_max2048/"
    "seed_sensitivity_55k_sample42_summary.csv"
)
DEFAULT_OUTPUT_DIR = "runs/m3_capacity_seed_sweep_55k_sample42_max2048/figures"

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
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot M3 capacity seed sweep summary.")
    parser.add_argument("--sweep-summary", default=DEFAULT_SWEEP_SUMMARY)
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-seeds", default="42,43,44")
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


def param_count(config: M3TinyTransformerConfig) -> int:
    return int(sum(parameter.numel() for parameter in M3TinyTransformer(config).parameters()))


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


def load_baseline(path: Path, baseline_seeds: set[int]) -> pd.DataFrame:
    baseline = pd.read_csv(path)
    baseline = baseline[baseline["train_seed"].astype(int).isin(baseline_seeds)].copy()
    config = M3TinyTransformerConfig(
        vocab_size=8000,
        max_length=2048,
        hidden_size=128,
        layers=4,
        heads=4,
        ffn_dim=512,
        dropout=0.1,
        pooling="mean",
        classifier_hidden_size=128,
    )
    baseline["config_label"] = "baseline_128_l4_h4_ffn512"
    baseline["display_name"] = "128d x 4L baseline"
    baseline["hidden_size"] = 128
    baseline["layers"] = 4
    baseline["heads"] = 4
    baseline["ffn_dim"] = 512
    baseline["classifier_hidden_size"] = 128
    baseline["parameter_count"] = param_count(config)
    baseline = baseline.rename(columns={"train_seed": "train_seed"})
    return baseline


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_auroc", "val_auprc", "test_auroc", "test_auprc"]
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(
        [
            "config_label",
            "display_name",
            "hidden_size",
            "layers",
            "heads",
            "ffn_dim",
            "classifier_hidden_size",
            "parameter_count",
        ],
        sort=False,
        dropna=False,
    )
    for keys, part in grouped:
        row = dict(
            zip(
                [
                    "config_label",
                    "display_name",
                    "hidden_size",
                    "layers",
                    "heads",
                    "ffn_dim",
                    "classifier_hidden_size",
                    "parameter_count",
                ],
                keys,
                strict=True,
            )
        )
        row["run_count"] = int(len(part))
        row["seeds"] = ",".join(str(seed) for seed in sorted(part["train_seed"].astype(int)))
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def draw_metric_panel(
    ax,
    aggregate_frame: pd.DataFrame,
    detail_frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    config_order: list[str],
    palette: dict[str, dict[str, str]],
) -> None:
    x_lookup = {label: index for index, label in enumerate(config_order)}
    for _, row in aggregate_frame.iterrows():
        label = row["config_label"]
        family = palette[label]
        x = x_lookup[label]
        y = float(row[f"{metric}_mean"])
        yerr = float(row[f"{metric}_std"])
        ax.bar(
            x,
            y,
            width=0.62,
            color=family["base"],
            edgecolor=family["dark"],
            linewidth=1.0,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=family["dark"],
            elinewidth=1.2,
            capsize=4,
            zorder=4,
        )
    for _, row in detail_frame.iterrows():
        label = row["config_label"]
        x = x_lookup[label]
        seed = int(row["train_seed"])
        offset = {-1: -0.16, 0: 0.0, 1: 0.16}[(seed % 3) - 1]
        family = palette[label]
        ax.scatter(
            x + offset,
            float(row[metric]),
            s=30,
            color=TOKENS["panel"],
            edgecolor=family["dark"],
            linewidth=1.0,
            zorder=5,
        )
    ax.set_title(title, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_xticks(
        range(len(config_order)),
        [
            str(
                aggregate_frame.loc[
                    aggregate_frame["config_label"].eq(label),
                    "display_name",
                ].iloc[0]
            )
            for label in config_order
        ],
        rotation=18,
        ha="right",
    )
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def main() -> None:
    args = parse_args()
    sweep_path = resolve_path(args.sweep_summary)
    baseline_path = resolve_path(args.baseline_summary)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(sweep_path)
    baseline = load_baseline(baseline_path, parse_seeds(args.baseline_seeds))
    columns = sorted(set(sweep.columns).intersection(baseline.columns))
    combined = pd.concat([baseline[columns], sweep[columns]], ignore_index=True)
    config_order = [
        "baseline_128_l4_h4_ffn512",
        "wide_192_l6_h6_ffn768",
        "deep_192_l8_h6_ffn768",
        "medium_256_l6_h8_ffn1024",
    ]
    combined = combined[combined["config_label"].isin(config_order)].copy()
    combined["config_label"] = pd.Categorical(
        combined["config_label"],
        categories=config_order,
        ordered=True,
    )
    combined = combined.sort_values(["config_label", "train_seed"], kind="mergesort")
    aggregate_frame = aggregate(combined)
    aggregate_frame["config_label"] = pd.Categorical(
        aggregate_frame["config_label"],
        categories=config_order,
        ordered=True,
    )
    aggregate_frame = aggregate_frame.sort_values("config_label", kind="mergesort")

    combined_csv = output_dir / "m3_capacity_seed_sweep_with_baseline.csv"
    aggregate_csv = output_dir / "m3_capacity_seed_sweep_aggregate_with_baseline.csv"
    combined.to_csv(combined_csv, index=False, encoding="utf-8")
    aggregate_frame.to_csv(aggregate_csv, index=False, encoding="utf-8")

    palette = {
        "baseline_128_l4_h4_ffn512": COLOR_FAMILIES["gold"],
        "wide_192_l6_h6_ffn768": COLOR_FAMILIES["blue"],
        "deep_192_l8_h6_ffn768": COLOR_FAMILIES["olive"],
        "medium_256_l6_h8_ffn1024": COLOR_FAMILIES["orange"],
    }
    use_chart_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    draw_metric_panel(
        axes[0, 0],
        aggregate_frame,
        combined,
        metric="val_auroc",
        title="Validation AUROC",
        config_order=config_order,
        palette=palette,
    )
    draw_metric_panel(
        axes[0, 1],
        aggregate_frame,
        combined,
        metric="val_auprc",
        title="Validation AUPRC",
        config_order=config_order,
        palette=palette,
    )
    draw_metric_panel(
        axes[1, 0],
        aggregate_frame,
        combined,
        metric="test_auroc",
        title="Test AUROC",
        config_order=config_order,
        palette=palette,
    )
    draw_metric_panel(
        axes[1, 1],
        aggregate_frame,
        combined,
        metric="test_auprc",
        title="Test AUPRC",
        config_order=config_order,
        palette=palette,
    )
    for ax in axes[:, 0]:
        ax.set_ylabel("Score")
    fig.subplots_adjust(top=0.82, left=0.08, right=0.985, bottom=0.15, hspace=0.32, wspace=0.18)
    add_chart_header(
        fig,
        axes[0, 0],
        "M3 capacity sweep on the fixed 55k train sample",
        (
            "Bars show mean across train seeds 42/43/44; whiskers show one sample standard deviation; "
            "open dots show individual seeds. Validation/test are full atomic held-out splits."
        ),
    )
    png_path = output_dir / "m3_capacity_seed_sweep_metrics.png"
    svg_path = output_dir / "m3_capacity_seed_sweep_metrics.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)

    summary = {
        "sweep_summary": str(sweep_path),
        "baseline_summary": str(baseline_path),
        "combined_csv": str(combined_csv),
        "aggregate_csv": str(aggregate_csv),
        "png": str(png_path),
        "svg": str(svg_path),
        "config_order": config_order,
    }
    write_json(summary, output_dir / "plot_manifest.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
