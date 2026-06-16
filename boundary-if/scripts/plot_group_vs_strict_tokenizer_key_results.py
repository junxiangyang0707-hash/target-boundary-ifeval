from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

DEFAULT_STRICT_AGGREGATE = "runs/strict_atomic_tokenizer_multiseed_key_runs/requested_multiseed_aggregate.csv"
DEFAULT_OLD_M1_CURVE = (
    "runs/model_comparisons/m1_vs_m3_learning_curve_atomic_constraint_heldout_seed42_max2048/"
    "combined_learning_curve.csv"
)
DEFAULT_OLD_M3_AGGREGATE = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_aggregate.csv"
)
DEFAULT_OLD_M4_AGGREGATE = (
    "runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_aggregate.csv"
)
DEFAULT_OUTPUT_DIR = "runs/model_comparisons/group_vs_strict_tokenizer_key_results"

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
TOKENIZER_COLORS = {
    "group-key tokenizer": {"fill": "#A3BEFA", "edge": "#2E4780"},
    "atomic-train tokenizer": {"fill": "#F0986E", "edge": "#804126"},
}
CONFIG_ORDER = [
    ("M1 TF-IDF", "full"),
    ("M3 mean", "40k"),
    ("M3 mean", "full"),
    ("M4 frozen", "20k"),
    ("M4 frozen", "40k"),
    ("M4 frozen", "full"),
]
CONFIG_LABELS = {
    ("M1 TF-IDF", "full"): "M1 TF-IDF full",
    ("M3 mean", "40k"): "M3 mean 40k",
    ("M3 mean", "full"): "M3 mean full",
    ("M4 frozen", "20k"): "M4 frozen 20k",
    ("M4 frozen", "40k"): "M4 frozen 40k",
    ("M4 frozen", "full"): "M4 frozen full",
}
PLOT_LABELS = {
    ("M1 TF-IDF", "full"): "M1 TF-IDF\nfull",
    ("M3 mean", "40k"): "M3 mean\n40k",
    ("M3 mean", "full"): "M3 mean\nfull",
    ("M4 frozen", "20k"): "M4 frozen\n20k",
    ("M4 frozen", "40k"): "M4 frozen\n40k",
    ("M4 frozen", "full"): "M4 frozen\nfull",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot group-key tokenizer versus strict atomic-train tokenizer test metrics."
    )
    parser.add_argument("--strict-aggregate", default=DEFAULT_STRICT_AGGREGATE)
    parser.add_argument("--old-m1-curve", default=DEFAULT_OLD_M1_CURVE)
    parser.add_argument("--old-m3-aggregate", default=DEFAULT_OLD_M3_AGGREGATE)
    parser.add_argument("--old-m4-aggregate", default=DEFAULT_OLD_M4_AGGREGATE)
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
    title = textwrap.fill(title.strip(), width=94, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=135, break_long_words=False)
    fig.subplots_adjust(top=0.75, left=0.06, right=0.985, bottom=0.2, wspace=0.16)
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


def read_strict_rows(path: Path) -> pd.DataFrame:
    aggregate = pd.read_csv(path)
    rows = []
    for model, curve_label in CONFIG_ORDER:
        match = aggregate[(aggregate["model"] == model) & (aggregate["curve_label"] == curve_label)]
        if match.empty:
            raise ValueError(f"Missing strict tokenizer result: {model} {curve_label}")
        row = match.iloc[0]
        rows.append(
            {
                "tokenizer": "atomic-train tokenizer",
                "model": model,
                "curve_label": curve_label,
                "config_label": CONFIG_LABELS[(model, curve_label)],
                "run_count": int(row["run_count"]),
                "seeds": row["seeds"],
                "actual_train_rows_mean": float(row["actual_train_rows_mean"]),
                "test_positive_rate": float(row["test_positive_rate"]),
                "test_auroc_mean": float(row["test_auroc_mean"]),
                "test_auroc_std": float(row["test_auroc_std"]),
                "test_auprc_mean": float(row["test_auprc_mean"]),
                "test_auprc_std": float(row["test_auprc_std"]),
                "test_auprc_lift_mean": float(row["test_auprc_lift_mean"]),
                "test_auprc_lift_std": float(row["test_auprc_lift_std"]),
            }
        )
    return pd.DataFrame(rows)


def add_old_row(
    rows: list[dict[str, object]],
    *,
    model: str,
    curve_label: str,
    row: pd.Series,
    seeds: str,
) -> None:
    test_positive_rate = float(row["test_positive_rate"])
    test_auprc_mean = float(row["test_auprc_mean"] if "test_auprc_mean" in row else row["test_auprc"])
    test_auprc_std = float(row["test_auprc_std"] if "test_auprc_std" in row else 0.0)
    rows.append(
        {
            "tokenizer": "group-key tokenizer",
            "model": model,
            "curve_label": curve_label,
            "config_label": CONFIG_LABELS[(model, curve_label)],
            "run_count": int(row["run_count"]) if "run_count" in row and pd.notna(row["run_count"]) else 1,
            "seeds": seeds,
            "actual_train_rows_mean": float(
                row["actual_train_rows_mean"] if "actual_train_rows_mean" in row else row["actual_train_rows"]
            ),
            "test_positive_rate": test_positive_rate,
            "test_auroc_mean": float(row["test_auroc_mean"] if "test_auroc_mean" in row else row["test_auroc"]),
            "test_auroc_std": float(row["test_auroc_std"] if "test_auroc_std" in row else 0.0),
            "test_auprc_mean": test_auprc_mean,
            "test_auprc_std": test_auprc_std,
            "test_auprc_lift_mean": test_auprc_mean / test_positive_rate,
            "test_auprc_lift_std": test_auprc_std / test_positive_rate,
        }
    )


def read_old_rows(m1_path: Path, m3_path: Path, m4_path: Path) -> pd.DataFrame:
    m1 = pd.read_csv(m1_path)
    m3 = pd.read_csv(m3_path)
    m4 = pd.read_csv(m4_path)
    rows: list[dict[str, object]] = []

    m1_full = m1[
        (m1["model"] == "M1 TF-IDF logreg")
        & ((m1["curve_label"] == "full") | (m1["curve_label"] == "80k/full") | (m1["source_label"].astype(str) == "80k"))
    ]
    if m1_full.empty:
        raise ValueError("Missing old group-key tokenizer result: M1 TF-IDF full")
    add_old_row(rows, model="M1 TF-IDF", curve_label="full", row=m1_full.iloc[0], seeds="42")

    for curve_label in ("40k", "full"):
        match = m3[(m3["pooling"] == "mean") & (m3["curve_label"] == curve_label)]
        if match.empty:
            raise ValueError(f"Missing old group-key tokenizer result: M3 mean {curve_label}")
        add_old_row(rows, model="M3 mean", curve_label=curve_label, row=match.iloc[0], seeds=str(match.iloc[0]["seeds"]))

    for curve_label in ("20k", "40k", "full"):
        match = m4[m4["curve_label"] == curve_label]
        if match.empty:
            raise ValueError(f"Missing old group-key tokenizer result: M4 frozen {curve_label}")
        add_old_row(rows, model="M4 frozen", curve_label=curve_label, row=match.iloc[0], seeds=str(match.iloc[0]["seeds"]))

    return pd.DataFrame(rows)


def build_plot_data(args: argparse.Namespace) -> pd.DataFrame:
    strict = read_strict_rows(resolve_path(args.strict_aggregate))
    old = read_old_rows(
        resolve_path(args.old_m1_curve),
        resolve_path(args.old_m3_aggregate),
        resolve_path(args.old_m4_aggregate),
    )
    data = pd.concat([old, strict], ignore_index=True)
    config_order = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    tokenizer_order = {name: idx for idx, name in enumerate(TOKENIZER_COLORS)}
    data["config_order"] = data.apply(lambda row: config_order[(row["model"], row["curve_label"])], axis=1)
    data["tokenizer_order"] = data["tokenizer"].map(tokenizer_order)
    data = data.sort_values(["config_order", "tokenizer_order"]).reset_index(drop=True)

    pivot = data.pivot(index=["model", "curve_label"], columns="tokenizer", values="test_auprc_mean")
    delta = (pivot["atomic-train tokenizer"] - pivot["group-key tokenizer"]).rename("test_auprc_delta_atomic_minus_group")
    data = data.merge(delta, left_on=["model", "curve_label"], right_index=True)
    pivot = data.pivot(index=["model", "curve_label"], columns="tokenizer", values="test_auroc_mean")
    delta = (pivot["atomic-train tokenizer"] - pivot["group-key tokenizer"]).rename("test_auroc_delta_atomic_minus_group")
    return data.merge(delta, left_on=["model", "curve_label"], right_index=True)


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle], values: pd.Series, y_offset: float) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=TOKENS["muted"],
            rotation=0,
        )


def add_metric_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylim: tuple[float, float],
) -> None:
    x_positions = list(range(len(CONFIG_ORDER)))
    width = 0.34
    offsets = {"group-key tokenizer": -width / 2, "atomic-train tokenizer": width / 2}
    for tokenizer, colors in TOKENIZER_COLORS.items():
        part = data[data["tokenizer"] == tokenizer].sort_values("config_order")
        xs = [x + offsets[tokenizer] for x in x_positions]
        values = part[f"{metric}_mean"]
        errors = part[f"{metric}_std"]
        bars = ax.bar(
            xs,
            values,
            width=width,
            label=tokenizer,
            color=colors["fill"],
            edgecolor=colors["edge"],
            linewidth=1.0,
            yerr=errors,
            ecolor=colors["edge"],
            capsize=3,
            error_kw={"elinewidth": 1.0},
            zorder=3,
        )
        annotate_bars(ax, list(bars), values, y_offset=(ylim[1] - ylim[0]) * 0.014)

    if metric == "test_auprc":
        baseline = float(data["test_positive_rate"].dropna().mean())
        ax.axhline(baseline, color=TOKENS["ink"], linestyle=":", linewidth=1.0, zorder=1)
        ax.text(
            -0.48,
            baseline + 0.006,
            f"baseline {baseline:.1%}",
            ha="left",
            va="bottom",
            fontsize=8,
            color=TOKENS["muted"],
        )

    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=TOKENS["ink"], pad=10)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x_positions, [PLOT_LABELS[config] for config in CONFIG_ORDER])
    ax.tick_params(axis="x", colors=TOKENS["muted"], labelsize=8, length=0)
    ax.tick_params(axis="y", colors=TOKENS["muted"], labelsize=8)
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.set_ylabel(title, color=TOKENS["ink"])
    ax.set_xlabel("")


def plot(data: pd.DataFrame, output_dir: Path) -> None:
    use_chart_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.8), sharex=True)
    add_metric_panel(axes[0], data, metric="test_auroc", title="Test AUROC", ylim=(0.0, 0.90))
    add_metric_panel(axes[1], data, metric="test_auprc", title="Test AUPRC", ylim=(0.0, 0.56))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1.17),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        fontsize=8.5,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    add_chart_header(
        fig,
        axes[0],
        "Group-key tokenizer vs atomic-train tokenizer on held-out atomic test",
        "Bars show test-set means for the same model/config. Error bars show cross-seed standard deviation when available; old M1 group-key tokenizer is seed 42 only.",
    )
    sns.despine(fig=fig)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"group_vs_strict_tokenizer_key_results.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_plot_data(args)
    data.to_csv(output_dir / "group_vs_strict_tokenizer_key_results_data.csv", index=False)
    plot(data, output_dir)
    columns = [
        "tokenizer",
        "model",
        "curve_label",
        "run_count",
        "seeds",
        "test_auroc_mean",
        "test_auroc_std",
        "test_auprc_mean",
        "test_auprc_std",
        "test_auprc_delta_atomic_minus_group",
    ]
    print(data[columns].to_string(index=False))


if __name__ == "__main__":
    main()
