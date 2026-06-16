from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRALS = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLORS = {
    "blue_xlight": "#EAF1FE",
    "blue_light": "#CEDFFE",
    "blue": "#A3BEFA",
    "blue_mid": "#5477C4",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_mid": "#B8A037",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
}

INSTRUCTION_ID_RE = re.compile(r"[A-Za-z_]+:[A-Za-z0-9_]+")
TRAILING_NUMERIC_RE = re.compile(r"([_:\-#])\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether max-token truncated outputs concentrate by cluster."
    )
    parser.add_argument(
        "--generation-root",
        default="data/generations",
    )
    parser.add_argument(
        "--generation-dirs",
        nargs="*",
        default=[
            "qwen3_4b_instruct_2507_under2048_0_10000_max2048",
            "qwen3_4b_instruct_2507_under2048_10000_20000_max2048",
            "qwen3_4b_instruct_2507_under2048_20000_30000_max2048",
            "qwen3_4b_instruct_2507_under2048_30000_40000_max2048",
            "qwen3_4b_instruct_2507_under2048_40000_50000_max2048",
            "qwen3_4b_instruct_2507_under2048_50000_60000_max2048",
            "qwen3_4b_instruct_2507_under2048_60000_70000_max2048",
            "qwen3_4b_instruct_2507_under2048_70000_80000_max2048",
            "qwen3_4b_instruct_2507_under2048_80000_94390_max2048",
        ],
    )
    parser.add_argument(
        "--clusters-file",
        default="data/reports/prompt_clusters_under2048_seed42.parquet",
    )
    parser.add_argument(
        "--cluster-summary-csv",
        default="data/reports/prompt_clusters_under2048_seed42.summary.csv",
    )
    parser.add_argument("--output-dir", default="data/reports/truncated_cluster_analysis")
    parser.add_argument("--min-category-count", type=int, default=200)
    parser.add_argument("--top-feature-count", type=int, default=18)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return Path.cwd() / raw


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", "monospace"]
    plt.rcParams["axes.unicode_minus"] = False


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.25, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=9)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=82, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=125, break_long_words=False)
    fig.text(
        0.075,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.075,
        0.915,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color=TOKENS["muted"],
    )


def fmt_int(value: float, _position: object = None) -> str:
    return f"{int(value):,}"


def fmt_pct_axis(value: float, _position: object = None) -> str:
    return f"{value * 100:.0f}%"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def read_generation_outputs(root: Path, names: list[str]) -> pd.DataFrame:
    columns = [
        "under2048_index",
        "prompt_id",
        "base_key",
        "constraint_signature",
        "constraint_family_signature",
        "instruction_ids_json",
        "num_constraints",
        "source_dataset",
        "constraint_type",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "finish_reason",
        "output_truncated",
        "output_hit_max_tokens",
        "context_window_error",
        "ok",
    ]
    frames: list[pd.DataFrame] = []
    for name in names:
        path = root / name / "outputs.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns)
        frame["generation_dir"] = name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["output_truncated"] = df["output_truncated"].fillna(False).astype(bool)
    df["context_window_error"] = df["context_window_error"].fillna(False).astype(bool)
    df["ok"] = df["ok"].fillna(False).astype(bool)
    return df


def normalize_key_family(value: Any) -> str:
    text = str(value) if value is not None else "<missing>"
    text = text.strip()
    if not text:
        return "<missing>"
    previous = None
    while previous != text:
        previous = text
        text = TRAILING_NUMERIC_RE.sub("", text)
    return text


def split_instruction_ids(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if "|" in text and "[" not in text:
        return [part for part in text.split("|") if part]
    return INSTRUCTION_ID_RE.findall(text)


def percentile(values: pd.Series, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return round(float(np.percentile(values.to_numpy(), q)), 3)


def summarize_category(
    df: pd.DataFrame,
    column: str,
    overall_rate: float,
    *,
    min_count: int = 1,
) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            row_count=("output_truncated", "size"),
            truncated_count=("output_truncated", "sum"),
            prompt_tokens_p50=("prompt_tokens", "median"),
            output_tokens_p50=("output_tokens", "median"),
            output_tokens_p90=("output_tokens", lambda s: percentile(s, 90)),
            num_constraints_mean=("num_constraints", "mean"),
        )
        .reset_index()
        .rename(columns={column: "category"})
    )
    grouped = grouped[grouped["row_count"] >= min_count].copy()
    grouped["category"] = grouped["category"].fillna("<missing>").astype(str)
    grouped["truncated_rate"] = grouped["truncated_count"] / grouped["row_count"]
    grouped["row_share"] = grouped["row_count"] / len(df)
    grouped["truncated_share"] = grouped["truncated_count"] / max(1, df["output_truncated"].sum())
    grouped["expected_truncated_count"] = grouped["row_count"] * overall_rate
    grouped["excess_truncated_count"] = (
        grouped["truncated_count"] - grouped["expected_truncated_count"]
    )
    grouped["lift_vs_overall"] = grouped["truncated_rate"] / overall_rate
    variance = grouped["row_count"] * overall_rate * (1.0 - overall_rate)
    grouped["binomial_z"] = np.where(
        variance > 0,
        grouped["excess_truncated_count"] / np.sqrt(variance),
        np.nan,
    )
    grouped["prompt_tokens_p50"] = grouped["prompt_tokens_p50"].round(2)
    grouped["output_tokens_p50"] = grouped["output_tokens_p50"].round(2)
    grouped["num_constraints_mean"] = grouped["num_constraints_mean"].round(3)
    return grouped.sort_values(
        ["lift_vs_overall", "truncated_count"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_feature_table(
    df: pd.DataFrame,
    columns: list[str],
    overall_rate: float,
    min_count: int,
) -> dict[str, pd.DataFrame]:
    return {
        column: summarize_category(df, column, overall_rate, min_count=min_count)
        for column in columns
    }


def summarize_instruction_ids(
    df: pd.DataFrame,
    overall_rate: float,
    min_count: int,
) -> pd.DataFrame:
    rows = df[["prompt_id", "constraint_signature", "output_truncated", "prompt_tokens"]].copy()
    rows["instruction_id"] = rows["constraint_signature"].map(split_instruction_ids)
    exploded = rows.explode("instruction_id")
    exploded = exploded.dropna(subset=["instruction_id"])
    exploded["instruction_id"] = exploded["instruction_id"].astype(str)
    grouped = (
        exploded.groupby("instruction_id")
        .agg(
            row_count=("prompt_id", "nunique"),
            truncated_count=("output_truncated", "sum"),
            prompt_tokens_p50=("prompt_tokens", "median"),
        )
        .reset_index()
        .rename(columns={"instruction_id": "category"})
    )
    grouped = grouped[grouped["row_count"] >= min_count].copy()
    grouped["truncated_rate"] = grouped["truncated_count"] / grouped["row_count"]
    grouped["lift_vs_overall"] = grouped["truncated_rate"] / overall_rate
    grouped["expected_truncated_count"] = grouped["row_count"] * overall_rate
    grouped["excess_truncated_count"] = (
        grouped["truncated_count"] - grouped["expected_truncated_count"]
    )
    variance = grouped["row_count"] * overall_rate * (1.0 - overall_rate)
    grouped["binomial_z"] = np.where(
        variance > 0,
        grouped["excess_truncated_count"] / np.sqrt(variance),
        np.nan,
    )
    grouped["row_share"] = grouped["row_count"] / len(df)
    grouped["truncated_share"] = grouped["truncated_count"] / max(1, df["output_truncated"].sum())
    grouped["prompt_tokens_p50"] = grouped["prompt_tokens_p50"].round(2)
    return grouped.sort_values(
        ["lift_vs_overall", "truncated_count"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def chi_square_stats(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    y = df["output_truncated"].astype(int)
    for column in columns:
        table = pd.crosstab(df[column].fillna("<missing>").astype(str), y)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, dof, _expected = chi2_contingency(table)
        n = table.to_numpy().sum()
        denom = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = math.sqrt((chi2 / n) / denom) if denom > 0 and n > 0 else float("nan")
        rows.append(
            {
                "feature": column,
                "levels": int(table.shape[0]),
                "chi2": round(float(chi2), 4),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": round(float(cramers_v), 6),
                "mutual_information": round(
                    float(mutual_info_score(df[column].fillna("<missing>").astype(str), y)),
                    6,
                ),
            }
        )
    return sorted(rows, key=lambda item: item["cramers_v"], reverse=True)


def make_top_level(value: Any, top_values: set[str]) -> str:
    text = str(value) if value is not None else "<missing>"
    return text if text in top_values else "__other__"


def make_auc_scores(df: pd.DataFrame, folds: int) -> list[dict[str, Any]]:
    eval_df = df.copy()
    eval_df["cluster_str"] = eval_df["cluster"].astype(str)
    top_key_families = set(eval_df["key_family"].value_counts().head(50).index.astype(str))
    top_family_sigs = set(
        eval_df["constraint_family_signature"].value_counts().head(120).index.astype(str)
    )
    eval_df["key_family_top"] = eval_df["key_family"].map(
        lambda value: make_top_level(value, top_key_families)
    )
    eval_df["constraint_family_signature_top"] = eval_df["constraint_family_signature"].map(
        lambda value: make_top_level(value, top_family_sigs)
    )
    y = eval_df["output_truncated"].astype(int).to_numpy()

    specs = [
        {
            "name": "cluster_only",
            "numeric": [],
            "categorical": ["cluster_str"],
        },
        {
            "name": "length_and_num_constraints",
            "numeric": ["prompt_tokens", "num_constraints"],
            "categorical": ["length_bin"],
        },
        {
            "name": "metadata_no_cluster",
            "numeric": ["prompt_tokens", "num_constraints"],
            "categorical": [
                "length_bin",
                "source_dataset",
                "constraint_type",
                "constraint_family_signature_top",
                "key_family_top",
            ],
        },
        {
            "name": "cluster_plus_metadata",
            "numeric": ["prompt_tokens", "num_constraints"],
            "categorical": [
                "cluster_str",
                "length_bin",
                "source_dataset",
                "constraint_type",
                "constraint_family_signature_top",
                "key_family_top",
            ],
        },
    ]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores: list[dict[str, Any]] = []
    for spec in specs:
        transformers = []
        if spec["numeric"]:
            transformers.append(("num", StandardScaler(), spec["numeric"]))
        if spec["categorical"]:
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
            transformers.append(("cat", encoder, spec["categorical"]))
        preprocessor = ColumnTransformer(transformers=transformers)
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=600,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        fold_scores = cross_val_score(
            model,
            eval_df[spec["numeric"] + spec["categorical"]],
            y,
            cv=splitter,
            scoring="roc_auc",
            n_jobs=1,
        )
        scores.append(
            {
                "feature_set": spec["name"],
                "roc_auc_mean": round(float(fold_scores.mean()), 6),
                "roc_auc_std": round(float(fold_scores.std()), 6),
                "cv_folds": int(folds),
            }
        )
    return scores


def top_values(series: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    counts = series.fillna("<missing>").astype(str).value_counts().head(limit)
    return [{"value": str(key), "count": int(value)} for key, value in counts.items()]


def plot_cluster_rates(
    cluster_rates: pd.DataFrame,
    overall_rate: float,
    output_dir: Path,
) -> Path:
    data = cluster_rates.sort_values("truncated_rate", ascending=True).copy()
    data["cluster_label"] = "C" + data["category"].astype(str)
    colors = np.where(data["truncated_rate"] >= overall_rate, COLORS["orange"], COLORS["blue"])
    fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    style_axis(ax)
    ax.barh(
        data["cluster_label"],
        data["truncated_rate"],
        color=colors,
        edgecolor=NEUTRALS["dark"],
        linewidth=0.45,
    )
    ax.axvline(overall_rate, color=COLORS["pink_dark"], linestyle="--", linewidth=1.2)
    ax.text(
        overall_rate + 0.002,
        len(data) - 0.3,
        f"overall {fmt_pct(overall_rate)}",
        color=COLORS["pink_dark"],
        fontsize=9,
        va="top",
    )
    for _, row in data.iterrows():
        ax.text(
            row["truncated_rate"] + 0.002,
            row["cluster_label"],
            f"{row['truncated_rate'] * 100:.1f}% n={int(row['row_count']):,}",
            va="center",
            ha="left",
            fontsize=7.7,
            color=TOKENS["muted"],
        )
    ax.set_xlabel("Truncated rate", color=TOKENS["ink"])
    ax.set_ylabel("Prompt cluster", color=TOKENS["ink"])
    ax.set_xlim(0, min(0.42, max(data["truncated_rate"].max() * 1.2, overall_rate * 1.4)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct_axis))
    add_header(
        fig,
        "Truncated outputs concentrate in specific prompt clusters",
        (
            "All 94,390 prompts under 2,048 input tokens; orange bars are clusters "
            "above the global truncated rate."
        ),
    )
    fig.subplots_adjust(left=0.14, right=0.94, top=0.82, bottom=0.12)
    path = output_dir / "truncated_rate_by_cluster.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_length_constraints(
    length_rates: pd.DataFrame,
    num_constraint_rates: pd.DataFrame,
    overall_rate: float,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax in axes:
        style_axis(ax)

    length_order = ["001-128", "129-256", "257-512", "513-1024", "1025-2047"]
    left = length_rates.set_index("category").reindex(length_order).reset_index()
    axes[0].bar(
        left["category"],
        left["truncated_rate"],
        color=COLORS["blue"],
        edgecolor=COLORS["blue_dark"],
        linewidth=0.5,
    )
    axes[0].axhline(overall_rate, color=COLORS["pink_dark"], linestyle="--", linewidth=1.1)
    axes[0].set_xlabel("Input token bin", color=TOKENS["ink"])
    axes[0].set_ylabel("Truncated rate", color=TOKENS["ink"])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct_axis))

    right = num_constraint_rates.sort_values("category", key=lambda s: s.astype(int))
    axes[1].bar(
        right["category"].astype(str),
        right["truncated_rate"],
        color=COLORS["gold"],
        edgecolor=COLORS["gold_dark"],
        linewidth=0.5,
    )
    axes[1].axhline(overall_rate, color=COLORS["pink_dark"], linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Number of constraints", color=TOKENS["ink"])
    axes[1].set_ylabel("Truncated rate", color=TOKENS["ink"])
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pct_axis))

    add_header(
        fig,
        "Length and constraint count have weaker signal than content cluster",
        (
            "Rates by input-token bin and number of constraints; the dashed line is "
            "the global truncated rate."
        ),
    )
    fig.subplots_adjust(left=0.075, right=0.97, top=0.81, bottom=0.19, wspace=0.26)
    path = output_dir / "truncated_rate_by_length_and_constraints.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_cluster_length_heatmap(
    df: pd.DataFrame,
    cluster_rates: pd.DataFrame,
    output_dir: Path,
) -> Path:
    ordered_clusters = cluster_rates.sort_values(
        "truncated_rate",
        ascending=False,
    )["category"].tolist()
    length_order = ["001-128", "129-256", "257-512", "513-1024", "1025-2047"]
    table = pd.pivot_table(
        df,
        values="output_truncated",
        index="cluster",
        columns="length_bin",
        aggfunc="mean",
    )
    counts = pd.pivot_table(
        df,
        values="output_truncated",
        index="cluster",
        columns="length_bin",
        aggfunc="count",
    )
    table = table.reindex(index=[int(x) for x in ordered_clusters], columns=length_order)
    counts = counts.reindex(index=[int(x) for x in ordered_clusters], columns=length_order)
    table = table.where(counts >= 30)
    annot = table.map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(10.8, 9.3), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])
    sns.heatmap(
        table,
        ax=ax,
        cmap=sns.light_palette(COLORS["orange_dark"], as_cmap=True),
        annot=annot,
        fmt="",
        linewidths=0.35,
        linecolor=TOKENS["grid"],
        cbar_kws={"label": "truncated rate"},
        vmin=0,
        vmax=max(0.25, float(table.max().max())),
    )
    ax.set_xlabel("Input token bin", color=TOKENS["ink"])
    ax.set_ylabel("Cluster ordered by truncated rate", color=TOKENS["ink"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
    add_header(
        fig,
        "High-truncation clusters remain visible across input-length bins",
        (
            "Cell values are truncated rates for each cluster and prompt-length bin; "
            "blank cells have fewer than 30 prompts."
        ),
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.13)
    path = output_dir / "truncated_rate_cluster_length_heatmap.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_top_instruction_lifts(
    instruction_rates: pd.DataFrame,
    output_dir: Path,
    limit: int,
) -> Path:
    data = instruction_rates[instruction_rates["row_count"] >= 200].copy()
    data = data.sort_values("lift_vs_overall", ascending=False).head(limit)
    data = data.sort_values("lift_vs_overall", ascending=True)
    fig, ax = plt.subplots(figsize=(12.5, 7.8), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    style_axis(ax)
    ax.barh(
        data["category"],
        data["lift_vs_overall"],
        color=COLORS["gold"],
        edgecolor=COLORS["gold_dark"],
        linewidth=0.55,
    )
    ax.axvline(1.0, color=COLORS["pink_dark"], linestyle="--", linewidth=1.1)
    for _, row in data.iterrows():
        ax.text(
            row["lift_vs_overall"] + 0.02,
            row["category"],
            f"{row['truncated_rate'] * 100:.1f}% n={int(row['row_count']):,}",
            va="center",
            fontsize=8,
            color=TOKENS["muted"],
        )
    ax.set_xlabel("Lift versus global truncated rate", color=TOKENS["ink"])
    ax.set_ylabel("Atomic instruction id", color=TOKENS["ink"])
    ax.set_xlim(0, max(2.5, data["lift_vs_overall"].max() * 1.18))
    add_header(
        fig,
        "Certain atomic constraints are much more likely to hit the output cap",
        (
            "Top frequent instruction ids by truncation lift; labels show truncated "
            "rate and number of prompts containing the instruction."
        ),
    )
    fig.subplots_adjust(left=0.34, right=0.94, top=0.82, bottom=0.13)
    path = output_dir / "top_instruction_truncation_lift.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_auc_scores(auc_scores: list[dict[str, Any]], output_dir: Path) -> Path:
    data = pd.DataFrame(auc_scores).sort_values("roc_auc_mean", ascending=True)
    labels = {
        "cluster_only": "Cluster only",
        "length_and_num_constraints": "Length + # constraints",
        "metadata_no_cluster": "Metadata, no cluster",
        "cluster_plus_metadata": "Cluster + metadata",
    }
    data["label"] = data["feature_set"].map(labels).fillna(data["feature_set"])
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    style_axis(ax)
    ax.barh(
        data["label"],
        data["roc_auc_mean"],
        xerr=data["roc_auc_std"],
        color=COLORS["blue"],
        edgecolor=COLORS["blue_dark"],
        linewidth=0.55,
        capsize=3,
    )
    ax.axvline(0.5, color=NEUTRALS["dark"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cross-validated ROC AUC", color=TOKENS["ink"])
    ax.set_ylabel("")
    ax.set_xlim(0.45, max(0.8, data["roc_auc_mean"].max() * 1.08))
    for _, row in data.iterrows():
        ax.text(
            row["roc_auc_mean"] + 0.008,
            row["label"],
            f"{row['roc_auc_mean']:.3f}",
            va="center",
            fontsize=9,
            color=TOKENS["muted"],
        )
    add_header(
        fig,
        "Simple metadata can predict truncation better than chance",
        "Five-fold ROC AUC using prompt-side features only; 0.5 is random guessing.",
    )
    fig.subplots_adjust(left=0.24, right=0.93, top=0.79, bottom=0.16)
    path = output_dir / "truncation_predictability_auc.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    cluster_rates: pd.DataFrame,
    instruction_rates: pd.DataFrame,
    category_tables: dict[str, pd.DataFrame],
) -> None:
    top_clusters = cluster_rates.sort_values("lift_vs_overall", ascending=False).head(8)
    top_instr = instruction_rates.sort_values("lift_vs_overall", ascending=False).head(10)
    top_family = category_tables["constraint_family_signature"].sort_values(
        "lift_vs_overall", ascending=False
    ).head(8)
    lines = [
        "# Truncated Cluster Analysis",
        "",
        "## Summary",
        "",
        f"- Rows analyzed: {summary['row_count']:,}",
        f"- Truncated rows: {summary['truncated_count']:,} ({summary['truncated_rate_pct']:.2f}%)",
        f"- Cluster Cramer's V: {summary['chi_square']['cluster']['cramers_v']:.4f}",
        f"- Cluster-only ROC AUC: {summary['auc_scores']['cluster_only']:.4f}",
        f"- Cluster + metadata ROC AUC: {summary['auc_scores']['cluster_plus_metadata']:.4f}",
        "",
        "## Top High-Lift Clusters",
        "",
        "| cluster | rows | truncated | rate | lift | top family signatures |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top_clusters.iterrows():
        top_family_text = str(row.get("top_family_signatures", ""))[:180].replace("\n", " ")
        lines.append(
            "| "
            f"{row['category']} | {int(row['row_count']):,} | "
            f"{int(row['truncated_count']):,} | {row['truncated_rate'] * 100:.2f}% | "
            f"{row['lift_vs_overall']:.2f} | {top_family_text} |"
        )
    lines.extend(
        [
            "",
            "## Top High-Lift Atomic Instructions",
            "",
            "| instruction_id | rows | truncated | rate | lift |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_instr.iterrows():
        lines.append(
            "| "
            f"{row['category']} | {int(row['row_count']):,} | "
            f"{int(row['truncated_count']):,} | {row['truncated_rate'] * 100:.2f}% | "
            f"{row['lift_vs_overall']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Top High-Lift Constraint Family Signatures",
            "",
            "| family signature | rows | truncated | rate | lift |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_family.iterrows():
        lines.append(
            "| "
            f"{row['category']} | {int(row['row_count']):,} | "
            f"{int(row['truncated_count']):,} | {row['truncated_rate'] * 100:.2f}% | "
            f"{row['lift_vs_overall']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            *[f"- {name}: `{plot_path}`" for name, plot_path in summary["plots"].items()],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_root = resolve_path(args.generation_root)
    outputs = read_generation_outputs(generation_root, args.generation_dirs)
    clusters = pd.read_parquet(resolve_path(args.clusters_file))
    cluster_summary = pd.read_csv(resolve_path(args.cluster_summary_csv))

    merged = outputs.merge(
        clusters[["prompt_id", "cluster", "length_bin"]],
        on="prompt_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_cluster"),
    )
    if merged["cluster"].isna().any():
        missing = int(merged["cluster"].isna().sum())
        raise ValueError(f"{missing} generated rows did not match cluster assignments")

    merged["cluster"] = merged["cluster"].astype(int)
    merged["length_bin"] = merged["length_bin"].astype(str)
    merged["key_family"] = merged["base_key"].map(normalize_key_family)
    merged["cluster_str"] = merged["cluster"].astype(str)

    if not merged["ok"].all():
        raise ValueError("Expected all generation rows to be ok for this analysis")
    if merged["context_window_error"].any():
        raise ValueError("Context window errors exist; this analysis assumes successful rows")

    row_count = int(len(merged))
    truncated_count = int(merged["output_truncated"].sum())
    overall_rate = truncated_count / row_count

    merged.to_parquet(output_dir / "truncated_cluster_merged.parquet", index=False)

    cluster_rates = summarize_category(merged, "cluster_str", overall_rate, min_count=1)
    cluster_detail_columns = [
        "cluster",
        "top_family_signatures",
        "top_constraint_signatures",
        "length_bin_counts",
    ]
    cluster_details = (
        cluster_summary[cluster_detail_columns]
        .rename(columns={"cluster": "category"})
        .copy()
    )
    cluster_details["category"] = cluster_details["category"].astype(str)
    cluster_rates = cluster_rates.merge(
        cluster_details,
        on="category",
        how="left",
    )
    cluster_rates.to_csv(output_dir / "truncated_rate_by_cluster.csv", index=False)

    category_columns = [
        "length_bin",
        "num_constraints",
        "source_dataset",
        "constraint_type",
        "constraint_family_signature",
        "key_family",
    ]
    category_tables = summarize_feature_table(
        merged,
        category_columns,
        overall_rate,
        args.min_category_count,
    )
    for column, table in category_tables.items():
        table.to_csv(output_dir / f"truncated_rate_by_{column}.csv", index=False)

    instruction_rates = summarize_instruction_ids(
        merged,
        overall_rate,
        args.min_category_count,
    )
    instruction_rates.to_csv(output_dir / "truncated_rate_by_instruction_id.csv", index=False)

    chi_columns = ["cluster_str", *category_columns]
    chi_rows = chi_square_stats(merged, chi_columns)
    chi_lookup = {row["feature"].replace("cluster_str", "cluster"): row for row in chi_rows}

    auc_rows = make_auc_scores(merged, args.cv_folds)
    auc_lookup = {row["feature_set"]: row["roc_auc_mean"] for row in auc_rows}

    plots = {
        "cluster_rates": str(plot_cluster_rates(cluster_rates, overall_rate, output_dir)),
        "length_constraints": str(
            plot_length_constraints(
                category_tables["length_bin"],
                category_tables["num_constraints"],
                overall_rate,
                output_dir,
            )
        ),
        "cluster_length_heatmap": str(
            plot_cluster_length_heatmap(merged, cluster_rates, output_dir)
        ),
        "top_instruction_lift": str(
            plot_top_instruction_lifts(instruction_rates, output_dir, args.top_feature_count)
        ),
        "predictability_auc": str(plot_auc_scores(auc_rows, output_dir)),
    }

    top_cluster_table = cluster_rates.sort_values(
        ["lift_vs_overall", "truncated_count"],
        ascending=[False, False],
    ).head(10)
    low_cluster_table = cluster_rates.sort_values(
        ["lift_vs_overall", "truncated_count"],
        ascending=[True, False],
    ).head(10)
    top_excess_clusters = cluster_rates.sort_values(
        ["excess_truncated_count"],
        ascending=False,
    ).head(10)
    summary = {
        "generation_root": str(generation_root),
        "generation_dirs": args.generation_dirs,
        "clusters_file": str(resolve_path(args.clusters_file)),
        "row_count": row_count,
        "truncated_count": truncated_count,
        "truncated_rate": round(overall_rate, 8),
        "truncated_rate_pct": round(overall_rate * 100, 4),
        "finish_reason_counts": {
            str(key): int(value)
            for key, value in merged["finish_reason"].value_counts().to_dict().items()
        },
        "cluster_concentration": {
            "top_3_cluster_truncated_share": round(
                float(
                    cluster_rates.sort_values("truncated_count", ascending=False)
                    .head(3)["truncated_count"]
                    .sum()
                    / truncated_count
                ),
                6,
            ),
            "top_3_cluster_row_share": round(
                float(
                    cluster_rates.sort_values("truncated_count", ascending=False)
                    .head(3)["row_count"]
                    .sum()
                    / row_count
                ),
                6,
            ),
            "top_5_cluster_truncated_share": round(
                float(
                    cluster_rates.sort_values("truncated_count", ascending=False)
                    .head(5)["truncated_count"]
                    .sum()
                    / truncated_count
                ),
                6,
            ),
            "top_5_cluster_row_share": round(
                float(
                    cluster_rates.sort_values("truncated_count", ascending=False)
                    .head(5)["row_count"]
                    .sum()
                    / row_count
                ),
                6,
            ),
        },
        "top_high_lift_clusters": top_cluster_table[
            [
                "category",
                "row_count",
                "truncated_count",
                "truncated_rate",
                "lift_vs_overall",
                "binomial_z",
                "top_family_signatures",
            ]
        ].to_dict("records"),
        "top_low_lift_clusters": low_cluster_table[
            ["category", "row_count", "truncated_count", "truncated_rate", "lift_vs_overall"]
        ].to_dict("records"),
        "top_excess_clusters": top_excess_clusters[
            [
                "category",
                "row_count",
                "truncated_count",
                "expected_truncated_count",
                "excess_truncated_count",
                "truncated_rate",
                "lift_vs_overall",
            ]
        ].to_dict("records"),
        "top_instruction_ids_by_lift": instruction_rates.head(15).to_dict("records"),
        "top_constraint_family_signatures_by_lift": category_tables[
            "constraint_family_signature"
        ].head(15).to_dict("records"),
        "length_bin_rates": category_tables["length_bin"].sort_values("category").to_dict(
            "records"
        ),
        "num_constraint_rates": category_tables["num_constraints"].sort_values(
            "category", key=lambda s: s.astype(int)
        ).to_dict("records"),
        "source_dataset_rates": category_tables["source_dataset"].to_dict("records"),
        "constraint_type_rates": category_tables["constraint_type"].to_dict("records"),
        "chi_square": chi_lookup,
        "auc_scores_raw": auc_rows,
        "auc_scores": auc_lookup,
        "plots": plots,
        "csv_outputs": {
            "merged": str(output_dir / "truncated_cluster_merged.parquet"),
            "cluster": str(output_dir / "truncated_rate_by_cluster.csv"),
            "instruction_id": str(output_dir / "truncated_rate_by_instruction_id.csv"),
            **{
                column: str(output_dir / f"truncated_rate_by_{column}.csv")
                for column in category_columns
            },
        },
        "top_values": {
            "source_dataset": top_values(merged["source_dataset"]),
            "constraint_type": top_values(merged["constraint_type"]),
            "key_family": top_values(merged["key_family"]),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(
        output_dir / "truncated_cluster_analysis.md",
        summary,
        cluster_rates,
        instruction_rates,
        category_tables,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
