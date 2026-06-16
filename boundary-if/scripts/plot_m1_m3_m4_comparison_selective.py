from __future__ import annotations

import argparse
import json
import math
import textwrap
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import torch

from boundary_if.models.m4_pretrained_encoder import (
    M4EncoderConfig,
    M4FrozenEncoderClassifier,
    make_m4_classifier_dataloader,
)
from boundary_if.models.tiny_transformer import (
    M3TinyTransformer,
    M3TinyTransformerConfig,
    make_m3_dataloader,
)

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_M1_SUMMARY = (
    "runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/"
    "learning_curve_summary.csv"
)
DEFAULT_M3_SUMMARY = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_summary.csv"
)
DEFAULT_M3_AGGREGATE = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_aggregate.csv"
)
DEFAULT_M4_SUMMARY = (
    "runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_summary.csv"
)
DEFAULT_M4_AGGREGATE = (
    "runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
    "multiseed_learning_curve_aggregate.csv"
)
DEFAULT_OUTPUT_DIR = "runs/model_comparisons/m1_m3_m4_selective_prediction"

REQUIRED_COLUMNS = [
    "prompt_id",
    "split",
    "label",
    "input_ids",
    "raw_prompt_bpe_token_count_full",
    "raw_prompt_bpe_token_count",
    "raw_prompt_bpe_truncated",
    "num_constraints",
    "cluster",
    "length_bin",
]
COMMON_LABELS = ["2k", "4k", "5k", "10k", "20k", "40k", "full"]
ORDER_MAP = {label: index for index, label in enumerate(COMMON_LABELS)}
COVERAGES = [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]
TOP_K_VALUES = [100, 500, 1000, 2000]

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "Consolas", "Menlo", "monospace"]
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
MODEL_STYLE = {
    "M1 TF-IDF": {"family": COLOR_FAMILIES["orange"], "marker": "o", "linestyle": "-"},
    "M3 mean": {"family": COLOR_FAMILIES["blue"], "marker": "s", "linestyle": "-"},
    "M4 frozen encoder": {"family": COLOR_FAMILIES["pink"], "marker": "^", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare M1/M3/M4 learning curves and compute M3/M4 selective "
            "prediction plus top-k precision on the shared atomic held-out test split."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--m1-summary", default=DEFAULT_M1_SUMMARY)
    parser.add_argument("--m3-summary", default=DEFAULT_M3_SUMMARY)
    parser.add_argument("--m3-aggregate", default=DEFAULT_M3_AGGREGATE)
    parser.add_argument("--m4-summary", default=DEFAULT_M4_SUMMARY)
    parser.add_argument("--m4-aggregate", default=DEFAULT_M4_AGGREGATE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-prediction-refresh", action="store_true")
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def resolve_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def localize_workspace_path(path: str | Path) -> Path:
    raw = str(path)
    if raw.startswith("/workspace/"):
        return Path.cwd() / raw.removeprefix("/workspace/")
    return resolve_path(raw)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(raw_device)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def add_chart_header(
    fig,
    ax,
    title: str,
    subtitle: str,
    *,
    title_width: int = 110,
    subtitle_width: int = 145,
) -> None:
    title = textwrap.fill(title.strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=subtitle_width, break_long_words=False)
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
        0.952,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.18,
    )


def normalize_curve_label(value: Any) -> str:
    label = str(value)
    if label in {"80k", "80k/full", "train_80k"}:
        return "full"
    return label.replace("train_", "")


def read_m1_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["curve_label"] = frame["curve_label"].map(normalize_curve_label)
    frame = frame[frame["curve_label"].isin(COMMON_LABELS)].copy()
    output = pd.DataFrame(
        {
            "model": "M1 TF-IDF",
            "curve_label": frame["curve_label"],
            "order": frame["curve_label"].map(ORDER_MAP).astype(int),
            "run_count": 1,
            "seeds": "42",
            "actual_train_rows_mean": frame["actual_train_rows"].astype(float),
            "val_positive_rate": frame["val_positive_rate"].astype(float),
            "test_positive_rate": frame["test_positive_rate"].astype(float),
            "val_auroc_mean": frame["val_auroc"].astype(float),
            "val_auroc_std": 0.0,
            "val_auprc_mean": frame["val_auprc"].astype(float),
            "val_auprc_std": 0.0,
            "test_auroc_mean": frame["test_auroc"].astype(float),
            "test_auroc_std": 0.0,
            "test_auprc_mean": frame["test_auprc"].astype(float),
            "test_auprc_std": 0.0,
            "source_file": str(path),
        }
    )
    return output


def read_aggregate_frame(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["curve_label"] = frame["curve_label"].map(normalize_curve_label)
    frame = frame[frame["curve_label"].isin(COMMON_LABELS)].copy()
    output = frame[
        [
            "curve_label",
            "order",
            "run_count",
            "seeds",
            "actual_train_rows_mean",
            "val_positive_rate",
            "test_positive_rate",
            "val_auroc_mean",
            "val_auroc_std",
            "val_auprc_mean",
            "val_auprc_std",
            "test_auroc_mean",
            "test_auroc_std",
            "test_auprc_mean",
            "test_auprc_std",
        ]
    ].copy()
    output["model"] = model_name
    output["source_file"] = str(path)
    return output


def build_learning_curve_comparison(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    frames = [
        read_m1_frame(resolve_path(args.m1_summary)),
        read_aggregate_frame(resolve_path(args.m3_aggregate), "M3 mean"),
        read_aggregate_frame(resolve_path(args.m4_aggregate), "M4 frozen encoder"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["order"] = combined["curve_label"].map(ORDER_MAP).astype(int)
    combined = combined.sort_values(["model", "order"], kind="mergesort")
    combined.to_csv(output_dir / "m1_m3_m4_learning_curve_common.csv", index=False, encoding="utf-8")
    write_json(
        combined.to_dict("records"),
        output_dir / "m1_m3_m4_learning_curve_common.json",
    )
    return combined


def draw_metric_line_panel(
    ax,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    baseline: float | None = None,
) -> None:
    for model_name, part in frame.groupby("model", sort=False):
        part = part.sort_values("order")
        style = MODEL_STYLE[model_name]
        family = style["family"]
        x_values = part["order"].astype(int).to_numpy()
        y_values = part[metric].astype(float).to_numpy()
        ax.plot(
            x_values,
            y_values,
            color=family["mid"],
            marker=style["marker"],
            markerfacecolor=family["base"],
            markeredgecolor=family["dark"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            markersize=6,
            label=model_name,
            zorder=4,
        )
        std_col = f"{metric.removesuffix('_mean')}_std"
        if std_col in part.columns:
            yerr = part[std_col].fillna(0).astype(float).to_numpy()
            if (yerr > 0).any():
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=yerr,
                    fmt="none",
                    ecolor=family["dark"],
                    elinewidth=1.0,
                    capsize=3,
                    zorder=3,
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
    ax.set_xticks(range(len(COMMON_LABELS)), COMMON_LABELS)
    ax.set_xlabel("Train sample size")
    ax.set_ylabel(title)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_learning_curve_comparison(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    use_chart_theme()
    val_baseline = float(frame["val_positive_rate"].dropna().iloc[0])
    test_baseline = float(frame["test_positive_rate"].dropna().iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    draw_metric_line_panel(axes[0, 0], frame, metric="val_auroc_mean", title="Validation AUROC")
    draw_metric_line_panel(
        axes[0, 1],
        frame,
        metric="val_auprc_mean",
        title="Validation AUPRC",
        baseline=val_baseline,
    )
    draw_metric_line_panel(axes[1, 0], frame, metric="test_auroc_mean", title="Test AUROC")
    draw_metric_line_panel(
        axes[1, 1],
        frame,
        metric="test_auprc_mean",
        title="Test AUPRC",
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
        ncol=4,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.985, bottom=0.095, hspace=0.32, wspace=0.18)
    add_chart_header(
        fig,
        axes[0, 0],
        "M1, M3, and M4 learning curves on the shared atomic held-out split",
        (
            "All points use the max2048 tokenized prompt set and full validation/test splits. "
            "M3 and M4 show mean+-std over train seeds 42/43/44; M1 is the seed42 TF-IDF baseline."
        ),
    )
    png_path = output_dir / "m1_m3_m4_learning_curve_metrics.png"
    svg_path = output_dir / "m1_m3_m4_learning_curve_metrics.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_best_model_summary(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    best_rows = (
        frame.sort_values(["model", "test_auprc_mean"], ascending=[True, False], kind="mergesort")
        .groupby("model", as_index=False)
        .head(1)
        .sort_values("test_auprc_mean", ascending=True)
        .copy()
    )
    best_rows.to_csv(output_dir / "m1_m3_m4_best_by_test_auprc.csv", index=False, encoding="utf-8")
    use_chart_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    y_labels = list(best_rows["model"] + " " + best_rows["curve_label"])
    y_positions = range(len(best_rows))
    for ax, metric, title in [
        (axes[0], "test_auroc_mean", "Best-config test AUROC"),
        (axes[1], "test_auprc_mean", "Best-config test AUPRC"),
    ]:
        for y, (_, row) in zip(y_positions, best_rows.iterrows(), strict=False):
            family = MODEL_STYLE[row["model"]]["family"]
            x = float(row[metric])
            std = float(row[metric.replace("_mean", "_std")])
            ax.barh(
                y,
                x,
                color=family["base"],
                edgecolor=family["dark"],
                linewidth=1.0,
                height=0.52,
            )
            if std > 0:
                ax.errorbar(x, y, xerr=std, fmt="none", ecolor=family["dark"], capsize=3)
            ax.text(
                min(x + 0.015, 0.94),
                y,
                f"{x:.1%}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=TOKENS["ink"],
                fontfamily=MONO_FONT_FAMILY[0],
            )
        ax.set_title(title, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
        ax.set_xlim(0.2, 0.95)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.tick_params(axis="x", colors=TOKENS["muted"], labelsize=8.5)
        ax.set_xlabel(title)
    axes[0].set_yticks(list(y_positions), y_labels)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(top=0.77, left=0.22, right=0.985, bottom=0.16, wspace=0.14)
    add_chart_header(
        fig,
        axes[0],
        "Best test AUPRC configuration by model family",
        "Bars show the best test-AUPRC train-size configuration within the shared M1/M3/M4 learning-curve comparison.",
    )
    png_path = output_dir / "m1_m3_m4_best_test_metrics.png"
    svg_path = output_dir / "m1_m3_m4_best_test_metrics.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def load_test_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    test_frame = frame[frame["split"].astype(str).eq("test")].copy()
    if test_frame.empty:
        raise ValueError(f"No test rows found in {path}.")
    return test_frame


def dataclass_config(raw_config: dict[str, Any], config_cls: type) -> Any:
    allowed = {field.name for field in fields(config_cls)}
    return config_cls(**{key: value for key, value in raw_config.items() if key in allowed})


def load_m3_model(model_path: Path, device: torch.device) -> tuple[M3TinyTransformer, M3TinyTransformerConfig]:
    checkpoint = torch_load(model_path)
    config = dataclass_config(checkpoint["config"], M3TinyTransformerConfig)
    model = M3TinyTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def load_m4_model(model_path: Path, device: torch.device) -> tuple[M4FrozenEncoderClassifier, M4EncoderConfig]:
    checkpoint = torch_load(model_path)
    config = dataclass_config(checkpoint["config"], M4EncoderConfig)
    model = M4FrozenEncoderClassifier(config)
    model.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
    model.pass_head.load_state_dict(checkpoint["pass_head_state_dict"])
    model.freeze_encoder()
    model.to(device)
    model.eval()
    return model, config


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    use_amp: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            probabilities = model.predict_pass_probability(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        probs = probabilities.detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy().astype(int)
        metadata = batch["metadata"]
        for row_index, probability in enumerate(probs):
            rows.append(
                {
                    "prompt_id": metadata["prompt_id"][row_index],
                    "split": metadata["split"][row_index],
                    "label": int(labels[row_index]),
                    "pred_proba": float(probability),
                }
            )
    return pd.DataFrame(rows)


def run_model_predictions(
    model_name: str,
    model_path: Path,
    test_frame: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> pd.DataFrame:
    if model_name == "M3 mean":
        model, config = load_m3_model(model_path, device)
        dataloader = make_m3_dataloader(
            test_frame,
            config,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    elif model_name == "M4 frozen encoder":
        model, config = load_m4_model(model_path, device)
        dataloader = make_m4_classifier_dataloader(
            test_frame,
            config,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}")
    try:
        return predict_probabilities(model, dataloader, device=device, use_amp=use_amp)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def collect_run_specs(summary_path: Path, model_name: str) -> list[dict[str, Any]]:
    summary = pd.read_csv(summary_path)
    summary["curve_label"] = summary["curve_label"].map(normalize_curve_label)
    summary = summary[summary["curve_label"].isin(COMMON_LABELS)].copy()
    summary["order"] = summary["curve_label"].map(ORDER_MAP).astype(int)
    specs: list[dict[str, Any]] = []
    for _, row in summary.sort_values(["train_seed", "order"], kind="mergesort").iterrows():
        output_dir = localize_workspace_path(row["output_dir"])
        model_path = output_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
        specs.append(
            {
                "model": model_name,
                "curve_label": str(row["curve_label"]),
                "order": int(ORDER_MAP[str(row["curve_label"])]),
                "train_seed": int(row["train_seed"]),
                "sample_seed": int(row["sample_seed"]),
                "actual_train_rows": int(row["actual_train_rows"]),
                "model_path": str(model_path),
            }
        )
    return specs


def compute_selective_metrics(
    predictions: pd.DataFrame,
    *,
    coverages: list[float],
) -> list[dict[str, Any]]:
    labels = predictions["label"].astype(int).to_numpy()
    probabilities = predictions["pred_proba"].astype(float).to_numpy()
    pred_labels = (probabilities >= 0.5).astype(int)
    confidence = abs(probabilities - 0.5)
    order = confidence.argsort()[::-1]
    rows: list[dict[str, Any]] = []
    for coverage in coverages:
        selected_count = max(1, min(len(order), math.ceil(len(order) * coverage)))
        selected = order[:selected_count]
        selected_pred_positive = pred_labels[selected] == 1
        positive_pred_count = int(selected_pred_positive.sum())
        selected_precision = (
            float(labels[selected][selected_pred_positive].mean())
            if positive_pred_count
            else float("nan")
        )
        accuracy = float((pred_labels[selected] == labels[selected]).mean())
        rows.append(
            {
                "coverage": coverage,
                "selected_count": selected_count,
                "selective_accuracy": accuracy,
                "selective_error_rate": 1.0 - accuracy,
                "selected_positive_prediction_count": positive_pred_count,
                "selected_positive_precision": selected_precision,
                "mean_confidence": float(confidence[selected].mean()),
            }
        )
    return rows


def compute_topk_metrics(
    predictions: pd.DataFrame,
    *,
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    labels = predictions["label"].astype(int).to_numpy()
    probabilities = predictions["pred_proba"].astype(float).to_numpy()
    order = probabilities.argsort()[::-1]
    rows: list[dict[str, Any]] = []
    for top_k in top_k_values:
        selected_count = max(1, min(len(order), top_k))
        selected = order[:selected_count]
        rows.append(
            {
                "top_k": selected_count,
                "top_fraction": selected_count / len(order),
                "precision_at_k": float(labels[selected].mean()),
                "positive_count_at_k": int(labels[selected].sum()),
                "mean_score_at_k": float(probabilities[selected].mean()),
            }
        )
    return rows


def summarize_predictions(
    run_specs: list[dict[str, Any]],
    test_frame: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(run_specs, start=1):
        start_time = now()
        print(
            json.dumps(
                {
                    "event": "predict_test",
                    "index": index,
                    "total": len(run_specs),
                    "model": spec["model"],
                    "curve_label": spec["curve_label"],
                    "train_seed": spec["train_seed"],
                    "model_path": spec["model_path"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        predictions = run_model_predictions(
            spec["model"],
            Path(spec["model_path"]),
            test_frame,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            use_amp=use_amp,
        )
        labels = predictions["label"].astype(int)
        probabilities = predictions["pred_proba"].astype(float)
        pred_labels = (probabilities >= 0.5).astype(int)
        run_record = {
            **spec,
            "row_count": int(len(predictions)),
            "positive_rate": float(labels.mean()),
            "accuracy": float((pred_labels == labels).mean()),
            "mean_score": float(probabilities.mean()),
            "predict_seconds": elapsed_since(start_time),
        }
        run_rows.append(run_record)
        for row in compute_selective_metrics(predictions, coverages=COVERAGES):
            selective_rows.append({**spec, **row})
        for row in compute_topk_metrics(predictions, top_k_values=TOP_K_VALUES):
            topk_rows.append({**spec, **row})

    run_frame = pd.DataFrame(run_rows)
    selective_frame = pd.DataFrame(selective_rows)
    topk_frame = pd.DataFrame(topk_rows)
    run_frame.to_csv(output_dir / "m3_m4_test_prediction_run_summary.csv", index=False, encoding="utf-8")
    selective_frame.to_csv(output_dir / "m3_m4_selective_prediction_by_run.csv", index=False, encoding="utf-8")
    topk_frame.to_csv(output_dir / "m3_m4_topk_precision_by_run.csv", index=False, encoding="utf-8")
    return run_frame, selective_frame, topk_frame


def aggregate_metric(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)
    rows: list[dict[str, Any]] = []
    for group_key, part in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key, strict=True))
        row["run_count"] = int(len(part))
        row["seeds"] = ",".join(str(seed) for seed in sorted(part["train_seed"].unique()))
        for column in value_columns:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{column}_min"] = float(part[column].min())
            row[f"{column}_max"] = float(part[column].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [column for column in ["model", "order", "coverage", "top_k"] if column in group_columns],
        kind="mergesort",
    )


def build_or_load_selective_outputs(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selective_aggregate_path = output_dir / "m3_m4_selective_prediction_aggregate.csv"
    topk_aggregate_path = output_dir / "m3_m4_topk_precision_aggregate.csv"
    if args.skip_prediction_refresh:
        if not selective_aggregate_path.exists() or not topk_aggregate_path.exists():
            raise FileNotFoundError("Selective/top-k aggregate files are missing; cannot skip refresh.")
        return pd.read_csv(selective_aggregate_path), pd.read_csv(topk_aggregate_path)

    device = resolve_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    test_frame = load_test_frame(resolve_path(args.input_file))
    run_specs = [
        *collect_run_specs(resolve_path(args.m3_summary), "M3 mean"),
        *collect_run_specs(resolve_path(args.m4_summary), "M4 frozen encoder"),
    ]
    run_frame, selective_frame, topk_frame = summarize_predictions(
        run_specs,
        test_frame,
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        output_dir=output_dir,
    )
    selective_aggregate = aggregate_metric(
        selective_frame,
        group_columns=["model", "curve_label", "order", "coverage"],
        value_columns=[
            "selective_accuracy",
            "selective_error_rate",
            "selected_positive_precision",
            "mean_confidence",
        ],
    )
    topk_aggregate = aggregate_metric(
        topk_frame,
        group_columns=["model", "curve_label", "order", "top_k", "top_fraction"],
        value_columns=["precision_at_k", "positive_count_at_k", "mean_score_at_k"],
    )
    selective_aggregate.to_csv(selective_aggregate_path, index=False, encoding="utf-8")
    topk_aggregate.to_csv(topk_aggregate_path, index=False, encoding="utf-8")
    write_json(
        {
            "device": str(device),
            "use_amp": use_amp,
            "eval_batch_size": args.eval_batch_size,
            "test_rows": int(len(test_frame)),
            "run_count": int(len(run_frame)),
        },
        output_dir / "m3_m4_prediction_refresh_manifest.json",
    )
    return selective_aggregate, topk_aggregate


def palette_for_curve_labels() -> dict[str, dict[str, str]]:
    roots = ["gold", "orange", "olive", "blue", "pink", "neutral", "gold"]
    return {
        label: COLOR_FAMILIES[roots[index]]
        for index, label in enumerate(COMMON_LABELS)
    }


def plot_selective_curves(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    use_chart_theme()
    curve_palette = palette_for_curve_labels()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, model_name in zip(axes, ["M3 mean", "M4 frozen encoder"], strict=True):
        part_model = frame[frame["model"].eq(model_name)].copy()
        for label in COMMON_LABELS:
            part = part_model[part_model["curve_label"].eq(label)].sort_values("coverage")
            if part.empty:
                continue
            family = curve_palette[label]
            ax.plot(
                part["coverage"],
                part["selective_accuracy_mean"],
                color=family["mid"],
                marker="o",
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                linewidth=1.2,
                markersize=4.8,
                label=label,
            )
        ax.set_title(model_name, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
        ax.set_xlabel("Accepted coverage")
        ax.set_ylabel("Selective accuracy")
        ax.set_ylim(0.45, 1.0)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=4,
        fontsize=8.5,
        title="Train size",
        title_fontsize=8.5,
    )
    fig.subplots_adjust(top=0.77, left=0.08, right=0.985, bottom=0.14, wspace=0.12)
    add_chart_header(
        fig,
        axes[0],
        "Selective prediction curves for M3 and M4",
        (
            "Test examples are accepted from highest to lowest confidence |P(pass)-0.5|. "
            "Lines show seed-averaged accuracy at each coverage level for every train-size configuration."
        ),
    )
    png_path = output_dir / "m3_m4_selective_accuracy_curves.png"
    svg_path = output_dir / "m3_m4_selective_accuracy_curves.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_selective_at_coverage(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    use_chart_theme()
    selected_coverages = [0.10, 0.20, 0.50, 1.00]
    coverage_style = {
        0.10: {"family": COLOR_FAMILIES["gold"], "marker": "o"},
        0.20: {"family": COLOR_FAMILIES["olive"], "marker": "s"},
        0.50: {"family": COLOR_FAMILIES["blue"], "marker": "^"},
        1.00: {"family": COLOR_FAMILIES["neutral"], "marker": "D"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, model_name in zip(axes, ["M3 mean", "M4 frozen encoder"], strict=True):
        part_model = frame[
            frame["model"].eq(model_name) & frame["coverage"].isin(selected_coverages)
        ].copy()
        for coverage in selected_coverages:
            part = part_model[part_model["coverage"].eq(coverage)].sort_values("order")
            style = coverage_style[coverage]
            family = style["family"]
            ax.plot(
                part["order"],
                part["selective_accuracy_mean"],
                color=family["mid"],
                marker=style["marker"],
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                linewidth=1.2,
                markersize=5,
                label=f"{coverage:.0%} coverage",
            )
            yerr = part["selective_accuracy_std"].fillna(0).to_numpy()
            if (yerr > 0).any():
                ax.errorbar(
                    part["order"],
                    part["selective_accuracy_mean"],
                    yerr=yerr,
                    fmt="none",
                    ecolor=family["dark"],
                    elinewidth=1.0,
                    capsize=3,
                )
        ax.set_title(model_name, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
        ax.set_xticks(range(len(COMMON_LABELS)), COMMON_LABELS)
        ax.set_xlabel("Train sample size")
        ax.set_ylabel("Selective accuracy")
        ax.set_ylim(0.45, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=2,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.77, left=0.08, right=0.985, bottom=0.14, wspace=0.12)
    add_chart_header(
        fig,
        axes[0],
        "Selective accuracy at fixed acceptance rates",
        (
            "Each line fixes the fraction of test rows accepted by confidence. "
            "The 100% line is ordinary full-coverage accuracy."
        ),
    )
    png_path = output_dir / "m3_m4_selective_accuracy_at_coverage.png"
    svg_path = output_dir / "m3_m4_selective_accuracy_at_coverage.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_topk_by_config(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    use_chart_theme()
    k_style = {
        100: {"family": COLOR_FAMILIES["gold"], "marker": "o"},
        500: {"family": COLOR_FAMILIES["olive"], "marker": "s"},
        1000: {"family": COLOR_FAMILIES["blue"], "marker": "^"},
        2000: {"family": COLOR_FAMILIES["pink"], "marker": "D"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, model_name in zip(axes, ["M3 mean", "M4 frozen encoder"], strict=True):
        part_model = frame[frame["model"].eq(model_name)].copy()
        for top_k in TOP_K_VALUES:
            part = part_model[part_model["top_k"].eq(top_k)].sort_values("order")
            style = k_style[top_k]
            family = style["family"]
            ax.plot(
                part["order"],
                part["precision_at_k_mean"],
                color=family["mid"],
                marker=style["marker"],
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                linewidth=1.2,
                markersize=5,
                label=f"top-{top_k:,}",
            )
            yerr = part["precision_at_k_std"].fillna(0).to_numpy()
            if (yerr > 0).any():
                ax.errorbar(
                    part["order"],
                    part["precision_at_k_mean"],
                    yerr=yerr,
                    fmt="none",
                    ecolor=family["dark"],
                    elinewidth=1.0,
                    capsize=3,
                )
        ax.set_title(model_name, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
        ax.set_xticks(range(len(COMMON_LABELS)), COMMON_LABELS)
        ax.set_xlabel("Train sample size")
        ax.set_ylabel("Precision@k")
        ax.set_ylim(0.0, 0.8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=2,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.77, left=0.08, right=0.985, bottom=0.14, wspace=0.12)
    add_chart_header(
        fig,
        axes[0],
        "Top-k precision for predicted pass samples",
        (
            "Rows are ranked by P(pass) on the full atomic held-out test split. "
            "Precision@k is averaged across seeds for each train-size configuration."
        ),
    )
    png_path = output_dir / "m3_m4_topk_precision_by_config.png"
    svg_path = output_dir / "m3_m4_topk_precision_by_config.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_topk_curves(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    use_chart_theme()
    curve_palette = palette_for_curve_labels()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, model_name in zip(axes, ["M3 mean", "M4 frozen encoder"], strict=True):
        part_model = frame[frame["model"].eq(model_name)].copy()
        for label in COMMON_LABELS:
            part = part_model[part_model["curve_label"].eq(label)].sort_values("top_k")
            if part.empty:
                continue
            family = curve_palette[label]
            ax.plot(
                part["top_k"],
                part["precision_at_k_mean"],
                color=family["mid"],
                marker="o",
                markerfacecolor=family["base"],
                markeredgecolor=family["dark"],
                linewidth=1.2,
                markersize=4.8,
                label=label,
            )
        ax.set_title(model_name, loc="left", fontsize=10.5, color=TOKENS["ink"], pad=8)
        ax.set_xlabel("Top k predicted pass rows")
        ax.set_ylabel("Precision@k")
        ax.set_ylim(0.0, 0.8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=8.5)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=4,
        fontsize=8.5,
        title="Train size",
        title_fontsize=8.5,
    )
    fig.subplots_adjust(top=0.77, left=0.08, right=0.985, bottom=0.14, wspace=0.12)
    add_chart_header(
        fig,
        axes[0],
        "Top-k precision curves for M3 and M4",
        (
            "Curves show whether the highest-scored predicted pass rows are enriched for true pass labels. "
            "The test positive-rate baseline is 14.8%."
        ),
    )
    png_path = output_dir / "m3_m4_topk_precision_curves.png"
    svg_path = output_dir / "m3_m4_topk_precision_curves.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_start = now()

    comparison = build_learning_curve_comparison(args, output_dir)
    learning_curve_plot = plot_learning_curve_comparison(comparison, output_dir)
    best_plot = plot_best_model_summary(comparison, output_dir)
    selective_aggregate, topk_aggregate = build_or_load_selective_outputs(args, output_dir)
    selective_curve_plot = plot_selective_curves(selective_aggregate, output_dir)
    selective_coverage_plot = plot_selective_at_coverage(selective_aggregate, output_dir)
    topk_config_plot = plot_topk_by_config(topk_aggregate, output_dir)
    topk_curve_plot = plot_topk_curves(topk_aggregate, output_dir)

    manifest = {
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed_since(total_start),
        "inputs": {
            "input_file": str(resolve_path(args.input_file)),
            "m1_summary": str(resolve_path(args.m1_summary)),
            "m3_summary": str(resolve_path(args.m3_summary)),
            "m3_aggregate": str(resolve_path(args.m3_aggregate)),
            "m4_summary": str(resolve_path(args.m4_summary)),
            "m4_aggregate": str(resolve_path(args.m4_aggregate)),
        },
        "tables": {
            "learning_curve_common": str(output_dir / "m1_m3_m4_learning_curve_common.csv"),
            "best_by_test_auprc": str(output_dir / "m1_m3_m4_best_by_test_auprc.csv"),
            "selective_by_run": str(output_dir / "m3_m4_selective_prediction_by_run.csv"),
            "selective_aggregate": str(output_dir / "m3_m4_selective_prediction_aggregate.csv"),
            "topk_by_run": str(output_dir / "m3_m4_topk_precision_by_run.csv"),
            "topk_aggregate": str(output_dir / "m3_m4_topk_precision_aggregate.csv"),
        },
        "figures": {
            "learning_curve": learning_curve_plot,
            "best_model_summary": best_plot,
            "selective_curves": selective_curve_plot,
            "selective_at_coverage": selective_coverage_plot,
            "topk_by_config": topk_config_plot,
            "topk_curves": topk_curve_plot,
        },
        "definitions": {
            "selective_prediction": (
                "Sort test rows by confidence |P(pass)-0.5| descending and evaluate accuracy "
                "on the accepted top coverage fraction."
            ),
            "top_k_precision": (
                "Sort test rows by P(pass) descending and compute the observed pass-label "
                "rate in the top-k rows."
            ),
        },
    }
    write_json(manifest, output_dir / "m1_m3_m4_selective_prediction_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
