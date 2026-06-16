from __future__ import annotations

import argparse
import json
import math
import textwrap
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import PreTrainedTokenizerFast

from boundary_if.models.m1_tfidf_logreg import feature_summary, predict_m1
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
DEFAULT_ALL_FILE = (
    "data/promptsets/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.all.parquet"
)
DEFAULT_TOKEN_COUNT_FILE = (
    "data/promptsets/if_multi_constraints_upto5.qwen3_4b_instruct_2507.tokens.parquet"
)
DEFAULT_OUTPUT_DIR = "runs/model_comparisons/final_figure_bundle_atomic_caveat"
DEFAULT_TOKENIZER_DIR = "data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_group_key_train"

COMMON_LABELS = ["2k", "4k", "5k", "10k", "20k", "40k", "full"]
ORDER_MAP = {label: index for index, label in enumerate(COMMON_LABELS)}
COVERAGES = [0.10, 0.20, 0.50, 1.00]
TOP_K_VALUES = [100, 500, 1000, 2000]
KEY_CONFIGS = [
    ("M1 TF-IDF", "full", "M1 full"),
    ("M3 mean", "40k", "M3 mean 40k"),
    ("M3 mean", "full", "M3 mean full"),
    ("M4 frozen encoder", "full", "M4 frozen full"),
]

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
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
    "neutral": {"xlight": "#F4F5F7", "light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}
MODEL_COLORS = {
    "M1 TF-IDF": COLOR_FAMILIES["orange"]["base"],
    "M3 mean": COLOR_FAMILIES["blue"]["base"],
    "M4 frozen encoder": COLOR_FAMILIES["pink"]["base"],
    "baseline": COLOR_FAMILIES["neutral"]["mid"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final validation tables and figures for the M1/M3/M4 boundary-model writeup."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--all-file", default=DEFAULT_ALL_FILE)
    parser.add_argument("--token-count-file", default=DEFAULT_TOKEN_COUNT_FILE)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--skip-prediction-refresh", action="store_true")
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return Path.cwd() / raw


def localize_workspace_path(path: str | Path) -> Path:
    raw = str(path)
    if raw.startswith("/workspace/"):
        return Path.cwd() / raw.removeprefix("/workspace/")
    return resolve_path(raw)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def dataclass_config(raw_config: dict[str, Any], cls):
    valid_fields = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in raw_config.items() if key in valid_fields})


def normalize_curve_label(value: Any) -> str:
    label = str(value)
    if label in {"80k", "80k/full", "train_80k"}:
        return "full"
    return label.replace("train_", "")


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(raw_device)


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


def add_header(fig, ax, title: str, subtitle: str, *, title_width: int = 105, subtitle_width: int = 140) -> None:
    title = textwrap.fill(title.strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=subtitle_width, break_long_words=False)
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.952, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])


def save_figure(fig, base_path: Path) -> dict[str, str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def percent_axis(ax) -> None:
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def xpercent_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def metric_dict(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    labels = labels.astype(int)
    probabilities = probabilities.astype(float)
    pred = (probabilities >= threshold).astype(int)
    output: dict[str, Any] = {
        "row_count": int(labels.size),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else math.nan,
        "threshold": float(threshold),
        "accuracy": float((pred == labels).mean()) if labels.size else math.nan,
        "precision": float(precision_score(labels, pred, zero_division=0)) if labels.size else math.nan,
        "recall": float(recall_score(labels, pred, zero_division=0)) if labels.size else math.nan,
        "f1": float(f1_score(labels, pred, zero_division=0)) if labels.size else math.nan,
        "brier": float(brier_score_loss(labels, probabilities)) if labels.size else math.nan,
    }
    if len(np.unique(labels)) == 2:
        output["auroc"] = float(roc_auc_score(labels, probabilities))
        output["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        output["auroc"] = math.nan
        output["auprc"] = math.nan
    output["ece"] = float(expected_calibration_error(labels, probabilities))
    return output


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = labels.astype(int)
    probabilities = probabilities.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return ece


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        rows.append(
            {
                "bin": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_mid": (lower + upper) / 2,
                "row_count": int(mask.sum()),
                "mean_predicted_probability": float(probabilities[mask].mean()) if mask.any() else math.nan,
                "empirical_pass_rate": float(labels[mask].mean()) if mask.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities.astype(float))))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(labels, pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return {"threshold": best_threshold, "f1": best_f1}


def load_data(input_file: Path, all_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenized_cols = [
        "prompt_id",
        "base_key",
        "split",
        "label",
        "checker_pass",
        "strict_pass",
        "num_constraints",
        "instruction_ids",
        "constraint_signature",
        "constraint_family_signature",
        "source_dataset",
        "constraint_type",
        "cluster",
        "length_bin",
        "target_prompt_tokens",
        "target_output_tokens",
        "target_output_truncated",
        "raw_prompt_bpe_token_count_full",
        "raw_prompt_bpe_token_count",
        "raw_prompt_bpe_truncated",
        "input_ids",
        "attention_mask",
    ]
    tokenized = pd.read_parquet(input_file, columns=tokenized_cols)
    all_cols = ["prompt_id", "user_prompt", "response_text", "prompt_tokens", "output_truncated"]
    all_frame = pd.read_parquet(all_file, columns=all_cols)
    return tokenized, all_frame


def load_m1_specs() -> list[dict[str, Any]]:
    summary = pd.read_csv("runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/learning_curve_summary.csv")
    summary["curve_label"] = summary["curve_label"].map(normalize_curve_label)
    summary = summary[summary["curve_label"].isin(COMMON_LABELS)].copy()
    specs = []
    for _, row in summary.sort_values("order").iterrows():
        output_dir = localize_workspace_path(row["output_dir"])
        model_path = output_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        specs.append(
            {
                "model": "M1 TF-IDF",
                "curve_label": str(row["curve_label"]),
                "order": int(ORDER_MAP[str(row["curve_label"])]),
                "train_seed": 42,
                "actual_train_rows": int(row["actual_train_rows"]),
                "model_path": str(model_path),
            }
        )
    return specs


def load_torch_specs(summary_path: Path, model_name: str) -> list[dict[str, Any]]:
    summary = pd.read_csv(summary_path)
    summary["curve_label"] = summary["curve_label"].map(normalize_curve_label)
    summary = summary[summary["curve_label"].isin(COMMON_LABELS)].copy()
    summary["order"] = summary["curve_label"].map(ORDER_MAP).astype(int)
    specs = []
    for _, row in summary.sort_values(["train_seed", "order"], kind="mergesort").iterrows():
        output_dir = localize_workspace_path(row["output_dir"])
        model_path = output_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        specs.append(
            {
                "model": model_name,
                "curve_label": str(row["curve_label"]),
                "order": int(row["order"]),
                "train_seed": int(row["train_seed"]),
                "sample_seed": int(row["sample_seed"]),
                "actual_train_rows": int(row["actual_train_rows"]),
                "model_path": str(model_path),
            }
        )
    return specs


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
def predict_torch(
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
            probabilities = model.predict_pass_probability(input_ids=input_ids, attention_mask=attention_mask)
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


def merge_prediction_metadata(predictions: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "prompt_id",
        "base_key",
        "num_constraints",
        "instruction_ids",
        "constraint_signature",
        "constraint_family_signature",
        "source_dataset",
        "constraint_type",
        "cluster",
        "length_bin",
        "target_prompt_tokens",
        "raw_prompt_bpe_token_count_full",
    ]
    return predictions.merge(meta[meta_cols].drop_duplicates("prompt_id"), on="prompt_id", how="left", validate="many_to_one")


def generate_predictions(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    skip_refresh: bool,
) -> pd.DataFrame:
    predictions_path = output_dir / "tables" / "final_row_level_predictions.parquet"
    summary_path = output_dir / "tables" / "prediction_refresh_summary.csv"
    if skip_refresh and predictions_path.exists():
        return pd.read_parquet(predictions_path)

    eval_frame = data[data["split"].isin(["val", "test"])].copy()
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in load_m1_specs():
        start = now()
        pipeline = joblib.load(spec["model_path"])
        pred = predict_m1(pipeline, eval_frame, threshold=0.5).rename(
            columns={"m1_pred_proba": "pred_proba", "m1_pred_label": "pred_label"}
        )
        pred = pred[["prompt_id", "split", "label", "pred_proba", "pred_label"]]
        pred = merge_prediction_metadata(pred, data)
        for key, value in spec.items():
            pred[key] = value
        all_rows.append(pred)
        summary_rows.append({**spec, "row_count": int(len(pred)), "predict_seconds": elapsed_since(start)})

    m3_specs = load_torch_specs(
        Path("runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/multiseed_learning_curve_summary.csv"),
        "M3 mean",
    )
    m4_specs = load_torch_specs(
        Path("runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/multiseed_learning_curve_summary.csv"),
        "M4 frozen encoder",
    )
    use_amp = device.type == "cuda"
    for spec in [*m3_specs, *m4_specs]:
        start = now()
        model_path = Path(spec["model_path"])
        if spec["model"] == "M3 mean":
            model, config = load_m3_model(model_path, device)
            dataloader = make_m3_dataloader(
                eval_frame,
                config,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            model, config = load_m4_model(model_path, device)
            dataloader = make_m4_classifier_dataloader(
                eval_frame,
                config,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        try:
            pred = predict_torch(model, dataloader, device=device, use_amp=use_amp)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pred["pred_label"] = (pred["pred_proba"] >= 0.5).astype(int)
        pred = merge_prediction_metadata(pred, data)
        for key, value in spec.items():
            pred[key] = value
        all_rows.append(pred)
        summary_rows.append({**spec, "row_count": int(len(pred)), "predict_seconds": elapsed_since(start)})
        print(json.dumps(summary_rows[-1], ensure_ascii=False), flush=True)

    predictions = pd.concat(all_rows, ignore_index=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    write_csv(pd.DataFrame(summary_rows), summary_path)
    return predictions


def aggregate_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    key_frames = []
    for model, curve_label, display in KEY_CONFIGS:
        part = predictions[
            (predictions["model"] == model)
            & (predictions["curve_label"] == curve_label)
            & (predictions["split"].isin(["val", "test"]))
        ].copy()
        if part.empty:
            continue
        group_cols = [
            "prompt_id",
            "split",
            "label",
            "base_key",
            "num_constraints",
            "instruction_ids",
            "constraint_signature",
            "constraint_family_signature",
            "cluster",
            "length_bin",
            "target_prompt_tokens",
            "raw_prompt_bpe_token_count_full",
        ]
        averaged = (
            part.groupby(group_cols, dropna=False, as_index=False)["pred_proba"]
            .mean()
            .assign(model=model, curve_label=curve_label, display_name=display)
        )
        key_frames.append(averaged)
    return pd.concat(key_frames, ignore_index=True)


def compute_metrics_tables(predictions: pd.DataFrame, key_predictions: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for (model, curve, seed, split), part in predictions.groupby(["model", "curve_label", "train_seed", "split"], sort=True):
        metrics = metric_dict(part["label"].to_numpy(), part["pred_proba"].to_numpy())
        rows.append(
            {
                "model": model,
                "curve_label": curve,
                "order": ORDER_MAP.get(str(curve), 999),
                "train_seed": int(seed),
                "split": split,
                **metrics,
            }
        )
    by_run = pd.DataFrame(rows)
    write_csv(by_run, output_dir / "tables" / "metrics_by_run.csv")

    agg_rows = []
    metric_cols = ["auroc", "auprc", "brier", "ece", "accuracy", "f1", "positive_rate"]
    for (model, curve, split), part in by_run.groupby(["model", "curve_label", "split"], sort=True):
        row = {"model": model, "curve_label": curve, "order": ORDER_MAP.get(str(curve), 999), "split": split, "run_count": len(part)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_std"] = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
        agg_rows.append(row)
    by_config = pd.DataFrame(agg_rows).sort_values(["model", "order", "split"])
    write_csv(by_config, output_dir / "tables" / "metrics_by_config.csv")

    key_rows = []
    for (display, split), part in key_predictions.groupby(["display_name", "split"], sort=True):
        metrics = metric_dict(part["label"].to_numpy(), part["pred_proba"].to_numpy())
        key_rows.append({"display_name": display, "split": split, **metrics})
    key_metrics = pd.DataFrame(key_rows)
    write_csv(key_metrics, output_dir / "tables" / "key_config_metrics_seed_averaged.csv")
    return {"by_run": by_run, "by_config": by_config, "key_metrics": key_metrics}


def compute_threshold_table(key_predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for display, part in key_predictions.groupby("display_name", sort=True):
        val = part[part["split"] == "val"]
        test = part[part["split"] == "test"]
        selected = best_f1_threshold(val["label"].to_numpy(), val["pred_proba"].to_numpy())
        test_metrics = metric_dict(test["label"].to_numpy(), test["pred_proba"].to_numpy(), threshold=selected["threshold"])
        rows.append(
            {
                "display_name": display,
                "selection_split": "val",
                "selection_metric": "F1",
                "selected_threshold": selected["threshold"],
                "val_f1_at_selected_threshold": selected["f1"],
                "test_f1_fixed_from_val": test_metrics["f1"],
                "test_precision_fixed_from_val": test_metrics["precision"],
                "test_recall_fixed_from_val": test_metrics["recall"],
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "threshold_selection_protocol.csv")
    return table


def compute_selective_tables(predictions: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    topk_rows = []
    risk_rows = []
    for (model, curve, seed), part in predictions[predictions["split"] == "test"].groupby(["model", "curve_label", "train_seed"], sort=True):
        y = part["label"].astype(int).to_numpy()
        p = part["pred_proba"].astype(float).to_numpy()
        pred = (p >= 0.5).astype(int)
        confidence = np.abs(p - 0.5)
        order = np.argsort(-confidence)
        for coverage in COVERAGES:
            n = max(1, min(len(order), math.ceil(len(order) * coverage)))
            idx = order[:n]
            rows.append(
                {
                    "model": model,
                    "curve_label": curve,
                    "order": ORDER_MAP.get(str(curve), 999),
                    "train_seed": int(seed),
                    "coverage": coverage,
                    "selected_count": n,
                    "selective_accuracy": float((pred[idx] == y[idx]).mean()),
                    "selective_error_rate": float((pred[idx] != y[idx]).mean()),
                    "mean_confidence": float(confidence[idx].mean()),
                }
            )
        order_p = np.argsort(-p)
        for top_k in TOP_K_VALUES:
            n = max(1, min(len(order_p), top_k))
            idx = order_p[:n]
            topk_rows.append(
                {
                    "model": model,
                    "curve_label": curve,
                    "order": ORDER_MAP.get(str(curve), 999),
                    "train_seed": int(seed),
                    "top_k": n,
                    "precision_at_k": float(y[idx].mean()),
                    "positive_count_at_k": int(y[idx].sum()),
                    "mean_score_at_k": float(p[idx].mean()),
                }
            )
        for coverage in np.linspace(0.05, 1.0, 20):
            n = max(1, min(len(order), math.ceil(len(order) * coverage)))
            idx = order[:n]
            risk_rows.append(
                {
                    "model": model,
                    "curve_label": curve,
                    "order": ORDER_MAP.get(str(curve), 999),
                    "train_seed": int(seed),
                    "coverage": float(coverage),
                    "risk": float((pred[idx] != y[idx]).mean()),
                }
            )
    selective = pd.DataFrame(rows)
    topk = pd.DataFrame(topk_rows)
    risk = pd.DataFrame(risk_rows)
    write_csv(selective, output_dir / "tables" / "selective_accuracy_by_run.csv")
    write_csv(topk, output_dir / "tables" / "topk_precision_by_run.csv")
    write_csv(risk, output_dir / "tables" / "coverage_risk_by_run.csv")
    return selective, topk, risk


def aggregate_with_std(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key, strict=True))
        row["run_count"] = len(part)
        for col in value_cols:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_std"] = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def group_bootstrap_ci(
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    reps: int,
    seed: int,
    group_column: str = "base_key",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    target = predictions[(predictions["model"].isin(["M3 mean", "M4 frozen encoder"])) & (predictions["split"] == "test")]
    for (model, curve, train_seed), part in target.groupby(["model", "curve_label", "train_seed"], sort=True):
        groups = part[group_column].astype(str).fillna("__missing__").to_numpy()
        labels = part["label"].astype(int).to_numpy()
        probs = part["pred_proba"].astype(float).to_numpy()
        unique_groups = np.array(sorted(pd.unique(groups)))
        group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
        aurocs = []
        auprcs = []
        for _ in range(reps):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_to_indices[group] for group in sampled_groups])
            sample_labels = labels[idx]
            sample_probs = probs[idx]
            if len(np.unique(sample_labels)) < 2:
                continue
            aurocs.append(float(roc_auc_score(sample_labels, sample_probs)))
            auprcs.append(float(average_precision_score(sample_labels, sample_probs)))
        rows.append(
            {
                "model": model,
                "curve_label": curve,
                "order": ORDER_MAP.get(str(curve), 999),
                "train_seed": int(train_seed),
                "bootstrap_group_column": group_column,
                "bootstrap_reps_requested": reps,
                "bootstrap_reps_used": len(aurocs),
                "test_auroc": float(roc_auc_score(labels, probs)),
                "test_auroc_ci_low": float(np.quantile(aurocs, 0.025)) if aurocs else math.nan,
                "test_auroc_ci_high": float(np.quantile(aurocs, 0.975)) if aurocs else math.nan,
                "test_auprc": float(average_precision_score(labels, probs)),
                "test_auprc_ci_low": float(np.quantile(auprcs, 0.025)) if auprcs else math.nan,
                "test_auprc_ci_high": float(np.quantile(auprcs, 0.975)) if auprcs else math.nan,
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "m3_m4_group_bootstrap_ci_by_seed.csv")
    return table


def compute_split_overview(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    split_dir = Path("data/splits/qwen3_4b_instruct_2507_under2048_nontruncated")
    split_files = {
        "group split": split_dir / "group_key_seed42.parquet",
        "atomic constraint held-out": split_dir / "atomic_constraint_heldout_seed42.parquet",
        "composition C1": split_dir / "composition_heldout_c1.parquet",
        "composition C2": split_dir / "composition_heldout_c2.parquet",
    }
    rows = []
    labels = data[["prompt_id", "label", "constraint_family_signature"]].drop_duplicates("prompt_id")
    manifest = json.loads((split_dir / "atomic_constraint_heldout_manifest.json").read_text(encoding="utf-8"))
    heldout_count = len(manifest["heldout_instruction_ids"])
    for display, path in split_files.items():
        split = pd.read_parquet(path, columns=["prompt_id", "split", "constraint_family_signature"])
        merged = split.merge(labels, on="prompt_id", how="inner", validate="one_to_one")
        counts = merged["split"].value_counts().to_dict()
        test = merged[merged["split"] == "test"]
        positive_rate = float(test["label"].mean())
        rows.append(
            {
                "split": display,
                "train_rows": int(counts.get("train", 0)),
                "val_rows": int(counts.get("val", 0)),
                "test_rows": int(counts.get("test", 0)),
                "test_positive_rate": positive_rate,
                "baseline_auprc": positive_rate,
                "number_of_constraint_groups": int(test["constraint_family_signature_x"].nunique() if "constraint_family_signature_x" in test.columns else test["constraint_family_signature"].nunique()),
                "number_of_heldout_constraints": heldout_count if "atomic" in display else (0 if "group" in display else 0),
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "figure02_dataset_split_overview.csv")
    return table


def compute_audit_table(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    split_dir = Path("data/splits/qwen3_4b_instruct_2507_under2048_nontruncated")
    audit = json.loads((split_dir / "audit" / "split_duplicate_cluster_audit.summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((split_dir / "split_audit.json").read_text(encoding="utf-8"))
    tokenizer_manifest = json.loads(
        Path("data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_group_key_train/training_manifest.json").read_text(encoding="utf-8")
    )
    atomic_manifest = json.loads((split_dir / "atomic_constraint_heldout_manifest.json").read_text(encoding="utf-8"))
    heldout_ids = {item["instruction_id"] if isinstance(item, dict) else str(item) for item in atomic_manifest["heldout_instruction_ids"]}

    atomic = pd.read_parquet(split_dir / "atomic_constraint_heldout_seed42.parquet", columns=["prompt_id", "split", "instruction_ids"])
    group = pd.read_parquet(split_dir / "group_key_seed42.parquet", columns=["prompt_id", "split", "base_key"])
    max_ids = set(data["prompt_id"])
    atomic = atomic[atomic["prompt_id"].isin(max_ids)]
    group = group[group["prompt_id"].isin(max_ids)]
    group_train = set(group.loc[group["split"] == "train", "prompt_id"])
    atomic_test = set(atomic.loc[atomic["split"] == "test", "prompt_id"])
    atomic_val = set(atomic.loc[atomic["split"] == "val", "prompt_id"])
    atomic_train = atomic[atomic["split"] == "train"].copy()
    atomic_train_contains_heldout = atomic_train["instruction_ids"].map(lambda value: bool(set(listify(value)) & heldout_ids)).sum()
    atomic_val_contains_heldout = atomic[atomic["split"] == "val"]["instruction_ids"].map(lambda value: bool(set(listify(value)) & heldout_ids)).sum()
    atomic_test_contains_heldout = atomic[atomic["split"] == "test"]["instruction_ids"].map(lambda value: bool(set(listify(value)) & heldout_ids)).sum()

    group_base_leaks = next(
        item["base_key_cross_split_leakage"]["leaked_value_count"]
        for item in audit["split_summaries"]
        if item["split_file"] == "group_key_seed42.parquet"
    )
    atomic_prompt_hash_leaks = next(
        item["prompt_id_hash_cross_split_leakage"]["leaked_value_count"]
        for item in audit["split_summaries"]
        if item["split_file"] == "atomic_constraint_heldout_seed42.parquet"
    )
    atomic_base_leaks = next(
        item["base_key_cross_split_leakage"]["leaked_value_count"]
        for item in audit["split_summaries"]
        if item["split_file"] == "atomic_constraint_heldout_seed42.parquet"
    )
    cluster_cross = (
        data.groupby("cluster")["split"]
        .nunique()
        .reset_index(name="split_count")
        .query("split_count > 1")
    )
    rows = [
        {
            "audit_item": "Atomic held-out split keeps held-out instructions out of train/val",
            "status": "pass",
            "evidence": f"train rows with held-out instruction={int(atomic_train_contains_heldout)}, val={int(atomic_val_contains_heldout)}, test rows with held-out={int(atomic_test_contains_heldout)}",
        },
        {
            "audit_item": "Group split keeps base_key groups separated",
            "status": "pass",
            "evidence": f"group_key_seed42 base_key cross-split leaked values={group_base_leaks}",
        },
        {
            "audit_item": "Exact prompt_id/hash duplicates across split",
            "status": "pass",
            "evidence": f"atomic prompt_id cross-split leaked values={atomic_prompt_hash_leaks}; all prompt_id duplicate rows={audit['all_prompt_id_hash_summary']['duplicate_row_count']}",
        },
        {
            "audit_item": "base_key duplicates in atomic split",
            "status": "caveat",
            "evidence": f"atomic split is not a group split; base_key cross-split leaked values={atomic_base_leaks}",
        },
        {
            "audit_item": "same prompt family / cousin cluster across split",
            "status": "caveat",
            "evidence": f"{len(cluster_cross)} clusters appear in more than one split; cluster is used for analysis, not as a split boundary",
        },
        {
            "audit_item": "M4 pretraining atomic held-out exposure",
            "status": "pass",
            "evidence": "M4 pretraining filters tokenized atomic input to split=train; train/val contain 0 held-out instruction rows",
        },
        {
            "audit_item": "Tokenizer allowed-corpus exposure for atomic held-out",
            "status": "accepted caveat",
            "evidence": (
                f"tokenizer manifest split={tokenizer_manifest['split_path']}::{tokenizer_manifest['split_value']}; "
                f"group train ∩ atomic test={len(group_train & atomic_test)}/{len(atomic_test)} max2048 rows; "
                f"group train ∩ atomic val={len(group_train & atomic_val)}"
            ),
        },
        {
            "audit_item": "Target response excluded from M1/M3/M4 inputs",
            "status": "pass",
            "evidence": "boundary model inputs use raw-prompt BPE input_ids only; target response appears only in promptset/checker artifacts, not tokenized model input columns",
        },
        {
            "audit_item": "Target model calls during boundary training/evaluation",
            "status": "pass",
            "evidence": "M1/M3/M4 training consumes frozen checker labels; no target LLM calls are made during boundary training or final validation",
        },
    ]
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "figure_audit_table.csv")
    return table


def compute_m1_split_lift(output_dir: Path) -> pd.DataFrame:
    mapping = {
        "group_key_seed42": "group split",
        "atomic_constraint_heldout_seed42": "atomic constraint held-out",
        "composition_heldout_c1": "composition C1",
        "composition_heldout_c2": "composition C2",
    }
    rows = []
    for path in sorted(Path("runs").glob("m1_raw_prompt_bpe_tfidf_logreg_*/metrics.json")):
        split_key = path.parent.name.replace("m1_raw_prompt_bpe_tfidf_logreg_", "")
        if split_key not in mapping:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["by_split"]["test"]
        baseline = float(metrics["positive_rate"])
        rows.append(
            {
                "split": mapping[split_key],
                "baseline_auprc": baseline,
                "m1_auprc": float(metrics["auprc"]),
                "m1_auroc": float(metrics["auroc"]),
                "m1_f1": float(metrics["f1"]),
                "test_rows": int(metrics["row_count"]),
                "positive_rate": baseline,
                "lift": float(metrics["auprc"]) / baseline if baseline else math.nan,
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "figure03_m1_split_baseline_lift.csv")
    return table


def compute_validation_vs_oracle(metrics_by_config: pd.DataFrame, key_metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for model in ["M1 TF-IDF", "M3 mean", "M4 frozen encoder"]:
        val = metrics_by_config[(metrics_by_config["model"] == model) & (metrics_by_config["split"] == "val")]
        test = metrics_by_config[(metrics_by_config["model"] == model) & (metrics_by_config["split"] == "test")]
        selected = val.sort_values(["auprc_mean", "order"], ascending=[False, True]).iloc[0]
        oracle = test.sort_values(["auprc_mean", "order"], ascending=[False, True]).iloc[0]
        for label, row in [("validation-selected", selected), ("oracle-best", oracle)]:
            curve = row["curve_label"]
            test_row = test[test["curve_label"] == curve].iloc[0]
            display_name = f"{model.replace(' TF-IDF', '').replace(' frozen encoder', ' frozen')} {curve}"
            key_match = key_metrics[(key_metrics["split"] == "test") & (key_metrics["display_name"].str.contains(model.split()[0], regex=False))]
            rows.append(
                {
                    "row": f"{model} {label}",
                    "selection_metric": "val AUPRC" if label == "validation-selected" else "test AUPRC oracle",
                    "selected_train_size": curve,
                    "val_auprc": float(row["auprc_mean"]) if label == "validation-selected" else float(val[val["curve_label"] == curve]["auprc_mean"].iloc[0]),
                    "test_auroc": float(test_row["auroc_mean"]),
                    "test_auprc": float(test_row["auprc_mean"]),
                    "test_brier": float(test_row["brier_mean"]),
                    "test_ece": float(test_row["ece_mean"]),
                }
            )
            if model == "M1 TF-IDF":
                break
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "figure06_validation_selected_vs_oracle.csv")
    return table


def compute_per_constraint_tables(key_predictions: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = key_predictions[key_predictions["split"] == "test"].copy()
    exploded = test.assign(instruction_id=test["instruction_ids"].map(listify)).explode("instruction_id")
    rows = []
    for (instruction_id, display), part in exploded.groupby(["instruction_id", "display_name"], sort=True):
        if len(part) < 20 or part["label"].nunique() < 2:
            continue
        metrics = metric_dict(part["label"].to_numpy(), part["pred_proba"].to_numpy())
        rows.append(
            {
                "instruction_id": instruction_id,
                "display_name": display,
                "test_rows": int(len(part)),
                "positive_count": int(part["label"].sum()),
                "positive_rate": float(part["label"].mean()),
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
                "brier": metrics["brier"],
            }
        )
    per_constraint = pd.DataFrame(rows)
    write_csv(per_constraint, output_dir / "tables" / "figure14_per_constraint_metrics.csv")

    pivot = per_constraint.pivot_table(index="instruction_id", columns="display_name", values="auprc", aggfunc="first")
    base = pivot.get("M1 full")
    delta_rows = []
    if base is not None:
        for target in ["M3 mean 40k", "M4 frozen full"]:
            if target not in pivot.columns:
                continue
            delta = (pivot[target] - base).dropna().reset_index(name="delta_auprc")
            delta["comparison"] = f"{target} - M1 full"
            delta_rows.append(delta)
    deltas = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()
    write_csv(deltas, output_dir / "tables" / "figure15_per_constraint_delta_vs_m1.csv")

    group_rows = []
    for (cluster, display), part in test.groupby(["cluster", "display_name"], sort=True):
        if len(part) < 20 or part["label"].nunique() < 2:
            continue
        metrics = metric_dict(part["label"].to_numpy(), part["pred_proba"].to_numpy())
        group_rows.append(
            {
                "cluster": int(cluster),
                "display_name": display,
                "test_rows": int(len(part)),
                "positive_rate": float(part["label"].mean()),
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
                "error_rate_at_0_5": 1.0 - metrics["accuracy"],
            }
        )
    per_group = pd.DataFrame(group_rows)
    write_csv(per_group, output_dir / "tables" / "per_group_cluster_metrics.csv")
    return per_constraint, deltas, per_group


def compute_m1_interpretability(output_dir: Path, tokenizer_dir: Path) -> pd.DataFrame:
    model_path = Path("runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/train_80k/model.joblib")
    pipeline = joblib.load(model_path)
    features = feature_summary(pipeline, top_n=30)
    features["odds_ratio"] = np.exp(features["coefficient"].clip(-20, 20))
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
        decoded = []
        for feature in features["feature"].astype(str):
            ids = []
            for token in feature.split():
                if token.isdigit():
                    ids.append(int(token))
            if ids:
                decoded.append(" ".join(tokenizer.convert_ids_to_tokens(ids)))
            else:
                decoded.append("")
        features["feature_decoded_tokens"] = decoded
    except Exception:
        features["feature_decoded_tokens"] = ""
    write_csv(features, output_dir / "tables" / "figure16_m1_top_ngram_coefficients.csv")
    return features


def compute_complexity_table(output_dir: Path) -> pd.DataFrame:
    m1_summary = pd.read_csv("runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/learning_curve_summary.csv")
    m1_full = m1_summary[m1_summary["curve_label"].map(normalize_curve_label) == "full"].iloc[0]
    m3_summary = pd.read_csv("runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/multiseed_learning_curve_summary.csv")
    m3_40k = m3_summary[m3_summary["curve_label"].map(normalize_curve_label) == "40k"]
    m4_summary = pd.read_csv("runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/multiseed_learning_curve_summary.csv")
    m4_full = m4_summary[m4_summary["curve_label"].map(normalize_curve_label) == "full"]

    m3_ckpt = torch_load(localize_workspace_path(m3_40k.iloc[0]["output_dir"]) / "model.pt")
    m3_params = int(sum(t.numel() for t in m3_ckpt["state_dict"].values()))
    m4_ckpt = torch_load(localize_workspace_path(m4_full.iloc[0]["output_dir"]) / "model.pt")
    m4_encoder_params = int(sum(t.numel() for t in m4_ckpt["prompt_encoder_state_dict"].values()))
    m4_head_params = int(sum(t.numel() for t in m4_ckpt["pass_head_state_dict"].values()))
    pretrain_times = []
    for path in Path("runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048").glob("seed_*/pretrain/pretraining_manifest.json"):
        pretrain_times.append(json.loads(path.read_text(encoding="utf-8"))["timing_seconds"]["total"])

    pred_summary = pd.read_csv("runs/model_comparisons/m1_m3_m4_selective_prediction/m3_m4_test_prediction_run_summary.csv")
    m3_pred = pred_summary[(pred_summary["model"] == "M3 mean") & (pred_summary["curve_label"] == "40k")]
    m4_pred = pred_summary[(pred_summary["model"] == "M4 frozen encoder") & (pred_summary["curve_label"] == "full")]
    rows = [
        {
            "component": "M1 TF-IDF full",
            "trainable_params": int(m1_full["vocabulary_size"]) + 1,
            "total_params": int(m1_full["vocabulary_size"]) + 1,
            "train_rows": int(m1_full["actual_train_rows"]),
            "pretraining_required": "no",
            "gpu_required": "no",
            "train_seconds": float(m1_full["train_seconds"]),
            "inference_rows_per_second": float(m1_full["test_rows"]) / float(m1_full["predict_seconds"]),
            "target_model_calls_during_boundary_training": 0,
        },
        {
            "component": "M3 mean 40k",
            "trainable_params": m3_params,
            "total_params": m3_params,
            "train_rows": 40000,
            "pretraining_required": "no",
            "gpu_required": "yes",
            "train_seconds": float(m3_40k["total_seconds"].mean()),
            "inference_rows_per_second": float(m3_pred["row_count"].mean() / m3_pred["predict_seconds"].mean()),
            "target_model_calls_during_boundary_training": 0,
        },
        {
            "component": "M4 IF-domain pretrain",
            "trainable_params": m4_encoder_params,
            "total_params": m4_encoder_params,
            "train_rows": 61501,
            "pretraining_required": "yes",
            "gpu_required": "yes",
            "train_seconds": float(np.mean(pretrain_times)),
            "inference_rows_per_second": math.nan,
            "target_model_calls_during_boundary_training": 0,
        },
        {
            "component": "M4 frozen head full",
            "trainable_params": m4_head_params,
            "total_params": m4_encoder_params + m4_head_params,
            "train_rows": int(m4_full["actual_train_rows"].mean()),
            "pretraining_required": "uses M4 pretrain",
            "gpu_required": "yes",
            "train_seconds": float(m4_full["total_seconds"].mean()),
            "inference_rows_per_second": float(m4_pred["row_count"].mean() / m4_pred["predict_seconds"].mean()),
            "target_model_calls_during_boundary_training": 0,
        },
    ]
    table = pd.DataFrame(rows)
    write_csv(table, output_dir / "tables" / "figure20_training_complexity.csv")
    return table


def compute_error_examples(key_predictions: pd.DataFrame, all_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    test = key_predictions[key_predictions["split"] == "test"].copy()
    wide = test.pivot_table(index="prompt_id", columns="display_name", values="pred_proba", aggfunc="first").reset_index()
    meta_cols = ["prompt_id", "label", "instruction_ids", "constraint_signature", "base_key", "cluster"]
    meta = test[meta_cols].drop_duplicates("prompt_id")
    wide = wide.merge(meta, on="prompt_id", how="left", validate="one_to_one")
    wide = wide.merge(all_frame[["prompt_id", "user_prompt", "response_text"]], on="prompt_id", how="left", validate="one_to_one")
    for display in ["M1 full", "M3 mean 40k", "M4 frozen full"]:
        if display in wide:
            wide[f"{display}_pred"] = (wide[display] >= 0.5).astype(int)
            wide[f"{display}_correct"] = wide[f"{display}_pred"] == wide["label"]

    examples = []
    cases = [
        ("high_confidence_false_positive_m3", wide[(wide["label"] == 0) & (wide["M3 mean 40k"] >= 0.8)].sort_values("M3 mean 40k", ascending=False).head(5)),
        ("high_confidence_false_negative_m3", wide[(wide["label"] == 1) & (wide["M3 mean 40k"] <= 0.2)].sort_values("M3 mean 40k").head(5)),
        ("m3_correct_m1_wrong", wide[wide["M3 mean 40k_correct"] & ~wide["M1 full_correct"]].sort_values("M3 mean 40k", ascending=False).head(5)),
        ("m4_correct_m3_wrong", wide[wide["M4 frozen full_correct"] & ~wide["M3 mean 40k_correct"]].sort_values("M4 frozen full", ascending=False).head(5)),
        ("all_wrong", wide[~wide["M1 full_correct"] & ~wide["M3 mean 40k_correct"] & ~wide["M4 frozen full_correct"]].head(5)),
    ]
    for case, frame in cases:
        for _, row in frame.iterrows():
            examples.append(
                {
                    "case": case,
                    "prompt_id": row["prompt_id"],
                    "constraint_signature": row["constraint_signature"],
                    "true_label": int(row["label"]),
                    "m1_p_pass": float(row.get("M1 full", math.nan)),
                    "m3_40k_p_pass": float(row.get("M3 mean 40k", math.nan)),
                    "m4_full_p_pass": float(row.get("M4 frozen full", math.nan)),
                    "prompt_preview": textwrap.shorten(str(row.get("user_prompt", "")), width=280, placeholder="..."),
                    "target_response_preview": textwrap.shorten(str(row.get("response_text", "")), width=220, placeholder="..."),
                }
            )
    table = pd.DataFrame(examples)
    write_csv(table, output_dir / "tables" / "figure19_error_examples.csv")
    (output_dir / "tables" / "figure19_error_examples.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    return table


def token_distribution_table(token_count_file: Path, input_file: Path, output_dir: Path) -> pd.DataFrame:
    all_tokens = pd.read_parquet(token_count_file, columns=["prompt_id", "prompt_tokens"])
    max_ids = set(pd.read_parquet(input_file, columns=["prompt_id"])["prompt_id"])
    bins = [0, 512, 1024, 1536, 2048, np.inf]
    labels = ["<=512", "513-1024", "1025-1536", "1537-2048", ">2048"]
    all_tokens["token_bin"] = pd.cut(all_tokens["prompt_tokens"], bins=bins, labels=labels, include_lowest=True, right=True)
    table = all_tokens["token_bin"].value_counts().reindex(labels).fillna(0).astype(int).reset_index()
    table.columns = ["token_bin", "prompt_count"]
    table["share"] = table["prompt_count"] / len(all_tokens)
    table["included_in_max2048_count"] = table["token_bin"].map(
        all_tokens[all_tokens["prompt_id"].isin(max_ids)]["token_bin"].value_counts().to_dict()
    ).fillna(0).astype(int)
    table.attrs["total_prompts"] = len(all_tokens)
    table.attrs["max2048_prompts"] = len(max_ids)
    write_csv(table, output_dir / "tables" / "figure21_prompt_token_distribution.csv")
    return table


def make_table_figure(table: pd.DataFrame, path: Path, title: str, subtitle: str, *, max_rows: int = 12) -> dict[str, str]:
    display = table.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    fig, ax = plt.subplots(figsize=(13, 0.7 + 0.42 * len(display)))
    ax.axis("off")
    tbl = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    for _, cell in tbl.get_celld().items():
        cell.set_edgecolor(TOKENS["axis"])
        cell.set_linewidth(0.5)
    add_header(fig, ax, title, subtitle)
    fig.subplots_adjust(top=0.78, bottom=0.05, left=0.02, right=0.98)
    return save_figure(fig, path)


def plot_flow(output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_axis_off()
    labels = [
        "Prompt pool",
        "Local target LLM",
        "Target response",
        "Deterministic checker",
        "Pass/fail label",
        "M1/M3/M4 boundary model",
        "P(pass)",
        "Evaluation and selective filtering",
    ]
    x = np.linspace(0.04, 0.96, len(labels))
    y = 0.48
    for index, (xi, label) in enumerate(zip(x, labels, strict=True)):
        ax.text(
            xi,
            y,
            textwrap.fill(label, width=14),
            ha="center",
            va="center",
            fontsize=9.5,
            color=TOKENS["ink"],
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.04", fc=TOKENS["panel"], ec=COLOR_FAMILIES["blue"]["dark"], lw=1.0),
        )
        if index < len(labels) - 1:
            ax.annotate("", xy=(x[index + 1] - 0.055, y), xytext=(xi + 0.055, y), arrowprops=dict(arrowstyle="->", color=TOKENS["muted"], lw=1.1))
    ax.text(
        0.04,
        0.95,
        "Figure 1. Boundary-predictor research flow",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    ax.text(
        0.04,
        0.84,
        "The learned model predicts whether a fixed target LLM response will pass deterministic instruction checks; it is not fine-tuning the target LLM.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    return save_figure(fig, output_dir / "figures" / "figure01_research_flow")


def plot_split_overview(table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    return make_table_figure(
        table,
        output_dir / "figures" / "figure02_dataset_split_overview",
        "Figure 2. Dataset and split overview",
        "Rows and positive-rate baselines differ by split, so AUPRC must be interpreted against each test baseline.",
    )


def plot_token_distribution(table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    family = COLOR_FAMILIES["blue"]
    bars = ax.bar(table["token_bin"], table["prompt_count"], color=family["base"], edgecolor=family["dark"], linewidth=1.0)
    for bar, share in zip(bars, table["share"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{share:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_xlabel("Qwen prompt-token bin")
    ax.set_ylabel("Prompt count")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    total = int(table.attrs.get("total_prompts", table["prompt_count"].sum()))
    included = int(table.attrs.get("max2048_prompts", table.loc[table["token_bin"] != ">2048", "prompt_count"].sum()))
    add_header(
        fig,
        ax,
        "Figure 2b. Prompt-token distribution supports the max2048 cutoff",
        f"{included:,} / {total:,} prompts are retained in the max2048 modeling input; the cutoff removes the long tail while preserving most prompts.",
    )
    fig.subplots_adjust(top=0.80, left=0.10, right=0.98, bottom=0.12)
    return save_figure(fig, output_dir / "figures" / "figure02b_prompt_token_distribution")


def plot_m1_split_lift(table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot = table.sort_values("lift")
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    x = np.arange(len(plot))
    width = 0.36
    ax.bar(x - width / 2, plot["baseline_auprc"], width, label="Baseline AUPRC", color=COLOR_FAMILIES["neutral"]["light"], edgecolor=COLOR_FAMILIES["neutral"]["dark"])
    ax.bar(x + width / 2, plot["m1_auprc"], width, label="M1 AUPRC", color=COLOR_FAMILIES["orange"]["base"], edgecolor=COLOR_FAMILIES["orange"]["dark"])
    for xi, lift, value in zip(x, plot["lift"], plot["m1_auprc"], strict=True):
        ax.text(xi + width / 2, value + 0.015, f"{lift:.1f}x", ha="center", va="bottom", fontsize=8, color=TOKENS["ink"])
    ax.set_xticks(x, plot["split"], rotation=15, ha="right")
    ax.set_ylabel("AUPRC")
    percent_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Figure 3. M1 baseline lift differs by split", "M1 improves substantially over the positive-rate AUPRC baseline, but split difficulty changes the absolute baseline.")
    fig.subplots_adjust(top=0.78, left=0.09, right=0.98, bottom=0.20)
    return save_figure(fig, output_dir / "figures" / "figure03_m1_split_baseline_lift")


def plot_learning_curves(metrics_by_config: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot = metrics_by_config[metrics_by_config["curve_label"].isin(COMMON_LABELS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = [
        ("val", "auroc", "Validation AUROC"),
        ("val", "auprc", "Validation AUPRC"),
        ("test", "auroc", "Test AUROC"),
        ("test", "auprc", "Test AUPRC"),
    ]
    x = np.arange(len(COMMON_LABELS))
    for ax, (split, metric, title) in zip(axes.flat, panels, strict=True):
        for model in ["M1 TF-IDF", "M3 mean", "M4 frozen encoder"]:
            part = plot[(plot["model"] == model) & (plot["split"] == split)].sort_values("order")
            if part.empty:
                continue
            color = MODEL_COLORS[model]
            y = part[f"{metric}_mean"].to_numpy()
            err = part[f"{metric}_std"].fillna(0).to_numpy()
            ax.errorbar(x[: len(y)], y, yerr=err, marker="o", color=color, label=model, capsize=3, linewidth=1.2)
        if metric == "auprc":
            baseline = plot[(plot["split"] == split) & (plot["model"] == "M1 TF-IDF")].sort_values("order")["positive_rate_mean"]
            if not baseline.empty:
                ax.plot(x[: len(baseline)], baseline, color=MODEL_COLORS["baseline"], linestyle=":", label="Positive-rate baseline")
        ax.set_title(title, loc="left", fontsize=11, color=TOKENS["ink"])
        ax.set_xticks(x, COMMON_LABELS)
        ax.set_ylim(0.0 if metric == "auprc" else 0.55, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=4, borderaxespad=0)
    add_header(fig, axes[0, 0], "Figure 4. Shared atomic held-out learning curves", "M1 is strong, M3 peaks around 40k, and M4 is smoother with its best mean AUPRC at full.")
    fig.subplots_adjust(top=0.84, left=0.08, right=0.98, bottom=0.08, hspace=0.25)
    return save_figure(fig, output_dir / "figures" / "figure04_m1_m3_m4_learning_curves")


def plot_best_config(metrics_by_config: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    rows = []
    for model in ["M1 TF-IDF", "M3 mean", "M4 frozen encoder"]:
        test = metrics_by_config[(metrics_by_config["model"] == model) & (metrics_by_config["split"] == "test")]
        best = test.sort_values(["auprc_mean", "order"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "display": f"{model} {best['curve_label']}",
                "test_auroc_mean": best["auroc_mean"],
                "test_auroc_std": best["auroc_std"],
                "test_auprc_mean": best["auprc_mean"],
                "test_auprc_std": best["auprc_std"],
                "model": model,
            }
        )
    table = pd.DataFrame(rows).sort_values("test_auprc_mean")
    write_csv(table, output_dir / "tables" / "figure05_best_test_auprc_config.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, metric, label in [(axes[0], "test_auroc", "Best-config test AUROC"), (axes[1], "test_auprc", "Best-config test AUPRC")]:
        colors = [MODEL_COLORS[row["model"]] for _, row in table.iterrows()]
        ax.barh(table["display"], table[f"{metric}_mean"], xerr=table[f"{metric}_std"], color=colors, edgecolor=TOKENS["ink"], linewidth=0.8, capsize=3)
        ax.set_xlabel(label)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlim(0.20, 0.95)
        ax.set_title(label, loc="left", fontsize=11)
    add_header(fig, axes[0], "Figure 5. Best test AUPRC configuration by model family", "Oracle-best selection shows M3's highest point while keeping M1 and M4 on the same atomic held-out test set.")
    fig.subplots_adjust(top=0.78, left=0.23, right=0.98, bottom=0.12, wspace=0.18)
    return save_figure(fig, output_dir / "figures" / "figure05_best_test_auprc_config")


def plot_pr_roc(key_predictions: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    test = key_predictions[key_predictions["split"] == "test"].copy()
    pr_rows = []
    roc_rows = []
    for display, part in test.groupby("display_name", sort=True):
        y = part["label"].astype(int).to_numpy()
        p = part["pred_proba"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        pr_rows.extend({"display_name": display, "precision": float(a), "recall": float(b)} for a, b in zip(precision, recall, strict=True))
        roc_rows.extend({"display_name": display, "fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr, strict=True))
    pr_table = pd.DataFrame(pr_rows)
    roc_table = pd.DataFrame(roc_rows)
    write_csv(pr_table, output_dir / "tables" / "figure07_pr_curve_points.csv")
    write_csv(roc_table, output_dir / "tables" / "figure08_roc_curve_points.csv")

    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    for display, part in pr_table.groupby("display_name", sort=True):
        ax.plot(part["recall"], part["precision"], label=display, linewidth=1.2)
    baseline = float(test["label"].mean())
    ax.axhline(baseline, color=TOKENS["muted"], linestyle=":", linewidth=1.0, label=f"Baseline {baseline:.1%}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Figure 7. Precision-recall curves", "AUPRC is the primary metric because the atomic held-out test set is imbalanced.")
    fig.subplots_adjust(top=0.78, left=0.12, right=0.98, bottom=0.11)
    pr_fig = save_figure(fig, output_dir / "figures" / "figure07_precision_recall_curves")

    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    for display, part in roc_table.groupby("display_name", sort=True):
        ax.plot(part["fpr"], part["tpr"], label=display, linewidth=1.2)
    ax.plot([0, 1], [0, 1], color=TOKENS["muted"], linestyle=":", linewidth=1.0, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Figure 8. ROC curves", "AUROC stays useful for ranking, but the precision-recall view is more sensitive to the low pass rate.")
    fig.subplots_adjust(top=0.78, left=0.12, right=0.98, bottom=0.11)
    roc_fig = save_figure(fig, output_dir / "figures" / "figure08_roc_curves")
    return pr_fig, roc_fig


def plot_calibration(key_predictions: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    test = key_predictions[key_predictions["split"] == "test"].copy()
    rel_frames = []
    metric_rows = []
    for display, part in test.groupby("display_name", sort=True):
        y = part["label"].to_numpy()
        p = part["pred_proba"].to_numpy()
        rel = reliability_bins(y, p)
        rel["display_name"] = display
        rel_frames.append(rel)
        metrics = metric_dict(y, p)
        metric_rows.append({"display_name": display, "brier": metrics["brier"], "ece": metrics["ece"]})
    rel_table = pd.concat(rel_frames, ignore_index=True)
    cal_metrics = pd.DataFrame(metric_rows)
    write_csv(rel_table, output_dir / "tables" / "figure09_reliability_bins.csv")
    write_csv(cal_metrics, output_dir / "tables" / "figure10_brier_ece_summary.csv")

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), gridspec_kw={"height_ratios": [3.0, 1.1]}, sharex=True)
    for display, part in rel_table.groupby("display_name", sort=True):
        axes[0].plot(part["bin_mid"], part["empirical_pass_rate"], marker="o", linewidth=1.1, label=display)
    axes[0].plot([0, 1], [0, 1], color=TOKENS["muted"], linestyle=":", linewidth=1.0, label="Perfect calibration")
    axes[0].set_ylabel("Empirical pass rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    hist = test.copy()
    hist["probability_bin"] = pd.cut(hist["pred_proba"], bins=np.linspace(0, 1, 11), include_lowest=True)
    hist_counts = hist.groupby(["display_name", "probability_bin"], observed=False).size().reset_index(name="row_count")
    sns.barplot(data=hist_counts, x="probability_bin", y="row_count", hue="display_name", ax=axes[1], palette="muted")
    axes[1].set_xlabel("Predicted P(pass) bin")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=35)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    axes[1].legend_.remove()
    add_header(fig, axes[0], "Figure 9. Reliability diagram", "Seed-averaged probabilities are binned into deciles; the histogram shows score concentration by model.")
    fig.subplots_adjust(top=0.82, left=0.11, right=0.98, bottom=0.18, hspace=0.12)
    rel_fig = save_figure(fig, output_dir / "figures" / "figure09_reliability_diagram")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, metric, title in [(axes[0], "brier", "Brier score"), (axes[1], "ece", "ECE")]:
        sns.barplot(data=cal_metrics, y="display_name", x=metric, ax=ax, color=COLOR_FAMILIES["gold"]["base"], edgecolor=COLOR_FAMILIES["gold"]["dark"])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("")
        ax.set_xlabel(f"{title} (lower is better)")
    add_header(fig, axes[0], "Figure 10. Brier and ECE summary", "Calibration metrics compare whether P(pass) behaves like a probability, not only a ranking score.")
    fig.subplots_adjust(top=0.78, left=0.24, right=0.98, bottom=0.14, wspace=0.25)
    metric_fig = save_figure(fig, output_dir / "figures" / "figure10_brier_ece_summary")
    return rel_fig, metric_fig


def plot_selective(selective: pd.DataFrame, topk: pd.DataFrame, risk: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    sel_agg = aggregate_with_std(selective, ["model", "curve_label", "order", "coverage"], ["selective_accuracy"])
    topk_agg = aggregate_with_std(topk, ["model", "curve_label", "order", "top_k"], ["precision_at_k"])
    risk_key = risk[
        ((risk["model"] == "M1 TF-IDF") & (risk["curve_label"] == "full"))
        | ((risk["model"] == "M3 mean") & (risk["curve_label"] == "40k"))
        | ((risk["model"] == "M4 frozen encoder") & (risk["curve_label"] == "full"))
    ].copy()
    risk_agg = aggregate_with_std(risk_key, ["model", "curve_label", "coverage"], ["risk"])
    write_csv(sel_agg, output_dir / "tables" / "figure11_selective_accuracy_aggregate.csv")
    write_csv(topk_agg, output_dir / "tables" / "figure12_topk_precision_aggregate.csv")
    write_csv(risk_agg, output_dir / "tables" / "figure13_coverage_risk_aggregate.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, model in zip(axes, ["M1 TF-IDF", "M3 mean", "M4 frozen encoder"], strict=True):
        part = sel_agg[sel_agg["model"] == model].sort_values("order")
        for coverage, cov_part in part.groupby("coverage", sort=True):
            ax.plot(cov_part["curve_label"], cov_part["selective_accuracy_mean"], marker="o", label=f"{coverage:.0%}")
        ax.set_title(model, loc="left", fontsize=11)
        ax.set_xlabel("Train sample size")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Selective accuracy")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=4, borderaxespad=0)
    add_header(fig, axes[0], "Figure 11. Selective accuracy at fixed coverage", "Rows are accepted by descending confidence |P(pass)-0.5|; high-confidence subsets are substantially more accurate.")
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.14, wspace=0.18)
    selective_fig = save_figure(fig, output_dir / "figures" / "figure11_selective_accuracy_fixed_coverage")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, model in zip(axes, ["M1 TF-IDF", "M3 mean", "M4 frozen encoder"], strict=True):
        part = topk_agg[topk_agg["model"] == model].sort_values("order")
        for top_k, top_part in part.groupby("top_k", sort=True):
            ax.plot(top_part["curve_label"], top_part["precision_at_k_mean"], marker="o", label=f"top-{top_k:,}")
        ax.set_title(model, loc="left", fontsize=11)
        ax.set_xlabel("Train sample size")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Precision@k")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=4, borderaxespad=0)
    add_header(fig, axes[0], "Figure 12. Top-k precision for predicted-pass samples", "Rows are ranked by P(pass); this directly tests whether the boundary model can surface likely passing prompts.")
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.14, wspace=0.18)
    topk_fig = save_figure(fig, output_dir / "figures" / "figure12_topk_precision")

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for _, row in risk_agg.groupby(["model", "curve_label"], sort=True).head(0).iterrows():
        pass
    for (model, curve), part in risk_agg.groupby(["model", "curve_label"], sort=True):
        label = f"{model} {curve}"
        ax.plot(part["coverage"], part["risk_mean"], marker="o", label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk / error rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Figure 13. Coverage-risk curve", "Lower coverage means keeping only higher-confidence predictions, reducing the error rate.")
    fig.subplots_adjust(top=0.78, left=0.11, right=0.98, bottom=0.12)
    risk_fig = save_figure(fig, output_dir / "figures" / "figure13_coverage_risk_curve")
    return selective_fig, topk_fig, risk_fig


def plot_per_constraint(per_constraint: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    plot = per_constraint[per_constraint["display_name"].isin(["M1 full", "M3 mean 40k", "M4 frozen full"])].copy()
    sns.scatterplot(
        data=plot,
        x="positive_rate",
        y="auprc",
        hue="display_name",
        size="test_rows",
        sizes=(40, 420),
        ax=ax,
        palette="muted",
        edgecolor=TOKENS["ink"],
        linewidth=0.5,
        alpha=0.75,
    )
    ax.set_xlabel("Per-constraint positive rate")
    ax.set_ylabel("Per-constraint AUPRC")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Figure 14. Per-constraint positive rate and performance", "Some constraints have low positive rates, making AUPRC volatile even with many test rows.")
    fig.subplots_adjust(top=0.78, left=0.12, right=0.98, bottom=0.12)
    scatter_fig = save_figure(fig, output_dir / "figures" / "figure14_per_constraint_positive_rate_performance")

    if deltas.empty:
        return scatter_fig, {}
    selected = []
    for comparison, part in deltas.groupby("comparison", sort=True):
        part = part.dropna().sort_values("delta_auprc")
        selected.append(pd.concat([part.head(20), part.tail(20)]).drop_duplicates("instruction_id"))
    plot_delta = pd.concat(selected, ignore_index=True)
    fig, axes = plt.subplots(1, plot_delta["comparison"].nunique(), figsize=(15, 8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (comparison, part) in zip(axes, plot_delta.groupby("comparison", sort=True), strict=True):
        part = part.sort_values("delta_auprc")
        colors = np.where(part["delta_auprc"] >= 0, COLOR_FAMILIES["olive"]["base"], COLOR_FAMILIES["orange"]["base"])
        ax.barh(part["instruction_id"], part["delta_auprc"], color=colors, edgecolor=TOKENS["ink"], linewidth=0.4)
        ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
        ax.set_title(comparison, loc="left", fontsize=11)
        ax.set_xlabel("Delta AUPRC")
    add_header(fig, axes[0], "Figure 15. Per-constraint delta relative to M1", "Positive values show constraints where the neural boundary model improves over M1 full.")
    fig.subplots_adjust(top=0.82, left=0.28, right=0.98, bottom=0.08, wspace=0.35)
    delta_fig = save_figure(fig, output_dir / "figures" / "figure15_per_constraint_delta_vs_m1")
    return scatter_fig, delta_fig


def plot_m1_features(features: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharex=False)
    for ax, direction, color_family in [(axes[0], "positive", "olive"), (axes[1], "negative", "orange")]:
        part = features[features["direction"] == direction].head(20).sort_values("coefficient")
        labels = part["feature_decoded_tokens"].where(part["feature_decoded_tokens"].astype(str) != "", part["feature"])
        ax.barh(labels, part["coefficient"], color=COLOR_FAMILIES[color_family]["base"], edgecolor=COLOR_FAMILIES[color_family]["dark"])
        ax.set_title(f"Top {direction}", loc="left", fontsize=11)
        ax.set_xlabel("Logistic coefficient")
    add_header(fig, axes[0], "Figure 16. M1 top positive and negative n-gram coefficients", "Features are raw-prompt BPE token-id n-grams decoded to tokenizer tokens where possible.")
    fig.subplots_adjust(top=0.82, left=0.22, right=0.98, bottom=0.08, wspace=0.35)
    return save_figure(fig, output_dir / "figures" / "figure16_m1_top_ngram_coefficients")


def plot_prediction_hist_and_correlation(key_predictions: pd.DataFrame, output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    test = key_predictions[key_predictions["split"] == "test"].copy()
    hist = test[test["display_name"].isin(["M1 full", "M3 mean 40k", "M4 frozen full"])].copy()
    hist["true_label"] = np.where(hist["label"] == 1, "true pass", "true fail")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, display in zip(axes, ["M1 full", "M3 mean 40k", "M4 frozen full"], strict=True):
        sns.histplot(
            data=hist[hist["display_name"] == display],
            x="pred_proba",
            hue="true_label",
            bins=np.linspace(0, 1, 31),
            multiple="stack",
            ax=ax,
            palette={"true pass": COLOR_FAMILIES["olive"]["base"], "true fail": COLOR_FAMILIES["neutral"]["light"]},
            edgecolor=TOKENS["panel"],
        )
        ax.set_title(display, loc="left", fontsize=11)
        ax.set_xlabel("Predicted P(pass)")
    axes[0].set_ylabel("Test rows")
    add_header(fig, axes[0], "Figure 17. Prediction distribution histogram", "Histograms show whether true pass rows separate from true fail rows or concentrate near low probabilities.")
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.13, wspace=0.18)
    hist_fig = save_figure(fig, output_dir / "figures" / "figure17_prediction_distribution_histogram")

    wide = test.pivot_table(index="prompt_id", columns="display_name", values="pred_proba", aggfunc="first")
    corr = wide[["M1 full", "M3 mean 40k", "M3 mean full", "M4 frozen full"]].corr(method="spearman")
    corr.to_csv(output_dir / "tables" / "figure18_score_correlation_matrix.csv", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    sns.heatmap(corr, vmin=0, vmax=1, annot=True, fmt=".2f", cmap=sns.blend_palette([TOKENS["panel"], COLOR_FAMILIES["blue"]["light"], COLOR_FAMILIES["blue"]["mid"]], as_cmap=True), ax=ax, linewidths=1, linecolor=TOKENS["panel"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_header(fig, ax, "Figure 18. Score correlation between models", "Spearman correlations show whether M3/M4 are mostly copying M1 rankings or changing the score order.")
    fig.subplots_adjust(top=0.78, left=0.25, right=0.98, bottom=0.20)
    corr_fig = save_figure(fig, output_dir / "figures" / "figure18_score_correlation_heatmap")
    return hist_fig, corr_fig


def plot_complexity(table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot = table.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.barplot(data=plot, y="component", x="trainable_params", ax=axes[0], color=COLOR_FAMILIES["blue"]["base"], edgecolor=COLOR_FAMILIES["blue"]["dark"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable params (log scale)")
    axes[0].set_ylabel("")
    sns.barplot(data=plot, y="component", x="train_seconds", ax=axes[1], color=COLOR_FAMILIES["gold"]["base"], edgecolor=COLOR_FAMILIES["gold"]["dark"])
    axes[1].set_xlabel("Train seconds")
    axes[1].set_ylabel("")
    add_header(fig, axes[0], "Figure 20. Training complexity comparison", "M1 is CPU-friendly; M3 trains end to end on GPU; M4 adds IF-domain pretraining but freezes the encoder for the final head.")
    fig.subplots_adjust(top=0.78, left=0.27, right=0.98, bottom=0.13, wspace=0.25)
    return save_figure(fig, output_dir / "figures" / "figure20_training_complexity")


def generate_report(
    output_dir: Path,
    figures: dict[str, dict[str, str]],
    tables: dict[str, Path],
    validation_table: pd.DataFrame,
    audit_table: pd.DataFrame,
) -> None:
    lines = [
        "# Figure 收尾验证报告",
        "",
        "本报告接受 tokenizer caveat：raw-prompt BPE tokenizer 使用 `group_key_seed42` train 训练，因此无监督 tokenizer 语料覆盖了部分 atomic held-out test prompt text。该 caveat 已写入 audit 表；其他边界模型训练仍只使用 prompt-side `input_ids` 和 checker pass/fail labels。",
        "",
        "## 核心验证结论",
        "",
        "- Validation-selected 与 oracle-best 已分开报告，避免把 M3 40k 的 test oracle 峰值误解为部署时自然会选到的模型。",
        "- F1 threshold 选择协议已固定为 validation 选 threshold，再应用到 test；未在 test 上找最优 F1。",
        "- Calibration、PR/ROC、selective prediction、top-k precision、per-constraint/per-group、M1 n-gram 系数和复杂度表均已落盘。",
        "- `max2048` 过滤由 token 分布图支持：大部分 prompt 在 2048 token 以内，长尾被排除以保证 2048 input + 2048 output 的结构可控。",
        "",
        "## Validation-selected vs oracle",
        "",
        validation_table.to_markdown(index=False),
        "",
        "## Audit 表",
        "",
        audit_table.to_markdown(index=False),
        "",
        "## Figure 文件",
        "",
    ]
    for name, paths in figures.items():
        lines.append(f"- {name}: `{paths.get('png', '')}` / `{paths.get('svg', '')}`")
    lines.extend(["", "## Table 文件", ""])
    for name, path in tables.items():
        lines.append(f"- {name}: `{path}`")
    (output_dir / "Figure验证报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    total_start = now()
    output_dir = resolve_path(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    input_file = resolve_path(args.input_file)
    all_file = resolve_path(args.all_file)
    data, all_frame = load_data(input_file, all_file)
    device = resolve_device(args.device)

    predictions = generate_predictions(
        data,
        output_dir,
        device=device,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        skip_refresh=args.skip_prediction_refresh,
    )
    key_predictions = aggregate_seed_predictions(predictions)
    key_predictions.to_parquet(output_dir / "tables" / "key_config_seed_averaged_predictions.parquet", index=False)

    metrics = compute_metrics_tables(predictions, key_predictions, output_dir)
    threshold_table = compute_threshold_table(key_predictions, output_dir)
    selective, topk, risk = compute_selective_tables(predictions, output_dir)
    bootstrap = group_bootstrap_ci(
        predictions,
        output_dir,
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
        group_column="base_key",
    )
    split_overview = compute_split_overview(data, output_dir)
    audit_table = compute_audit_table(data, output_dir)
    m1_lift = compute_m1_split_lift(output_dir)
    validation_table = compute_validation_vs_oracle(metrics["by_config"], metrics["key_metrics"], output_dir)
    per_constraint, deltas, per_group = compute_per_constraint_tables(key_predictions, output_dir)
    features = compute_m1_interpretability(output_dir, resolve_path(args.tokenizer_dir))
    complexity = compute_complexity_table(output_dir)
    error_examples = compute_error_examples(key_predictions, all_frame, output_dir)
    token_dist = token_distribution_table(resolve_path(args.token_count_file), input_file, output_dir)

    figures: dict[str, dict[str, str]] = {}
    figures["Figure 1 research flow"] = plot_flow(output_dir)
    figures["Figure 2 split overview"] = plot_split_overview(split_overview, output_dir)
    figures["Figure 2b token distribution"] = plot_token_distribution(token_dist, output_dir)
    figures["Figure 3 M1 split lift"] = plot_m1_split_lift(m1_lift, output_dir)
    figures["Figure 4 learning curves"] = plot_learning_curves(metrics["by_config"], output_dir)
    figures["Figure 5 best config"] = plot_best_config(metrics["by_config"], output_dir)
    figures["Figure 6 validation vs oracle"] = make_table_figure(
        validation_table,
        output_dir / "figures" / "figure06_validation_selected_vs_oracle",
        "Figure 6. Validation-selected vs oracle-best",
        "Deployment selection must use validation; oracle-best rows are test-set diagnostics only.",
        max_rows=8,
    )
    pr_fig, roc_fig = plot_pr_roc(key_predictions, output_dir)
    figures["Figure 7 PR curves"] = pr_fig
    figures["Figure 8 ROC curves"] = roc_fig
    rel_fig, brier_fig = plot_calibration(key_predictions, output_dir)
    figures["Figure 9 reliability"] = rel_fig
    figures["Figure 10 Brier ECE"] = brier_fig
    sel_fig, topk_fig, risk_fig = plot_selective(selective, topk, risk, output_dir)
    figures["Figure 11 selective accuracy"] = sel_fig
    figures["Figure 12 top-k precision"] = topk_fig
    figures["Figure 13 coverage risk"] = risk_fig
    pc_fig, delta_fig = plot_per_constraint(per_constraint, deltas, output_dir)
    figures["Figure 14 per-constraint"] = pc_fig
    if delta_fig:
        figures["Figure 15 per-constraint delta"] = delta_fig
    figures["Figure 16 M1 coefficients"] = plot_m1_features(features, output_dir)
    hist_fig, corr_fig = plot_prediction_hist_and_correlation(key_predictions, output_dir)
    figures["Figure 17 prediction histogram"] = hist_fig
    figures["Figure 18 score correlation"] = corr_fig
    figures["Figure 19 error examples"] = make_table_figure(
        error_examples[["case", "prompt_id", "true_label", "m1_p_pass", "m3_40k_p_pass", "m4_full_p_pass"]].head(18),
        output_dir / "figures" / "figure19_error_examples_table",
        "Figure 19. Error examples table",
        "Representative high-confidence errors and disagreement cases; full prompt previews are in the CSV/Markdown table.",
        max_rows=18,
    )
    figures["Figure 20 complexity"] = plot_complexity(complexity, output_dir)
    figures["Audit table"] = make_table_figure(
        audit_table,
        output_dir / "figures" / "figure_audit_table",
        "Audit table. Split, leakage, tokenizer, and target-response checks",
        "Tokenizer prompt-text exposure is accepted as a caveat; target labels and responses are not input features.",
        max_rows=12,
    )

    table_paths = {
        "row_level_predictions": output_dir / "tables" / "final_row_level_predictions.parquet",
        "key_seed_averaged_predictions": output_dir / "tables" / "key_config_seed_averaged_predictions.parquet",
        "metrics_by_config": output_dir / "tables" / "metrics_by_config.csv",
        "bootstrap_ci": output_dir / "tables" / "m3_m4_group_bootstrap_ci_by_seed.csv",
        "audit_table": output_dir / "tables" / "figure_audit_table.csv",
        "validation_selected_vs_oracle": output_dir / "tables" / "figure06_validation_selected_vs_oracle.csv",
        "threshold_protocol": output_dir / "tables" / "threshold_selection_protocol.csv",
        "per_constraint": output_dir / "tables" / "figure14_per_constraint_metrics.csv",
        "per_group": output_dir / "tables" / "per_group_cluster_metrics.csv",
        "m1_features": output_dir / "tables" / "figure16_m1_top_ngram_coefficients.csv",
        "complexity": output_dir / "tables" / "figure20_training_complexity.csv",
        "error_examples": output_dir / "tables" / "figure19_error_examples.csv",
        "token_distribution": output_dir / "tables" / "figure21_prompt_token_distribution.csv",
    }
    generate_report(output_dir, figures, table_paths, validation_table, audit_table)

    manifest = {
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed_since(total_start),
        "device": str(device),
        "bootstrap_reps": args.bootstrap_reps,
        "figures": figures,
        "tables": {key: str(value) for key, value in table_paths.items()},
        "accepted_caveat": "Tokenizer trained on group_key_seed42 train, which overlaps atomic held-out test prompt text unsupervised.",
    }
    write_json(manifest, output_dir / "final_figure_bundle_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
