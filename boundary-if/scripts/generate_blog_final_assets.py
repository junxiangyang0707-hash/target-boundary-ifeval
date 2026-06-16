from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
import time
from dataclasses import dataclass
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
    precision_score,
    roc_auc_score,
)

from boundary_if.models.m1_tfidf_logreg import (
    M1TfidfLogRegConfig,
    feature_summary,
    fit_m1_pipeline,
    predict_m1,
)
from boundary_if.models.m4_pretrained_encoder import (
    M4EncoderConfig,
    M4FrozenEncoderClassifier,
    make_m4_classifier_dataloader,
    predict_m4,
)
from boundary_if.models.tiny_transformer import (
    M3TinyTransformer,
    M3TinyTransformerConfig,
    make_m3_dataloader,
    predict_m3,
)

STRICT_INPUT = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k_atomic_train.max2048.parquet"
)
STRICT_FULL_TOKENIZED = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k_atomic_train.parquet"
)
OLD_FULL_TOKENIZED = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.parquet"
)
PER_CONSTRAINT_FILE = "data/checks/qwen3_4b_instruct_2507_under2048_ifevalg_deterministic/per_constraint.parquet"
M1_SPLIT_FILE = "runs/m1_split_comparison/m1_split_comparison_test.csv"
SPLIT_AUDIT_FILE = "data/splits/qwen3_4b_instruct_2507_under2048_nontruncated/split_audit.json"
STRICT_AGGREGATE = "runs/strict_atomic_tokenizer_multiseed_key_runs/requested_multiseed_aggregate.csv"
STRICT_DETAIL = "runs/strict_atomic_tokenizer_multiseed_key_runs/requested_multiseed_summary.csv"
OUTPUT_DIR = "runs/blog_final_assets"

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#DD74B8",
    "pink_dark": "#8C3E74",
}


@dataclass(frozen=True)
class StrictRun:
    run_id: str
    tokenizer_protocol: str
    model_family: str
    model_variant: str
    train_size_label: str
    requested_train_size: str
    effective_train_size: int
    seed: int
    model_path: str | None
    manifest_path: str | None
    pred_kind: str
    source_prediction_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final blog analysis assets.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--input-file", default=STRICT_INPUT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(raw_device)


def load_eval_frame(path: Path) -> pd.DataFrame:
    columns = [
        "prompt_id",
        "split",
        "label",
        "input_ids",
        "attention_mask",
        "raw_prompt_bpe_token_count_full",
        "raw_prompt_bpe_token_count",
        "raw_prompt_bpe_truncated",
        "num_constraints",
        "cluster",
        "length_bin",
        "constraint_signature",
        "constraint_family_signature",
        "instruction_ids",
    ]
    frame = pd.read_parquet(path, columns=columns)
    return frame[frame["split"].isin(["val", "test"])].copy()


def load_train_frame_for_m1(path: Path) -> pd.DataFrame:
    columns = [
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
    frame = pd.read_parquet(path, columns=columns)
    return frame[frame["split"] == "train"].copy()


def config_from_checkpoint(raw_config: dict[str, Any], cls: type) -> Any:
    field_names = cls.__dataclass_fields__.keys()
    payload = {key: raw_config[key] for key in field_names if key in raw_config}
    return cls(**payload)


def make_strict_runs() -> list[StrictRun]:
    runs: list[StrictRun] = []
    for seed in (42, 43, 44):
        if seed == 42:
            manifest = "runs/m1_strict_atomic_tokenizer_atomic_constraint_heldout_seed42_max2048/train_full/manifest.json"
            source_pred = None
        else:
            manifest = f"runs/strict_atomic_tokenizer_multiseed_key_runs/m1_tfidf_full/seed_{seed}/train_full/manifest.json"
            source_pred = None
        runs.append(
            StrictRun(
                run_id=f"strict__M1_TFIDF__baseline__full__seed{seed}",
                tokenizer_protocol="strict_atomic_train",
                model_family="M1_TFIDF",
                model_variant="baseline",
                train_size_label="full",
                requested_train_size="full",
                effective_train_size=61504,
                seed=seed,
                model_path=None,
                manifest_path=manifest,
                pred_kind="m1",
                source_prediction_path=source_pred,
            )
        )
    for seed in (42, 43, 44):
        for train_size_label, effective_train_size in (("40k", 40000), ("full", 61504)):
            if seed == 42:
                base = f"runs/strict_atomic_tokenizer_key_runs/m3_mean_{train_size_label}_seed42"
            else:
                base = f"runs/strict_atomic_tokenizer_multiseed_key_runs/m3_mean/seed_{seed}/train_{train_size_label}"
            runs.append(
                StrictRun(
                    run_id=f"strict__M3_mean__mean_pooling__{train_size_label}__seed{seed}",
                    tokenizer_protocol="strict_atomic_train",
                    model_family="M3_mean",
                    model_variant="mean_pooling",
                    train_size_label=train_size_label,
                    requested_train_size=train_size_label,
                    effective_train_size=effective_train_size,
                    seed=seed,
                    model_path=f"{base}/model.pt",
                    manifest_path=f"{base}/manifest.json",
                    pred_kind="m3",
                    source_prediction_path=f"{base}/predictions.parquet" if seed == 42 else None,
                )
            )
    for seed in (42, 43, 44):
        for train_size_label, effective_train_size in (("20k", 20000), ("40k", 40000), ("full", 61504)):
            if seed == 42 and train_size_label == "full":
                base = "runs/strict_atomic_tokenizer_key_runs/m4_frozen_full_seed42"
            else:
                base = f"runs/strict_atomic_tokenizer_multiseed_key_runs/m4/seed_{seed}/train_{train_size_label}"
            runs.append(
                StrictRun(
                    run_id=f"strict__M4_frozen__frozen_encoder__{train_size_label}__seed{seed}",
                    tokenizer_protocol="strict_atomic_train",
                    model_family="M4_frozen",
                    model_variant="frozen_encoder",
                    train_size_label=train_size_label,
                    requested_train_size=train_size_label,
                    effective_train_size=effective_train_size,
                    seed=seed,
                    model_path=f"{base}/model.pt",
                    manifest_path=f"{base}/manifest.json",
                    pred_kind="m4",
                    source_prediction_path=f"{base}/predictions.parquet" if seed == 42 and train_size_label == "full" else None,
                )
            )
    return runs


def standardize_prediction_frame(run: StrictRun, frame: pd.DataFrame) -> pd.DataFrame:
    proba_col = {
        "m1": "m1_pred_proba",
        "m3": "m3_pred_proba",
        "m4": "m4_pred_proba",
    }[run.pred_kind]
    pred_col = {
        "m1": "m1_pred_label",
        "m3": "m3_pred_label",
        "m4": "m4_pred_label",
    }[run.pred_kind]
    output = frame[frame["split"].isin(["val", "test"])].copy()
    output = output.rename(columns={proba_col: "p_pred", pred_col: "pred_label"})
    output["run_id"] = run.run_id
    output["tokenizer_protocol"] = run.tokenizer_protocol
    output["model_family"] = run.model_family
    output["model_variant"] = run.model_variant
    output["train_size_label"] = run.train_size_label
    output["requested_train_size"] = run.requested_train_size
    output["effective_train_size"] = run.effective_train_size
    output["seed"] = run.seed
    columns = [
        "run_id",
        "tokenizer_protocol",
        "model_family",
        "model_variant",
        "train_size_label",
        "requested_train_size",
        "effective_train_size",
        "seed",
        "prompt_id",
        "split",
        "label",
        "p_pred",
        "pred_label",
        "raw_prompt_bpe_token_count_full",
        "raw_prompt_bpe_token_count",
        "raw_prompt_bpe_truncated",
        "num_constraints",
        "cluster",
        "length_bin",
    ]
    return output[columns]


def generate_m1_predictions(
    run: StrictRun,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    config = M1TfidfLogRegConfig(random_state=run.seed)
    pipeline = fit_m1_pipeline(train_frame, config)
    predictions = predict_m1(pipeline, eval_frame, config.threshold)
    predictions.to_parquet(cache_path, index=False)
    if run.seed == 42:
        features = feature_summary(pipeline, top_n=60)
        features_path = cache_path.parent / "m1_seed42_feature_coefficients.csv"
        features.to_csv(features_path, index=False, encoding="utf-8")
        model_path = cache_path.parent / "m1_seed42_model.joblib"
        joblib.dump(pipeline, model_path)
    return predictions


def generate_m3_predictions(
    run: StrictRun,
    eval_frame: pd.DataFrame,
    cache_path: Path,
    device: torch.device,
    eval_batch_size: int,
) -> pd.DataFrame:
    checkpoint = torch.load(resolve(run.model_path or ""), map_location="cpu")
    config = config_from_checkpoint(checkpoint["config"], M3TinyTransformerConfig)
    model = M3TinyTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    loader = make_m3_dataloader(
        eval_frame,
        config,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = predict_m3(model, loader, device=device)
    predictions.to_parquet(cache_path, index=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def generate_m4_predictions(
    run: StrictRun,
    eval_frame: pd.DataFrame,
    cache_path: Path,
    device: torch.device,
    eval_batch_size: int,
) -> pd.DataFrame:
    checkpoint = torch.load(resolve(run.model_path or ""), map_location="cpu")
    config = config_from_checkpoint(checkpoint["config"], M4EncoderConfig)
    model = M4FrozenEncoderClassifier(config)
    model.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
    model.pass_head.load_state_dict(checkpoint["pass_head_state_dict"])
    model.to(device)
    loader = make_m4_classifier_dataloader(
        eval_frame,
        config,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = predict_m4(model, loader, device=device)
    predictions.to_parquet(cache_path, index=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def ensure_predictions(args: argparse.Namespace, output_dir: Path, runs: list[StrictRun]) -> pd.DataFrame:
    predictions_dir = output_dir / "prediction_cache"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    eval_frame = load_eval_frame(resolve(args.input_file))
    train_frame = load_train_frame_for_m1(resolve(args.input_file))
    device = resolve_device(args.device)
    print(json.dumps({"stage": "prediction_device", "device": str(device)}, ensure_ascii=False), flush=True)

    frames = []
    for run in runs:
        cache_path = predictions_dir / f"{run.run_id}.parquet"
        if cache_path.exists() and not args.force_predictions:
            raw_predictions = pd.read_parquet(cache_path)
        elif run.source_prediction_path and resolve(run.source_prediction_path).exists() and not args.force_predictions:
            source_frame = pd.read_parquet(resolve(run.source_prediction_path))
            raw_predictions = source_frame[source_frame["split"].isin(["val", "test"])].copy()
            raw_predictions.to_parquet(cache_path, index=False)
        elif args.skip_predictions:
            raise FileNotFoundError(f"Missing cached predictions for {run.run_id}: {cache_path}")
        elif run.pred_kind == "m1":
            raw_predictions = generate_m1_predictions(run, train_frame, eval_frame, cache_path)
        elif run.pred_kind == "m3":
            raw_predictions = generate_m3_predictions(run, eval_frame, cache_path, device, args.eval_batch_size)
        elif run.pred_kind == "m4":
            raw_predictions = generate_m4_predictions(run, eval_frame, cache_path, device, args.eval_batch_size)
        else:
            raise ValueError(f"Unsupported pred_kind: {run.pred_kind}")
        frames.append(standardize_prediction_frame(run, raw_predictions))
        print(json.dumps({"stage": "prediction_ready", "run_id": run.run_id, "rows": len(frames[-1])}, ensure_ascii=False), flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    return predictions


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bins, right=True) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    total = len(labels)
    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        left = float(bins[idx])
        right = float(bins[idx + 1])
        if count == 0:
            avg_pred = np.nan
            empirical = np.nan
        else:
            avg_pred = float(probs[mask].mean())
            empirical = float(labels[mask].mean())
            ece += (count / total) * abs(avg_pred - empirical)
        rows.append(
            {
                "bin": idx,
                "bin_left": left,
                "bin_right": right,
                "bin_midpoint": (left + right) / 2,
                "row_count": count,
                "avg_pred": avg_pred,
                "empirical_pass_rate": empirical,
            }
        )
    return float(ece), pd.DataFrame(rows)


def binary_metrics(labels: pd.Series, probs: pd.Series, threshold: float = 0.5) -> dict[str, float | int | None]:
    y = labels.astype(int).to_numpy()
    p = probs.astype(float).to_numpy()
    pred = (p >= threshold).astype(int)
    payload: dict[str, float | int | None] = {
        "row_count": int(len(y)),
        "positive_count": int(y.sum()),
        "baseline_AUPRC": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) == 2:
        payload["AUROC"] = float(roc_auc_score(y, p))
        payload["AUPRC"] = float(average_precision_score(y, p))
    else:
        payload["AUROC"] = None
        payload["AUPRC"] = None
    payload["ECE"], _ = expected_calibration_error(y, p, n_bins=10)
    return payload


def build_run_metrics(predictions: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    calibration_rows = []
    group_cols = [
        "run_id",
        "tokenizer_protocol",
        "model_family",
        "model_variant",
        "train_size_label",
        "requested_train_size",
        "effective_train_size",
        "seed",
        "split",
    ]
    for keys, part in predictions.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        metrics = binary_metrics(part["label"], part["p_pred"])
        row.update(metrics)
        rows.append(row)
        ece, bins = expected_calibration_error(
            part["label"].astype(int).to_numpy(),
            part["p_pred"].astype(float).to_numpy(),
            n_bins=10,
        )
        bins.insert(0, "split", row["split"])
        bins.insert(0, "seed", row["seed"])
        bins.insert(0, "train_size_label", row["train_size_label"])
        bins.insert(0, "model_variant", row["model_variant"])
        bins.insert(0, "model_family", row["model_family"])
        bins.insert(0, "run_id", row["run_id"])
        bins["ECE"] = ece
        calibration_rows.append(bins)
    metrics_frame = pd.DataFrame(rows)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    metrics_frame.to_csv(output_dir / "run_metrics.csv", index=False, encoding="utf-8")
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False, encoding="utf-8")
    return metrics_frame, calibration


def summarize_by_config(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["AUROC", "AUPRC", "f1", "brier", "ECE", "baseline_AUPRC", "accuracy"]
    group_cols = [
        "tokenizer_protocol",
        "model_family",
        "model_variant",
        "train_size_label",
        "requested_train_size",
        "effective_train_size",
        "split",
    ]
    rows = []
    for keys, part in run_metrics.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        row["run_count"] = int(part["seed"].nunique())
        row["seeds"] = ",".join(map(str, sorted(part["seed"].unique())))
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_selection_table(run_metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = summarize_by_config(run_metrics)
    rows = []
    for family, family_summary in summary.groupby("model_family", sort=True):
        split_wide = {}
        key_cols = ["tokenizer_protocol", "model_family", "model_variant", "train_size_label", "effective_train_size"]
        for _, row in family_summary.iterrows():
            key = tuple(row[col] for col in key_cols)
            split_wide.setdefault(key, {})
            split_wide[key][str(row["split"])] = row
        candidates = []
        for key, by_split in split_wide.items():
            if "val" in by_split and "test" in by_split:
                candidates.append((key, by_split))
        if not candidates:
            continue
        for rule, split_name, metric in [
            ("validation-selected by val AUPRC", "val", "AUPRC_mean"),
            ("oracle-best by test AUPRC", "test", "AUPRC_mean"),
        ]:
            key, by_split = max(candidates, key=lambda item: item[1][split_name][metric])
            val_row = by_split["val"]
            test_row = by_split["test"]
            rows.append(
                {
                    "model_family": family,
                    "tokenizer_protocol": key[0],
                    "selection_rule": rule,
                    "selected_config": f"{key[2]} {key[3]}",
                    "val_AUROC": val_row["AUROC_mean"],
                    "val_AUPRC": val_row["AUPRC_mean"],
                    "test_AUROC": test_row["AUROC_mean"],
                    "test_AUPRC": test_row["AUPRC_mean"],
                    "test_Brier": test_row["brier_mean"],
                    "test_ECE": test_row["ECE_mean"],
                    "run_count": test_row["run_count"],
                    "seeds": test_row["seeds"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "selection_table.csv", index=False, encoding="utf-8")
    return table


def build_selective_and_topk(predictions: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    include = {
        ("M1_TFIDF", "full"),
        ("M3_mean", "40k"),
        ("M3_mean", "full"),
        ("M4_frozen", "40k"),
        ("M4_frozen", "full"),
    }
    test = predictions[predictions["split"] == "test"].copy()
    test = test[test.apply(lambda row: (row["model_family"], row["train_size_label"]) in include, axis=1)]
    selective_rows = []
    for keys, part in test.groupby(
        ["run_id", "model_family", "model_variant", "train_size_label", "seed"],
        sort=True,
    ):
        run_id, family, variant, train_size, seed = keys
        part = part.copy()
        part["confidence"] = np.maximum(part["p_pred"], 1.0 - part["p_pred"])
        part["correct"] = ((part["p_pred"] >= 0.5).astype(int) == part["label"].astype(int)).astype(int)
        ordered = part.sort_values("confidence", ascending=False)
        for coverage in [0.1, 0.2, 0.5, 1.0]:
            k = max(1, int(round(len(ordered) * coverage)))
            selected = ordered.head(k)
            selective_rows.append(
                {
                    "run_id": run_id,
                    "model_family": family,
                    "model_variant": variant,
                    "train_size_label": train_size,
                    "seed": seed,
                    "coverage": coverage,
                    "selected_rows": k,
                    "selective_accuracy": float(selected["correct"].mean()),
                    "selective_risk": float(1.0 - selected["correct"].mean()),
                    "selected_positive_precision": float(selected["label"].mean()),
                    "mean_confidence": float(selected["confidence"].mean()),
                }
            )
    selective = pd.DataFrame(selective_rows)
    selective.to_csv(output_dir / "selective_metrics_by_run.csv", index=False, encoding="utf-8")
    selective_agg = aggregate_metric(
        selective,
        ["model_family", "model_variant", "train_size_label", "coverage"],
        ["selective_accuracy", "selective_risk", "selected_positive_precision", "mean_confidence"],
    )
    selective_agg.to_csv(output_dir / "selective_metrics.csv", index=False, encoding="utf-8")

    topk_rows = []
    for keys, part in test.groupby(
        ["run_id", "model_family", "model_variant", "train_size_label", "seed"],
        sort=True,
    ):
        run_id, family, variant, train_size, seed = keys
        ordered = part.sort_values("p_pred", ascending=False)
        positives = int(part["label"].sum())
        for k in [100, 500, 1000, 2000]:
            selected = ordered.head(min(k, len(ordered)))
            positive_count = int(selected["label"].sum())
            topk_rows.append(
                {
                    "run_id": run_id,
                    "model_family": family,
                    "model_variant": variant,
                    "train_size_label": train_size,
                    "seed": seed,
                    "k": k,
                    "selected_rows": int(len(selected)),
                    "precision_at_k": float(selected["label"].mean()),
                    "recall_at_k": float(positive_count / positives) if positives else np.nan,
                    "positive_count_at_k": positive_count,
                    "mean_score_at_k": float(selected["p_pred"].mean()),
                }
            )
    topk = pd.DataFrame(topk_rows)
    topk.to_csv(output_dir / "topk_metrics_by_run.csv", index=False, encoding="utf-8")
    topk_agg = aggregate_metric(
        topk,
        ["model_family", "model_variant", "train_size_label", "k"],
        ["precision_at_k", "recall_at_k", "positive_count_at_k", "mean_score_at_k"],
    )
    topk_agg.to_csv(output_dir / "topk_metrics.csv", index=False, encoding="utf-8")
    return selective_agg, topk_agg


def aggregate_metric(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=True))
        row["run_count"] = int(part["seed"].nunique()) if "seed" in part.columns else int(len(part))
        row["seeds"] = ",".join(map(str, sorted(part["seed"].unique()))) if "seed" in part.columns else ""
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_tokenizer_audit(output_dir: Path) -> pd.DataFrame:
    rows = []
    for protocol, path, vocab_size, fit_prompt_count, fit_split in [
        ("old_group_key", resolve(OLD_FULL_TOKENIZED), 8000, np.nan, "group_key train"),
        ("strict_atomic_train", resolve(STRICT_FULL_TOKENIZED), 8000, 61655, "atomic train"),
    ]:
        frame = pd.read_parquet(
            path,
            columns=["split", "input_ids", "raw_prompt_bpe_token_count_full", "raw_prompt_bpe_truncated"],
        )
        train = frame[frame["split"] == "train"]
        test = frame[frame["split"] == "test"]
        train_ids = set()
        for ids in train["input_ids"]:
            train_ids.update(int(x) for x in ids)
        unseen_test_tokens = 0
        total_test_tokens = 0
        unseen_ids = set()
        for ids in test["input_ids"]:
            values = [int(x) for x in ids]
            total_test_tokens += len(values)
            for token_id in values:
                if token_id not in train_ids:
                    unseen_test_tokens += 1
                    unseen_ids.add(token_id)
        rows.append(
            {
                "tokenizer_protocol": protocol,
                "vocab_size": vocab_size,
                "fit_prompt_count": fit_prompt_count,
                "fit_split": fit_split,
                "row_count": int(len(frame)),
                "mean_token_length": float(frame["raw_prompt_bpe_token_count_full"].mean()),
                "p95_token_length": float(frame["raw_prompt_bpe_token_count_full"].quantile(0.95)),
                "max_token_length": int(frame["raw_prompt_bpe_token_count_full"].max()),
                "truncation_rate": float(frame["raw_prompt_bpe_truncated"].mean()),
                "test_token_ids_unseen_in_atomic_train": int(len(unseen_ids)),
                "test_token_mass_unseen_in_atomic_train": float(unseen_test_tokens / max(1, total_test_tokens)),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "tokenizer_audit.csv", index=False, encoding="utf-8")
    return audit


def build_per_constraint(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    selected = predictions[
        (predictions["split"] == "test")
        & (
            ((predictions["model_family"] == "M1_TFIDF") & (predictions["train_size_label"] == "full"))
            | ((predictions["model_family"] == "M3_mean") & (predictions["train_size_label"] == "full"))
            | ((predictions["model_family"] == "M4_frozen") & (predictions["train_size_label"] == "40k"))
        )
    ].copy()
    averaged = (
        selected.groupby(["model_family", "train_size_label", "prompt_id"], as_index=False)
        .agg(label=("label", "first"), p_pred=("p_pred", "mean"))
    )
    pivot = averaged.pivot(index="prompt_id", columns="model_family", values="p_pred").reset_index()
    labels = averaged.drop_duplicates("prompt_id")[["prompt_id", "label"]]
    per_constraint = pd.read_parquet(resolve(PER_CONSTRAINT_FILE), columns=["prompt_id", "instruction_id"])
    joined = per_constraint.merge(labels, on="prompt_id", how="inner").merge(pivot, on="prompt_id", how="inner")
    rows = []
    for instruction_id, part in joined.groupby("instruction_id", sort=True):
        row: dict[str, Any] = {
            "constraint_id": instruction_id,
            "test_rows": int(len(part)),
            "positive_rate": float(part["label"].mean()),
        }
        for model in ["M1_TFIDF", "M3_mean", "M4_frozen"]:
            if model in part.columns and part["label"].nunique() == 2:
                row[f"{model}_AUPRC"] = float(average_precision_score(part["label"], part[model]))
                row[f"{model}_AUROC"] = float(roc_auc_score(part["label"], part[model]))
            else:
                row[f"{model}_AUPRC"] = np.nan
                row[f"{model}_AUROC"] = np.nan
        scores = {model: row.get(f"{model}_AUPRC", np.nan) for model in ["M1_TFIDF", "M3_mean", "M4_frozen"]}
        valid_scores = {key: value for key, value in scores.items() if pd.notna(value)}
        row["best_model"] = max(valid_scores, key=valid_scores.get) if valid_scores else None
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["test_rows", "constraint_id"], ascending=[False, True])
    result.to_csv(output_dir / "per_constraint_metrics.csv", index=False, encoding="utf-8")
    return result


def build_split_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = json.loads(resolve(SPLIT_AUDIT_FILE).read_text(encoding="utf-8"))
    split_rows = []
    name_map = {
        "group_key_seed42": ("group-key split", "base_key"),
        "atomic_constraint_heldout_seed42": ("atomic constraint held-out", "instruction_id"),
        "composition_heldout_c1": ("composition C1", "num_constraints"),
        "composition_heldout_c2": ("composition C2", "num_constraints"),
    }
    for split_name, (label, heldout_unit) in name_map.items():
        payload = audit["splits"][split_name]
        by_split = payload["by_split"]
        split_rows.append(
            {
                "split_name": label,
                "train_rows": by_split["train"]["row_count"],
                "val_rows": by_split["val"]["row_count"],
                "test_rows": by_split["test"]["row_count"],
                "test_positive_rate": by_split["test"]["pass_rate"],
                "baseline_AUPRC": by_split["test"]["pass_rate"],
                "heldout_unit": heldout_unit,
            }
        )
    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8")

    m1_split = pd.read_csv(resolve(M1_SPLIT_FILE))
    m1_split["split_label"] = m1_split["dataset_split"].map(
        {
            "group_key_seed42": "group-key",
            "atomic_constraint_heldout_seed42": "constraint-id held-out",
            "composition_heldout_c1": "composition C1",
            "composition_heldout_c2": "composition C2",
        }
    )
    m1_split.to_csv(output_dir / "m1_split_performance.csv", index=False, encoding="utf-8")
    return split_table, m1_split


def copy_existing_figures(output_dir: Path) -> None:
    copies = {
        "figure4_strict_main_result": "runs/model_comparisons/strict_atomic_tokenizer_multiseed_key_runs/strict_atomic_tokenizer_multiseed_test_metrics.png",
        "figure5_tokenizer_protocol_audit": "runs/model_comparisons/group_vs_strict_tokenizer_key_results/group_vs_strict_tokenizer_key_results.png",
    }
    for name, source in copies.items():
        source_path = resolve(source)
        if source_path.exists():
            shutil.copy2(source_path, output_dir / f"{name}.png")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_flow(output_dir: Path) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    labels = [
        "Prompt",
        "Local target LLM",
        "Response",
        "Deterministic checker",
        "Pass/fail label",
        "Small boundary model",
        "P(pass)",
    ]
    xs = np.linspace(0.055, 0.945, len(labels))
    for x, label in zip(xs, labels, strict=True):
        ax.text(
            x,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color=TOKENS["ink"],
            bbox=dict(boxstyle="round,pad=0.38", facecolor=TOKENS["panel"], edgecolor=TOKENS["axis"]),
            transform=ax.transAxes,
        )
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        ax.annotate(
            "",
            xy=(right - 0.07, 0.52),
            xytext=(left + 0.07, 0.52),
            arrowprops=dict(arrowstyle="->", color=TOKENS["muted"], lw=1.2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )
    ax.set_title("Boundary-prediction pipeline", loc="left", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    save_figure(fig, output_dir, "figure1_boundary_pipeline")


def render_table(table: pd.DataFrame, output_dir: Path, name: str, title: str, percent_cols: list[str]) -> None:
    display = table.copy()
    for col in percent_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.1%}")
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(15.5, 0.75 + 0.5 * len(display)))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="semibold", color=TOKENS["ink"], pad=14)
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    tbl.scale(1, 1.45)
    tbl.auto_set_column_width(col=list(range(len(display.columns))))
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(TOKENS["axis"])
        if row == 0:
            cell.set_facecolor("#EEF3FF")
            cell.set_text_props(weight="bold", color=TOKENS["ink"])
        else:
            cell.set_facecolor(TOKENS["panel"])
            cell.set_text_props(color=TOKENS["ink"])
    save_figure(fig, output_dir, name)


def plot_m1_split(m1_split: pd.DataFrame, output_dir: Path) -> None:
    use_chart_theme()
    order = ["group-key", "constraint-id held-out", "composition C1", "composition C2"]
    frame = m1_split.copy()
    frame["split_label"] = pd.Categorical(frame["split_label"], categories=order, ordered=True)
    frame = frame.sort_values("split_label")
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars1 = ax.bar(
        x - width / 2,
        frame["positive_rate_baseline_auprc"],
        width=width,
        color=TOKENS["blue"],
        edgecolor=TOKENS["blue_dark"],
        label="positive-rate baseline",
    )
    bars2 = ax.bar(
        x + width / 2,
        frame["auprc"],
        width=width,
        color=TOKENS["orange"],
        edgecolor=TOKENS["orange_dark"],
        label="M1 TF-IDF",
    )
    for bar, lift in zip(bars2, frame["auprc_lift_vs_baseline"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{lift:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TOKENS["muted"],
        )
    ax.set_title("M1 TF-IDF remains above baseline across split protocols", loc="left", fontsize=14, fontweight="semibold")
    ax.set_ylabel("Test AUPRC")
    ax.set_xticks(x, frame["split_label"])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, max(frame["auprc"].max(), frame["positive_rate_baseline_auprc"].max()) + 0.12)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "figure3_m1_split_auprc")


def plot_strict_learning_curve(run_metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_by_config(run_metrics)
    test = summary[summary["split"] == "test"].copy()
    mapping = {"20k": 20000, "40k": 40000, "full": 61504}
    test["x"] = test["train_size_label"].map(mapping)
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    m1 = test[(test["model_family"] == "M1_TFIDF") & (test["train_size_label"] == "full")]
    if not m1.empty:
        baseline_value = float(m1.iloc[0]["AUPRC_mean"])
        ax.axhline(baseline_value, color=TOKENS["orange_dark"], linestyle="--", lw=1.2, label=f"M1 full {baseline_value:.1%}")
    positive_rate = float(test["baseline_AUPRC_mean"].dropna().iloc[0])
    ax.axhline(positive_rate, color=TOKENS["ink"], linestyle=":", lw=1.0, label=f"baseline {positive_rate:.1%}")
    for family, color, edge in [
        ("M3_mean", TOKENS["blue"], TOKENS["blue_dark"]),
        ("M4_frozen", TOKENS["olive"], TOKENS["olive_dark"]),
    ]:
        part = test[test["model_family"] == family].dropna(subset=["x"]).sort_values("x")
        ax.errorbar(
            part["x"],
            part["AUPRC_mean"],
            yerr=part["AUPRC_std"],
            marker="o",
            color=edge,
            markerfacecolor=color,
            markeredgecolor=edge,
            capsize=3,
            lw=1.8,
            label=family.replace("_", " "),
        )
    ax.set_title("Strict-tokenizer key points are sparse, but show the clean trend", loc="left", fontsize=14, fontweight="semibold")
    ax.set_ylabel("Test AUPRC")
    ax.set_xlabel("Target-labeled train rows")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks([20000, 40000, 61504], ["20k", "40k", "full"])
    ax.set_ylim(0.12, 0.56)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "figure7_strict_tokenizer_key_curve")


def plot_selective(selective: pd.DataFrame, output_dir: Path) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    labels = label_for_config(selective)
    selective = selective.assign(label=labels)
    palette = sns.color_palette("colorblind", n_colors=selective["label"].nunique())
    for (label, part), color in zip(selective.groupby("label", sort=False), palette, strict=False):
        part = part.sort_values("coverage")
        ax.errorbar(
            part["coverage"],
            part["selective_accuracy_mean"],
            yerr=part["selective_accuracy_std"],
            marker="o",
            lw=1.8,
            capsize=3,
            label=label,
            color=color,
        )
    ax.set_title("Strict-tokenizer selective accuracy by confidence coverage", loc="left", fontsize=14, fontweight="semibold")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_xticks([0.1, 0.2, 0.5, 1.0], ["10%", "20%", "50%", "100%"])
    ax.set_xlim(0.08, 1.02)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0.68, 1.01)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "figure8_strict_selective_accuracy")


def plot_topk(topk: pd.DataFrame, output_dir: Path) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    topk = topk.assign(label=label_for_config(topk))
    palette = sns.color_palette("colorblind", n_colors=topk["label"].nunique())
    for (label, part), color in zip(topk.groupby("label", sort=False), palette, strict=False):
        part = part.sort_values("k")
        ax.errorbar(
            part["k"],
            part["precision_at_k_mean"],
            yerr=part["precision_at_k_std"],
            marker="o",
            lw=1.8,
            capsize=3,
            label=label,
            color=color,
        )
    ax.set_title("Strict-tokenizer precision among top predicted-pass prompts", loc="left", fontsize=14, fontweight="semibold")
    ax.set_xlabel("Top-k by predicted P(pass)")
    ax.set_ylabel("Precision@k")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks([100, 500, 1000, 2000])
    ax.set_ylim(0.15, 0.75)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "figure9_strict_topk_precision")


def label_for_config(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: {
            "M1_TFIDF": "M1 strict full",
            "M3_mean": f"M3 strict {row['train_size_label']}",
            "M4_frozen": f"M4 strict {row['train_size_label']}",
        }.get(row["model_family"], f"{row['model_family']} {row['train_size_label']}"),
        axis=1,
    )


def plot_calibration(calibration: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    selected = {
        ("M1_TFIDF", "full"),
        ("M3_mean", "40k"),
        ("M3_mean", "full"),
        ("M4_frozen", "full"),
    }
    cal = calibration[
        (calibration["split"] == "test")
        & (calibration.apply(lambda row: (row["model_family"], row["train_size_label"]) in selected, axis=1))
    ].copy()
    cal = (
        cal.groupby(["model_family", "train_size_label", "bin", "bin_midpoint"], as_index=False)
        .agg(
            avg_pred=("avg_pred", "mean"),
            empirical_pass_rate=("empirical_pass_rate", "mean"),
            row_count=("row_count", "sum"),
        )
    )
    pred = predictions[
        (predictions["split"] == "test")
        & (predictions.apply(lambda row: (row["model_family"], row["train_size_label"]) in selected, axis=1))
    ].copy()
    use_chart_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    axes[0].plot([0, 1], [0, 1], linestyle=":", color=TOKENS["ink"], label="perfect calibration")
    cal = cal.assign(label=label_for_config(cal))
    pred = pred.assign(model_label=label_for_config(pred))
    palette = sns.color_palette("colorblind", n_colors=cal["label"].nunique())
    for (label, part), color in zip(cal.groupby("label", sort=False), palette, strict=False):
        part = part.dropna(subset=["avg_pred", "empirical_pass_rate"]).sort_values("bin_midpoint")
        axes[0].plot(part["avg_pred"], part["empirical_pass_rate"], marker="o", lw=1.6, label=label, color=color)
    axes[0].set_title("Reliability diagram", loc="left", fontsize=12, fontweight="semibold")
    axes[0].set_xlabel("Predicted P(pass)")
    axes[0].set_ylabel("Empirical pass rate")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=8)
    sns.histplot(
        data=pred,
        x="p_pred",
        hue="model_label",
        bins=np.linspace(0, 1, 21),
        element="step",
        stat="count",
        common_norm=False,
        ax=axes[1],
    )
    legend = axes[1].get_legend()
    if legend is not None:
        legend.set_title("model")
    axes[1].set_title("Prediction histogram", loc="left", fontsize=12, fontweight="semibold")
    axes[1].set_xlabel("Predicted P(pass)")
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.suptitle("Strict-tokenizer calibration on atomic held-out test", x=0.04, y=1.02, ha="left", fontsize=14, fontweight="semibold")
    save_figure(fig, output_dir, "figure10_strict_calibration")


def plot_m1_features(output_dir: Path) -> None:
    candidates = [
        output_dir / "prediction_cache" / "m1_seed42_feature_coefficients.csv",
        resolve("runs/m1_strict_atomic_tokenizer_atomic_constraint_heldout_seed42_max2048/train_full/top_features.csv"),
    ]
    feature_path = next(path for path in candidates if path.exists())
    features = pd.read_csv(feature_path).copy()
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(
            str(resolve("data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_atomic_train/tokenizer.json"))
        )

        def decode_feature(raw_feature: str) -> str:
            try:
                token_ids = [int(piece) for piece in str(raw_feature).split()]
                decoded = tokenizer.decode(token_ids).strip()
                decoded = decoded.replace("\n", "\\n")
                return textwrap.shorten(decoded or str(raw_feature), width=42, placeholder="...")
            except Exception:
                return str(raw_feature)

        features["decoded_feature"] = features["feature"].map(decode_feature)
    except Exception:
        features["decoded_feature"] = features["feature"].astype(str)
    features.to_csv(output_dir / "feature_coefficients.csv", index=False, encoding="utf-8")
    use_chart_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    for ax, direction, color, edge in [
        (axes[0], "positive", TOKENS["orange"], TOKENS["orange_dark"]),
        (axes[1], "negative", TOKENS["blue"], TOKENS["blue_dark"]),
    ]:
        part = features[features["direction"] == direction].head(15).iloc[::-1]
        ax.barh(part["decoded_feature"], part["coefficient"], color=color, edgecolor=edge)
        ax.set_title(f"Top {direction} n-grams", loc="left", fontsize=12, fontweight="semibold")
        ax.set_xlabel("Logistic coefficient")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("M1 TF-IDF feature interpretation", x=0.04, y=1.02, ha="left", fontsize=14, fontweight="semibold")
    save_figure(fig, output_dir, "figure11_m1_feature_coefficients")


def plot_per_constraint(per_constraint: pd.DataFrame, output_dir: Path) -> None:
    top = per_constraint[
        (per_constraint["test_rows"] >= 300)
        & per_constraint[["M1_TFIDF_AUPRC", "M3_mean_AUPRC", "M4_frozen_AUPRC"]].notna().all(axis=1)
    ].head(20)
    long = top.melt(
        id_vars=["constraint_id", "test_rows", "positive_rate"],
        value_vars=["M1_TFIDF_AUPRC", "M3_mean_AUPRC", "M4_frozen_AUPRC"],
        var_name="model",
        value_name="AUPRC",
    )
    long["model"] = long["model"].str.replace("_AUPRC", "", regex=False)
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    pivot = long.pivot(index="constraint_id", columns="model", values="AUPRC")
    sns.heatmap(pivot, cmap="YlGnBu", vmin=0.0, vmax=min(1.0, float(np.nanmax(pivot.values)) + 0.05), annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Per-constraint AUPRC on top held-out atomic constraints", loc="left", fontsize=14, fontweight="semibold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, output_dir, "figure12_per_constraint_auprc")


def plot_appendix(output_dir: Path) -> None:
    use_chart_theme()
    # Appendix A1: old tokenizer common learning curves.
    common = pd.read_csv(resolve("runs/model_comparisons/m1_m3_m4_selective_prediction/m1_m3_m4_learning_curve_common.csv"))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    mapping = {"2k": 2000, "4k": 4000, "5k": 5000, "10k": 10000, "20k": 20000, "40k": 40000, "full": 61501}
    common["x"] = common["curve_label"].map(mapping)
    for model, part in common.groupby("model", sort=False):
        part = part.dropna(subset=["x"]).sort_values("x")
        ax.errorbar(part["x"], part["test_auprc_mean"], yerr=part["test_auprc_std"], marker="o", capsize=3, label=model)
    ax.set_title("Appendix: old-tokenizer learning curves", loc="left", fontsize=14, fontweight="semibold")
    ax.set_xlabel("Train rows")
    ax.set_ylabel("Test AUPRC")
    ax.set_xticks([2000, 4000, 5000, 10000, 20000, 40000, 61501], ["2k", "4k", "5k", "10k", "20k", "40k", "full"])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "appendix_a1_old_tokenizer_learning_curves")

    # Appendix A2/A3/A4 use existing aggregate tables.
    capacity = pd.read_csv(resolve("runs/m3_capacity_seed_sweep_55k_sample42_max2048/capacity_seed_sweep_aggregate.csv"))
    cls = pd.read_csv(resolve("runs/m3_initial_cls_55k_multiseed_atomic_constraint_heldout_max2048/multiseed_learning_curve_aggregate.csv"))
    mean55 = pd.read_csv(resolve("runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/figures/m3_55k_pooling_multiseed_aggregate.csv"))
    rows = []
    for _, row in capacity.iterrows():
        rows.append({"variant": row["display_name"], "test_AUPRC": row["test_auprc_mean"], "std": row["test_auprc_std"]})
    for _, row in mean55.iterrows():
        rows.append({"variant": f"{row['pooling']} pooling", "test_AUPRC": row["test_auprc_mean"], "std": row["test_auprc_std"]})
    cap_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(cap_plot["variant"], cap_plot["test_AUPRC"], yerr=cap_plot["std"], capsize=3, color=TOKENS["pink"], edgecolor=TOKENS["pink_dark"])
    ax.set_title("Appendix: old-tokenizer M3 pooling and capacity ablations", loc="left", fontsize=14, fontweight="semibold")
    ax.set_ylabel("Test AUPRC")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, output_dir, "appendix_a2_m3_capacity_pooling")

    seed_path = resolve("runs/m3_seed_sensitivity_55k_fixedsample42_max2048/seed_sensitivity_55k_sample42_summary.csv")
    if seed_path.exists():
        seed = pd.read_csv(seed_path)
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.plot(seed["train_seed"].astype(str), seed["test_auprc"], marker="o", color=TOKENS["blue_dark"])
        ax.set_title("Appendix: fixed 55k sample, different training seeds", loc="left", fontsize=14, fontweight="semibold")
        ax.set_xlabel("Training seed")
        ax.set_ylabel("Test AUPRC")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        save_figure(fig, output_dir, "appendix_a4_m3_55k_seed_stability")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    runs = make_strict_runs()
    predictions = ensure_predictions(args, output_dir, runs)
    run_metrics, calibration = build_run_metrics(predictions, output_dir)
    summary = summarize_by_config(run_metrics)
    summary.to_csv(output_dir / "run_metrics_by_config.csv", index=False, encoding="utf-8")
    selection = build_selection_table(run_metrics, output_dir)
    selective, topk = build_selective_and_topk(predictions, output_dir)
    tokenizer_audit = build_tokenizer_audit(output_dir)
    per_constraint = build_per_constraint(predictions, output_dir)
    split_table, m1_split = build_split_tables(output_dir)
    copy_existing_figures(output_dir)

    plot_flow(output_dir)
    render_table(
        split_table,
        output_dir,
        "figure2_split_summary_table",
        "Task and split summary",
        ["test_positive_rate", "baseline_AUPRC"],
    )
    plot_m1_split(m1_split, output_dir)
    render_table(
        selection[
            [
                "model_family",
                "selection_rule",
                "selected_config",
                "val_AUPRC",
                "test_AUROC",
                "test_AUPRC",
                "test_Brier",
                "test_ECE",
            ]
        ].assign(
            model_family=lambda df: df["model_family"].map(
                {"M1_TFIDF": "M1 TF-IDF", "M3_mean": "M3 mean", "M4_frozen": "M4 frozen"}
            ),
            selection_rule=lambda df: df["selection_rule"].map(
                {
                    "validation-selected by val AUPRC": "val-selected",
                    "oracle-best by test AUPRC": "test oracle",
                }
            ),
            selected_config=lambda df: df["selected_config"].str.replace("_", " ", regex=False),
        ),
        output_dir,
        "figure6_validation_vs_oracle_table",
        "Validation-selected vs oracle-best strict-tokenizer results",
        ["val_AUPRC", "test_AUROC", "test_AUPRC", "test_Brier", "test_ECE"],
    )
    plot_strict_learning_curve(run_metrics, output_dir)
    plot_selective(selective, output_dir)
    plot_topk(topk, output_dir)
    plot_calibration(calibration, predictions, output_dir)
    plot_m1_features(output_dir)
    plot_per_constraint(per_constraint, output_dir)
    render_table(
        tokenizer_audit,
        output_dir,
        "appendix_a5_tokenizer_diagnostics_table",
        "Tokenizer diagnostics",
        ["truncation_rate", "test_token_mass_unseen_in_atomic_train"],
    )
    plot_appendix(output_dir)

    manifest = {
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "prediction_rows": int(len(predictions)),
        "created_files": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
        "total_seconds": round(time.perf_counter() - start, 3),
    }
    write_json(manifest, output_dir / "blog_final_assets_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
