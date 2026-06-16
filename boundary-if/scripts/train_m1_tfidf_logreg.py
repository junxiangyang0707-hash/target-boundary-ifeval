from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from boundary_if.models.m1_tfidf_logreg import (
    MODEL_ID,
    MODEL_NAME,
    M1TfidfLogRegConfig,
    evaluate_by_split,
    feature_summary,
    fit_m1_pipeline,
    make_raw_bpe_length_bucket,
    pipeline_size_summary,
    predict_m1,
)

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "group_key_seed42.raw_prompt_byte_bpe_v8k.parquet"
)
DEFAULT_OUTPUT_DIR = "runs/m1_raw_prompt_bpe_tfidf_logreg_group_key_seed42"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train M1: raw-prompt BPE token n-gram TF-IDF + logistic regression."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=500_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="saga")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--drop-truncated", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--wandb-name", default="M1_raw_prompt_bpe_tfidf_logreg_group_key_seed42")
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "runs/wandb"))
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


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


def parse_eval_splits(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def load_data(path: Path, *, drop_truncated: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    if frame["prompt_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate prompt_id values.")
    if drop_truncated:
        frame = frame[~frame["raw_prompt_bpe_truncated"]].copy()
    frame["raw_bpe_length_bucket"] = make_raw_bpe_length_bucket(
        frame["raw_prompt_bpe_token_count_full"]
    ).astype(str)
    return frame


def select_split(frame: pd.DataFrame, split: str, limit: int | None) -> pd.DataFrame:
    selected = frame[frame["split"].astype(str) == split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split={split!r}.")
    if limit is not None:
        selected = selected.head(limit).copy()
    return selected


def split_count_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return (
        frame["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="row_count")
        .sort_values("split")
        .to_dict("records")
    )


def label_count_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return (
        frame["label"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="row_count")
        .sort_values("label")
        .to_dict("records")
    )


def init_wandb(args: argparse.Namespace, config: M1TfidfLogRegConfig):
    if args.no_wandb:
        return None

    import wandb

    wandb_dir = resolve_path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        dir=str(wandb_dir),
        name=args.wandb_name,
        job_type="train_m1",
        tags=["M1", "tfidf", "logistic-regression", "raw-prompt-bpe"],
        config={
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "train_split": args.train_split,
            "eval_splits": parse_eval_splits(args.eval_splits),
            "m1_config": config.to_dict(),
        },
    )


def flatten_metrics_for_wandb(metrics: dict[str, Any]) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for split, split_metrics in metrics["by_split"].items():
        for key, value in split_metrics.items():
            if key == "confusion":
                for confusion_key, confusion_value in value.items():
                    flattened[f"m1/{split}/confusion/{confusion_key}"] = int(confusion_value)
            elif isinstance(value, int | float) and value is not None:
                flattened[f"m1/{split}/{key}"] = value
    return flattened


def save_wandb_files(run: Any, paths: list[Path]) -> None:
    if run is None:
        return
    for path in paths:
        if path.exists():
            run.save(str(path), base_path=str(path.parent))


def main() -> None:
    args = parse_args()
    total_start = now()
    input_file = resolve_path(args.input_file)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = parse_eval_splits(args.eval_splits)

    config = M1TfidfLogRegConfig(
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        class_weight=None if args.class_weight in ("none", "None", "") else args.class_weight,
        threshold=args.threshold,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        drop_truncated=args.drop_truncated,
    )
    run = init_wandb(args, config)

    load_start = now()
    data = load_data(input_file, drop_truncated=args.drop_truncated)
    train_frame = select_split(data, args.train_split, args.limit_train)
    eval_frames = [
        select_split(data, split, args.limit_eval)
        for split in eval_splits
    ]
    fit_eval_frame = pd.concat([train_frame, *eval_frames], ignore_index=True)
    load_seconds = elapsed_since(load_start)
    if run is not None:
        run.log(
            {
                "m1/stage": 0,
                "m1/loaded_rows": len(data),
                "m1/train_rows": len(train_frame),
                "m1/load_data_seconds": load_seconds,
            },
            step=0,
        )

    train_start = now()
    pipeline = fit_m1_pipeline(train_frame, config)
    train_seconds = elapsed_since(train_start)
    if run is not None:
        run.log(
            {
                "m1/stage": 1,
                "m1/train_seconds": train_seconds,
            },
            step=1,
        )

    predict_start = now()
    prediction_frames = [
        predict_m1(pipeline, split_frame, config.threshold)
        for split_frame in [train_frame, *eval_frames]
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predict_seconds = elapsed_since(predict_start)
    if run is not None:
        run.log(
            {
                "m1/stage": 2,
                "m1/predict_seconds": predict_seconds,
            },
            step=2,
        )

    metrics = {
        "by_split": evaluate_by_split(predictions, config.threshold),
    }
    features = feature_summary(pipeline, top_n=args.top_features)

    model_path = output_dir / "model.joblib"
    predictions_path = output_dir / "predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    features_path = output_dir / "top_features.csv"
    manifest_path = output_dir / "manifest.json"

    joblib.dump(pipeline, model_path)
    if not args.no_save_predictions:
        predictions.to_parquet(predictions_path, index=False)
    features.to_csv(features_path, index=False, encoding="utf-8")

    manifest = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "config": config.to_dict(),
        "drop_truncated": args.drop_truncated,
        "data_counts": {
            "loaded_rows": int(len(data)),
            "fit_eval_rows": int(len(fit_eval_frame)),
            "split_counts_loaded": split_count_table(data),
            "split_counts_fit_eval": split_count_table(fit_eval_frame),
            "label_counts_fit_eval": label_count_table(fit_eval_frame),
            "train_rows": int(len(train_frame)),
        },
        "feature_space": pipeline_size_summary(pipeline),
        "output_files": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "manifest": str(manifest_path),
            "top_features": str(features_path),
            "predictions": None if args.no_save_predictions else str(predictions_path),
        },
        "timing_seconds": {
            "load_data": load_seconds,
            "train": train_seconds,
            "predict": predict_seconds,
            "total": elapsed_since(total_start),
        },
    }
    write_json(metrics, metrics_path)
    write_json(manifest, manifest_path)
    if run is not None:
        run.summary.update(
            {
                "model_id": MODEL_ID,
                "train_rows": int(len(train_frame)),
                "vocabulary_size": manifest["feature_space"]["vocabulary_size"],
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": manifest["timing_seconds"]["total"],
                "output_dir": str(output_dir),
            }
        )
        run.log(flatten_metrics_for_wandb(metrics), step=3)
        save_wandb_files(
            run,
            [
                metrics_path,
                manifest_path,
                features_path,
                model_path,
                *([] if args.no_save_predictions else [predictions_path]),
            ],
        )
        run.finish()
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
