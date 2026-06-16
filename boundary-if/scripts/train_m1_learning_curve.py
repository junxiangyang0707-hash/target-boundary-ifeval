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
from boundary_if.training.sampling import stable_sample_frame

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_OUTPUT_DIR = "runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048"
DEFAULT_TRAIN_SIZES = "2k,4k,5k,10k,20k,40k,50k,55k,80k"
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
            "Run M1 raw-prompt BPE TF-IDF + logistic-regression train-size "
            "learning curve with stable nested train sampling."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--train-sizes", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-key-column", default="prompt_id")
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
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument(
        "--wandb-group",
        default="M1_learning_curve_atomic_constraint_heldout_seed42_max2048",
    )
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


def parse_size_token(token: str) -> tuple[str, int | None]:
    normalized = token.strip().lower()
    if not normalized:
        raise ValueError("Empty train size token.")
    if normalized == "full":
        return "full", None
    multiplier = 1
    number_text = normalized
    if normalized.endswith("k"):
        multiplier = 1000
        number_text = normalized[:-1]
    size = int(number_text) * multiplier
    if size <= 0:
        raise ValueError(f"Train size must be positive, got {token!r}.")
    return normalized, size


def parse_train_sizes(raw_value: str) -> list[tuple[str, int | None]]:
    return [parse_size_token(token) for token in raw_value.split(",") if token.strip()]


def load_data(path: Path, *, drop_truncated: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    if frame["prompt_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate prompt_id values.")
    if drop_truncated:
        frame = frame[~frame["raw_prompt_bpe_truncated"]].copy()
    if frame["raw_prompt_bpe_truncated"].any():
        raise ValueError(
            "Input still contains raw_prompt_bpe_truncated rows. "
            "Use the max2048 non-truncated parquet or pass --drop-truncated."
        )
    frame["split"] = frame["split"].astype(str)
    frame["raw_bpe_length_bucket"] = make_raw_bpe_length_bucket(
        frame["raw_prompt_bpe_token_count_full"]
    ).astype(str)
    return frame


def select_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = frame[frame["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split={split!r}.")
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


def metric_value(metrics: dict[str, Any], split: str, metric: str) -> Any:
    return metrics.get("by_split", {}).get(split, {}).get(metric)


def init_wandb(
    args: argparse.Namespace,
    config: M1TfidfLogRegConfig,
    *,
    curve_label: str,
    source_label: str,
    requested_train_rows: int | str,
    actual_train_rows: int,
):
    if args.no_wandb:
        return None

    import wandb

    wandb_dir = resolve_path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        name=f"M1_learning_curve_{source_label}_atomic_constraint_heldout_seed{args.random_state}",
        group=args.wandb_group,
        job_type="train_m1_learning_curve",
        tags=[
            "M1",
            "tfidf",
            "logistic-regression",
            "raw-prompt-bpe",
            "learning-curve",
            "atomic-constraint-heldout",
        ],
        config={
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "train_split": args.train_split,
            "eval_splits": parse_eval_splits(args.eval_splits),
            "curve_label": curve_label,
            "source_label": source_label,
            "requested_train_rows": requested_train_rows,
            "actual_train_rows": actual_train_rows,
            "sample_seed": args.sample_seed,
            "sample_key_column": args.sample_key_column,
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


def summarize_run(
    *,
    order: int,
    curve_label: str,
    source_label: str,
    requested_train_rows: int | str,
    actual_train_rows: int,
    metrics: dict[str, Any],
    train_seconds: float,
    predict_seconds: float,
    total_seconds: float,
    output_dir: Path,
    vocabulary_size: int | None,
) -> dict[str, Any]:
    val_positive_rate = metric_value(metrics, "val", "positive_rate")
    test_positive_rate = metric_value(metrics, "test", "positive_rate")
    val_auprc = metric_value(metrics, "val", "auprc")
    test_auprc = metric_value(metrics, "test", "auprc")
    return {
        "curve_label": curve_label,
        "source_label": source_label,
        "order": order,
        "requested_train_rows": requested_train_rows,
        "actual_train_rows": actual_train_rows,
        "val_rows": metric_value(metrics, "val", "row_count"),
        "test_rows": metric_value(metrics, "test", "row_count"),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "vocabulary_size": vocabulary_size,
        "train_auroc": metric_value(metrics, "train", "auroc"),
        "train_auprc": metric_value(metrics, "train", "auprc"),
        "val_auroc": metric_value(metrics, "val", "auroc"),
        "val_auprc": val_auprc,
        "test_auroc": metric_value(metrics, "test", "auroc"),
        "test_auprc": test_auprc,
        "val_positive_rate": val_positive_rate,
        "test_positive_rate": test_positive_rate,
        "test_auprc_lift": (
            None if not test_positive_rate else float(test_auprc) / float(test_positive_rate)
        ),
        "val_auprc_lift": (
            None if not val_positive_rate else float(val_auprc) / float(val_positive_rate)
        ),
        "output_dir": str(output_dir),
    }


def train_one_size(
    *,
    args: argparse.Namespace,
    data: pd.DataFrame,
    train_full: pd.DataFrame,
    eval_frames: list[pd.DataFrame],
    config: M1TfidfLogRegConfig,
    output_dir: Path,
    order: int,
    source_label: str,
    train_size: int | None,
) -> dict[str, Any]:
    total_start = now()
    if train_size is None:
        train_frame = train_full.copy()
        requested_train_rows: int | str = "full"
    else:
        train_frame = stable_sample_frame(
            train_full,
            sample_size=train_size,
            sample_seed=args.sample_seed,
            key_column=args.sample_key_column,
        )
        requested_train_rows = train_size

    actual_train_rows = int(len(train_frame))
    curve_label = source_label
    if isinstance(requested_train_rows, int) and actual_train_rows < requested_train_rows:
        curve_label = f"{source_label}/full"

    run_output_dir = output_dir / f"train_{source_label}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run = init_wandb(
        args,
        config,
        curve_label=curve_label,
        source_label=source_label,
        requested_train_rows=requested_train_rows,
        actual_train_rows=actual_train_rows,
    )

    train_start = now()
    pipeline = fit_m1_pipeline(train_frame, config)
    train_seconds = elapsed_since(train_start)
    if run is not None:
        run.log(
            {
                "m1/stage": 1,
                "m1/train_rows": actual_train_rows,
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
        run.log({"m1/stage": 2, "m1/predict_seconds": predict_seconds}, step=2)

    metrics = {"by_split": evaluate_by_split(predictions, config.threshold)}
    features = feature_summary(pipeline, top_n=args.top_features)
    feature_space = pipeline_size_summary(pipeline)

    model_path = run_output_dir / "model.joblib"
    predictions_path = run_output_dir / "predictions.parquet"
    metrics_path = run_output_dir / "metrics.json"
    features_path = run_output_dir / "top_features.csv"
    manifest_path = run_output_dir / "manifest.json"

    if not args.no_save_model:
        joblib.dump(pipeline, model_path)
    if not args.no_save_predictions:
        predictions.to_parquet(predictions_path, index=False)
    features.to_csv(features_path, index=False, encoding="utf-8")

    total_seconds = elapsed_since(total_start)
    summary_row = summarize_run(
        order=order,
        curve_label=curve_label,
        source_label=source_label,
        requested_train_rows=requested_train_rows,
        actual_train_rows=actual_train_rows,
        metrics=metrics,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        total_seconds=total_seconds,
        output_dir=run_output_dir,
        vocabulary_size=feature_space["vocabulary_size"],
    )
    manifest = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "input_file": str(resolve_path(args.input_file)),
        "output_dir": str(run_output_dir),
        "curve_label": curve_label,
        "source_label": source_label,
        "train_split": args.train_split,
        "eval_splits": parse_eval_splits(args.eval_splits),
        "config": config.to_dict(),
        "drop_truncated": args.drop_truncated,
        "data_counts": {
            "loaded_rows": int(len(data)),
            "split_counts_loaded": split_count_table(data),
            "train_full_rows": int(len(train_full)),
            "requested_train_rows": requested_train_rows,
            "actual_train_rows": actual_train_rows,
            "label_counts_train": label_count_table(train_frame),
            "label_counts_eval": label_count_table(pd.concat(eval_frames, ignore_index=True)),
        },
        "feature_space": feature_space,
        "summary": summary_row,
        "output_files": {
            "model": None if args.no_save_model else str(model_path),
            "metrics": str(metrics_path),
            "manifest": str(manifest_path),
            "top_features": str(features_path),
            "predictions": None if args.no_save_predictions else str(predictions_path),
        },
        "timing_seconds": {
            "train": train_seconds,
            "predict": predict_seconds,
            "total": total_seconds,
        },
    }
    write_json(metrics, metrics_path)
    write_json(manifest, manifest_path)
    if run is not None:
        run.summary.update(
            {
                "model_id": MODEL_ID,
                "curve_label": curve_label,
                "source_label": source_label,
                "actual_train_rows": actual_train_rows,
                "vocabulary_size": feature_space["vocabulary_size"],
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": total_seconds,
                "output_dir": str(run_output_dir),
            }
        )
        run.log(flatten_metrics_for_wandb(metrics), step=3)
        save_wandb_files(run, [metrics_path, manifest_path, features_path])
        run.finish()

    print(json.dumps({"summary": summary_row}, ensure_ascii=False), flush=True)
    return summary_row


def main() -> None:
    args = parse_args()
    total_start = now()
    input_file = resolve_path(args.input_file)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = parse_eval_splits(args.eval_splits)
    sizes = parse_train_sizes(args.train_sizes)

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

    load_start = now()
    data = load_data(input_file, drop_truncated=args.drop_truncated)
    train_full = select_split(data, args.train_split)
    eval_frames = [select_split(data, split) for split in eval_splits]
    load_seconds = elapsed_since(load_start)

    summary_rows: list[dict[str, Any]] = []
    for order, (source_label, train_size) in enumerate(sizes):
        row = train_one_size(
            args=args,
            data=data,
            train_full=train_full,
            eval_frames=eval_frames,
            config=config,
            output_dir=output_dir,
            order=order,
            source_label=source_label,
            train_size=train_size,
        )
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "learning_curve_summary.csv",
            index=False,
            encoding="utf-8",
        )
        write_json(summary_rows, output_dir / "learning_curve_summary.json")

    manifest = {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "train_sizes": [
            {"source_label": label, "requested_train_rows": size if size is not None else "full"}
            for label, size in sizes
        ],
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "sample_seed": args.sample_seed,
        "sample_key_column": args.sample_key_column,
        "wandb_mode": args.wandb_mode,
        "wandb_group": None if args.no_wandb else args.wandb_group,
        "load_seconds": load_seconds,
        "total_seconds": elapsed_since(total_start),
        "m1_config": config.to_dict(),
        "summary_file": str(output_dir / "learning_curve_summary.csv"),
    }
    write_json(manifest, output_dir / "learning_curve_manifest.json")
    print(json.dumps({"manifest": manifest, "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
