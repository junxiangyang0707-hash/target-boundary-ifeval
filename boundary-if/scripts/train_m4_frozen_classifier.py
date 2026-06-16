from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from boundary_if.models.m4_pretrained_encoder import (
    MODEL_ID,
    MODEL_NAME,
    M4EncoderConfig,
    M4FrozenEncoderClassifier,
    compute_pos_weight,
    evaluate_m4_by_split,
    make_m4_classifier_dataloader,
    predict_m4,
    set_torch_seed,
    train_m4_classifier_epoch,
)
from boundary_if.training.sampling import stable_sample_frame

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_ENCODER_FILE = (
    "runs/m4_pretraining_atomic_constraint_heldout_seed42_max2048/pretrained_encoder.pt"
)
DEFAULT_OUTPUT_DIR = "runs/m4_frozen_classifier_atomic_constraint_heldout_seed42_max2048"

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
        description="Train M4 frozen pretrained encoder + MLP boundary classifier."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--encoder-file", default=DEFAULT_ENCODER_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--classifier-hidden-size", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--selection-metric", default="auprc", choices=["auprc", "auroc", "brier"])
    parser.add_argument("--class-balance", default="pos_weight", choices=["none", "pos_weight"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--train-sample-size", type=int, default=None)
    parser.add_argument("--test-sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-wandb-save-model", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_eval_splits(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(raw_device)


def make_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        remaining = num_training_steps - current_step
        decay_steps = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    if frame["prompt_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate prompt_id values.")
    return frame


def select_split(
    frame: pd.DataFrame,
    split: str,
    *,
    limit: int | None,
    sample_size: int | None,
    sample_seed: int,
) -> pd.DataFrame:
    if limit is not None and sample_size is not None:
        raise ValueError("Use either limit or sample_size, not both.")
    selected = frame[frame["split"].astype(str) == split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split={split!r}.")
    selected = stable_sample_frame(
        selected,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
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


def load_encoder_checkpoint(path: Path, args: argparse.Namespace) -> tuple[M4EncoderConfig, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu")
    raw_config = checkpoint["config"]
    config = M4EncoderConfig(
        vocab_size=int(raw_config["vocab_size"]),
        max_length=int(args.max_length),
        hidden_size=int(raw_config["hidden_size"]),
        layers=int(raw_config["layers"]),
        heads=int(raw_config["heads"]),
        ffn_dim=int(raw_config["ffn_dim"]),
        dropout=float(raw_config["dropout"]),
        pooling="mean",
        classifier_hidden_size=args.classifier_hidden_size,
        classifier_dropout=args.classifier_dropout,
        pad_token_id=raw_config.get("pad_token_id"),
        mask_token_id=int(raw_config["mask_token_id"]),
        threshold=args.threshold,
    )
    return config, checkpoint["prompt_encoder_state_dict"]


def selection_score(metrics: dict[str, Any], metric: str) -> float:
    value = metrics.get(metric)
    if value is None:
        raise ValueError(f"selection_metric={metric!r} is unavailable.")
    score = float(value)
    if metric == "brier":
        return -score
    return score


def init_wandb(args: argparse.Namespace, config: M4EncoderConfig):
    if args.no_wandb:
        return None

    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    wandb_dir = resolve_path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        name=args.wandb_name or f"M4_frozen_classifier_seed{args.seed}",
        group=args.wandb_group,
        job_type="m4_frozen_classifier",
        tags=["M4", "frozen-encoder", "MLP", "raw-prompt-bpe", "atomic-constraint-heldout"],
        config={
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "input_file": args.input_file,
            "encoder_file": args.encoder_file,
            "output_dir": args.output_dir,
            "m4_config": config.to_dict(),
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "grad_clip_norm": args.grad_clip_norm,
                "early_stop_patience": args.early_stop_patience,
                "selection_metric": args.selection_metric,
                "selection_split": args.selection_split,
                "seed": args.seed,
                "sample_seed": args.sample_seed if args.sample_seed is not None else args.seed,
                "train_sample_size": args.train_sample_size,
            },
        },
    )


def flatten_metrics_for_wandb(metrics: dict[str, Any], *, prefix: str) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for split, split_metrics in metrics["by_split"].items():
        for key, value in split_metrics.items():
            if key == "confusion":
                for confusion_key, confusion_value in value.items():
                    flattened[f"{prefix}/{split}/confusion/{confusion_key}"] = int(confusion_value)
            elif isinstance(value, int | float) and value is not None:
                flattened[f"{prefix}/{split}/{key}"] = value
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
    set_torch_seed(args.seed)
    input_file = resolve_path(args.input_file)
    encoder_file = resolve_path(args.encoder_file)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_splits = parse_eval_splits(args.eval_splits)
    device = resolve_device(args.device)
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed

    config, encoder_state_dict = load_encoder_checkpoint(encoder_file, args)
    run = init_wandb(args, config)

    load_start = now()
    data = load_data(input_file)
    train_frame = select_split(
        data,
        args.train_split,
        limit=args.limit_train,
        sample_size=args.train_sample_size,
        sample_seed=sample_seed,
    )
    eval_frames = [
        select_split(
            data,
            split,
            limit=args.limit_eval,
            sample_size=args.test_sample_size if split == "test" else None,
            sample_seed=sample_seed,
        )
        for split in eval_splits
    ]
    fit_eval_frame = pd.concat([train_frame, *eval_frames], ignore_index=True)
    load_seconds = elapsed_since(load_start)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = make_m4_classifier_dataloader(
        train_frame,
        config,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    eval_loaders = {
        split: make_m4_classifier_dataloader(
            split_frame,
            config,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for split, split_frame in zip(eval_splits, eval_frames, strict=True)
    }
    if args.selection_split not in eval_loaders:
        raise ValueError(
            f"selection_split={args.selection_split!r} is not in eval_splits={eval_splits!r}."
        )

    model = M4FrozenEncoderClassifier(config).to(device)
    model.load_pretrained_encoder(encoder_state_dict)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.pass_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_training_steps = max(1, args.epochs * len(train_loader))
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = make_linear_warmup_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    if args.class_balance == "pos_weight":
        pos_weight = compute_pos_weight(train_frame["label"])
    else:
        pos_weight = 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    best_score = float("-inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = now()
        train_stats = train_m4_classifier_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=args.grad_clip_norm,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=amp_enabled,
        )
        selection_predictions = predict_m4(
            model,
            eval_loaders[args.selection_split],
            device=device,
        )
        selection_metrics = evaluate_m4_by_split(
            selection_predictions,
            threshold=config.threshold,
        )[args.selection_split]
        score = selection_score(selection_metrics, args.selection_metric)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.pass_head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "selection_split": args.selection_split,
            "selection_metric": args.selection_metric,
            "selection_score": score,
            "selection_metric_raw": selection_metrics[args.selection_metric],
            "best_score": best_score,
            "best_epoch": best_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": elapsed_since(epoch_start),
        }
        history.append(epoch_record)
        if run is not None:
            run.log(
                {
                    "m4/epoch": epoch,
                    "m4/train/loss": train_stats["loss"],
                    f"m4/{args.selection_split}/{args.selection_metric}": selection_metrics[
                        args.selection_metric
                    ],
                    "m4/best_score": best_score,
                    "m4/learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= args.early_stop_patience:
            break

    if best_state_dict is not None:
        model.pass_head.load_state_dict(best_state_dict)

    predict_start = now()
    prediction_frames = [
        predict_m4(
            model,
            make_m4_classifier_dataloader(
                train_frame,
                config,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            ),
            device=device,
        ),
        *[predict_m4(model, eval_loaders[split], device=device) for split in eval_splits],
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predict_seconds = elapsed_since(predict_start)
    metrics = {"by_split": evaluate_m4_by_split(predictions, threshold=config.threshold)}

    model_path = output_dir / "model.pt"
    predictions_path = output_dir / "predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    history_path = output_dir / "history.csv"

    torch.save(
        {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "config": config.to_dict(),
            "encoder_file": str(encoder_file),
            "prompt_encoder_state_dict": model.prompt_encoder.state_dict(),
            "pass_head_state_dict": model.pass_head.state_dict(),
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        model_path,
    )
    if not args.no_save_predictions:
        predictions.to_parquet(predictions_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")

    manifest = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "input_file": str(input_file),
        "encoder_file": str(encoder_file),
        "output_dir": str(output_dir),
        "train_split": args.train_split,
        "eval_splits": eval_splits,
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "config": config.to_dict(),
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": num_warmup_steps,
            "grad_clip_norm": args.grad_clip_norm,
            "early_stop_patience": args.early_stop_patience,
            "class_balance": args.class_balance,
            "pos_weight": pos_weight,
            "seed": args.seed,
            "device": str(device),
            "amp_enabled": amp_enabled,
        },
        "best": {
            "epoch": best_epoch,
            "score": best_score,
        },
        "data_counts": {
            "loaded_rows": int(len(data)),
            "fit_eval_rows": int(len(fit_eval_frame)),
            "split_counts_loaded": split_count_table(data),
            "split_counts_fit_eval": split_count_table(fit_eval_frame),
            "label_counts_fit_eval": label_count_table(fit_eval_frame),
            "train_rows": int(len(train_frame)),
            "requested_train_sample_size": args.train_sample_size,
            "requested_test_sample_size": args.test_sample_size,
            "sample_seed": sample_seed,
        },
        "output_files": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "manifest": str(manifest_path),
            "history": str(history_path),
            "predictions": None if args.no_save_predictions else str(predictions_path),
        },
        "timing_seconds": {
            "load_data": load_seconds,
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
                "best_epoch": best_epoch,
                "best_score": best_score,
                "total_seconds": manifest["timing_seconds"]["total"],
                "output_dir": str(output_dir),
            }
        )
        run.log(flatten_metrics_for_wandb(metrics, prefix="m4/final"), step=len(history) + 1)
        save_wandb_files(
            run,
            [
                metrics_path,
                manifest_path,
                history_path,
                *([] if args.no_wandb_save_model else [model_path]),
                *([] if args.no_save_predictions else [predictions_path]),
            ],
        )
        run.finish()
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
