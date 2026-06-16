from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_PROMPTSET_FILE = (
    "data/promptsets/if_multi_constraints_upto5.qwen3_4b_instruct_2507."
    "under2048_nontruncated.all.parquet"
)
DEFAULT_TOKENIZER_DIR = "data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_group_key_train"
DEFAULT_OUTPUT_DIR = "runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048"
DEFAULT_TRAIN_SIZES = "2k,4k,5k,10k,20k,40k,full"
DEFAULT_SEEDS = "42,43,44"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M4 pretraining and frozen-classifier learning curves across seeds."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--promptset-file", default=DEFAULT_PROMPTSET_FILE)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-sizes", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-learning-rate", type=float, default=5e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.01)
    parser.add_argument("--pretrain-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--digit-quote-mask-rate", type=float, default=0.30)
    parser.add_argument("--lambda-struct", type=float, default=0.3)
    parser.add_argument("--classifier-hidden-size", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--classifier-eval-batch-size", type=int, default=16)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-3)
    parser.add_argument("--classifier-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--selection-metric", default="auprc", choices=["auprc", "auroc", "brier"])
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "runs/wandb"))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seeds(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_size_token(token: str) -> tuple[str, int | None]:
    normalized = token.strip().lower()
    if not normalized:
        raise ValueError("Empty train-size token.")
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


def metric_value(metrics: dict[str, Any], split: str, metric: str) -> Any:
    return metrics.get("by_split", {}).get(split, {}).get(metric)


def build_pretrain_command(args: argparse.Namespace, *, seed: int, output_dir: Path) -> list[str]:
    script = Path(__file__).with_name("train_m4_pretrain_encoder.py")
    wandb_group = args.wandb_group or "M4_initial_pretrain_multiseed_atomic_max2048"
    command = [
        sys.executable,
        str(script),
        "--input-file",
        args.input_file,
        "--promptset-file",
        args.promptset_file,
        "--tokenizer-dir",
        args.tokenizer_dir,
        "--output-dir",
        str(output_dir),
        "--vocab-size",
        str(args.vocab_size),
        "--max-length",
        str(args.max_length),
        "--hidden-size",
        str(args.hidden_size),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--ffn-dim",
        str(args.ffn_dim),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.pretrain_epochs),
        "--batch-size",
        str(args.pretrain_batch_size),
        "--learning-rate",
        str(args.pretrain_learning_rate),
        "--weight-decay",
        str(args.pretrain_weight_decay),
        "--warmup-ratio",
        str(args.pretrain_warmup_ratio),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--mask-rate",
        str(args.mask_rate),
        "--digit-quote-mask-rate",
        str(args.digit_quote_mask_rate),
        "--lambda-struct",
        str(args.lambda_struct),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-name",
        f"M4_pretrain_seed{seed}_atomic_max2048",
        "--wandb-group",
        wandb_group,
        "--wandb-dir",
        args.wandb_dir,
        "--no-wandb-save-model",
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.no_wandb:
        command.append("--no-wandb")
    return command


def build_classifier_command(
    args: argparse.Namespace,
    *,
    seed: int,
    label: str,
    train_size: int | None,
    encoder_file: Path,
    output_dir: Path,
) -> list[str]:
    script = Path(__file__).with_name("train_m4_frozen_classifier.py")
    wandb_group = args.wandb_group or "M4_initial_frozen_classifier_learning_curve_atomic_max2048"
    command = [
        sys.executable,
        str(script),
        "--input-file",
        args.input_file,
        "--encoder-file",
        str(encoder_file),
        "--output-dir",
        str(output_dir),
        "--max-length",
        str(args.max_length),
        "--classifier-hidden-size",
        str(args.classifier_hidden_size),
        "--classifier-dropout",
        str(args.classifier_dropout),
        "--epochs",
        str(args.classifier_epochs),
        "--batch-size",
        str(args.classifier_batch_size),
        "--eval-batch-size",
        str(args.classifier_eval_batch_size),
        "--learning-rate",
        str(args.classifier_learning_rate),
        "--weight-decay",
        str(args.classifier_weight_decay),
        "--warmup-ratio",
        str(args.classifier_warmup_ratio),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--selection-metric",
        args.selection_metric,
        "--selection-split",
        args.selection_split,
        "--seed",
        str(seed),
        "--sample-seed",
        str(args.sample_seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-name",
        f"M4_frozen_{label}_seed{seed}_sample{args.sample_seed}",
        "--wandb-group",
        wandb_group,
        "--wandb-dir",
        args.wandb_dir,
        "--no-wandb-save-model",
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.no_wandb:
        command.append("--no-wandb")
    if not args.save_predictions:
        command.append("--no-save-predictions")
    if train_size is not None:
        command.extend(["--train-sample-size", str(train_size)])
    return command


def pretrain_complete(run_dir: Path) -> bool:
    return (run_dir / "pretrained_encoder.pt").exists() and (
        run_dir / "pretraining_manifest.json"
    ).exists()


def classifier_complete(run_dir: Path) -> bool:
    return (run_dir / "manifest.json").exists() and (run_dir / "metrics.json").exists()


def summarize_classifier_run(
    *,
    run_dir: Path,
    seed: int,
    label: str,
    train_size: int | None,
    encoder_file: Path,
) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    counts = manifest["data_counts"]
    training = manifest["training"]
    best = manifest["best"]
    timing = manifest["timing_seconds"]
    test_positive_rate = metric_value(metrics, "test", "positive_rate")
    val_positive_rate = metric_value(metrics, "val", "positive_rate")
    test_auprc = metric_value(metrics, "test", "auprc")
    val_auprc = metric_value(metrics, "val", "auprc")
    test_accuracy = metric_value(metrics, "test", "accuracy")
    return {
        "curve_label": label,
        "requested_train_rows": train_size if train_size is not None else "full",
        "actual_train_rows": counts["train_rows"],
        "train_seed": seed,
        "sample_seed": counts["sample_seed"],
        "encoder_file": str(encoder_file),
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "epochs_run": training["epochs_run"],
        "batch_size": training["batch_size"],
        "eval_batch_size": training["eval_batch_size"],
        "val_auroc": metric_value(metrics, "val", "auroc"),
        "val_auprc": val_auprc,
        "test_auroc": metric_value(metrics, "test", "auroc"),
        "test_auprc": test_auprc,
        "val_positive_rate": val_positive_rate,
        "test_positive_rate": test_positive_rate,
        "val_auprc_lift": None if not val_positive_rate else val_auprc / val_positive_rate,
        "test_auprc_lift": None if not test_positive_rate else test_auprc / test_positive_rate,
        "test_accuracy": test_accuracy,
        "test_error_rate": None if test_accuracy is None else 1.0 - test_accuracy,
        "test_fp": metric_value(metrics, "test", "confusion")["fp"],
        "test_fn": metric_value(metrics, "test", "confusion")["fn"],
        "test_tp": metric_value(metrics, "test", "confusion")["tp"],
        "test_tn": metric_value(metrics, "test", "confusion")["tn"],
        "total_seconds": timing["total"],
        "output_dir": str(run_dir),
    }


def aggregate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
        "val_auprc_lift",
        "test_auprc_lift",
        "test_accuracy",
        "test_error_rate",
        "total_seconds",
    ]
    label_order = {
        label: index
        for index, label in enumerate(["2k", "4k", "5k", "10k", "20k", "40k", "full"])
    }
    rows: list[dict[str, Any]] = []
    for label, part in frame.groupby("curve_label", sort=False):
        row: dict[str, Any] = {
            "curve_label": label,
            "order": label_order.get(str(label), 999),
            "run_count": int(len(part)),
            "seeds": ",".join(str(seed) for seed in sorted(part["train_seed"].astype(int))),
            "actual_train_rows_mean": float(part["actual_train_rows"].mean()),
            "actual_train_rows_min": int(part["actual_train_rows"].min()),
            "actual_train_rows_max": int(part["actual_train_rows"].max()),
            "val_positive_rate": float(part["val_positive_rate"].dropna().iloc[0]),
            "test_positive_rate": float(part["test_positive_rate"].dropna().iloc[0]),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("order", kind="mergesort")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multiseed_learning_curve_summary.csv"
    aggregate_path = output_dir / "multiseed_learning_curve_aggregate.csv"
    seeds = parse_seeds(args.seeds)
    sizes = parse_train_sizes(args.train_sizes)
    summary_rows: list[dict[str, Any]] = []

    for seed in seeds:
        pretrain_dir = output_dir / f"seed_{seed}" / "pretrain"
        encoder_file = pretrain_dir / "pretrained_encoder.pt"
        if pretrain_complete(pretrain_dir) and not args.force:
            print(json.dumps({"skipped_existing_pretrain": str(pretrain_dir)}, ensure_ascii=False))
        else:
            pretrain_command = build_pretrain_command(args, seed=seed, output_dir=pretrain_dir)
            print(json.dumps({"pretrain_command": pretrain_command}, ensure_ascii=False), flush=True)
            if not args.dry_run:
                env = os.environ.copy()
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                completed = subprocess.run(pretrain_command, env=env, check=False)
                if completed.returncode != 0:
                    failure = {
                        "stage": "pretrain",
                        "seed": seed,
                        "return_code": completed.returncode,
                        "output_dir": str(pretrain_dir),
                    }
                    write_json(failure, output_dir / "failure.json")
                    raise SystemExit(json.dumps({"failed": failure}, ensure_ascii=False))

        for label, train_size in sizes:
            run_output_dir = output_dir / f"seed_{seed}" / f"train_{label}"
            if classifier_complete(run_output_dir) and not args.force:
                row = summarize_classifier_run(
                    run_dir=run_output_dir,
                    seed=seed,
                    label=label,
                    train_size=train_size,
                    encoder_file=encoder_file,
                )
                summary_rows.append(row)
                print(json.dumps({"skipped_existing_classifier": row}, ensure_ascii=False), flush=True)
                continue

            classifier_command = build_classifier_command(
                args,
                seed=seed,
                label=label,
                train_size=train_size,
                encoder_file=encoder_file,
                output_dir=run_output_dir,
            )
            print(json.dumps({"classifier_command": classifier_command}, ensure_ascii=False), flush=True)
            if args.dry_run:
                continue
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            completed = subprocess.run(classifier_command, env=env, check=False)
            if completed.returncode != 0:
                failure = {
                    "stage": "classifier",
                    "seed": seed,
                    "curve_label": label,
                    "return_code": completed.returncode,
                    "output_dir": str(run_output_dir),
                }
                write_json(failure, output_dir / "failure.json")
                raise SystemExit(json.dumps({"failed": failure}, ensure_ascii=False))
            row = summarize_classifier_run(
                run_dir=run_output_dir,
                seed=seed,
                label=label,
                train_size=train_size,
                encoder_file=encoder_file,
            )
            summary_rows.append(row)
            frame = pd.DataFrame(summary_rows)
            frame.to_csv(summary_path, index=False, encoding="utf-8")
            write_json(summary_rows, output_dir / "multiseed_learning_curve_summary.json")
            aggregate_summary(frame).to_csv(aggregate_path, index=False, encoding="utf-8")
            print(json.dumps({"summary": row}, ensure_ascii=False), flush=True)

    if summary_rows:
        frame = pd.DataFrame(summary_rows)
        frame.to_csv(summary_path, index=False, encoding="utf-8")
        write_json(summary_rows, output_dir / "multiseed_learning_curve_summary.json")
        aggregate_summary(frame).to_csv(aggregate_path, index=False, encoding="utf-8")

    manifest = {
        "input_file": args.input_file,
        "promptset_file": args.promptset_file,
        "tokenizer_dir": args.tokenizer_dir,
        "output_dir": str(output_dir),
        "seeds": seeds,
        "train_sizes": [
            {"label": label, "requested_train_rows": size if size is not None else "full"}
            for label, size in sizes
        ],
        "sample_seed": args.sample_seed,
        "model": {
            "vocab_size": args.vocab_size,
            "max_length": args.max_length,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "heads": args.heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "pooling": "mean",
        },
        "pretraining": {
            "epochs": args.pretrain_epochs,
            "batch_size": args.pretrain_batch_size,
            "learning_rate": args.pretrain_learning_rate,
            "weight_decay": args.pretrain_weight_decay,
            "warmup_ratio": args.pretrain_warmup_ratio,
            "mask_rate": args.mask_rate,
            "digit_quote_mask_rate": args.digit_quote_mask_rate,
            "lambda_struct": args.lambda_struct,
        },
        "classifier_training": {
            "epochs": args.classifier_epochs,
            "batch_size": args.classifier_batch_size,
            "eval_batch_size": args.classifier_eval_batch_size,
            "learning_rate": args.classifier_learning_rate,
            "weight_decay": args.classifier_weight_decay,
            "warmup_ratio": args.classifier_warmup_ratio,
            "early_stop_patience": args.early_stop_patience,
            "selection_metric": args.selection_metric,
        },
        "wandb_mode": args.wandb_mode,
        "summary_path": str(summary_path),
        "aggregate_path": str(aggregate_path),
    }
    write_json(manifest, output_dir / "multiseed_learning_curve_manifest.json")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "manifest": manifest}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
