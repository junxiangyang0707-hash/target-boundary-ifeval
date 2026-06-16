from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from boundary_if.models.tiny_transformer import M3TinyTransformer, M3TinyTransformerConfig

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_OUTPUT_DIR = "runs/m3_capacity_seed_sweep_55k_sample42_max2048"
DEFAULT_CONFIGS = [
    {
        "config_label": "wide_192_l6_h6_ffn768",
        "display_name": "192d x 6L",
        "hidden_size": 192,
        "layers": 6,
        "heads": 6,
        "ffn_dim": 768,
        "classifier_hidden_size": 192,
    },
    {
        "config_label": "deep_192_l8_h6_ffn768",
        "display_name": "192d x 8L",
        "hidden_size": 192,
        "layers": 8,
        "heads": 6,
        "ffn_dim": 768,
        "classifier_hidden_size": 192,
    },
    {
        "config_label": "medium_256_l6_h8_ffn1024",
        "display_name": "256d x 6L",
        "hidden_size": 256,
        "layers": 6,
        "heads": 8,
        "ffn_dim": 1024,
        "classifier_hidden_size": 256,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3 capacity configs with train seeds on the fixed 55k sample."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--train-sample-size", type=int, default=55_000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--retry-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument(
        "--wandb-group",
        default="M3_capacity_seed_sweep_55k_sample42_max2048",
    )
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seeds(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def param_count(config_payload: dict[str, Any], args: argparse.Namespace) -> int:
    config = M3TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        hidden_size=int(config_payload["hidden_size"]),
        layers=int(config_payload["layers"]),
        heads=int(config_payload["heads"]),
        ffn_dim=int(config_payload["ffn_dim"]),
        dropout=args.dropout,
        pooling=args.pooling,
        classifier_hidden_size=int(config_payload["classifier_hidden_size"]),
    )
    return int(sum(parameter.numel() for parameter in M3TinyTransformer(config).parameters()))


def metric_value(metrics: dict[str, Any], split: str, metric: str) -> Any:
    return metrics.get("by_split", {}).get(split, {}).get(metric)


def build_command(
    *,
    args: argparse.Namespace,
    config_payload: dict[str, Any],
    seed: int,
    output_dir: Path,
    batch_size: int,
) -> list[str]:
    train_script = Path(__file__).with_name("train_m3_tiny_transformer.py")
    wandb_name = f"M3_capacity_{config_payload['config_label']}_seed{seed}"
    command = [
        sys.executable,
        str(train_script),
        "--input-file",
        args.input_file,
        "--output-dir",
        str(output_dir),
        "--max-length",
        str(args.max_length),
        "--vocab-size",
        str(args.vocab_size),
        "--hidden-size",
        str(config_payload["hidden_size"]),
        "--layers",
        str(config_payload["layers"]),
        "--heads",
        str(config_payload["heads"]),
        "--ffn-dim",
        str(config_payload["ffn_dim"]),
        "--dropout",
        str(args.dropout),
        "--pooling",
        args.pooling,
        "--classifier-hidden-size",
        str(config_payload["classifier_hidden_size"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
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
        "--train-sample-size",
        str(args.train_sample_size),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-name",
        wandb_name,
        "--wandb-group",
        args.wandb_group,
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
    return command


def summarize_run(
    *,
    config_payload: dict[str, Any],
    seed: int,
    output_dir: Path,
    parameter_count: int,
    batch_size: int,
) -> dict[str, Any]:
    manifest = read_json(output_dir / "manifest.json")
    metrics = read_json(output_dir / "metrics.json")
    training = manifest["training"]
    data_counts = manifest["data_counts"]
    best = manifest["best"]
    timing = manifest["timing_seconds"]
    return {
        "config_label": config_payload["config_label"],
        "display_name": config_payload["display_name"],
        "train_seed": seed,
        "sample_seed": data_counts["sample_seed"],
        "requested_train_sample_size": data_counts["requested_train_sample_size"],
        "actual_train_rows": data_counts["train_rows"],
        "hidden_size": config_payload["hidden_size"],
        "layers": config_payload["layers"],
        "heads": config_payload["heads"],
        "ffn_dim": config_payload["ffn_dim"],
        "classifier_hidden_size": config_payload["classifier_hidden_size"],
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "eval_batch_size": training["eval_batch_size"],
        "best_epoch": best["epoch"],
        "epochs_run": training["epochs_run"],
        "best_val_auprc": best["score"],
        "val_auroc": metric_value(metrics, "val", "auroc"),
        "val_auprc": metric_value(metrics, "val", "auprc"),
        "test_auroc": metric_value(metrics, "test", "auroc"),
        "test_auprc": metric_value(metrics, "test", "auprc"),
        "test_auprc_lift": (
            None
            if not metric_value(metrics, "test", "positive_rate")
            else metric_value(metrics, "test", "auprc")
            / metric_value(metrics, "test", "positive_rate")
        ),
        "test_accuracy": metric_value(metrics, "test", "accuracy"),
        "test_error_rate": (
            None
            if metric_value(metrics, "test", "accuracy") is None
            else 1.0 - metric_value(metrics, "test", "accuracy")
        ),
        "test_fp": metric_value(metrics, "test", "confusion")["fp"],
        "test_fn": metric_value(metrics, "test", "confusion")["fn"],
        "test_tp": metric_value(metrics, "test", "confusion")["tp"],
        "test_tn": metric_value(metrics, "test", "confusion")["tn"],
        "total_seconds": timing["total"],
        "output_dir": str(output_dir),
    }


def aggregate_summary(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(summary_rows)
    numeric_metrics = [
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
        "test_auprc_lift",
        "test_accuracy",
        "test_error_rate",
        "total_seconds",
    ]
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
        dropna=False,
        sort=False,
    )
    rows: list[dict[str, Any]] = []
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
        for metric in numeric_metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "capacity_seed_sweep_summary.csv"
    aggregate_path = output_dir / "capacity_seed_sweep_aggregate.csv"
    failed_path = output_dir / "capacity_seed_sweep_failures.json"
    seeds = parse_seeds(args.seeds)
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for config_payload in DEFAULT_CONFIGS:
        parameter_count = param_count(config_payload, args)
        for seed in seeds:
            run_output_dir = output_dir / config_payload["config_label"] / f"seed_{seed}"
            manifest_path = run_output_dir / "manifest.json"
            metrics_path = run_output_dir / "metrics.json"
            if manifest_path.exists() and metrics_path.exists() and not args.force:
                row = summarize_run(
                    config_payload=config_payload,
                    seed=seed,
                    output_dir=run_output_dir,
                    parameter_count=parameter_count,
                    batch_size=args.batch_size,
                )
                summary_rows.append(row)
                print(json.dumps({"skipped_existing": row}, ensure_ascii=False), flush=True)
                continue

            attempted_batch_sizes = [args.batch_size]
            if args.retry_batch_size and args.retry_batch_size < args.batch_size:
                attempted_batch_sizes.append(args.retry_batch_size)
            last_return_code: int | None = None
            used_batch_size: int | None = None
            for batch_size in attempted_batch_sizes:
                command = build_command(
                    args=args,
                    config_payload=config_payload,
                    seed=seed,
                    output_dir=run_output_dir,
                    batch_size=batch_size,
                )
                print(
                    json.dumps(
                        {
                            "config_label": config_payload["config_label"],
                            "seed": seed,
                            "batch_size": batch_size,
                            "parameter_count": parameter_count,
                            "command": command,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.dry_run:
                    used_batch_size = batch_size
                    last_return_code = 0
                    break
                env = os.environ.copy()
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                completed = subprocess.run(command, env=env, check=False)
                last_return_code = completed.returncode
                if completed.returncode == 0:
                    used_batch_size = batch_size
                    break

            if args.dry_run:
                continue
            if last_return_code != 0 or used_batch_size is None:
                failure = {
                    "config_label": config_payload["config_label"],
                    "seed": seed,
                    "return_code": last_return_code,
                    "output_dir": str(run_output_dir),
                    "attempted_batch_sizes": attempted_batch_sizes,
                }
                failures.append(failure)
                write_json(failures, failed_path)
                print(json.dumps({"failed": failure}, ensure_ascii=False), flush=True)
                continue

            row = summarize_run(
                config_payload=config_payload,
                seed=seed,
                output_dir=run_output_dir,
                parameter_count=parameter_count,
                batch_size=used_batch_size,
            )
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")
            aggregate_summary(summary_rows).to_csv(aggregate_path, index=False, encoding="utf-8")
            print(json.dumps({"summary": row}, ensure_ascii=False), flush=True)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")
        aggregate_summary(summary_rows).to_csv(aggregate_path, index=False, encoding="utf-8")
    manifest = {
        "input_file": args.input_file,
        "output_dir": str(output_dir),
        "configs": DEFAULT_CONFIGS,
        "seeds": seeds,
        "train_sample_size": args.train_sample_size,
        "sample_seed": args.sample_seed,
        "max_length": args.max_length,
        "pooling": args.pooling,
        "batch_size": args.batch_size,
        "retry_batch_size": args.retry_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "wandb_mode": args.wandb_mode,
        "wandb_group": None if args.no_wandb else args.wandb_group,
        "summary_path": str(summary_path),
        "aggregate_path": str(aggregate_path),
        "failed_path": str(failed_path) if failures else None,
        "failures": failures,
    }
    write_json(manifest, output_dir / "capacity_seed_sweep_manifest.json")
    if failures:
        raise SystemExit(f"{len(failures)} capacity sweep run(s) failed.")
    print(json.dumps({"manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
