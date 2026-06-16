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
DEFAULT_OUTPUT_DIR = (
    "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048"
)
DEFAULT_TRAIN_SIZES = "2k,4k,5k,10k,20k,40k,full"
DEFAULT_SEEDS = "42,43,44"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run initial M3 tiny Transformer learning curves across train seeds."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-sizes", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier-hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
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
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "runs/wandb"))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--reuse-known-runs", action=argparse.BooleanOptionalAction, default=True)
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


def known_reuse_candidates(*, pooling: str, seed: int, label: str) -> list[Path]:
    if pooling != "mean":
        return []
    candidates: list[Path] = []
    if seed == 42:
        previous_label = "80k" if label == "full" else label
        candidates.append(
            Path("runs/m3_learning_curve_atomic_constraint_heldout_seed42_mean_pooling_max2048")
            / f"train_{previous_label}"
        )
    if label == "full" and seed in {43, 44}:
        candidates.append(
            Path("runs/m3_seed_sensitivity_55k_atomic_constraint_heldout_seed42_mean_pooling_max2048")
            / f"train_55k_seed{seed}_sample42"
        )
    return candidates


def validate_run(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    seed: int,
    train_size: int | None,
) -> tuple[bool, str]:
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        return False, "missing manifest.json or metrics.json"
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:  # pragma: no cover - defensive around old runs
        return False, f"cannot read manifest: {exc}"
    config = manifest.get("config", {})
    training = manifest.get("training", {})
    counts = manifest.get("data_counts", {})
    checks = {
        "vocab_size": args.vocab_size,
        "max_length": args.max_length,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "heads": args.heads,
        "ffn_dim": args.ffn_dim,
        "pooling": args.pooling,
        "classifier_hidden_size": args.classifier_hidden_size,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            return False, f"config {key}={config.get(key)!r}, expected {expected!r}"
    if int(training.get("seed")) != int(seed):
        return False, f"seed={training.get('seed')!r}, expected {seed}"
    if int(counts.get("sample_seed")) != int(args.sample_seed):
        return False, f"sample_seed={counts.get('sample_seed')!r}, expected {args.sample_seed}"
    actual_train_rows = int(counts.get("train_rows"))
    requested = counts.get("requested_train_sample_size")
    if train_size is None:
        if requested is not None and int(requested) < actual_train_rows:
            return False, f"requested_train_sample_size={requested!r} is not full"
    else:
        if requested is None or int(requested) != int(train_size):
            return False, f"requested_train_sample_size={requested!r}, expected {train_size}"
        if actual_train_rows != int(train_size):
            return False, f"actual_train_rows={actual_train_rows}, expected {train_size}"
    return True, "ok"


def build_command(
    *,
    args: argparse.Namespace,
    seed: int,
    label: str,
    train_size: int | None,
    output_dir: Path,
) -> list[str]:
    train_script = Path(__file__).with_name("train_m3_tiny_transformer.py")
    wandb_group = args.wandb_group or (
        f"M3_initial_{args.pooling}_multiseed_learning_curve_sample{args.sample_seed}_max{args.max_length}"
    )
    wandb_name = f"M3_initial_{args.pooling}_{label}_seed{seed}_sample{args.sample_seed}"
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
        str(args.hidden_size),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--ffn-dim",
        str(args.ffn_dim),
        "--dropout",
        str(args.dropout),
        "--pooling",
        args.pooling,
        "--classifier-hidden-size",
        str(args.classifier_hidden_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
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


def summarize_run(
    *,
    run_dir: Path,
    pooling: str,
    seed: int,
    label: str,
    train_size: int | None,
    reused_from: Path | None,
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
    row = {
        "pooling": pooling,
        "curve_label": label,
        "requested_train_rows": train_size if train_size is not None else "full",
        "actual_train_rows": counts["train_rows"],
        "train_seed": seed,
        "sample_seed": counts["sample_seed"],
        "best_epoch": best["epoch"],
        "best_val_auprc": best["score"],
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
        "output_dir": str(run_dir),
        "reused_from": None if reused_from is None else str(reused_from),
    }
    return row


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
    label_order = {label: index for index, label in enumerate(["2k", "4k", "5k", "10k", "20k", "40k", "full"])}
    rows: list[dict[str, Any]] = []
    for (pooling, label), part in frame.groupby(["pooling", "curve_label"], sort=False):
        row: dict[str, Any] = {
            "pooling": pooling,
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
    return pd.DataFrame(rows).sort_values(["pooling", "order"], kind="mergesort")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multiseed_learning_curve_summary.csv"
    aggregate_path = output_dir / "multiseed_learning_curve_aggregate.csv"
    failures_path = output_dir / "multiseed_learning_curve_failures.json"
    seeds = parse_seeds(args.seeds)
    sizes = parse_train_sizes(args.train_sizes)
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for seed in seeds:
        for label, train_size in sizes:
            run_output_dir = output_dir / f"seed_{seed}" / f"train_{label}"
            run_dir_to_summarize = run_output_dir
            reused_from: Path | None = None

            valid, reason = validate_run(
                run_dir=run_output_dir,
                args=args,
                seed=seed,
                train_size=train_size,
            )
            if not valid and args.reuse_known_runs and not args.force:
                for candidate in known_reuse_candidates(pooling=args.pooling, seed=seed, label=label):
                    candidate_path = resolve_path(str(candidate))
                    candidate_valid, _ = validate_run(
                        run_dir=candidate_path,
                        args=args,
                        seed=seed,
                        train_size=train_size,
                    )
                    if candidate_valid:
                        run_dir_to_summarize = candidate_path
                        reused_from = candidate_path
                        valid = True
                        reason = "reused_known_run"
                        break

            if valid and not args.force:
                row = summarize_run(
                    run_dir=run_dir_to_summarize,
                    pooling=args.pooling,
                    seed=seed,
                    label=label,
                    train_size=train_size,
                    reused_from=reused_from,
                )
                summary_rows.append(row)
                print(json.dumps({"skipped_existing": row, "reason": reason}, ensure_ascii=False), flush=True)
                continue

            command = build_command(
                args=args,
                seed=seed,
                label=label,
                train_size=train_size,
                output_dir=run_output_dir,
            )
            print(
                json.dumps(
                    {
                        "pooling": args.pooling,
                        "seed": seed,
                        "curve_label": label,
                        "train_size": train_size,
                        "command": command,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.dry_run:
                continue
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            completed = subprocess.run(command, env=env, check=False)
            if completed.returncode != 0:
                failure = {
                    "pooling": args.pooling,
                    "seed": seed,
                    "curve_label": label,
                    "return_code": completed.returncode,
                    "output_dir": str(run_output_dir),
                }
                failures.append(failure)
                write_json(failures, failures_path)
                print(json.dumps({"failed": failure}, ensure_ascii=False), flush=True)
                continue
            row = summarize_run(
                run_dir=run_output_dir,
                pooling=args.pooling,
                seed=seed,
                label=label,
                train_size=train_size,
                reused_from=None,
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
        "output_dir": str(output_dir),
        "pooling": args.pooling,
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
            "classifier_hidden_size": args.classifier_hidden_size,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "early_stop_patience": args.early_stop_patience,
        },
        "wandb_mode": args.wandb_mode,
        "wandb_group": None
        if args.no_wandb
        else args.wandb_group
        or f"M3_initial_{args.pooling}_multiseed_learning_curve_sample{args.sample_seed}_max{args.max_length}",
        "summary_path": str(summary_path),
        "aggregate_path": str(aggregate_path),
        "failures_path": str(failures_path) if failures else None,
        "failures": failures,
    }
    write_json(manifest, output_dir / "multiseed_learning_curve_manifest.json")
    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed.")
    print(json.dumps({"manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
