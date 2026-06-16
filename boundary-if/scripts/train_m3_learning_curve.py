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
DEFAULT_OUTPUT_DIR = "runs/m3_learning_curve_atomic_constraint_heldout_seed42_mean_pooling_max2048"
DEFAULT_TRAIN_SIZES = "2k,5k,10k,20k,40k,80k,full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3 train-size learning curve experiments with W&B grouping."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-sizes", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=None,
        help="Optional debug-only test sample size. By default, val/test are evaluated in full.",
    )
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--selection-metric", default="auprc")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "runs/wandb"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], split: str, metric: str) -> Any:
    return metrics.get("by_split", {}).get(split, {}).get(metric)


def build_train_command(
    *,
    args: argparse.Namespace,
    label: str,
    train_size: int | None,
    run_output_dir: Path,
    train_script: Path,
    wandb_group: str,
) -> list[str]:
    wandb_name = (
        f"M3_{args.pooling}_learning_curve_{label}_atomic_constraint_heldout_seed{args.seed}"
    )
    command = [
        sys.executable,
        str(train_script),
        "--input-file",
        args.input_file,
        "--output-dir",
        str(run_output_dir),
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
        str(args.seed),
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
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.no_wandb:
        command.append("--no-wandb")
    if not args.save_predictions:
        command.append("--no-save-predictions")
    if train_size is not None:
        command.extend(["--train-sample-size", str(train_size)])
    if args.test_sample_size is not None:
        command.extend(["--test-sample-size", str(args.test_sample_size)])
    return command


def summarize_run(label: str, train_size: int | None, run_output_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_output_dir / "manifest.json")
    metrics = load_json(run_output_dir / "metrics.json")
    data_counts = manifest["data_counts"]
    best = manifest["best"]
    timing = manifest["timing_seconds"]
    return {
        "curve_label": label,
        "requested_train_rows": train_size if train_size is not None else "full",
        "actual_train_rows": data_counts["train_rows"],
        "requested_test_sample_size": data_counts["requested_test_sample_size"],
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "val_auroc": metric_value(metrics, "val", "auroc"),
        "val_auprc": metric_value(metrics, "val", "auprc"),
        "test_auroc": metric_value(metrics, "test", "auroc"),
        "test_auprc": metric_value(metrics, "test", "auprc"),
        "test_row_count": metric_value(metrics, "test", "row_count"),
        "total_seconds": timing["total"],
        "run_output_dir": str(run_output_dir),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train_m3_tiny_transformer.py")
    wandb_group = args.wandb_group or (
        f"M3_{args.pooling}_learning_curve_atomic_constraint_heldout_seed{args.seed}_max2048"
    )
    sizes = parse_train_sizes(args.train_sizes)

    commands: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, train_size in sizes:
        run_output_dir = output_dir / f"train_{label}"
        command = build_train_command(
            args=args,
            label=label,
            train_size=train_size,
            run_output_dir=run_output_dir,
            train_script=train_script,
            wandb_group=wandb_group,
        )
        commands.append(
            {
                "curve_label": label,
                "requested_train_rows": train_size if train_size is not None else "full",
                "output_dir": str(run_output_dir),
                "command": command,
            }
        )
        print(json.dumps(commands[-1], ensure_ascii=False), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        summary_rows.append(summarize_run(label, train_size, run_output_dir))
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "learning_curve_summary.csv",
            index=False,
            encoding="utf-8",
        )
        write_json(summary_rows, output_dir / "learning_curve_summary.json")

    write_json(
        {
            "input_file": args.input_file,
            "output_dir": str(output_dir),
            "train_sizes": [
                {"label": label, "requested_train_rows": size if size is not None else "full"}
                for label, size in sizes
            ],
            "test_sample_size": args.test_sample_size,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "wandb_mode": args.wandb_mode,
            "wandb_group": wandb_group,
            "commands": commands,
        },
        output_dir / "learning_curve_manifest.json",
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "commands": commands}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
