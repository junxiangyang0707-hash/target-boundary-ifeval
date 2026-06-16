from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k_atomic_train.max2048.parquet"
)
PROMPTSET_FILE = (
    "data/promptsets/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.all.parquet"
)
TOKENIZER_DIR = "data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_atomic_train"
DEFAULT_OUTPUT_DIR = "runs/strict_atomic_tokenizer_multiseed_key_runs"
SEEDS = [42, 43, 44]
SAMPLE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run requested strict atomic-tokenizer multiseed key experiments: "
            "M1 full, M3 40k/full, and M4 20k/40k/full."
        )
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def complete_run(run_dir: Path) -> bool:
    return (run_dir / "manifest.json").exists() and (run_dir / "metrics.json").exists()


def complete_m1(output_dir: Path) -> bool:
    return (output_dir / "learning_curve_summary.csv").exists() and complete_run(
        output_dir / "train_full"
    )


def complete_pretrain(run_dir: Path) -> bool:
    return (run_dir / "pretrained_encoder.pt").exists() and (
        run_dir / "pretraining_manifest.json"
    ).exists()


def run_command(
    *,
    name: str,
    command: list[str],
    log_dir: Path,
    dry_run: bool,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(json.dumps({"start": name, "command": command, "log": str(log_path)}, ensure_ascii=False), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(json.dumps({"end": name, "return_code": completed.returncode}, ensure_ascii=False), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {completed.returncode}; see {log_path}")


def wandb_args(no_wandb: bool, *, name: str, group: str) -> list[str]:
    args = [
        "--wandb-name",
        name,
        "--wandb-group",
        group,
    ]
    if no_wandb:
        args.append("--no-wandb")
    return args


def m1_command(seed: int, output_dir: Path, no_wandb: bool) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_m1_learning_curve.py",
        "--input-file",
        INPUT_FILE,
        "--output-dir",
        str(output_dir),
        "--train-sizes",
        "full",
        "--sample-seed",
        str(SAMPLE_SEED),
        "--random-state",
        str(seed),
        "--no-save-model",
        "--no-save-predictions",
        "--wandb-group",
        "strict_atomic_tokenizer_multiseed_m1_full",
    ]
    if no_wandb:
        command.append("--no-wandb")
    return command


def m3_command(seed: int, label: str, train_size: int | None, output_dir: Path, no_wandb: bool) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_m3_tiny_transformer.py",
        "--input-file",
        INPUT_FILE,
        "--output-dir",
        str(output_dir),
        "--pooling",
        "mean",
        "--max-length",
        "2048",
        "--batch-size",
        "8",
        "--eval-batch-size",
        "16",
        "--seed",
        str(seed),
        "--sample-seed",
        str(SAMPLE_SEED),
        "--no-wandb-save-model",
        "--no-save-predictions",
        *wandb_args(
            no_wandb,
            name=f"M3_strict_atomic_tokenizer_mean_{label}_seed{seed}_sample{SAMPLE_SEED}",
            group="strict_atomic_tokenizer_multiseed_m3",
        ),
    ]
    if train_size is not None:
        command.extend(["--train-sample-size", str(train_size)])
    return command


def m4_pretrain_command(seed: int, output_dir: Path, no_wandb: bool) -> list[str]:
    return [
        sys.executable,
        "scripts/train_m4_pretrain_encoder.py",
        "--input-file",
        INPUT_FILE,
        "--promptset-file",
        PROMPTSET_FILE,
        "--tokenizer-dir",
        TOKENIZER_DIR,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--batch-size",
        "8",
        "--no-wandb-save-model",
        *wandb_args(
            no_wandb,
            name=f"M4_strict_atomic_tokenizer_pretrain_seed{seed}",
            group="strict_atomic_tokenizer_multiseed_m4_pretrain",
        ),
    ]


def m4_classifier_command(
    seed: int,
    label: str,
    train_size: int | None,
    encoder_file: Path,
    output_dir: Path,
    no_wandb: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_m4_frozen_classifier.py",
        "--input-file",
        INPUT_FILE,
        "--encoder-file",
        str(encoder_file),
        "--output-dir",
        str(output_dir),
        "--max-length",
        "2048",
        "--batch-size",
        "8",
        "--eval-batch-size",
        "16",
        "--seed",
        str(seed),
        "--sample-seed",
        str(SAMPLE_SEED),
        "--no-wandb-save-model",
        "--no-save-predictions",
        *wandb_args(
            no_wandb,
            name=f"M4_strict_atomic_tokenizer_frozen_{label}_seed{seed}_sample{SAMPLE_SEED}",
            group="strict_atomic_tokenizer_multiseed_m4_frozen",
        ),
    ]
    if train_size is not None:
        command.extend(["--train-sample-size", str(train_size)])
    return command


def summarize_m1(seed: int, output_dir: Path, reused_from: Path | None) -> dict[str, Any]:
    frame = pd.read_csv(output_dir / "learning_curve_summary.csv")
    row = frame[frame["source_label"].astype(str).eq("full")].iloc[0].to_dict()
    manifest = read_json(output_dir / "train_full" / "manifest.json")
    config = manifest.get("m1_config") or manifest.get("config") or {}
    return {
        "model": "M1 TF-IDF",
        "curve_label": "full",
        "train_seed": seed,
        "sample_seed": SAMPLE_SEED,
        "requested_train_rows": "full",
        "actual_train_rows": int(row["actual_train_rows"]),
        "best_epoch": None,
        "epochs_run": None,
        "val_auroc": float(row["val_auroc"]),
        "val_auprc": float(row["val_auprc"]),
        "test_auroc": float(row["test_auroc"]),
        "test_auprc": float(row["test_auprc"]),
        "test_positive_rate": float(row["test_positive_rate"]),
        "test_auprc_lift": float(row["test_auprc_lift"]),
        "total_seconds": float(row["total_seconds"]),
        "output_dir": str(output_dir),
        "reused_from": None if reused_from is None else str(reused_from),
        "random_state": config.get("random_state"),
    }


def summarize_neural(
    *,
    model: str,
    seed: int,
    label: str,
    requested_train_rows: int | str,
    run_dir: Path,
    reused_from: Path | None,
    encoder_file: Path | None = None,
) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    manifest = read_json(run_dir / "manifest.json")
    test = metrics["by_split"]["test"]
    val = metrics["by_split"]["val"]
    return {
        "model": model,
        "curve_label": label,
        "train_seed": seed,
        "sample_seed": int(manifest["data_counts"]["sample_seed"]),
        "requested_train_rows": requested_train_rows,
        "actual_train_rows": int(manifest["data_counts"]["train_rows"]),
        "best_epoch": int(manifest["best"]["epoch"]),
        "epochs_run": int(manifest["training"]["epochs_run"]),
        "val_auroc": float(val["auroc"]),
        "val_auprc": float(val["auprc"]),
        "test_auroc": float(test["auroc"]),
        "test_auprc": float(test["auprc"]),
        "test_positive_rate": float(test["positive_rate"]),
        "test_auprc_lift": float(test["auprc"]) / float(test["positive_rate"]),
        "total_seconds": float(manifest["timing_seconds"]["total"]),
        "output_dir": str(run_dir),
        "reused_from": None if reused_from is None else str(reused_from),
        "encoder_file": None if encoder_file is None else str(encoder_file),
    }


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
        "test_auprc_lift",
        "total_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for (model, label), part in frame.groupby(["model", "curve_label"], sort=False):
        row: dict[str, Any] = {
            "model": model,
            "curve_label": label,
            "run_count": int(len(part)),
            "seeds": ",".join(str(seed) for seed in sorted(part["train_seed"].astype(int))),
            "actual_train_rows_mean": float(part["actual_train_rows"].mean()),
            "test_positive_rate": float(part["test_positive_rate"].dropna().iloc[0]),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_summaries(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "requested_multiseed_summary.csv", index=False, encoding="utf-8")
    write_json(rows, output_dir / "requested_multiseed_summary.json")
    aggregate(frame).to_csv(output_dir / "requested_multiseed_aggregate.csv", index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    try:
        for seed in SEEDS:
            if seed == 42:
                m1_dir = Path("runs/m1_strict_atomic_tokenizer_atomic_constraint_heldout_seed42_max2048")
                reused = m1_dir
            else:
                m1_dir = output_dir / "m1_tfidf_full" / f"seed_{seed}"
                reused = None
            if not complete_m1(m1_dir) or args.force:
                run_command(
                    name=f"m1_full_seed{seed}",
                    command=m1_command(seed, m1_dir, args.no_wandb),
                    log_dir=log_dir,
                    dry_run=args.dry_run,
                )
                if args.dry_run and not complete_m1(m1_dir):
                    continue
            rows.append(summarize_m1(seed, m1_dir, reused if seed == 42 else None))
            write_summaries(rows, output_dir)

        for seed in SEEDS:
            for label, train_size in [("40k", 40000), ("full", None)]:
                if seed == 42:
                    run_dir = Path(f"runs/strict_atomic_tokenizer_key_runs/m3_mean_{label}_seed42")
                    reused = run_dir
                else:
                    run_dir = output_dir / "m3_mean" / f"seed_{seed}" / f"train_{label}"
                    reused = None
                if not complete_run(run_dir) or args.force:
                    run_command(
                        name=f"m3_mean_{label}_seed{seed}",
                        command=m3_command(seed, label, train_size, run_dir, args.no_wandb),
                        log_dir=log_dir,
                        dry_run=args.dry_run,
                    )
                    if args.dry_run and not complete_run(run_dir):
                        continue
                rows.append(
                    summarize_neural(
                        model="M3 mean",
                        seed=seed,
                        label=label,
                        requested_train_rows=train_size if train_size is not None else "full",
                        run_dir=run_dir,
                        reused_from=reused if seed == 42 else None,
                    )
                )
                write_summaries(rows, output_dir)

        for seed in SEEDS:
            if seed == 42:
                pretrain_dir = Path("runs/strict_atomic_tokenizer_key_runs/m4_pretrain_seed42")
                pretrain_reused = pretrain_dir
            else:
                pretrain_dir = output_dir / "m4" / f"seed_{seed}" / "pretrain"
                pretrain_reused = None
            if not complete_pretrain(pretrain_dir) or args.force:
                run_command(
                    name=f"m4_pretrain_seed{seed}",
                    command=m4_pretrain_command(seed, pretrain_dir, args.no_wandb),
                    log_dir=log_dir,
                    dry_run=args.dry_run,
                )
                if args.dry_run and not complete_pretrain(pretrain_dir):
                    continue
            encoder_file = pretrain_dir / "pretrained_encoder.pt"
            if pretrain_reused is not None:
                print(json.dumps({"reused_pretrain": str(pretrain_reused)}, ensure_ascii=False), flush=True)
            for label, train_size in [("20k", 20000), ("40k", 40000), ("full", None)]:
                if seed == 42 and label == "full":
                    run_dir = Path("runs/strict_atomic_tokenizer_key_runs/m4_frozen_full_seed42")
                    reused = run_dir
                else:
                    run_dir = output_dir / "m4" / f"seed_{seed}" / f"train_{label}"
                    reused = None
                if not complete_run(run_dir) or args.force:
                    run_command(
                        name=f"m4_frozen_{label}_seed{seed}",
                        command=m4_classifier_command(
                            seed,
                            label,
                            train_size,
                            encoder_file,
                            run_dir,
                            args.no_wandb,
                        ),
                        log_dir=log_dir,
                        dry_run=args.dry_run,
                    )
                    if args.dry_run and not complete_run(run_dir):
                        continue
                rows.append(
                    summarize_neural(
                        model="M4 frozen",
                        seed=seed,
                        label=label,
                        requested_train_rows=train_size if train_size is not None else "full",
                        run_dir=run_dir,
                        reused_from=reused if seed == 42 and label == "full" else None,
                        encoder_file=encoder_file,
                    )
                )
                write_summaries(rows, output_dir)
    except Exception as exc:
        failure = {"error": str(exc), "rows_completed": len(rows)}
        write_json(failure, output_dir / "failure.json")
        write_summaries(rows, output_dir)
        raise

    manifest = {
        "input_file": INPUT_FILE,
        "promptset_file": PROMPTSET_FILE,
        "tokenizer_dir": TOKENIZER_DIR,
        "output_dir": str(output_dir),
        "seeds": SEEDS,
        "sample_seed": SAMPLE_SEED,
        "requested_runs": [
            "M1 TF-IDF full",
            "M3 mean 40k",
            "M3 mean full",
            "M4 frozen 20k",
            "M4 frozen 40k",
            "M4 frozen full",
        ],
    }
    write_json(manifest, output_dir / "requested_multiseed_manifest.json")
    write_summaries(rows, output_dir)
    print(json.dumps({"manifest": manifest, "row_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
