from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from boundary_if.checkers.ifevalg import ensure_nltk_data, run_checker

DEFAULT_GENERATION_DIRS = [
    "qwen3_4b_instruct_2507_under2048_0_10000_max2048",
    "qwen3_4b_instruct_2507_under2048_10000_20000_max2048",
    "qwen3_4b_instruct_2507_under2048_20000_30000_max2048",
    "qwen3_4b_instruct_2507_under2048_30000_40000_max2048",
    "qwen3_4b_instruct_2507_under2048_40000_50000_max2048",
    "qwen3_4b_instruct_2507_under2048_50000_60000_max2048",
    "qwen3_4b_instruct_2507_under2048_60000_70000_max2048",
    "qwen3_4b_instruct_2507_under2048_70000_80000_max2048",
    "qwen3_4b_instruct_2507_under2048_80000_94390_max2048",
]


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vendored IFEvalG checker on all under-2048 generations."
    )
    parser.add_argument("--generation-root", default="data/generations")
    parser.add_argument("--generation-dirs", nargs="*", default=DEFAULT_GENERATION_DIRS)
    parser.add_argument(
        "--clusters-file",
        default="data/reports/prompt_clusters_under2048_seed42.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/checks/qwen3_4b_instruct_2507_under2048_ifevalg",
    )
    parser.add_argument("--nltk-cache-dir", default=".cache/nltk")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_generation_outputs(root: Path, names: list[str]) -> pd.DataFrame:
    columns = [
        "under2048_index",
        "prompt_id",
        "base_key",
        "ground_truth_spec",
        "constraint_signature",
        "constraint_family_signature",
        "num_constraints",
        "source_dataset",
        "constraint_type",
        "prompt_tokens",
        "response_text",
        "finish_reason",
        "output_tokens",
        "total_tokens",
        "output_hit_max_tokens",
        "output_truncated",
        "context_window_error",
        "ok",
    ]
    frames = []
    for name in names:
        path = root / name / "outputs.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns)
        frame["generation_dir"] = name
        frames.append(frame)
    outputs = pd.concat(frames, ignore_index=True)
    outputs = outputs.sort_values(["under2048_index", "prompt_id"], kind="mergesort")
    outputs = outputs.reset_index(drop=True)
    outputs["ok"] = outputs["ok"].fillna(False).astype(bool)
    outputs["context_window_error"] = outputs["context_window_error"].fillna(False).astype(bool)
    outputs["output_truncated"] = outputs["output_truncated"].fillna(False).astype(bool)
    outputs["output_hit_max_tokens"] = outputs["output_hit_max_tokens"].fillna(False).astype(bool)
    return outputs


def check_row(row: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response_text = "" if pd.isna(row["response_text"]) else str(row["response_text"])
    checker = run_checker(str(row["ground_truth_spec"]), response_text)
    base_record = {
        "under2048_index": int(row["under2048_index"]),
        "prompt_id": str(row["prompt_id"]),
        "base_key": str(row["base_key"]),
        "constraint_signature": str(row["constraint_signature"]),
        "constraint_family_signature": str(row["constraint_family_signature"]),
        "num_constraints": int(row["num_constraints"]),
        "source_dataset": str(row["source_dataset"]),
        "constraint_type": str(row["constraint_type"]),
        "prompt_tokens": int(row["prompt_tokens"]),
        "response_chars": len(response_text),
        "finish_reason": str(row["finish_reason"]),
        "output_tokens": int(row["output_tokens"]),
        "total_tokens": int(row["total_tokens"]),
        "output_hit_max_tokens": bool(row["output_hit_max_tokens"]),
        "output_truncated": bool(row["output_truncated"]),
        "context_window_error": bool(row["context_window_error"]),
        "generation_ok": bool(row["ok"]),
        "generation_dir": str(row["generation_dir"]),
        "checker_name": checker["checker_name"],
        "checker_pass": bool(checker["strict_pass"]),
        "strict_pass": bool(checker["strict_pass"]),
        "num_constraints_checked": int(checker["num_constraints_checked"]),
        "followed_count": int(checker["followed_count"]),
        "failed_count": int(checker["failed_count"]),
        "checker_error_count": int(checker["checker_error_count"]),
        "checker_seconds": float(checker["checker_seconds"]),
    }
    constraint_records = []
    for item in checker["per_constraint"]:
        constraint_records.append(
            {
                "under2048_index": base_record["under2048_index"],
                "prompt_id": base_record["prompt_id"],
                "constraint_index": int(item["constraint_index"]),
                "instruction_id": str(item["instruction_id"]),
                "kwargs_json": json_string(item["kwargs"]),
                "followed": bool(item["followed"]),
                "checker_error": bool(item["checker_error"]),
                "checker_error_type": str(item["checker_error_type"]),
                "checker_error_message": str(item["checker_error_message"]),
                "checker_seconds": float(item["checker_seconds"]),
            }
        )
    return base_record, constraint_records


def write_shard(shard: pd.DataFrame, checks_path: Path, constraints_path: Path) -> None:
    check_records = []
    constraint_records = []
    for _index, row in tqdm(
        shard.iterrows(),
        total=len(shard),
        desc=f"checking {shard['under2048_index'].min()}..{shard['under2048_index'].max()}",
    ):
        check_record, row_constraint_records = check_row(row)
        check_records.append(check_record)
        constraint_records.extend(row_constraint_records)

    checks = pd.DataFrame(check_records)
    constraints = pd.DataFrame(constraint_records)
    checks.to_parquet(checks_path, index=False)
    constraints.to_parquet(constraints_path, index=False)


def combine_shards(shard_dir: Path, pattern: str, output_path: Path) -> pd.DataFrame:
    shard_paths = sorted(shard_dir.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for {pattern} in {shard_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in shard_paths], ignore_index=True)
    sort_columns = ["under2048_index", "prompt_id"]
    if "constraint_index" in frame.columns:
        sort_columns.append("constraint_index")
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    frame.to_parquet(output_path, index=False)
    return frame


def mean_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return round(float(series.astype(bool).mean()), 8)


def rate_table(
    df: pd.DataFrame,
    group_columns: list[str],
    pass_column: str = "checker_pass",
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_columns, dropna=False)
        .agg(
            row_count=(pass_column, "size"),
            pass_count=(pass_column, "sum"),
            checker_error_rows=("checker_error_count", lambda s: int((s > 0).sum())),
            truncated_count=("output_truncated", "sum"),
            prompt_tokens_p50=("prompt_tokens", "median"),
            output_tokens_p50=("output_tokens", "median"),
            output_tokens_p90=("output_tokens", lambda s: float(np.percentile(s, 90))),
            num_constraints_mean=("num_constraints", "mean"),
        )
        .reset_index()
    )
    grouped["pass_rate"] = grouped["pass_count"] / grouped["row_count"]
    grouped["truncated_rate"] = grouped["truncated_count"] / grouped["row_count"]
    return grouped.sort_values(["pass_rate", "row_count"], ascending=[False, False])


def constraint_rate_table(constraints: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        constraints.groupby("instruction_id", dropna=False)
        .agg(
            row_count=("followed", "size"),
            pass_count=("followed", "sum"),
            checker_error_count=("checker_error", "sum"),
        )
        .reset_index()
    )
    grouped["pass_rate"] = grouped["pass_count"] / grouped["row_count"]
    grouped["checker_error_rate"] = grouped["checker_error_count"] / grouped["row_count"]
    return grouped.sort_values(["pass_rate", "row_count"], ascending=[False, False])


def build_summary(
    checks: pd.DataFrame,
    constraints: pd.DataFrame,
    clusters_file: Path,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    clusters = pd.read_parquet(clusters_file, columns=["prompt_id", "cluster", "length_bin"])
    checks_with_clusters = checks.merge(
        clusters,
        on="prompt_id",
        how="left",
        validate="one_to_one",
    )
    missing_cluster_count = int(checks_with_clusters["cluster"].isna().sum())
    if missing_cluster_count:
        raise ValueError(f"{missing_cluster_count} check rows did not match clusters")

    checks_with_clusters["cluster"] = checks_with_clusters["cluster"].astype(int)
    checks_with_clusters.to_parquet(output_dir / "checks_with_clusters.parquet", index=False)

    cluster_rates = rate_table(checks_with_clusters, ["cluster"])
    cluster_rates.to_csv(output_dir / "pass_rate_by_cluster.csv", index=False, encoding="utf-8")

    cluster_length_rates = rate_table(checks_with_clusters, ["cluster", "length_bin"])
    cluster_length_rates.to_csv(
        output_dir / "pass_rate_by_cluster_length_bin.csv",
        index=False,
        encoding="utf-8",
    )

    length_rates = rate_table(checks_with_clusters, ["length_bin"])
    length_rates.to_csv(output_dir / "pass_rate_by_length_bin.csv", index=False, encoding="utf-8")

    num_constraint_rates = rate_table(checks_with_clusters, ["num_constraints"])
    num_constraint_rates.to_csv(
        output_dir / "pass_rate_by_num_constraints.csv",
        index=False,
        encoding="utf-8",
    )

    family_rates = rate_table(checks_with_clusters, ["constraint_family_signature"])
    family_rates.to_csv(
        output_dir / "pass_rate_by_constraint_family_signature.csv",
        index=False,
        encoding="utf-8",
    )

    truncation_rates = rate_table(checks_with_clusters, ["output_truncated"])
    truncation_rates.to_csv(
        output_dir / "pass_rate_by_output_truncated.csv",
        index=False,
        encoding="utf-8",
    )

    instruction_rates = constraint_rate_table(constraints)
    instruction_rates.to_csv(
        output_dir / "pass_rate_by_instruction_id.csv",
        index=False,
        encoding="utf-8",
    )

    pass_count = int(checks["checker_pass"].sum())
    row_count = int(len(checks))
    checker_error_rows = int((checks["checker_error_count"] > 0).sum())
    checker_names = sorted(checks["checker_name"].dropna().unique().tolist())
    summary = {
        "checker_name": checker_names[0] if len(checker_names) == 1 else checker_names,
        "row_count": row_count,
        "pass_count": pass_count,
        "fail_count": row_count - pass_count,
        "pass_rate": round(pass_count / row_count, 8),
        "checker_error_rows": checker_error_rows,
        "checker_error_row_rate": round(checker_error_rows / row_count, 8),
        "constraint_check_count": int(len(constraints)),
        "constraint_followed_count": int(constraints["followed"].sum()),
        "constraint_pass_rate": mean_bool(constraints["followed"]),
        "output_truncated_count": int(checks["output_truncated"].sum()),
        "output_truncated_pass_rate": mean_bool(
            checks.loc[checks["output_truncated"], "checker_pass"]
        ),
        "not_output_truncated_pass_rate": mean_bool(
            checks.loc[~checks["output_truncated"], "checker_pass"]
        ),
        "cluster_count": int(cluster_rates["cluster"].nunique()),
        "top_pass_rate_clusters": cluster_rates.head(10).to_dict("records"),
        "lowest_pass_rate_clusters": cluster_rates.sort_values(
            ["pass_rate", "row_count"],
            ascending=[True, False],
        )
        .head(10)
        .to_dict("records"),
        "top_instruction_pass_rates": instruction_rates.head(15).to_dict("records"),
        "lowest_instruction_pass_rates": instruction_rates.sort_values(
            ["pass_rate", "row_count"],
            ascending=[True, False],
        )
        .head(15)
        .to_dict("records"),
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "checks": str(output_dir / "checks.parquet"),
            "checks_with_clusters": str(output_dir / "checks_with_clusters.parquet"),
            "per_constraint": str(output_dir / "per_constraint.parquet"),
            "pass_rate_by_cluster": str(output_dir / "pass_rate_by_cluster.csv"),
            "pass_rate_by_instruction_id": str(output_dir / "pass_rate_by_instruction_id.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    total_start = now()
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    shard_dir = output_dir / "shards"
    constraint_shard_dir = output_dir / "per_constraint_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    constraint_shard_dir.mkdir(parents=True, exist_ok=True)

    ensure_nltk_data(resolve_path(args.nltk_cache_dir))

    outputs = read_generation_outputs(resolve_path(args.generation_root), args.generation_dirs)
    if args.limit is not None:
        outputs = outputs.head(args.limit).copy()
    if not outputs["ok"].all():
        raise ValueError(
            "Generation outputs contain failed rows; checker expects successful outputs."
        )
    if outputs["context_window_error"].any():
        raise ValueError("Generation outputs contain context-window-error rows.")

    resume = not args.no_resume
    total_rows = len(outputs)
    for start in range(0, total_rows, args.shard_size):
        end = min(start + args.shard_size, total_rows)
        checks_path = shard_dir / f"checks_{start:06d}_{end:06d}.parquet"
        constraints_path = constraint_shard_dir / f"constraints_{start:06d}_{end:06d}.parquet"
        if resume and checks_path.exists() and constraints_path.exists():
            continue
        write_shard(outputs.iloc[start:end].copy(), checks_path, constraints_path)

    checks = combine_shards(shard_dir, "checks_*.parquet", output_dir / "checks.parquet")
    constraints = combine_shards(
        constraint_shard_dir,
        "constraints_*.parquet",
        output_dir / "per_constraint.parquet",
    )
    expected_rows = len(outputs)
    if len(checks) != expected_rows:
        raise ValueError(f"Expected {expected_rows} check rows, got {len(checks)}")
    if checks["under2048_index"].duplicated().any():
        raise ValueError("Duplicate under2048_index values in checks")

    summary = build_summary(
        checks,
        constraints,
        resolve_path(args.clusters_file),
        output_dir,
        elapsed_since(total_start),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
