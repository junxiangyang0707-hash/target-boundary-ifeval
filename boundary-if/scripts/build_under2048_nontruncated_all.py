from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the under-2048, non-truncated deterministic checker all set."
    )
    parser.add_argument(
        "--normalized-path",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument("--generation-root", default="data/generations")
    parser.add_argument("--generation-dirs", nargs="*", default=DEFAULT_GENERATION_DIRS)
    parser.add_argument(
        "--checks-path",
        default=(
            "data/checks/qwen3_4b_instruct_2507_under2048_ifevalg_deterministic/"
            "checks_with_clusters.parquet"
        ),
    )
    parser.add_argument(
        "--output-path",
        default=(
            "data/promptsets/"
            "if_multi_constraints_upto5.qwen3_4b_instruct_2507."
            "under2048_nontruncated.all.parquet"
        ),
    )
    parser.add_argument("--audit-path", default=None)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def read_generations(root: Path, names: list[str]) -> pd.DataFrame:
    columns = [
        "under2048_index",
        "prompt_id",
        "response_text",
        "context_limit",
        "max_output_tokens",
        "prompt_plus_max_output_tokens",
        "within_context_budget",
        "model",
        "temperature",
        "vllm_input_tokens",
        "seconds",
        "completion_tokens_per_second",
        "total_tokens_per_second",
        "status_code",
        "attempts",
        "error",
    ]
    frames = []
    for name in names:
        path = root / name / "outputs.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns)
        frame["generation_dir_from_outputs"] = name
        frames.append(frame)
    generations = pd.concat(frames, ignore_index=True)
    generations = generations.sort_values("under2048_index", kind="mergesort")
    return generations.reset_index(drop=True)


def sorted_count_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 8)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_unique(df: pd.DataFrame, column: str, name: str) -> None:
    duplicates = int(df[column].duplicated().sum())
    if duplicates:
        raise ValueError(f"{name} has {duplicates} duplicate {column} values")


def main() -> None:
    args = parse_args()
    normalized_path = resolve_path(args.normalized_path)
    generation_root = resolve_path(args.generation_root)
    checks_path = resolve_path(args.checks_path)
    output_path = resolve_path(args.output_path)
    audit_path = (
        resolve_path(args.audit_path)
        if args.audit_path
        else output_path.with_suffix(".audit.json")
    )

    normalized = pd.read_parquet(normalized_path)
    checks = pd.read_parquet(checks_path)
    generations = read_generations(generation_root, args.generation_dirs)

    validate_unique(normalized, "prompt_id", "normalized promptset")
    validate_unique(checks, "prompt_id", "checker outputs")
    validate_unique(generations, "prompt_id", "generation outputs")
    validate_unique(generations, "under2048_index", "generation outputs")

    filtered_checks = checks[~checks["output_truncated"]].copy()
    if filtered_checks["context_window_error"].any():
        raise ValueError("Filtered checks contain context-window-error rows")
    if not filtered_checks["generation_ok"].all():
        raise ValueError("Filtered checks contain failed generation rows")
    if (filtered_checks["prompt_tokens"] >= 2048).any():
        raise ValueError("Filtered checks contain prompt_tokens >= 2048")

    check_columns = [
        "prompt_id",
        "under2048_index",
        "prompt_tokens",
        "response_chars",
        "finish_reason",
        "output_tokens",
        "total_tokens",
        "output_hit_max_tokens",
        "output_truncated",
        "context_window_error",
        "generation_ok",
        "generation_dir",
        "checker_name",
        "checker_pass",
        "strict_pass",
        "num_constraints_checked",
        "followed_count",
        "failed_count",
        "checker_error_count",
        "checker_seconds",
        "cluster",
        "length_bin",
    ]
    generation_columns = [
        "prompt_id",
        "response_text",
        "context_limit",
        "max_output_tokens",
        "prompt_plus_max_output_tokens",
        "within_context_budget",
        "model",
        "temperature",
        "vllm_input_tokens",
        "seconds",
        "completion_tokens_per_second",
        "total_tokens_per_second",
        "status_code",
        "attempts",
        "error",
        "generation_dir_from_outputs",
    ]
    all_set = (
        normalized.merge(
            filtered_checks[check_columns],
            on="prompt_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            generations[generation_columns],
            on="prompt_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("under2048_index", kind="mergesort")
        .reset_index(drop=True)
    )

    if len(all_set) != len(filtered_checks):
        raise ValueError(f"Expected {len(filtered_checks)} rows, got {len(all_set)}")
    if all_set["response_text"].isna().any():
        raise ValueError("Missing response_text after generation join")
    if all_set["num_constraints_checked"].ne(all_set["num_constraints"]).any():
        raise ValueError("num_constraints_checked differs from num_constraints")
    if all_set["checker_error_count"].gt(0).any():
        raise ValueError("Checker errors present in all set")
    if all_set["generation_dir"].ne(all_set["generation_dir_from_outputs"]).any():
        raise ValueError("generation_dir mismatch between checks and generation outputs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_set.to_parquet(output_path, index=False)

    pass_count = int(all_set["checker_pass"].sum())
    row_count = int(len(all_set))
    constraint_followed = int(all_set["followed_count"].sum())
    constraint_total = int(all_set["num_constraints_checked"].sum())
    audit = {
        "dataset_name": "if_multi_constraints_upto5_qwen3_4b_under2048_nontruncated_all",
        "source_normalized_path": str(normalized_path),
        "source_checks_path": str(checks_path),
        "source_generation_root": str(generation_root),
        "output_path": str(output_path),
        "row_count": row_count,
        "source_normalized_row_count": int(len(normalized)),
        "source_under2048_checked_row_count": int(len(checks)),
        "excluded_output_truncated_count": int(checks["output_truncated"].sum()),
        "output_truncated_count": int(all_set["output_truncated"].sum()),
        "prompt_tokens_min": int(all_set["prompt_tokens"].min()),
        "prompt_tokens_max": int(all_set["prompt_tokens"].max()),
        "checker_name": sorted(all_set["checker_name"].dropna().unique().tolist()),
        "pass_count": pass_count,
        "fail_count": row_count - pass_count,
        "pass_rate": safe_rate(pass_count, row_count),
        "constraint_check_count": constraint_total,
        "constraint_followed_count": constraint_followed,
        "constraint_pass_rate": safe_rate(constraint_followed, constraint_total),
        "num_constraints_distribution": sorted_count_dict(all_set["num_constraints"]),
        "checker_pass_distribution": sorted_count_dict(all_set["checker_pass"]),
        "cluster_distribution": sorted_count_dict(all_set["cluster"]),
        "length_bin_distribution": sorted_count_dict(all_set["length_bin"]),
        "source_dataset_distribution": sorted_count_dict(all_set["source_dataset"]),
        "constraint_type_distribution": sorted_count_dict(all_set["constraint_type"]),
        "columns": list(all_set.columns),
    }
    write_json(audit, audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
