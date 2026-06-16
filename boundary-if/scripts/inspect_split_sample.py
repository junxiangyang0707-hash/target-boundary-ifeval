from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample rows from a split parquet for inspection.")
    parser.add_argument(
        "--split-file",
        default="data/splits/group_key_seed42.parquet",
        help="Path to split parquet, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
        help="Path to normalized promptset parquet, relative to project root unless absolute.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="Inspect one or more exact prompt_id values. Can be passed multiple times.",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Omit it for a fresh random sample each run.",
    )
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Print full text fields. Equivalent to setting --max-chars 0.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def truncate(value: object, max_chars: int | None) -> object:
    if max_chars is None or max_chars <= 0:
        return value
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def to_jsonable(value: object, max_chars: int | None) -> object:
    if isinstance(value, str):
        return truncate(value, max_chars)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item, max_chars) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, max_chars) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    split_path = resolve_path(args.split_file)
    promptset_path = resolve_path(args.promptset_file)

    split_df = pd.read_parquet(split_path)
    promptset_df = pd.read_parquet(promptset_path)
    max_chars = None if args.no_truncate or args.max_chars <= 0 else args.max_chars
    merged = split_df.merge(
        promptset_df[["prompt_id", "user_prompt", "constraint_text", "ground_truth_spec"]],
        on="prompt_id",
        how="left",
        validate="one_to_one",
    )

    if args.split != "all":
        merged = merged[merged["split"] == args.split]

    if args.prompt_id:
        requested_prompt_ids = set(args.prompt_id)
        sampled = merged[merged["prompt_id"].isin(requested_prompt_ids)]
        sample_size = len(sampled)
    else:
        sample_size = min(args.num_samples, len(merged))
        sampled = merged.sample(n=sample_size, random_state=args.seed) if sample_size else merged

    print(
        json.dumps(
            {
                "split_file": str(split_path),
                "promptset_file": str(promptset_path),
                "requested_split": args.split,
                "requested_prompt_ids": args.prompt_id,
                "available_rows": int(len(merged)),
                "sampled_rows": int(sample_size),
                "seed": args.seed,
                "max_chars": max_chars,
                "split_counts": split_df["split"].value_counts().sort_index().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    inspect_columns = [
        "prompt_id",
        "base_key",
        "split",
        "num_constraints",
        "constraint_signature",
        "constraint_family_signature",
        "instruction_ids",
        "heldout_instruction_ids",
        "user_prompt",
        "constraint_text",
        "ground_truth_spec",
    ]
    records = []
    for record in sampled[inspect_columns].to_dict(orient="records"):
        records.append({key: to_jsonable(value, max_chars) for key, value in record.items()})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
