from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.parquet"
)
DEFAULT_OUTPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_AUDIT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter tokenized prompt parquet rows by full raw-prompt BPE token length."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--audit-file", default=DEFAULT_AUDIT_FILE)
    parser.add_argument("--max-token-count", type=int, default=2048)
    parser.add_argument("--token-count-column", default="raw_prompt_bpe_token_count_full")
    parser.add_argument("--truncated-column", default="raw_prompt_bpe_truncated")
    parser.add_argument(
        "--drop-tokenizer-truncated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also remove rows already marked truncated by the raw-prompt BPE tokenizer.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def count_by(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    return (
        frame[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="row_count")
        .sort_values(column)
        .to_dict("records")
    )


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = resolve_path(args.input_file)
    output_file = resolve_path(args.output_file)
    audit_file = resolve_path(args.audit_file)

    frame = pd.read_parquet(input_file)
    if args.token_count_column not in frame.columns:
        raise ValueError(f"Missing token count column: {args.token_count_column!r}")

    keep_mask = frame[args.token_count_column].astype(int) <= args.max_token_count
    if args.drop_tokenizer_truncated:
        if args.truncated_column not in frame.columns:
            raise ValueError(f"Missing truncated column: {args.truncated_column!r}")
        keep_mask &= ~frame[args.truncated_column].astype(bool)

    kept = frame[keep_mask].copy()
    removed = frame[~keep_mask].copy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(output_file, index=False)

    audit = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "max_token_count": args.max_token_count,
        "token_count_column": args.token_count_column,
        "truncated_column": args.truncated_column,
        "drop_tokenizer_truncated": args.drop_tokenizer_truncated,
        "input_rows": int(len(frame)),
        "output_rows": int(len(kept)),
        "removed_rows": int(len(removed)),
        "max_token_count_before": int(frame[args.token_count_column].max()),
        "max_token_count_after": int(kept[args.token_count_column].max()) if len(kept) else None,
        "split_counts_before": count_by(frame, "split"),
        "split_counts_after": count_by(kept, "split"),
        "split_counts_removed": count_by(removed, "split"),
        "label_counts_before": count_by(frame, "label"),
        "label_counts_after": count_by(kept, "label"),
        "label_counts_removed": count_by(removed, "label"),
        "num_constraints_removed": count_by(removed, "num_constraints"),
    }
    write_json(audit, audit_file)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
