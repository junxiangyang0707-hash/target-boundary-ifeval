from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

DEFAULT_PROMPTSET_FILE = (
    "data/promptsets/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.all.parquet"
)
DEFAULT_SPLIT_FILE = (
    "data/splits/qwen3_4b_instruct_2507_under2048_nontruncated/group_key_seed42.parquet"
)
DEFAULT_TOKENIZER_DIR = "data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_group_key_train"
DEFAULT_OUTPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "group_key_seed42.raw_prompt_byte_bpe_v8k.parquet"
)
DEFAULT_AUDIT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "group_key_seed42.raw_prompt_byte_bpe_v8k.audit.json"
)

BASE_COLUMNS = [
    "prompt_id",
    "base_key",
    "user_prompt",
    "checker_pass",
    "strict_pass",
    "num_constraints",
    "instruction_ids",
    "constraint_signature",
    "constraint_family_signature",
    "source_dataset",
    "constraint_type",
    "cluster",
    "length_bin",
    "prompt_tokens",
    "output_tokens",
    "output_truncated",
]
SPLIT_COLUMNS = ["prompt_id", "split", "split_name", "split_type", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize the under2048 nontruncated promptset with the raw-prompt byte-level "
            "BPE tokenizer and attach group_key split labels."
        )
    )
    parser.add_argument("--promptset-file", default=DEFAULT_PROMPTSET_FILE)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--audit-file", default=DEFAULT_AUDIT_FILE)
    parser.add_argument("--text-column", default="user_prompt")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-truncate", action="store_true")
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compression", default="zstd")
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


def percentile(values: list[int], q: float) -> float:
    return round(float(np.percentile(values, q)), 2)


def summarize_counts(values: list[int]) -> dict[str, Any]:
    counts = np.array(values, dtype=np.int64)
    return {
        "row_count": int(counts.size),
        "min": int(counts.min()),
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": int(counts.max()),
        "mean": round(float(counts.mean()), 4),
    }


def load_rows(promptset_file: Path, split_file: Path, text_column: str) -> pd.DataFrame:
    prompt_columns = BASE_COLUMNS.copy()
    if text_column != "user_prompt":
        prompt_columns[prompt_columns.index("user_prompt")] = text_column
    promptset = pd.read_parquet(promptset_file, columns=prompt_columns)
    splits = pd.read_parquet(split_file, columns=SPLIT_COLUMNS)
    if promptset["prompt_id"].duplicated().any():
        raise ValueError(f"{promptset_file} has duplicate prompt_id values.")
    if splits["prompt_id"].duplicated().any():
        raise ValueError(f"{split_file} has duplicate prompt_id values.")

    merged = promptset.merge(splits, on="prompt_id", how="inner", validate="one_to_one")
    if len(merged) != len(promptset):
        raise ValueError(
            f"Split coverage mismatch: promptset rows={len(promptset)}, merged rows={len(merged)}."
        )
    return merged


def normalize_instruction_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_schema(*, include_text: bool) -> pa.Schema:
    fields = [
        pa.field("prompt_id", pa.string()),
        pa.field("base_key", pa.string()),
        pa.field("split", pa.string()),
        pa.field("split_name", pa.string()),
        pa.field("split_type", pa.string()),
        pa.field("seed", pa.int32()),
        pa.field("label", pa.int8()),
        pa.field("checker_pass", pa.bool_()),
        pa.field("strict_pass", pa.bool_()),
        pa.field("num_constraints", pa.int16()),
        pa.field("instruction_ids", pa.list_(pa.string())),
        pa.field("constraint_signature", pa.string()),
        pa.field("constraint_family_signature", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("constraint_type", pa.string()),
        pa.field("cluster", pa.int16()),
        pa.field("length_bin", pa.string()),
        pa.field("target_prompt_tokens", pa.int32()),
        pa.field("target_output_tokens", pa.int32()),
        pa.field("target_output_truncated", pa.bool_()),
        pa.field("raw_prompt_char_count", pa.int32()),
        pa.field("raw_prompt_bpe_token_count_full", pa.int32()),
        pa.field("raw_prompt_bpe_token_count", pa.int32()),
        pa.field("raw_prompt_bpe_truncated", pa.bool_()),
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("attention_mask", pa.list_(pa.int8())),
    ]
    if include_text:
        fields.insert(2, pa.field("user_prompt", pa.string()))
    return pa.schema(fields)


def encode_batch(
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
    *,
    add_special_tokens: bool,
    truncate: bool,
    max_length: int,
) -> tuple[list[list[int]], list[int], list[int], list[bool]]:
    full_encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
        verbose=False,
    )
    full_lengths = [len(ids) for ids in full_encoded["input_ids"]]
    if truncate:
        encoded = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=True,
            max_length=max_length,
            verbose=False,
        )
        input_ids = [[int(token_id) for token_id in ids] for ids in encoded["input_ids"]]
    else:
        input_ids = [[int(token_id) for token_id in ids] for ids in full_encoded["input_ids"]]
    token_counts = [len(ids) for ids in input_ids]
    truncated = [full > kept for full, kept in zip(full_lengths, token_counts, strict=True)]
    return input_ids, full_lengths, token_counts, truncated


def batch_to_table(
    batch: pd.DataFrame,
    input_ids: list[list[int]],
    full_counts: list[int],
    token_counts: list[int],
    truncated: list[bool],
    *,
    text_column: str,
    include_text: bool,
) -> pa.Table:
    output: dict[str, Any] = {
        "prompt_id": batch["prompt_id"].astype(str).tolist(),
        "base_key": batch["base_key"].astype(str).tolist(),
        "split": batch["split"].astype(str).tolist(),
        "split_name": batch["split_name"].astype(str).tolist(),
        "split_type": batch["split_type"].astype(str).tolist(),
        "seed": batch["seed"].astype("int32").tolist(),
        "label": batch["checker_pass"].astype("int8").tolist(),
        "checker_pass": batch["checker_pass"].astype(bool).tolist(),
        "strict_pass": batch["strict_pass"].astype(bool).tolist(),
        "num_constraints": batch["num_constraints"].astype("int16").tolist(),
        "instruction_ids": batch["instruction_ids"].map(normalize_instruction_ids).tolist(),
        "constraint_signature": batch["constraint_signature"].astype(str).tolist(),
        "constraint_family_signature": batch["constraint_family_signature"].astype(str).tolist(),
        "source_dataset": batch["source_dataset"].astype(str).tolist(),
        "constraint_type": batch["constraint_type"].astype(str).tolist(),
        "cluster": batch["cluster"].astype("int16").tolist(),
        "length_bin": batch["length_bin"].astype(str).tolist(),
        "target_prompt_tokens": batch["prompt_tokens"].astype("int32").tolist(),
        "target_output_tokens": batch["output_tokens"].astype("int32").tolist(),
        "target_output_truncated": batch["output_truncated"].astype(bool).tolist(),
        "raw_prompt_char_count": batch[text_column].astype(str).str.len().astype("int32").tolist(),
        "raw_prompt_bpe_token_count_full": full_counts,
        "raw_prompt_bpe_token_count": token_counts,
        "raw_prompt_bpe_truncated": truncated,
        "input_ids": input_ids,
        "attention_mask": [[1] * len(ids) for ids in input_ids],
    }
    if include_text:
        output["user_prompt"] = batch[text_column].astype(str).tolist()
    return pa.table(output, schema=build_schema(include_text=include_text))


def counts_by_split(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return (
        frame["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="row_count")
        .sort_values("split")
        .to_dict("records")
    )


def main() -> None:
    args = parse_args()
    total_start = now()
    promptset_file = resolve_path(args.promptset_file)
    split_file = resolve_path(args.split_file)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    output_file = resolve_path(args.output_file)
    audit_file = resolve_path(args.audit_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    data_start = now()
    data = load_rows(promptset_file, split_file, args.text_column)
    if args.limit is not None:
        data = data.head(args.limit).copy()
    data_seconds = elapsed_since(data_start)

    tokenizer_start = now()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    tokenizer_seconds = elapsed_since(tokenizer_start)

    schema = build_schema(include_text=args.include_text).with_metadata(
        {
            b"tokenizer_dir": str(tokenizer_dir).encode("utf-8"),
            b"input_view": b"raw_prompt_only",
            b"source_promptset": str(promptset_file).encode("utf-8"),
            b"split_file": str(split_file).encode("utf-8"),
            b"max_length": str(args.max_length).encode("utf-8"),
            b"truncate": str(not args.no_truncate).lower().encode("utf-8"),
            b"add_special_tokens": str(args.add_special_tokens).lower().encode("utf-8"),
        }
    )

    full_counts: list[int] = []
    token_counts: list[int] = []
    truncated_flags: list[bool] = []
    tokenize_start = now()

    with pq.ParquetWriter(
        output_file,
        schema=schema,
        compression=args.compression,
        use_dictionary=["prompt_id", "base_key", "split", "split_name", "split_type"],
    ) as writer:
        for start in range(0, len(data), args.batch_size):
            batch = data.iloc[start : start + args.batch_size].copy()
            texts = batch[args.text_column].astype(str).tolist()
            input_ids, batch_full_counts, batch_token_counts, batch_truncated = encode_batch(
                tokenizer,
                texts,
                add_special_tokens=args.add_special_tokens,
                truncate=not args.no_truncate,
                max_length=args.max_length,
            )
            table = batch_to_table(
                batch,
                input_ids,
                batch_full_counts,
                batch_token_counts,
                batch_truncated,
                text_column=args.text_column,
                include_text=args.include_text,
            )
            writer.write_table(table)
            full_counts.extend(batch_full_counts)
            token_counts.extend(batch_token_counts)
            truncated_flags.extend(batch_truncated)

            completed = min(start + args.batch_size, len(data))
            if completed % 10000 < args.batch_size or completed == len(data):
                print(f"tokenized {completed}/{len(data)} rows", flush=True)

    tokenize_seconds = elapsed_since(tokenize_start)
    output_meta = pq.read_metadata(output_file)
    split_counts = counts_by_split(data)
    label_counts = (
        data["checker_pass"]
        .astype("int8")
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="row_count")
        .sort_values("label")
        .to_dict("records")
    )
    truncated_by_split = []
    for split_name in sorted(data["split"].astype(str).unique()):
        mask = data["split"].astype(str).to_numpy() == split_name
        split_truncated = np.array(truncated_flags, dtype=bool)[mask]
        truncated_by_split.append(
            {
                "split": split_name,
                "row_count": int(mask.sum()),
                "raw_prompt_bpe_truncated_count": int(split_truncated.sum()),
                "raw_prompt_bpe_truncated_rate": round(float(split_truncated.mean()), 8),
            }
        )

    audit = {
        "promptset_file": str(promptset_file),
        "split_file": str(split_file),
        "tokenizer_dir": str(tokenizer_dir),
        "output_file": str(output_file),
        "row_count": int(len(data)),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "input_view": "raw_prompt_only",
        "text_column": args.text_column,
        "max_length": args.max_length,
        "truncate": not args.no_truncate,
        "add_special_tokens": args.add_special_tokens,
        "include_text": args.include_text,
        "compression": args.compression,
        "output_metadata": {
            "num_rows": int(output_meta.num_rows),
            "num_row_groups": int(output_meta.num_row_groups),
        },
        "raw_prompt_bpe_token_count_full": summarize_counts(full_counts),
        "raw_prompt_bpe_token_count": summarize_counts(token_counts),
        "raw_prompt_bpe_truncated_count": int(np.sum(truncated_flags)),
        "raw_prompt_bpe_truncated_rate": round(float(np.mean(truncated_flags)), 8),
        "raw_prompt_bpe_truncated_by_split": truncated_by_split,
        "timing_seconds": {
            "load_data": data_seconds,
            "load_tokenizer": tokenizer_seconds,
            "tokenize_and_write": tokenize_seconds,
            "total": elapsed_since(total_start),
        },
    }
    audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
