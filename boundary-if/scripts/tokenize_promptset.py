from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a normalized promptset with a target chat model tokenizer and save token "
            "ids plus counts to Parquet."
        )
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument(
        "--output-file",
        default=(
            "data/promptsets/"
            "if_multi_constraints_upto5.qwen3_4b_instruct_2507.tokens.parquet"
        ),
    )
    parser.add_argument(
        "--audit-file",
        default=(
            "data/promptsets/"
            "if_multi_constraints_upto5.qwen3_4b_instruct_2507.tokens.audit.json"
        ),
    )
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--threshold", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def load_tokenizer(tokenizer_model: str) -> Any:
    return AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        local_files_only=True,
    )


def chat_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        token_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        token_ids = encoded.input_ids
    else:
        token_ids = encoded
    return [int(token_id) for token_id in token_ids]


def percentile(values: list[int], q: float) -> float:
    return round(float(np.percentile(values, q)), 2)


def write_batch(
    writer: pq.ParquetWriter,
    prompt_ids: list[str],
    token_counts: list[int],
    input_ids: list[list[int]],
) -> None:
    table = pa.table(
        {
            "prompt_id": pa.array(prompt_ids, type=pa.string()),
            "prompt_tokens": pa.array(token_counts, type=pa.int32()),
            "input_ids": pa.array(input_ids, type=pa.list_(pa.int32())),
        }
    )
    writer.write_table(table)


def summarize_counts(token_counts: list[int], threshold: int) -> dict[str, Any]:
    counts_array = np.array(token_counts, dtype=np.int64)
    below_threshold = int(np.sum(counts_array < threshold))
    return {
        "row_count": int(counts_array.size),
        "min": int(counts_array.min()),
        "p01": percentile(token_counts, 1),
        "p05": percentile(token_counts, 5),
        "p10": percentile(token_counts, 10),
        "p25": percentile(token_counts, 25),
        "p50": percentile(token_counts, 50),
        "p75": percentile(token_counts, 75),
        "p90": percentile(token_counts, 90),
        "p95": percentile(token_counts, 95),
        "p99": percentile(token_counts, 99),
        "max": int(counts_array.max()),
        "mean": round(float(counts_array.mean()), 2),
        "threshold": threshold,
        "lt_threshold_count": below_threshold,
        "lt_threshold_share": round(below_threshold / int(counts_array.size), 6),
        "ge_threshold_count": int(np.sum(counts_array >= threshold)),
    }


def main() -> None:
    args = parse_args()
    total_start = now()
    promptset_file = resolve_path(args.promptset_file)
    output_file = resolve_path(args.output_file)
    audit_file = resolve_path(args.audit_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_start = now()
    tokenizer = load_tokenizer(args.tokenizer_model)
    tokenizer_seconds = elapsed_since(tokenizer_start)

    data_start = now()
    promptset = pd.read_parquet(promptset_file, columns=["prompt_id", "user_prompt"])
    data_seconds = elapsed_since(data_start)

    schema = pa.schema(
        [
            pa.field("prompt_id", pa.string()),
            pa.field("prompt_tokens", pa.int32()),
            pa.field("input_ids", pa.list_(pa.int32())),
        ],
        metadata={
            b"tokenizer_model": args.tokenizer_model.encode("utf-8"),
            b"chat_template_add_generation_prompt": b"true",
            b"chat_template_enable_thinking": b"false",
        },
    )

    token_counts: list[int] = []
    batch_prompt_ids: list[str] = []
    batch_counts: list[int] = []
    batch_input_ids: list[list[int]] = []
    tokenize_start = now()

    with pq.ParquetWriter(
        output_file,
        schema=schema,
        compression=args.compression,
        use_dictionary=["prompt_id"],
    ) as writer:
        for index, row in enumerate(promptset.itertuples(index=False), start=1):
            token_ids = chat_token_ids(tokenizer, str(row.user_prompt))
            token_count = len(token_ids)
            token_counts.append(token_count)
            batch_prompt_ids.append(str(row.prompt_id))
            batch_counts.append(token_count)
            batch_input_ids.append(token_ids)

            if len(batch_prompt_ids) >= args.batch_size:
                write_batch(writer, batch_prompt_ids, batch_counts, batch_input_ids)
                batch_prompt_ids.clear()
                batch_counts.clear()
                batch_input_ids.clear()

            if index % 10000 == 0:
                print(f"tokenized {index}/{len(promptset)} rows", flush=True)

        if batch_prompt_ids:
            write_batch(writer, batch_prompt_ids, batch_counts, batch_input_ids)

    tokenize_seconds = elapsed_since(tokenize_start)
    audit = {
        "promptset_file": str(promptset_file),
        "output_file": str(output_file),
        "tokenizer_model": args.tokenizer_model,
        "chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "compression": args.compression,
        "timing_seconds": {
            "load_tokenizer": tokenizer_seconds,
            "load_promptset": data_seconds,
            "tokenize_and_write": tokenize_seconds,
            "total": elapsed_since(total_start),
        },
        "token_counts": summarize_counts(token_counts, args.threshold),
    }
    audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
