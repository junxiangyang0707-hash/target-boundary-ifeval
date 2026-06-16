from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable vLLM batch generation for normalized prompts whose tokenizer length "
            "is below a configured input limit."
        )
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument(
        "--tokens-file",
        default=(
            "data/promptsets/"
            "if_multi_constraints_upto5.qwen3_4b_instruct_2507.tokens.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/generations/qwen3_4b_instruct_2507_under2048_0_10000_max2048",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10000)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b-instruct-2507"))
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-combined", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def json_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def wait_for_vllm(base_url: str, context_limit: int, timeout_seconds: int) -> dict[str, Any]:
    start_time = now()
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/models"
    last_error = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                models = response.json().get("data", [])
                max_model_len = models[0].get("max_model_len") if models else None
                if max_model_len != context_limit:
                    raise RuntimeError(
                        f"vLLM max_model_len={max_model_len}, expected {context_limit}"
                    )
                return {
                    "attempts": attempts,
                    "seconds": elapsed_since(start_time),
                    "models": models,
                    "max_model_len": max_model_len,
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def load_selected_rows(args: argparse.Namespace) -> pd.DataFrame:
    promptset_file = resolve_path(args.promptset_file)
    tokens_file = resolve_path(args.tokens_file)

    prompt_columns = [
        "prompt_id",
        "base_key",
        "raw_messages",
        "user_prompt",
        "constraint_text",
        "ground_truth_spec",
        "instruction_ids",
        "constraint_signature",
        "constraint_family_signature",
        "num_constraints",
        "source_dataset",
        "constraint_type",
    ]
    promptset = pd.read_parquet(promptset_file, columns=prompt_columns)
    tokens = pd.read_parquet(tokens_file, columns=["prompt_id", "prompt_tokens"])
    merged = promptset.merge(tokens, on="prompt_id", how="left", validate="one_to_one", sort=False)

    missing_tokens = int(merged["prompt_tokens"].isna().sum())
    if missing_tokens:
        raise ValueError(f"{missing_tokens} prompt rows are missing token counts")

    eligible = merged[merged["prompt_tokens"] < args.max_input_tokens].copy().reset_index(drop=True)
    eligible.insert(0, "under2048_index", np.arange(len(eligible), dtype=np.int64))

    if args.start_index < 0 or args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index, both non-negative")
    if args.end_index > len(eligible):
        raise ValueError(
            f"Requested end-index {args.end_index}, but only {len(eligible)} rows "
            f"have prompt_tokens < {args.max_input_tokens}"
        )

    selected = eligible.iloc[args.start_index : args.end_index].copy().reset_index(drop=True)
    selected.insert(1, "batch_position", np.arange(len(selected), dtype=np.int64))
    return selected


def is_context_window_error(error: str) -> bool:
    lowered = str(error or "").lower()
    patterns = [
        "maximum context",
        "context length",
        "max model len",
        "maximum model length",
        "tokens exceed",
        "too long",
    ]
    return any(pattern in lowered for pattern in patterns)


def call_vllm(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        start_time = now()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                timeout=timeout_seconds,
            )
            seconds = elapsed_since(start_time)
            if response.status_code >= 400:
                last_result = {
                    "ok": False,
                    "seconds": seconds,
                    "status_code": response.status_code,
                    "error": response.text[:2000],
                    "attempts": attempt + 1,
                }
                if response.status_code < 500:
                    break
            else:
                data = response.json()
                usage = data.get("usage") or {}
                choice = (data.get("choices") or [{}])[0]
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                return {
                    "ok": True,
                    "seconds": seconds,
                    "status_code": response.status_code,
                    "error": "",
                    "attempts": attempt + 1,
                    "finish_reason": choice.get("finish_reason") or "",
                    "response_text": choice.get("message", {}).get("content") or "",
                    "vllm_input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "completion_tokens_per_second": (
                        round(completion_tokens / seconds, 4)
                        if isinstance(completion_tokens, int) and seconds > 0
                        else None
                    ),
                    "total_tokens_per_second": (
                        round(total_tokens / seconds, 4)
                        if isinstance(total_tokens, int) and seconds > 0
                        else None
                    ),
                }
        except requests.RequestException as exc:
            last_result = {
                "ok": False,
                "seconds": elapsed_since(start_time),
                "status_code": None,
                "error": str(exc)[:2000],
                "attempts": attempt + 1,
            }

        if attempt < max_retries:
            time.sleep(min(2**attempt, 8))

    return last_result or {
        "ok": False,
        "seconds": 0.0,
        "status_code": None,
        "error": "request failed without result",
        "attempts": max_retries + 1,
    }


def build_output_record(
    row: Any,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_tokens = result.get("output_tokens") if result.get("ok") else None
    finish_reason = result.get("finish_reason") or ""
    output_hit_max_tokens = isinstance(output_tokens, int) and output_tokens >= args.max_tokens
    output_truncated = bool(finish_reason == "length" or output_hit_max_tokens)
    error = str(result.get("error") or "")
    prompt_tokens = int(row.prompt_tokens)

    return {
        "under2048_index": int(row.under2048_index),
        "batch_position": int(row.batch_position),
        "prompt_id": str(row.prompt_id),
        "base_key": str(row.base_key),
        "raw_messages_json": json_string(row.raw_messages),
        "user_prompt": str(row.user_prompt),
        "constraint_text": str(row.constraint_text),
        "ground_truth_spec": str(row.ground_truth_spec),
        "instruction_ids_json": json_string(row.instruction_ids),
        "constraint_signature": str(row.constraint_signature),
        "constraint_family_signature": str(row.constraint_family_signature),
        "num_constraints": int(row.num_constraints),
        "source_dataset": str(row.source_dataset),
        "constraint_type": str(row.constraint_type),
        "prompt_tokens": prompt_tokens,
        "context_limit": args.context_limit,
        "max_output_tokens": args.max_tokens,
        "prompt_plus_max_output_tokens": prompt_tokens + args.max_tokens,
        "within_context_budget": bool(prompt_tokens + args.max_tokens <= args.context_limit),
        "model": args.model,
        "temperature": args.temperature,
        "response_text": str(result.get("response_text") or ""),
        "ok": bool(result.get("ok")),
        "finish_reason": finish_reason,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, int) else None,
        "vllm_input_tokens": (
            int(result["vllm_input_tokens"])
            if isinstance(result.get("vllm_input_tokens"), int)
            else None
        ),
        "total_tokens": (
            int(result["total_tokens"]) if isinstance(result.get("total_tokens"), int) else None
        ),
        "output_hit_max_tokens": output_hit_max_tokens,
        "output_truncated": output_truncated,
        "output_limit_status": "truncated" if output_truncated else "completed",
        "context_window_error": bool((not result.get("ok")) and is_context_window_error(error)),
        "seconds": float(result.get("seconds") or 0.0),
        "completion_tokens_per_second": result.get("completion_tokens_per_second"),
        "total_tokens_per_second": result.get("total_tokens_per_second"),
        "status_code": result.get("status_code"),
        "attempts": int(result.get("attempts") or 0),
        "error": error[:1000],
    }


def shard_path(shards_dir: Path, start_index: int, end_index: int) -> Path:
    return shards_dir / f"part-{start_index:06d}-{end_index:06d}.parquet"


def valid_existing_shard(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        return pq.ParquetFile(path).metadata.num_rows == expected_rows
    except Exception:
        return False


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    successes = frame[frame["ok"]].copy()
    completion_tokens = successes["output_tokens"].dropna().astype(int)
    total_tokens = successes["total_tokens"].dropna().astype(int)
    seconds = successes["seconds"].dropna().astype(float)
    return {
        "row_count": int(len(frame)),
        "success_count": int(frame["ok"].sum()),
        "failure_count": int((~frame["ok"]).sum()),
        "context_window_error_count": int(frame["context_window_error"].sum()),
        "output_truncated_count": int(frame["output_truncated"].sum()),
        "finish_reason_counts": {
            str(key): int(value)
            for key, value in successes["finish_reason"].fillna("<missing>").value_counts().items()
        },
        "aggregate_completion_tokens": int(completion_tokens.sum()),
        "aggregate_total_tokens": int(total_tokens.sum()),
        "request_seconds_p50": (
            round(float(np.percentile(seconds, 50)), 4) if len(seconds) else None
        ),
        "request_seconds_max": round(float(seconds.max()), 4) if len(seconds) else None,
    }


def summarize_shard_file(path: Path) -> dict[str, Any]:
    columns = [
        "ok",
        "context_window_error",
        "output_truncated",
        "finish_reason",
        "output_tokens",
        "total_tokens",
        "seconds",
    ]
    return summarize_frame(pd.read_parquet(path, columns=columns))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path, compression: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False, compression=compression)
    tmp_path.replace(path)


def write_combined(shard_files: list[Path], output_file: Path, compression: str) -> None:
    pieces = [pd.read_parquet(path) for path in shard_files]
    combined = pd.concat(pieces, ignore_index=True).sort_values(
        ["under2048_index", "prompt_id"], kind="mergesort"
    )
    write_parquet_atomic(combined, output_file, compression=compression)


def run_shard(shard: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_row = {
            executor.submit(
                call_vllm,
                base_url=args.base_url,
                model=args.model,
                prompt=str(row.user_prompt),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            ): row
            for row in shard.itertuples(index=False)
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            results.append(build_output_record(row, future.result(), args))

    return pd.DataFrame(results).sort_values(["under2048_index", "prompt_id"], kind="mergesort")


def main() -> None:
    total_start = now()
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    shards_dir = output_dir / "shards"
    manifest_file = output_dir / "manifest.json"
    combined_file = output_dir / "outputs.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    wait_info = wait_for_vllm(args.base_url, args.context_limit, args.timeout_seconds)
    selected = load_selected_rows(args)
    total_rows = int(len(selected))
    shard_summaries: list[dict[str, Any]] = []
    completed_shard_files: list[Path] = []

    initial_manifest = {
        "status": "running",
        "output_dir": str(output_dir),
        "shards_dir": str(shards_dir),
        "combined_file": str(combined_file),
        "args": vars(args),
        "vllm": wait_info,
        "selected_rows": total_rows,
        "started_at_unix": time.time(),
    }
    write_json_atomic(manifest_file, initial_manifest)
    print(
        json.dumps(
            {"phase": "start", "selected_rows": total_rows, "vllm": wait_info},
            ensure_ascii=False,
        )
    )

    for offset in range(0, total_rows, args.shard_size):
        shard = selected.iloc[offset : offset + args.shard_size].copy()
        shard_start = args.start_index + offset
        shard_end = shard_start + len(shard)
        path = shard_path(shards_dir, shard_start, shard_end)
        shard_started = now()

        if args.resume and valid_existing_shard(path, len(shard)):
            summary = summarize_shard_file(path)
            status = "skipped_existing"
        else:
            frame = run_shard(shard, args)
            write_parquet_atomic(frame, path, compression=args.compression)
            summary = summarize_frame(frame)
            status = "completed"

        shard_summary = {
            "status": status,
            "path": str(path),
            "range": {"start": shard_start, "end": shard_end},
            "seconds": elapsed_since(shard_started),
            **summary,
        }
        shard_summaries.append(shard_summary)
        completed_shard_files.append(path)

        progress = {
            **initial_manifest,
            "status": "running",
            "elapsed_seconds": elapsed_since(total_start),
            "completed_rows": int(sum(item["row_count"] for item in shard_summaries)),
            "completed_shards": len(shard_summaries),
            "latest_shard": shard_summary,
            "shards": shard_summaries,
        }
        write_json_atomic(manifest_file, progress)
        print(json.dumps({"phase": "shard", **shard_summary}, ensure_ascii=False), flush=True)

    if args.write_combined:
        combine_start = now()
        write_combined(completed_shard_files, combined_file, compression=args.compression)
        combine_seconds = elapsed_since(combine_start)
    else:
        combine_seconds = None

    shard_frames = [
        pd.read_parquet(
            path,
            columns=[
                "ok",
                "context_window_error",
                "output_truncated",
                "finish_reason",
                "output_tokens",
                "total_tokens",
                "seconds",
            ],
        )
        for path in completed_shard_files
    ]
    aggregate = summarize_frame(pd.concat(shard_frames, ignore_index=True))
    elapsed = elapsed_since(total_start)
    aggregate["elapsed_seconds"] = elapsed
    aggregate["aggregate_completion_tokens_per_second"] = (
        round(aggregate["aggregate_completion_tokens"] / elapsed, 4) if elapsed > 0 else None
    )
    aggregate["aggregate_total_tokens_per_second"] = (
        round(aggregate["aggregate_total_tokens"] / elapsed, 4) if elapsed > 0 else None
    )

    final_manifest = {
        **initial_manifest,
        "status": "complete",
        "completed_at_unix": time.time(),
        "elapsed_seconds": elapsed,
        "combine_seconds": combine_seconds,
        "aggregate": aggregate,
        "shards": shard_summaries,
    }
    write_json_atomic(manifest_file, final_manifest)
    print(
        json.dumps({"phase": "complete", **final_manifest}, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
