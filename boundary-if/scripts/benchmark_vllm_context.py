from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure prompt token length distribution and benchmark vLLM with length-stratified "
            "requests. Does not persist sampled prompts or responses."
        )
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b-instruct-2507"))
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def percentile(values: pd.Series | list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2)


def load_promptset(path: Path) -> pd.DataFrame:
    columns = [
        "prompt_id",
        "user_prompt",
        "num_constraints",
        "constraint_family_signature",
        "constraint_signature",
    ]
    return pd.read_parquet(path, columns=columns)


def load_tokenizer(tokenizer_model: str) -> Any:
    return AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        local_files_only=True,
    )


def count_chat_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
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
    return len(token_ids)


def add_prompt_lengths(df: pd.DataFrame, tokenizer: Any) -> tuple[pd.DataFrame, float]:
    start_time = now()
    lengths: list[int] = []
    for prompt in df["user_prompt"].astype(str):
        lengths.append(count_chat_prompt_tokens(tokenizer, prompt))
    output = df.copy()
    output["prompt_tokens"] = lengths
    return output, elapsed_since(start_time)


def summarize_lengths(df: pd.DataFrame) -> dict[str, Any]:
    lengths = df["prompt_tokens"]
    return {
        "row_count": int(len(df)),
        "distinct_prompt_token_lengths": int(lengths.nunique()),
        "min": int(lengths.min()),
        "p01": percentile(lengths, 1),
        "p05": percentile(lengths, 5),
        "p10": percentile(lengths, 10),
        "p25": percentile(lengths, 25),
        "p50": percentile(lengths, 50),
        "p75": percentile(lengths, 75),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
    }


def select_length_stratified_rows(df: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    by_length = (
        df.sort_values(["prompt_tokens", "prompt_id"], kind="mergesort")
        .groupby("prompt_tokens", as_index=False)
        .first()
        .sort_values("prompt_tokens", kind="mergesort")
        .reset_index(drop=True)
    )
    if sample_count >= len(by_length):
        return by_length.copy()

    positions = np.linspace(0, len(by_length) - 1, sample_count)
    selected_positions = sorted({int(round(position)) for position in positions})
    cursor = 0
    while len(selected_positions) < sample_count and cursor < len(by_length):
        if cursor not in selected_positions:
            selected_positions.append(cursor)
        cursor += 1
    selected_positions = sorted(selected_positions[:sample_count])
    return by_length.iloc[selected_positions].copy().reset_index(drop=True)


def wait_for_vllm(base_url: str, timeout_seconds: int) -> dict[str, Any]:
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
                data = response.json()
                return {
                    "attempts": attempts,
                    "seconds": elapsed_since(start_time),
                    "models": data.get("data", []),
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def resolve_model_max_len(models_payload: list[dict[str, Any]], model_name: str) -> int | None:
    for model in models_payload:
        if model.get("id") == model_name:
            value = model.get("max_model_len")
            return int(value) if isinstance(value, int) else None
    if models_payload:
        value = models_payload[0].get("max_model_len")
        return int(value) if isinstance(value, int) else None
    return None


def call_vllm(
    *,
    base_url: str,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    start_time = now()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": str(row["user_prompt"])}],
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
            return {
                "prompt_id": row["prompt_id"],
                "prompt_tokens_estimated": int(row["prompt_tokens"]),
                "ok": False,
                "seconds": seconds,
                "status_code": response.status_code,
                "error": response.text[:1000],
            }
        data = response.json()
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        return {
            "prompt_id": row["prompt_id"],
            "prompt_tokens_estimated": int(row["prompt_tokens"]),
            "ok": True,
            "seconds": seconds,
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
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
        return {
            "prompt_id": row["prompt_id"],
            "prompt_tokens_estimated": int(row["prompt_tokens"]),
            "ok": False,
            "seconds": elapsed_since(start_time),
            "status_code": None,
            "error": str(exc)[:1000],
        }


def is_context_window_error(result: dict[str, Any]) -> bool:
    error = str(result.get("error") or "").lower()
    patterns = [
        "maximum context",
        "context length",
        "max model len",
        "maximum model length",
        "tokens exceed",
        "too long",
    ]
    return any(pattern in error for pattern in patterns)


def summarize_results(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successes = [result for result in results if result.get("ok")]
    failures = [result for result in results if not result.get("ok")]
    completion_tokens = [
        result.get("usage", {}).get("completion_tokens")
        for result in successes
        if isinstance(result.get("usage", {}).get("completion_tokens"), int)
    ]
    total_tokens = [
        result.get("usage", {}).get("total_tokens")
        for result in successes
        if isinstance(result.get("usage", {}).get("total_tokens"), int)
    ]
    request_seconds = [float(result["seconds"]) for result in results]
    per_request_completion_tps = [
        float(result["completion_tokens_per_second"])
        for result in successes
        if isinstance(result.get("completion_tokens_per_second"), float)
    ]
    finish_reasons: dict[str, int] = {}
    for result in successes:
        reason = str(result.get("finish_reason"))
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    total_completion_tokens = int(sum(completion_tokens))
    total_all_tokens = int(sum(total_tokens))
    return {
        "request_count": len(results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "context_window_error_count": sum(
            1 for result in failures if is_context_window_error(result)
        ),
        "finish_reason_counts": finish_reasons,
        "wall_seconds": round(wall_seconds, 4),
        "aggregate_completion_tokens": total_completion_tokens,
        "aggregate_total_tokens": total_all_tokens,
        "aggregate_completion_tokens_per_second": (
            round(total_completion_tokens / wall_seconds, 4) if wall_seconds > 0 else None
        ),
        "aggregate_total_tokens_per_second": (
            round(total_all_tokens / wall_seconds, 4) if wall_seconds > 0 else None
        ),
        "request_latency_seconds": {
            "min": round(min(request_seconds), 4) if request_seconds else None,
            "p50": round(median(request_seconds), 4) if request_seconds else None,
            "mean": round(mean(request_seconds), 4) if request_seconds else None,
            "p90": percentile(request_seconds, 90) if request_seconds else None,
            "max": round(max(request_seconds), 4) if request_seconds else None,
        },
        "per_request_completion_tokens_per_second": {
            "p50": round(median(per_request_completion_tps), 4)
            if per_request_completion_tps
            else None,
            "mean": round(mean(per_request_completion_tps), 4)
            if per_request_completion_tps
            else None,
            "p90": percentile(per_request_completion_tps, 90)
            if per_request_completion_tps
            else None,
        },
        "failure_examples": failures[:5],
    }


def main() -> None:
    total_start = now()
    args = parse_args()

    promptset = load_promptset(resolve_path(args.promptset_file))
    tokenizer = load_tokenizer(args.tokenizer_model)
    promptset_with_lengths, tokenization_seconds = add_prompt_lengths(promptset, tokenizer)
    length_summary = summarize_lengths(promptset_with_lengths)
    selected = select_length_stratified_rows(promptset_with_lengths, args.requests)
    selected_summary = summarize_lengths(selected)

    output: dict[str, Any] = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "tokenizer_model": args.tokenizer_model,
            "requested_count": args.requests,
            "actual_selected_count": int(len(selected)),
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "timing_seconds": {
            "tokenize_all_prompts": tokenization_seconds,
        },
        "all_prompt_token_distribution": length_summary,
        "selected_prompt_token_distribution": selected_summary,
        "selected_extremes": {
            "shortest": {
                "prompt_id": selected.loc[selected["prompt_tokens"].idxmin(), "prompt_id"],
                "prompt_tokens": int(selected["prompt_tokens"].min()),
            },
            "longest": {
                "prompt_id": selected.loc[selected["prompt_tokens"].idxmax(), "prompt_id"],
                "prompt_tokens": int(selected["prompt_tokens"].max()),
            },
        },
    }

    if args.distribution_only:
        output["timing_seconds"]["total_script"] = elapsed_since(total_start)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    wait_timing = wait_for_vllm(args.base_url, args.timeout_seconds)
    model_max_len = resolve_model_max_len(wait_timing["models"], args.model)
    output["vllm"] = {
        "wait_for_ready_seconds": wait_timing["seconds"],
        "model_max_len": model_max_len,
    }
    if model_max_len is not None:
        selected = selected.copy()
        selected["requested_total_tokens_estimate"] = selected["prompt_tokens"] + args.max_tokens
        output["context_window_preflight"] = {
            "max_model_len": model_max_len,
            "requested_completion_tokens": args.max_tokens,
            "would_exceed_count": int(
                (selected["requested_total_tokens_estimate"] > model_max_len).sum()
            ),
            "max_requested_total_tokens_estimate": int(
                selected["requested_total_tokens_estimate"].max()
            ),
        }

    rows = selected.to_dict(orient="records")
    start_benchmark = now()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                call_vllm,
                base_url=args.base_url,
                model=args.model,
                row=row,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
            )
            for row in rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(
                    f"completed {index}/{len(futures)} requests in "
                    f"{elapsed_since(start_benchmark)}s",
                    file=sys.stderr,
                    flush=True,
                )

    benchmark_wall_seconds = elapsed_since(start_benchmark)
    output["benchmark"] = summarize_results(results, benchmark_wall_seconds)
    output["timing_seconds"]["benchmark_wall"] = benchmark_wall_seconds
    output["timing_seconds"]["total_script"] = elapsed_since(total_start)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
