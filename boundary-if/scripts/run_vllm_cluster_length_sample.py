from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample prompts by cluster and length, run vLLM concurrently, and save token metrics "
            "without prompt or response text."
        )
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument(
        "--clusters-file",
        default="data/reports/prompt_clusters_under2048_seed42.parquet",
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--output-length-threshold", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b-instruct-2507"))
    parser.add_argument(
        "--output-csv",
        default="data/reports/vllm_under2048_cluster_len_seed42.csv",
    )
    parser.add_argument(
        "--sample-file",
        default="data/reports/vllm_under2048_cluster_len_seed42.sample.parquet",
    )
    parser.add_argument(
        "--summary-json",
        default="data/reports/vllm_under2048_cluster_len_seed42.summary.json",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


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
                return {
                    "attempts": attempts,
                    "seconds": elapsed_since(start_time),
                    "models": response.json().get("data", []),
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def allocate_by_cluster(assignments: pd.DataFrame, sample_count: int) -> dict[int, int]:
    counts = assignments["cluster"].value_counts().sort_index()
    weights = np.sqrt(counts.to_numpy(dtype=np.float64))
    raw = weights / weights.sum() * sample_count
    allocations = np.floor(raw).astype(int)
    allocations = np.maximum(allocations, 1)

    while allocations.sum() > sample_count:
        candidates = np.where(allocations > 1)[0]
        if len(candidates) == 0:
            break
        reduce_index = candidates[np.argmax(allocations[candidates] - raw[candidates])]
        allocations[reduce_index] -= 1

    while allocations.sum() < sample_count:
        add_index = int(np.argmax(raw - allocations))
        allocations[add_index] += 1

    return {
        int(cluster): int(count)
        for cluster, count in zip(counts.index, allocations, strict=True)
    }


def sample_group_by_length(
    group: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sorted_group = group.sort_values(["prompt_tokens", "prompt_id"], kind="mergesort").reset_index(
        drop=True
    )
    if count >= len(sorted_group):
        return sorted_group.copy()
    chunks = np.array_split(np.arange(len(sorted_group)), count)
    selected_positions: list[int] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        selected_positions.append(int(rng.choice(chunk)))
    return sorted_group.iloc[sorted(selected_positions)].copy()


def select_cluster_length_sample(
    assignments: pd.DataFrame,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    allocations = allocate_by_cluster(assignments, sample_count)
    pieces = []
    for cluster_id, count in allocations.items():
        group = assignments[assignments["cluster"] == cluster_id]
        pieces.append(sample_group_by_length(group, count, rng))
    sample = pd.concat(pieces, ignore_index=True)

    extremes = assignments.loc[
        [assignments["prompt_tokens"].idxmin(), assignments["prompt_tokens"].idxmax()]
    ]
    sample = (
        pd.concat([sample, extremes], ignore_index=True)
        .drop_duplicates("prompt_id", keep="last")
        .sort_values(["cluster", "prompt_tokens", "prompt_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(sample) > sample_count:
        protected = set(extremes["prompt_id"])
        removable = sample[~sample["prompt_id"].isin(protected)].copy()
        remove_count = len(sample) - sample_count
        remove_positions = np.linspace(0, len(removable) - 1, remove_count, dtype=int)
        remove_ids = set(removable.iloc[remove_positions]["prompt_id"])
        sample = sample[~sample["prompt_id"].isin(remove_ids)].reset_index(drop=True)
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def call_vllm(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
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
            return {
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


def summarize(
    metrics: pd.DataFrame,
    args: argparse.Namespace,
    wall_seconds: float,
) -> dict[str, Any]:
    successes = metrics[metrics["ok"]].copy()
    completion_tokens = successes["output_tokens"].fillna(0).astype(int)
    total_tokens = successes["total_tokens"].fillna(0).astype(int)
    finish_reason_counts = successes["finish_reason"].fillna("<missing>").value_counts().to_dict()
    return {
        "sample_count": int(len(metrics)),
        "success_count": int(metrics["ok"].sum()),
        "failure_count": int((~metrics["ok"]).sum()),
        "context_window_error_count": int(metrics["context_window_error"].sum()),
        "output_gt_threshold_count": int(
            (successes["output_tokens"] > args.output_length_threshold).sum()
        ),
        "output_eq_max_tokens_count": int((successes["output_tokens"] == args.max_tokens).sum()),
        "finish_reason_length_count": int((successes["finish_reason"] == "length").sum()),
        "finish_reason_counts": {
            str(key): int(value) for key, value in finish_reason_counts.items()
        },
        "max_tokens": args.max_tokens,
        "output_length_threshold": args.output_length_threshold,
        "context_limit": args.context_limit,
        "concurrency": args.concurrency,
        "sample_seed": args.sample_seed,
        "wall_seconds": round(wall_seconds, 4),
        "aggregate_completion_tokens": int(completion_tokens.sum()),
        "aggregate_total_tokens": int(total_tokens.sum()),
        "aggregate_completion_tokens_per_second": (
            round(float(completion_tokens.sum()) / wall_seconds, 4) if wall_seconds > 0 else None
        ),
        "output_tokens_success_only": {
            "min": int(completion_tokens.min()) if len(completion_tokens) else None,
            "p50": round(float(np.percentile(completion_tokens, 50)), 2)
            if len(completion_tokens)
            else None,
            "p90": round(float(np.percentile(completion_tokens, 90)), 2)
            if len(completion_tokens)
            else None,
            "p95": round(float(np.percentile(completion_tokens, 95)), 2)
            if len(completion_tokens)
            else None,
            "max": int(completion_tokens.max()) if len(completion_tokens) else None,
        },
        "input_tokens": {
            "min": int(metrics["prompt_tokens"].min()),
            "p50": round(float(np.percentile(metrics["prompt_tokens"], 50)), 2),
            "p90": round(float(np.percentile(metrics["prompt_tokens"], 90)), 2),
            "max": int(metrics["prompt_tokens"].max()),
        },
        "cluster_coverage": {
            str(key): int(value)
            for key, value in metrics["cluster"].value_counts().sort_index().items()
        },
        "length_bin_coverage": {
            str(key): int(value)
            for key, value in metrics["length_bin"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    total_start = now()
    args = parse_args()
    clusters_file = resolve_path(args.clusters_file)
    promptset_file = resolve_path(args.promptset_file)
    output_csv = resolve_path(args.output_csv)
    sample_file = resolve_path(args.sample_file)
    summary_json = resolve_path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_parquet(clusters_file)
    sample = select_cluster_length_sample(assignments, args.sample_count, args.sample_seed)
    promptset = pd.read_parquet(promptset_file, columns=["prompt_id", "user_prompt"])
    sample_for_requests = sample.merge(
        promptset,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )
    sample.drop(columns=[], errors="ignore").to_parquet(sample_file, index=False)

    wait_info = wait_for_vllm(args.base_url, args.timeout_seconds)
    print(
        json.dumps(
            {
                "vllm_ready_seconds": wait_info["seconds"],
                "sample_count": int(len(sample_for_requests)),
                "concurrency": args.concurrency,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    start_requests = now()
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
            ): row
            for row in sample_for_requests.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(future_to_row), start=1):
            row = future_to_row[future]
            result = future.result()
            usage = result.get("usage") or {}
            output_tokens = usage.get("completion_tokens") if result.get("ok") else None
            input_tokens = usage.get("prompt_tokens") if result.get("ok") else None
            total_tokens = usage.get("total_tokens") if result.get("ok") else None
            context_error = (not result.get("ok")) and is_context_window_error(result)
            results.append(
                {
                    "prompt_id": row.prompt_id,
                    "cluster": int(row.cluster),
                    "length_bin": row.length_bin,
                    "prompt_tokens": int(row.prompt_tokens),
                    "vllm_input_tokens": int(input_tokens)
                    if isinstance(input_tokens, int)
                    else None,
                    "output_tokens": int(output_tokens)
                    if isinstance(output_tokens, int)
                    else None,
                    "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
                    "finish_reason": result.get("finish_reason") or "",
                    "ok": bool(result.get("ok")),
                    "context_window_error": bool(context_error),
                    "output_gt_threshold": bool(
                        isinstance(output_tokens, int)
                        and output_tokens > args.output_length_threshold
                    ),
                    "output_eq_max_tokens": bool(
                        isinstance(output_tokens, int) and output_tokens == args.max_tokens
                    ),
                    "seconds": float(result.get("seconds") or 0.0),
                    "completion_tokens_per_second": result.get("completion_tokens_per_second"),
                    "total_tokens_per_second": result.get("total_tokens_per_second"),
                    "status_code": result.get("status_code"),
                    "error": str(result.get("error") or "")[:500],
                    "sample_seed": args.sample_seed,
                }
            )
            if index % 10 == 0:
                print(
                    f"completed {index}/{len(sample_for_requests)} in "
                    f"{elapsed_since(start_requests)}s",
                    flush=True,
                )
    wall_seconds = elapsed_since(start_requests)
    metrics = pd.DataFrame(results).sort_values(
        ["cluster", "prompt_tokens", "prompt_id"], kind="mergesort"
    )
    metrics.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "clusters_file": str(clusters_file),
        "sample_file": str(sample_file),
        "output_csv": str(output_csv),
        "model": args.model,
        "base_url": args.base_url,
        "timing_seconds": {
            "requests_wall": wall_seconds,
            "total": elapsed_since(total_start),
        },
        "result": summarize(metrics, args, wall_seconds),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
