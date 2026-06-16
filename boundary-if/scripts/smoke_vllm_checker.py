from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

SUPPORTED_INSTRUCTIONS = {
    "change_case:english_capital",
    "change_case:english_lowercase",
    "detectable_format:title",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:exclude_word_harder",
    "last_word:last_word_answer",
    "punctuation:no_comma",
    "startend:end_checker",
    "startend:quotation",
}


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample a prompt, run vLLM inference, and check supported constraints."
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument("--split-file", default="data/splits/group_key_seed42.parquet")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b-instruct-2507"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--print-response", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


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
                    "ready": True,
                    "attempts": attempts,
                    "seconds": elapsed_since(start_time),
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")


def flatten_ground_truth(ground_truth_spec: str) -> list[dict[str, Any]]:
    parsed = json.loads(ground_truth_spec)
    flattened: list[dict[str, Any]] = []
    for item in parsed:
        instruction_ids = item.get("instruction_id") or []
        kwargs_list = item.get("kwargs") or []
        for index, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[index] if index < len(kwargs_list) else None
            flattened.append({"instruction_id": instruction_id, "kwargs": kwargs or {}})
    return flattened


def load_candidate_rows(args: argparse.Namespace) -> pd.DataFrame:
    promptset = pd.read_parquet(resolve_path(args.promptset_file))
    split_df = pd.read_parquet(resolve_path(args.split_file))
    merged = split_df[["prompt_id", "split"]].merge(
        promptset,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )
    if args.split != "all":
        merged = merged[merged["split"] == args.split]
    if args.prompt_id:
        merged = merged[merged["prompt_id"] == args.prompt_id]
    else:
        merged = merged[
            merged["instruction_ids"].map(
                lambda ids: set(as_list(ids)).issubset(SUPPORTED_INSTRUCTIONS)
            )
        ]
    if merged.empty:
        raise ValueError("No rows matched the requested split/prompt_id and smoke-checker support.")
    return merged


def choose_row(df: pd.DataFrame, seed: int | None) -> pd.Series:
    random_state = random.Random(seed)
    offset = random_state.randrange(len(df))
    return df.iloc[offset]


def generate_response(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    response = requests.post(url, json=payload, timeout=args.timeout_seconds)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return content, data.get("usage", {})


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def last_word(text: str) -> str:
    words = normalized_words(text.rstrip(string.whitespace + string.punctuation))
    return words[-1].lower() if words else ""


def check_instruction(instruction_id: str, kwargs: dict[str, Any], response: str) -> dict[str, Any]:
    followed = False
    reason = ""

    if instruction_id == "detectable_format:title":
        followed = bool(re.search(r"<<[^<>]+>>", response))
        reason = "requires a title wrapped in <<...>>"
    elif instruction_id == "change_case:english_capital":
        followed = not any(char.isalpha() and char.islower() for char in response)
        reason = "requires no lowercase alphabetic characters"
    elif instruction_id == "change_case:english_lowercase":
        followed = not any(char.isalpha() and char.isupper() for char in response)
        reason = "requires no uppercase alphabetic characters"
    elif instruction_id == "startend:end_checker":
        end_phrase = str(kwargs.get("end_phrase", ""))
        followed = bool(end_phrase) and response.rstrip().endswith(end_phrase)
        reason = f"requires response to end with {end_phrase!r}"
    elif instruction_id == "startend:quotation":
        stripped = response.strip()
        followed = len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')
        reason = "requires wrapping the full response in double quotation marks"
    elif instruction_id == "punctuation:no_comma":
        followed = "," not in response
        reason = "requires no comma characters"
    elif instruction_id in {"keywords:forbidden_words", "keywords:exclude_word_harder"}:
        forbidden = kwargs.get("forbidden_words") or kwargs.get("keyword") or []
        if isinstance(forbidden, str) or not isinstance(forbidden, Sequence):
            forbidden_words = [str(forbidden)]
        else:
            forbidden_words = [str(word) for word in forbidden]
        lower_response = response.lower()
        followed = all(word.lower() not in lower_response for word in forbidden_words)
        reason = f"forbids {forbidden_words}"
    elif instruction_id == "keywords:existence":
        raw_keywords = kwargs.get("keywords") or kwargs.get("keyword") or []
        if isinstance(raw_keywords, str) or not isinstance(raw_keywords, Sequence):
            keywords = [str(raw_keywords)]
        else:
            keywords = [str(keyword) for keyword in raw_keywords]
        lower_response = response.lower()
        followed = bool(keywords) and all(keyword.lower() in lower_response for keyword in keywords)
        reason = f"requires keywords {keywords}"
    elif instruction_id == "last_word:last_word_answer":
        expected = str(kwargs.get("last_word", "")).lower()
        followed = bool(expected) and last_word(response) == expected
        reason = f"requires final word {expected!r}"
    else:
        raise KeyError(f"Unsupported smoke checker instruction: {instruction_id}")

    return {
        "instruction_id": instruction_id,
        "kwargs": kwargs,
        "followed": followed,
        "reason": reason,
    }


def run_checker(ground_truth_spec: str, response: str) -> dict[str, Any]:
    checks = [
        check_instruction(item["instruction_id"], item["kwargs"], response)
        for item in flatten_ground_truth(ground_truth_spec)
    ]
    return {
        "checker_name": "local_smoke_subset",
        "supported_instruction_count": len(SUPPORTED_INSTRUCTIONS),
        "strict_pass": all(item["followed"] for item in checks),
        "per_constraint": checks,
    }


def main() -> None:
    total_start = now()
    args = parse_args()

    wait_timing = wait_for_vllm(args.base_url, args.timeout_seconds)

    data_start = now()
    candidates = load_candidate_rows(args)
    data_seconds = elapsed_since(data_start)

    sample_start = now()
    row = choose_row(candidates, args.seed)
    sample_seconds = elapsed_since(sample_start)

    infer_start = now()
    response_text, usage = generate_response(args, str(row["user_prompt"]))
    inference_seconds = elapsed_since(infer_start)

    checker_start = now()
    checker_result = run_checker(str(row["ground_truth_spec"]), response_text)
    checker_seconds = elapsed_since(checker_start)

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    completion_tokens_per_second = (
        round(completion_tokens / inference_seconds, 4)
        if isinstance(completion_tokens, int) and inference_seconds > 0
        else None
    )
    total_tokens_per_second = (
        round(total_tokens / inference_seconds, 4)
        if isinstance(total_tokens, int) and inference_seconds > 0
        else None
    )

    output = {
        "model": args.model,
        "base_url": args.base_url,
        "timing_seconds": {
            "wait_for_vllm_ready": wait_timing["seconds"],
            "load_candidates": data_seconds,
            "sample_row": sample_seconds,
            "vllm_chat_completion": inference_seconds,
            "checker": checker_seconds,
            "total_script": elapsed_since(total_start),
        },
        "sample": {
            "prompt_id": row["prompt_id"],
            "split": row["split"],
            "num_constraints": int(row["num_constraints"]),
            "instruction_ids": to_jsonable(row["instruction_ids"]),
            "constraint_text": row["constraint_text"],
        },
        "inference": {
            "response_chars": len(response_text),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                **{
                    key: value
                    for key, value in usage.items()
                    if key not in {"prompt_tokens", "completion_tokens", "total_tokens"}
                },
            },
            "completion_tokens_per_second": completion_tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second,
        },
        "checker": checker_result,
    }
    if args.print_response:
        output["inference"]["response_text"] = response_text
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
