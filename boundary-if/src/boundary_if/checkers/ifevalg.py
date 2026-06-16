from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nltk
from langdetect import DetectorFactory

from open_instruct.IFEvalG import instructions_registry

DetectorFactory.seed = 0
CHECKER_NAME = "ifevalg_official_vendored_deterministic"


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 6)


def stable_seed(instruction_id: str, kwargs: dict[str, Any]) -> int:
    payload = json.dumps(
        {"instruction_id": instruction_id, "kwargs": kwargs},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


@contextmanager
def deterministic_checker_context(instruction_id: str, kwargs: dict[str, Any]) -> Iterator[None]:
    state = random.getstate()
    DetectorFactory.seed = 0
    random.seed(stable_seed(instruction_id, kwargs))
    try:
        yield
    finally:
        random.setstate(state)


def ensure_nltk_data(cache_dir: str | Path = ".cache/nltk") -> Path:
    """Ensure tokenizer data needed by the vendored IFEvalG checker exists."""
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(path))
    if str(path) not in nltk.data.path:
        nltk.data.path.insert(0, str(path))

    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, package_name in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, download_dir=str(path), quiet=True)
            nltk.data.find(resource_path)
    return path


def flatten_ground_truth(ground_truth_spec: str) -> list[dict[str, Any]]:
    parsed = json.loads(ground_truth_spec)
    flattened: list[dict[str, Any]] = []
    for item in parsed:
        instruction_ids = item.get("instruction_id") or []
        kwargs_list = item.get("kwargs") or []
        for index, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[index] if index < len(kwargs_list) else None
            flattened.append(
                {
                    "constraint_index": len(flattened),
                    "instruction_id": str(instruction_id),
                    "kwargs": kwargs or {},
                }
            )
    return flattened


def check_instruction(
    instruction_id: str,
    kwargs: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    start_time = now()
    result: dict[str, Any] = {
        "instruction_id": instruction_id,
        "kwargs": kwargs,
        "followed": False,
        "checker_error": False,
        "checker_error_type": "",
        "checker_error_message": "",
        "checker_seconds": 0.0,
    }
    try:
        with deterministic_checker_context(instruction_id, kwargs):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            instruction.build_description(**kwargs)
            result["followed"] = bool(instruction.check_following(response))
    except Exception as exc:  # noqa: BLE001 - per-row checker errors must be materialized.
        result["checker_error"] = True
        result["checker_error_type"] = type(exc).__name__
        result["checker_error_message"] = str(exc)
    finally:
        result["checker_seconds"] = elapsed_since(start_time)
    return result


def run_checker(ground_truth_spec: str, response: str) -> dict[str, Any]:
    start_time = now()
    constraints = flatten_ground_truth(ground_truth_spec)
    per_constraint = [
        {
            "constraint_index": item["constraint_index"],
            **check_instruction(item["instruction_id"], item["kwargs"], response),
        }
        for item in constraints
    ]
    checker_error_count = sum(1 for item in per_constraint if item["checker_error"])
    followed_count = sum(1 for item in per_constraint if item["followed"])
    failed_count = len(per_constraint) - followed_count
    strict_pass = bool(per_constraint) and checker_error_count == 0 and failed_count == 0
    return {
        "checker_name": CHECKER_NAME,
        "num_constraints_checked": len(per_constraint),
        "followed_count": followed_count,
        "failed_count": failed_count,
        "checker_error_count": checker_error_count,
        "strict_pass": strict_pass,
        "checker_seconds": elapsed_since(start_time),
        "per_constraint": per_constraint,
    }
