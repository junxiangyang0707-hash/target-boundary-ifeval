"""Normalize raw prompt rows into the project promptset schema."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk
from omegaconf import DictConfig

from boundary_if.common.config import load_config
from boundary_if.common.data_io import write_json


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def parse_ground_truth(raw_ground_truth: Any) -> tuple[str, list[str], bool, str | None]:
    try:
        if isinstance(raw_ground_truth, str):
            parsed = ast.literal_eval(raw_ground_truth)
        else:
            parsed = raw_ground_truth

        if not isinstance(parsed, list):
            raise ValueError(f"ground_truth must parse to list, got {type(parsed).__name__}")

        instruction_ids: list[str] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError(f"ground_truth item must be dict, got {type(item).__name__}")
            raw_ids = item.get("instruction_id", [])
            if raw_ids is None:
                continue
            if isinstance(raw_ids, str):
                instruction_ids.append(raw_ids)
            else:
                instruction_ids.extend(str(instruction_id) for instruction_id in raw_ids)

        return canonical_json(parsed), instruction_ids, True, None
    except Exception as exc:
        return str(raw_ground_truth), [], False, str(exc)


def flatten_user_prompt(messages: Any) -> str:
    if not messages:
        return ""

    user_contents = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_contents) == 1:
        return user_contents[0]

    flattened: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            flattened.append(str(message))
            continue
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        flattened.append(f"{role}: {content}")
    return "\n".join(flattened)


def normalize_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool, str | None]:
    messages = row.get("messages") or []
    ground_truth = row.get("ground_truth")
    constraint = row.get("constraint")
    ground_truth_spec, instruction_ids, parse_ok, parse_error = parse_ground_truth(ground_truth)

    sorted_instruction_ids = sorted(instruction_ids)
    sorted_constraint_families = sorted(
        instruction_id.split(":", 1)[0] for instruction_id in instruction_ids
    )

    normalized = {
        "prompt_id": sha256_text(
            {
                "messages": messages,
                "ground_truth": ground_truth,
                "constraint": constraint,
            }
        ),
        "base_key": row.get("key"),
        "raw_messages": messages,
        "user_prompt": flatten_user_prompt(messages),
        "constraint_text": constraint,
        "ground_truth_spec": ground_truth_spec,
        "instruction_ids": instruction_ids,
        "constraint_signature": "|".join(sorted_instruction_ids),
        "constraint_family_signature": "|".join(sorted_constraint_families),
        "num_constraints": len(instruction_ids),
        "source_dataset": row.get("dataset"),
        "constraint_type": row.get("constraint_type"),
    }
    return normalized, parse_ok, parse_error


def _counter_to_sorted_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def _counter_to_frequency(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def _duplicate_stats(values: list[Any]) -> dict[str, int]:
    counts = Counter(values)
    duplicate_values = sum(1 for value in counts.values() if value > 1)
    duplicate_rows = sum(value - 1 for value in counts.values() if value > 1)
    return {
        "duplicate_values": int(duplicate_values),
        "duplicate_rows": int(duplicate_rows),
    }


def _count_ratio(count: int, denominator: int) -> dict[str, float | int]:
    return {
        "count": int(count),
        "ratio": 0.0 if denominator == 0 else count / denominator,
    }


def build_audit(
    raw_dataset: Dataset,
    normalized_rows: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, Any]:
    num_rows = len(normalized_rows)
    expected_rows = int(cfg.data.expected_num_rows)

    num_constraints = Counter(row["num_constraints"] for row in normalized_rows)
    dataset_distribution = Counter(row["source_dataset"] for row in normalized_rows)
    constraint_type_distribution = Counter(row["constraint_type"] for row in normalized_rows)
    instruction_id_frequency: Counter[str] = Counter()
    constraint_family_frequency: Counter[str] = Counter()

    for row in normalized_rows:
        instruction_id_frequency.update(row["instruction_ids"])
        constraint_family_frequency.update(
            instruction_id.split(":", 1)[0] for instruction_id in row["instruction_ids"]
        )

    empty_messages = 0
    empty_constraint = 0
    for row in raw_dataset:
        if not row.get("messages"):
            empty_messages += 1
        if row.get("constraint") in (None, ""):
            empty_constraint += 1

    return {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_dataset_path": str(cfg.data.raw_dataset_dir),
        "normalized_path": str(cfg.data.normalized_path),
        "num_rows": num_rows,
        "expected_num_rows": expected_rows,
        "expected_num_rows_match": num_rows == expected_rows,
        "num_constraints_distribution": _counter_to_sorted_dict(num_constraints),
        "dataset_distribution": _counter_to_frequency(dataset_distribution),
        "constraint_type_distribution": _counter_to_frequency(constraint_type_distribution),
        "instruction_id_frequency": _counter_to_frequency(instruction_id_frequency),
        "constraint_family_frequency": _counter_to_frequency(constraint_family_frequency),
        "duplicates": {
            "prompt_id": _duplicate_stats([row["prompt_id"] for row in normalized_rows]),
            "base_key": _duplicate_stats([row["base_key"] for row in normalized_rows]),
        },
        "empty_messages": _count_ratio(empty_messages, num_rows),
        "empty_constraint": _count_ratio(empty_constraint, num_rows),
        "ground_truth_parse_failures": _count_ratio(len(parse_errors), num_rows),
        "ground_truth_parse_failure_examples": parse_errors[:20],
    }


def load_raw_promptset(cfg: DictConfig) -> Dataset:
    raw_dataset_dir = Path(str(cfg.data.raw_dataset_dir))
    if raw_dataset_dir.exists():
        return load_from_disk(str(raw_dataset_dir))

    raw_parquet_path = Path(str(cfg.data.raw_parquet_path))
    if raw_parquet_path.exists():
        return Dataset.from_parquet(str(raw_parquet_path))

    raise FileNotFoundError(
        f"Raw dataset not found at {raw_dataset_dir} or {raw_parquet_path}. "
        "Run `python -m boundary_if.data.ingest_dataset` first."
    )


def run(cfg: DictConfig) -> None:
    raw_dataset = load_raw_promptset(cfg)

    normalized_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for index, row in enumerate(raw_dataset):
        normalized, parse_ok, parse_error = normalize_row(row)
        normalized_rows.append(normalized)
        if not parse_ok:
            parse_errors.append(
                {
                    "row_index": index,
                    "key": row.get("key"),
                    "error": parse_error,
                    "ground_truth": row.get("ground_truth"),
                }
            )

    normalized_dataset = Dataset.from_list(normalized_rows)

    normalized_path = Path(str(cfg.data.normalized_path))
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_dataset.to_parquet(str(normalized_path))

    audit = build_audit(raw_dataset, normalized_rows, parse_errors, cfg)
    write_json(audit, cfg.data.audit_path)

    if not audit["expected_num_rows_match"]:
        raise ValueError(
            f"Normalized row count mismatch: got {audit['num_rows']}, "
            f"expected {audit['expected_num_rows']}"
        )
    if parse_errors:
        raise ValueError(f"Failed to parse {len(parse_errors)} ground_truth rows")

    print(f"Normalized rows: {audit['num_rows']}")
    print(f"Saved normalized Parquet: {normalized_path}")
    print(f"Saved audit JSON: {cfg.data.audit_path}")
    print(f"num_constraints distribution: {audit['num_constraints_distribution']}")
    print(f"duplicate prompt_id: {audit['duplicates']['prompt_id']}")
    print(f"duplicate base_key: {audit['duplicates']['base_key']}")


def main() -> None:
    cfg = load_config(overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
