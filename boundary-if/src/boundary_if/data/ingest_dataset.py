"""Ingest allenai/IF_multi_constraints_upto5 from Hugging Face Datasets."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf

from boundary_if.common.config import load_config
from boundary_if.common.data_io import write_json


def _optional_str(value: Any) -> str | None:
    if value is None or value in ("", "null", "None"):
        return None
    return str(value)


def load_raw_dataset(cfg: DictConfig) -> Dataset:
    data_cfg = cfg.data
    kwargs: dict[str, Any] = {
        "path": str(data_cfg.hf_path),
        "split": str(data_cfg.hf_split),
        "cache_dir": str(data_cfg.cache_dir),
    }

    hf_name = _optional_str(data_cfg.get("hf_name"))
    if hf_name is not None:
        kwargs["name"] = hf_name

    revision = _optional_str(data_cfg.get("revision"))
    if revision is not None:
        kwargs["revision"] = revision

    dataset = load_dataset(**kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset split, got {type(dataset)!r}")
    return dataset


def save_raw_dataset(dataset: Dataset, cfg: DictConfig) -> dict[str, Any]:
    data_cfg = cfg.data
    raw_dataset_dir = Path(str(data_cfg.raw_dataset_dir))
    raw_parquet_path = Path(str(data_cfg.raw_parquet_path))

    raw_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(raw_dataset_dir))
    dataset.to_parquet(str(raw_parquet_path))

    info = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "hf_path": str(data_cfg.hf_path),
        "hf_name": _optional_str(data_cfg.get("hf_name")),
        "hf_split": str(data_cfg.hf_split),
        "revision": _optional_str(data_cfg.get("revision")),
        "num_rows": dataset.num_rows,
        "columns": list(dataset.column_names),
        "features": str(dataset.features),
        "cache_files": dataset.cache_files,
        "raw_dataset_dir": str(raw_dataset_dir),
        "raw_parquet_path": str(raw_parquet_path),
    }
    write_json(info, data_cfg.raw_info_path)
    return info


def run(cfg: DictConfig) -> None:
    print("Resolved data ingest config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dataset = load_raw_dataset(cfg)
    info = save_raw_dataset(dataset, cfg)

    expected_rows = int(cfg.data.expected_num_rows)
    status = "OK" if info["num_rows"] == expected_rows else "MISMATCH"
    print(f"Downloaded rows: {info['num_rows']} expected: {expected_rows} [{status}]")
    print(f"Saved Arrow dataset: {info['raw_dataset_dir']}")
    print(f"Saved raw Parquet: {info['raw_parquet_path']}")
    print(f"Saved ingest info: {cfg.data.raw_info_path}")

    if info["num_rows"] != expected_rows:
        raise ValueError(
            f"Dataset row count mismatch: got {info['num_rows']}, expected {expected_rows}"
        )


def main() -> None:
    cfg = load_config(overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
