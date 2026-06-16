from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: Any, path: str | Path) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_dataset_parquet(path: str | Path) -> Dataset:
    return Dataset.from_parquet(str(path))


def write_dataset_parquet(dataset: Dataset, path: str | Path) -> None:
    output_path = ensure_parent(path)
    dataset.to_parquet(str(output_path))


def read_dataframe_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_dataframe_parquet(dataframe: pd.DataFrame, path: str | Path) -> None:
    output_path = ensure_parent(path)
    dataframe.to_parquet(output_path, index=False)
