from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def stable_sample_frame(
    frame: pd.DataFrame,
    *,
    sample_size: int | None,
    sample_seed: int,
    key_column: str = "prompt_id",
) -> pd.DataFrame:
    """Return a deterministic hash-ranked sample.

    With the same seed, smaller sample sizes are nested inside larger ones because
    all rows are ranked once by a stable key hash and then truncated.
    """
    if sample_size is None or sample_size >= len(frame):
        return frame.copy()
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")
    if key_column not in frame.columns:
        raise ValueError(f"Cannot sample by missing key column: {key_column!r}.")

    seed_prefix = f"{sample_seed}:".encode()

    def score_key(value: Any) -> str:
        return hashlib.sha256(seed_prefix + str(value).encode()).hexdigest()

    sampled = (
        frame.assign(_sample_rank=frame[key_column].map(score_key))
        .sort_values(["_sample_rank", key_column], kind="mergesort")
        .head(sample_size)
        .drop(columns=["_sample_rank"])
    )
    return sampled.copy()
