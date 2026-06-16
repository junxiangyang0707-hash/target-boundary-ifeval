"""Create IID, group-key, composition held-out, and atomic held-out splits."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from boundary_if.common.config import load_config
from boundary_if.common.data_io import write_json

SPLIT_COLUMNS = [
    "prompt_id",
    "base_key",
    "split",
    "split_name",
    "split_type",
    "seed",
    "num_constraints",
    "constraint_signature",
    "constraint_family_signature",
    "source_dataset",
    "constraint_type",
    "instruction_ids",
    "heldout_instruction_ids",
]


@dataclass(frozen=True)
class AtomicSelectionConfig:
    seed: int = 42
    heldout_min_frequency: int = 500
    heldout_max_frequency_quantile: float = 0.9
    heldout_min_family_frequency: int = 5000
    heldout_fraction: float = 0.1
    val_fraction_from_train_candidate: float = 0.1
    avoid_extreme_special_constraints: bool = True


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    return [str(value)]


def load_promptset(cfg: DictConfig) -> pd.DataFrame:
    normalized_path = Path(str(cfg.data.normalized_path))
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized promptset not found at {normalized_path}. "
            "Run `python -m boundary_if.data.normalize_promptset` first."
        )
    df = pd.read_parquet(normalized_path)
    df["instruction_ids"] = df["instruction_ids"].map(as_list)
    return df


def load_atomic_selection_config(cfg: DictConfig) -> AtomicSelectionConfig:
    config_path = Path(str(cfg.paths.project_root)) / "configs" / "split" / (
        "atomic_constraint_heldout.yaml"
    )
    raw_cfg = OmegaConf.load(config_path) if config_path.exists() else {}
    defaults = AtomicSelectionConfig()
    return AtomicSelectionConfig(
        seed=int(raw_cfg.get("seed", cfg.seed)),
        heldout_min_frequency=int(
            raw_cfg.get("heldout_min_frequency", defaults.heldout_min_frequency)
        ),
        heldout_max_frequency_quantile=float(
            raw_cfg.get(
                "heldout_max_frequency_quantile",
                defaults.heldout_max_frequency_quantile,
            )
        ),
        heldout_min_family_frequency=int(
            raw_cfg.get(
                "heldout_min_family_frequency",
                defaults.heldout_min_family_frequency,
            )
        ),
        heldout_fraction=float(raw_cfg.get("heldout_fraction", defaults.heldout_fraction)),
        val_fraction_from_train_candidate=float(
            raw_cfg.get(
                "val_fraction_from_train_candidate",
                defaults.val_fraction_from_train_candidate,
            )
        ),
        avoid_extreme_special_constraints=bool(
            raw_cfg.get(
                "avoid_extreme_special_constraints",
                defaults.avoid_extreme_special_constraints,
            )
        ),
    )


def make_stratify_labels(
    df: pd.DataFrame,
    *,
    min_count: int = 20,
    include_family_signature: bool = True,
) -> pd.Series:
    num = df["num_constraints"].astype(str)
    if include_family_signature:
        raw_labels = num + "::" + df["constraint_family_signature"].fillna("__missing__")
    else:
        raw_labels = num

    counts = raw_labels.value_counts()
    rare_labels = set(counts[counts < min_count].index)
    labels = raw_labels.mask(
        raw_labels.isin(rare_labels),
        num + "::__rare_constraint_family_signature__",
    )

    if labels.value_counts().min() < 2:
        labels = num
    return labels


def stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return labels


def split_indices(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_family_signature: bool = True,
) -> dict[str, pd.Index]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train, val, and test ratios must sum to 1.0")

    labels = make_stratify_labels(
        df,
        min_count=20,
        include_family_signature=include_family_signature,
    )
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_or_none(labels),
    )

    temp_df = df.loc[temp_idx]
    temp_labels = make_stratify_labels(
        temp_df,
        min_count=2,
        include_family_signature=include_family_signature,
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_df.index,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=stratify_or_none(temp_labels),
    )

    return {
        "train": pd.Index(train_idx),
        "val": pd.Index(val_idx),
        "test": pd.Index(test_idx),
    }


def base_split_frame(
    df: pd.DataFrame,
    split_name: str,
    split_type: str,
    seed: int,
) -> pd.DataFrame:
    split_df = df[
        [
            "prompt_id",
            "base_key",
            "num_constraints",
            "constraint_signature",
            "constraint_family_signature",
            "source_dataset",
            "constraint_type",
            "instruction_ids",
        ]
    ].copy()
    split_df["split"] = ""
    split_df["split_name"] = split_name
    split_df["split_type"] = split_type
    split_df["seed"] = seed
    split_df["heldout_instruction_ids"] = [[] for _ in range(len(split_df))]
    return split_df


def finalize_split_columns(split_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SPLIT_COLUMNS if column not in split_df.columns]
    if missing:
        raise ValueError(f"Split dataframe missing columns: {missing}")
    return split_df[SPLIT_COLUMNS].sort_values("prompt_id").reset_index(drop=True)


def build_iid_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    split_df = base_split_frame(df, "iid_seed42", "iid_prompt", seed)
    assignments = split_indices(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        include_family_signature=True,
    )
    for split, indices in assignments.items():
        split_df.loc[indices, "split"] = split
    return finalize_split_columns(split_df)


def build_group_key_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    row_labels = make_stratify_labels(df, min_count=20, include_family_signature=True)
    group_df = (
        pd.DataFrame(
            {
                "base_key": df["base_key"],
                "group_label": row_labels,
                "num_constraints": df["num_constraints"],
                "constraint_family_signature": df["constraint_family_signature"],
            }
        )
        .groupby("base_key", sort=False)
        .agg(
            group_label=("group_label", lambda values: values.value_counts().index[0]),
            num_constraints=("num_constraints", "max"),
            constraint_family_signature=(
                "constraint_family_signature",
                lambda values: values.value_counts().index[0],
            ),
        )
    )

    labels = group_df["group_label"].reset_index(drop=True)
    group_keys = list(group_df.index)
    train_groups, temp_groups = train_test_split(
        group_keys,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_or_none(labels),
    )
    temp_df = group_df.loc[temp_groups]
    val_groups, test_groups = train_test_split(
        list(temp_df.index),
        test_size=0.5,
        random_state=seed,
        stratify=stratify_or_none(temp_df["group_label"].reset_index(drop=True)),
    )

    group_to_split = {
        **{group: "train" for group in train_groups},
        **{group: "val" for group in val_groups},
        **{group: "test" for group in test_groups},
    }

    split_df = base_split_frame(df, "group_key_seed42", "base_key_group", seed)
    split_df["split"] = split_df["base_key"].map(group_to_split)
    return finalize_split_columns(split_df)


def build_composition_c1_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    split_df = base_split_frame(df, "composition_heldout_c1", "composition_heldout", seed)
    split_df.loc[df["num_constraints"].isin([1, 2, 3]), "split"] = "train"
    split_df.loc[df["num_constraints"] == 4, "split"] = "val"
    split_df.loc[df["num_constraints"] == 5, "split"] = "test"
    return finalize_split_columns(split_df)


def build_composition_c2_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    split_df = base_split_frame(df, "composition_heldout_c2", "composition_heldout", seed)
    train_mask = df["num_constraints"].isin([1, 2, 3, 4])
    heldout_df = df.loc[df["num_constraints"] == 5]

    labels = make_stratify_labels(
        heldout_df,
        min_count=2,
        include_family_signature=True,
    )
    val_idx, test_idx = train_test_split(
        heldout_df.index,
        test_size=0.8,
        random_state=seed,
        stratify=stratify_or_none(labels),
    )

    split_df.loc[train_mask, "split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_idx, "split"] = "test"
    return finalize_split_columns(split_df)


def instruction_counter(df: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for instruction_ids in df["instruction_ids"]:
        counter.update(as_list(instruction_ids))
    return counter


def family_counter_from_instruction_counts(instruction_counts: Counter[str]) -> Counter[str]:
    family_counts: Counter[str] = Counter()
    for instruction_id, count in instruction_counts.items():
        family_counts[instruction_id.split(":", 1)[0]] += count
    return family_counts


def select_atomic_heldout_ids(
    df: pd.DataFrame,
    selection_cfg: AtomicSelectionConfig,
) -> tuple[list[str], dict[str, Any]]:
    instruction_counts = instruction_counter(df)
    family_counts = family_counter_from_instruction_counts(instruction_counts)
    frequencies = np.array(list(instruction_counts.values()), dtype=float)
    max_frequency = int(
        np.quantile(frequencies, selection_cfg.heldout_max_frequency_quantile)
    )

    candidate_records: list[dict[str, Any]] = []
    for instruction_id, frequency in instruction_counts.items():
        family = instruction_id.split(":", 1)[0]
        family_frequency = family_counts[family]
        if frequency < selection_cfg.heldout_min_frequency:
            continue
        if frequency > max_frequency:
            continue
        if (
            selection_cfg.avoid_extreme_special_constraints
            and family_frequency < selection_cfg.heldout_min_family_frequency
        ):
            continue
        candidate_records.append(
            {
                "instruction_id": instruction_id,
                "frequency": int(frequency),
                "family": family,
                "family_frequency": int(family_frequency),
            }
        )

    if not candidate_records:
        raise ValueError("No candidate atomic constraints matched the held-out filters")

    candidate_records = sorted(candidate_records, key=lambda item: item["instruction_id"])
    rng = np.random.default_rng(selection_cfg.seed)
    heldout_count = max(1, int(round(len(candidate_records) * selection_cfg.heldout_fraction)))
    selected_indices = sorted(
        rng.choice(len(candidate_records), size=heldout_count, replace=False).tolist()
    )
    selected_records = [candidate_records[index] for index in selected_indices]
    selected_ids = [record["instruction_id"] for record in selected_records]

    manifest = {
        "selection": {
            "seed": selection_cfg.seed,
            "heldout_min_frequency": selection_cfg.heldout_min_frequency,
            "heldout_max_frequency_quantile": selection_cfg.heldout_max_frequency_quantile,
            "heldout_max_frequency": max_frequency,
            "heldout_min_family_frequency": selection_cfg.heldout_min_family_frequency,
            "heldout_fraction": selection_cfg.heldout_fraction,
            "avoid_extreme_special_constraints": (
                selection_cfg.avoid_extreme_special_constraints
            ),
            "candidate_instruction_count": len(candidate_records),
            "selected_instruction_count": len(selected_records),
        },
        "heldout_instruction_ids": selected_records,
    }
    return selected_ids, manifest


def heldout_overlap(instruction_ids: Any, heldout_ids: set[str]) -> list[str]:
    return sorted(set(as_list(instruction_ids)).intersection(heldout_ids))


def build_atomic_constraint_split(
    df: pd.DataFrame,
    selection_cfg: AtomicSelectionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    heldout_ids, manifest = select_atomic_heldout_ids(df, selection_cfg)
    heldout_set = set(heldout_ids)
    overlaps = df["instruction_ids"].map(lambda ids: heldout_overlap(ids, heldout_set))
    test_mask = overlaps.map(bool)
    train_val_df = df.loc[~test_mask]

    labels = make_stratify_labels(
        train_val_df,
        min_count=20,
        include_family_signature=True,
    )
    train_idx, val_idx = train_test_split(
        train_val_df.index,
        test_size=selection_cfg.val_fraction_from_train_candidate,
        random_state=selection_cfg.seed,
        stratify=stratify_or_none(labels),
    )

    split_df = base_split_frame(
        df,
        "atomic_constraint_heldout_seed42",
        "atomic_constraint_heldout",
        selection_cfg.seed,
    )
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_mask, "split"] = "test"
    split_df["heldout_instruction_ids"] = overlaps

    split_df = finalize_split_columns(split_df)
    manifest["created_at"] = datetime.now(tz=UTC).isoformat()
    manifest["split_name"] = "atomic_constraint_heldout_seed42"
    manifest["split_type"] = "atomic_constraint_heldout"
    manifest["split_counts"] = split_counts(split_df)
    manifest["num_constraints_by_split"] = split_num_constraints(split_df)
    manifest["test_candidate_rows"] = int(test_mask.sum())
    manifest["train_val_candidate_rows"] = int((~test_mask).sum())
    return split_df, manifest


def split_counts(split_df: pd.DataFrame) -> dict[str, int]:
    return {
        split: int(count)
        for split, count in split_df["split"].value_counts().sort_index().items()
    }


def split_num_constraints(split_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split, group in split_df.groupby("split"):
        result[str(split)] = {
            str(key): int(value)
            for key, value in group["num_constraints"].value_counts().sort_index().items()
        }
    return result


def validate_complete_assignment(split_df: pd.DataFrame, expected_rows: int) -> None:
    if len(split_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} split rows, got {len(split_df)}")
    if split_df["prompt_id"].duplicated().any():
        raise ValueError(f"{split_df['split_name'].iloc[0]} contains duplicate prompt_id")
    if split_df["split"].isna().any() or (split_df["split"] == "").any():
        raise ValueError(f"{split_df['split_name'].iloc[0]} contains unassigned rows")


def validate_group_no_leak(split_df: pd.DataFrame) -> None:
    leaks = (
        split_df.groupby("base_key")["split"]
        .nunique()
        .loc[lambda values: values > 1]
    )
    if not leaks.empty:
        raise ValueError(f"base_key leakage across splits: {leaks.head().to_dict()}")


def validate_atomic_no_leak(split_df: pd.DataFrame, heldout_ids: list[str]) -> None:
    heldout_set = set(heldout_ids)
    non_test = split_df[split_df["split"].isin(["train", "val"])]
    leaked = [
        prompt_id
        for prompt_id, instruction_ids in zip(
            non_test["prompt_id"],
            non_test["instruction_ids"],
            strict=True,
        )
        if set(as_list(instruction_ids)).intersection(heldout_set)
    ]
    if leaked:
        raise ValueError(f"Held-out instruction leaked into train/val: {leaked[:5]}")


def write_split(split_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(output_path, index=False)


def run(cfg: DictConfig) -> None:
    df = load_promptset(cfg)
    expected_rows = int(cfg.data.expected_num_rows)
    splits_dir = Path(str(cfg.paths.splits_dir))

    outputs: list[tuple[str, pd.DataFrame, Path]] = [
        ("iid_seed42", build_iid_split(df), splits_dir / "iid_seed42.parquet"),
        ("group_key_seed42", build_group_key_split(df), splits_dir / "group_key_seed42.parquet"),
        (
            "composition_heldout_c1",
            build_composition_c1_split(df),
            splits_dir / "composition_heldout_c1.parquet",
        ),
        (
            "composition_heldout_c2",
            build_composition_c2_split(df),
            splits_dir / "composition_heldout_c2.parquet",
        ),
    ]

    atomic_cfg = load_atomic_selection_config(cfg)
    atomic_split, atomic_manifest = build_atomic_constraint_split(df, atomic_cfg)
    outputs.append(
        (
            "atomic_constraint_heldout_seed42",
            atomic_split,
            splits_dir / "atomic_constraint_heldout_seed42.parquet",
        )
    )

    for split_name, split_df, output_path in outputs:
        validate_complete_assignment(split_df, expected_rows)
        if split_name == "group_key_seed42":
            validate_group_no_leak(split_df)
        write_split(split_df, output_path)
        print(f"Saved {split_name}: {output_path}")
        print(f"  counts: {split_counts(split_df)}")
        print(f"  num_constraints: {split_num_constraints(split_df)}")

    heldout_ids = [
        record["instruction_id"]
        for record in atomic_manifest["heldout_instruction_ids"]
    ]
    validate_atomic_no_leak(atomic_split, heldout_ids)
    write_json(atomic_manifest, splits_dir / "atomic_constraint_heldout_manifest.json")
    print(f"Saved atomic manifest: {splits_dir / 'atomic_constraint_heldout_manifest.json'}")
    print(f"  heldout ids: {heldout_ids}")


def main() -> None:
    cfg = load_config(overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
