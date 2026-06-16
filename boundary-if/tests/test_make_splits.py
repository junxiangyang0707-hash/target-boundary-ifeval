from __future__ import annotations

import pandas as pd

from boundary_if.data.make_splits import (
    AtomicSelectionConfig,
    build_atomic_constraint_split,
    build_composition_c1_split,
    build_group_key_split,
    build_iid_split,
)


def sample_promptset() -> pd.DataFrame:
    rows = []
    for index in range(60):
        num_constraints = (index % 5) + 1
        family = "keywords" if index % 2 == 0 else "detectable_format"
        instruction_id = f"{family}:rule_{index % 6}"
        rows.append(
            {
                "prompt_id": f"prompt-{index:03d}",
                "base_key": f"base-{index // 2:03d}",
                "num_constraints": num_constraints,
                "constraint_signature": instruction_id,
                "constraint_family_signature": family,
                "source_dataset": "ifeval",
                "constraint_type": "multi",
                "instruction_ids": [instruction_id],
            }
        )
    return pd.DataFrame(rows)


def test_iid_split_assigns_all_rows() -> None:
    df = sample_promptset()
    split_df = build_iid_split(df)

    assert len(split_df) == len(df)
    assert set(split_df["split"]) == {"train", "val", "test"}
    assert split_df["prompt_id"].is_unique


def test_group_key_split_has_no_base_key_leakage() -> None:
    df = sample_promptset()
    split_df = build_group_key_split(df)

    assert split_df.groupby("base_key")["split"].nunique().max() == 1


def test_composition_c1_holds_out_larger_compositions() -> None:
    df = sample_promptset()
    split_df = build_composition_c1_split(df)

    assert set(split_df.loc[split_df["split"] == "train", "num_constraints"]) <= {1, 2, 3}
    assert set(split_df.loc[split_df["split"] == "val", "num_constraints"]) == {4}
    assert set(split_df.loc[split_df["split"] == "test", "num_constraints"]) == {5}


def test_atomic_split_keeps_heldout_ids_out_of_train_val() -> None:
    df = sample_promptset()
    split_df, manifest = build_atomic_constraint_split(
        df,
        AtomicSelectionConfig(
            heldout_min_frequency=1,
            heldout_max_frequency_quantile=1.0,
            heldout_min_family_frequency=1,
            heldout_fraction=0.2,
        ),
    )
    heldout_ids = {
        record["instruction_id"] for record in manifest["heldout_instruction_ids"]
    }
    non_test = split_df[split_df["split"].isin(["train", "val"])]

    assert heldout_ids
    assert all(not set(ids).intersection(heldout_ids) for ids in non_test["instruction_ids"])
