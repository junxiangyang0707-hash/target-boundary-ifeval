from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/predictions/strict_atomic_blog_predictions.parquet"
SPLIT = ROOT / "data/splits/atomic_constraint_heldout_seed42.parquet"
PER_CONSTRAINT = ROOT / "results/tables/per_constraint_metrics.csv"
OUT = ROOT / "results/tables/prompt_level_error_examples.csv"
GENERATION_ROOT_ENV = "BOUNDARY_IF_GENERATIONS_ROOT"


def instruction_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, (list, tuple)):
            return "|".join(str(item) for item in items)
    return str(value)


def normalize_multiline_text(value: object) -> str:
    if value is None:
        return ""
    return "\n".join(line.rstrip() for line in str(value).splitlines())


def outcome(row: pd.Series) -> str:
    if row["label"] == 1 and row["pred_label"] == 1:
        return "TP"
    if row["label"] == 0 and row["pred_label"] == 1:
        return "FP"
    if row["label"] == 1 and row["pred_label"] == 0:
        return "FN"
    return "TN"


def append_unique(rows: list[pd.Series], candidates: pd.DataFrame, n: int) -> None:
    seen = {row["prompt_id"] for row in rows}
    for _, row in candidates.iterrows():
        if row["prompt_id"] in seen:
            continue
        rows.append(row)
        seen.add(row["prompt_id"])
        if len([r for r in rows if r["case_group"] == row["case_group"]]) >= n:
            break


def candidate_generation_roots() -> list[Path]:
    roots: list[Path] = []
    env_value = os.environ.get(GENERATION_ROOT_ENV)
    if env_value:
        roots.append(Path(env_value))
    roots.extend(
        [
            ROOT.parent / "boundary-if/data/generations",
            Path("/workspace/data/generations"),
        ]
    )
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def load_prompt_responses(prompt_ids: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for generation_root in candidate_generation_roots():
        if not generation_root.exists():
            continue
        for path in sorted(generation_root.glob("qwen3_4b_instruct_2507_under2048_*_max2048/outputs.parquet")):
            part = pd.read_parquet(path, columns=["prompt_id", "user_prompt", "response_text"])
            part = part[part["prompt_id"].isin(prompt_ids)].copy()
            if part.empty:
                continue
            rows.append(part)
        if rows:
            break
    if not rows:
        return pd.DataFrame(columns=["prompt_id", "input_prompt", "model_response"])
    text = pd.concat(rows, ignore_index=True).drop_duplicates("prompt_id", keep="first")
    for col in ["user_prompt", "response_text"]:
        text[col] = text[col].map(normalize_multiline_text)
    return text.rename(columns={"user_prompt": "input_prompt", "response_text": "model_response"})[
        ["prompt_id", "input_prompt", "model_response"]
    ]


def load_existing_prompt_responses() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=["prompt_id", "input_prompt", "model_response"])
    existing = pd.read_csv(OUT)
    wanted = ["prompt_id", "input_prompt", "model_response"]
    if not set(wanted).issubset(existing.columns):
        return pd.DataFrame(columns=wanted)
    existing = existing[wanted].copy()
    for col in ["input_prompt", "model_response"]:
        existing[col] = existing[col].map(normalize_multiline_text)
    return existing


def add_prompt_responses(table: pd.DataFrame) -> pd.DataFrame:
    prompt_ids = set(table["prompt_id"].astype(str))
    text = load_prompt_responses(prompt_ids)
    if text.empty:
        text = load_existing_prompt_responses()
    table = table.merge(text, on="prompt_id", how="left")
    for col in ["input_prompt", "model_response"]:
        table[col] = table[col].fillna("")
    return table


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS)
    split = pd.read_parquet(SPLIT)
    per_constraint = pd.read_csv(PER_CONSTRAINT)

    part = predictions[
        (predictions["model_family"] == "M3_mean")
        & (predictions["model_variant"] == "mean_pooling")
        & (predictions["train_size_label"] == "full")
        & (predictions["split"] == "test")
    ].copy()

    grouped = (
        part.groupby("prompt_id", as_index=False)
        .agg(
            label=("label", "first"),
            p_pass_mean=("p_pred", "mean"),
            p_pass_std=("p_pred", "std"),
            num_constraints=("num_constraints", "first"),
            length_bin=("length_bin", "first"),
            cluster=("cluster", "first"),
        )
    )
    grouped["pred_label"] = (grouped["p_pass_mean"] >= 0.5).astype(int)
    grouped["outcome"] = grouped.apply(outcome, axis=1)
    grouped["confidence"] = (grouped["p_pass_mean"] - 0.5).abs()

    meta = split[
        [
            "prompt_id",
            "base_key",
            "source_dataset",
            "constraint_signature",
            "constraint_family_signature",
            "instruction_ids",
        ]
    ].copy()
    meta["instruction_ids"] = meta["instruction_ids"].map(instruction_text)
    grouped = grouped.merge(meta, on="prompt_id", how="left")

    selected: list[pd.Series] = []
    recipes = [
        ("High-confidence TP", grouped[grouped["outcome"] == "TP"].sort_values("p_pass_mean", ascending=False), 2),
        ("High-confidence FP", grouped[grouped["outcome"] == "FP"].sort_values("p_pass_mean", ascending=False), 2),
        ("High-confidence FN", grouped[grouped["outcome"] == "FN"].sort_values("p_pass_mean", ascending=True), 2),
        ("High-confidence TN", grouped[grouped["outcome"] == "TN"].sort_values("p_pass_mean", ascending=True), 2),
    ]
    for case_group, candidates, n in recipes:
        candidates = candidates.copy()
        candidates["case_group"] = case_group
        candidates["hard_constraint_focus"] = ""
        candidates["case_note"] = case_group
        append_unique(selected, candidates, n)

    hard_constraints = (
        per_constraint[per_constraint["test_rows"] >= 100]
        .sort_values(["M3_mean_AUPRC", "positive_rate"], ascending=[True, True])
        .head(6)["constraint_id"]
        .tolist()
    )
    seen = {row["prompt_id"] for row in selected}
    hard_rows: list[pd.Series] = []
    for constraint_id in hard_constraints:
        candidates = grouped[
            grouped["instruction_ids"].fillna("").str.contains(constraint_id, regex=False)
            & (~grouped["prompt_id"].isin(seen))
        ].copy()
        if candidates.empty:
            continue
        candidates["is_error"] = candidates["outcome"].isin(["FP", "FN"])
        candidates = candidates.sort_values(["is_error", "confidence"], ascending=[False, False])
        row = candidates.iloc[0].copy()
        row["case_group"] = "Hard constraint"
        row["hard_constraint_focus"] = constraint_id
        row["case_note"] = f"Low per-constraint M3 AUPRC region: {constraint_id}"
        hard_rows.append(row)
        seen.add(row["prompt_id"])
        if len(hard_rows) == 4:
            break
    selected.extend(hard_rows)

    table = pd.DataFrame(selected).head(12).copy()
    table.insert(0, "case_id", [f"E{i:02d}" for i in range(1, len(table) + 1)])
    table["prompt_id_short"] = table["prompt_id"].str.slice(0, 12)
    table["p_pass_mean"] = table["p_pass_mean"].round(4)
    table["p_pass_std"] = table["p_pass_std"].round(4)
    table = add_prompt_responses(table)
    table = table[
        [
            "case_id",
            "case_group",
            "outcome",
            "prompt_id_short",
            "prompt_id",
            "label",
            "pred_label",
            "p_pass_mean",
            "p_pass_std",
            "num_constraints",
            "length_bin",
            "cluster",
            "hard_constraint_focus",
            "instruction_ids",
            "constraint_family_signature",
            "source_dataset",
            "base_key",
            "input_prompt",
            "model_response",
            "case_note",
        ]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(table)} rows")


if __name__ == "__main__":
    main()
