from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ALL_PATH = (
    "data/promptsets/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.all.parquet"
)
DEFAULT_SPLITS_DIR = "data/splits/qwen3_4b_instruct_2507_under2048_nontruncated"
SPLIT_ORDER = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit split assignment files for base_key duplication/leakage, "
            "prompt_id hash duplication/leakage, and cluster distribution drift."
        )
    )
    parser.add_argument("--all-path", default=DEFAULT_ALL_PATH)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit non-zero if prompt hash leakage/coverage issues or group key leakage are found.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(payload: dict[str, Any], path: Path) -> str:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    return str(path)


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False, encoding="utf-8")
    return str(path)


def require_columns(frame: pd.DataFrame, columns: list[str], *, path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def duplicate_summary(frame: pd.DataFrame, column: str) -> dict[str, int]:
    values = frame[column]
    duplicate_mask = values.duplicated(keep=False)
    duplicated_values = values[duplicate_mask]
    return {
        "row_count": int(len(frame)),
        "unique_count": int(values.nunique(dropna=False)),
        "duplicate_row_count": int(duplicate_mask.sum()),
        "duplicated_value_count": int(duplicated_values.nunique(dropna=False)),
        "null_count": int(values.isna().sum()),
    }


def duplicate_preview(
    frame: pd.DataFrame,
    column: str,
    *,
    split_file: str,
    top_n: int,
) -> pd.DataFrame:
    counts = (
        frame.groupby(column, dropna=False)
        .agg(
            row_count=(column, "size"),
            splits=("split", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    counts = counts[counts["row_count"] > 1].sort_values(
        ["row_count", column],
        ascending=[False, True],
    )
    if counts.empty:
        return pd.DataFrame()
    counts.insert(0, "split_file", split_file)
    return counts.head(top_n)


def cross_partition_leakage(
    frame: pd.DataFrame,
    column: str,
    *,
    split_file: str,
    top_n: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            row_count=(column, "size"),
            partition_count=("split", lambda s: len(set(map(str, s)))),
            partitions=("split", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    leaked = grouped[grouped["partition_count"] > 1].sort_values(
        ["partition_count", "row_count", column],
        ascending=[False, False, True],
    )
    summary = {
        "leaked_value_count": int(len(leaked)),
        "leaked_row_count": int(leaked["row_count"].sum()) if not leaked.empty else 0,
    }
    if leaked.empty:
        return summary, pd.DataFrame()
    leaked.insert(0, "split_file", split_file)
    return summary, leaked.head(top_n)


def duplicate_by_partition(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, part in frame.groupby("split", dropna=False):
        summary = duplicate_summary(part, column)
        summary["split"] = str(split)
        rows.append(summary)
    rows.sort(
        key=lambda item: (
            SPLIT_ORDER.index(item["split"]) if item["split"] in SPLIT_ORDER else 99
        )
    )
    return rows


def coverage_summary(
    split_prompt_ids: pd.Series,
    all_prompt_ids: pd.Series,
    *,
    split_file: str,
    top_n: int,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    split_set = set(split_prompt_ids.tolist())
    all_set = set(all_prompt_ids.tolist())
    missing = sorted(all_set - split_set)
    extra = sorted(split_set - all_set)
    summary = {
        "expected_all_row_count": int(len(all_prompt_ids)),
        "split_row_count": int(len(split_prompt_ids)),
        "missing_prompt_id_count": int(len(missing)),
        "extra_prompt_id_count": int(len(extra)),
    }
    missing_preview = pd.DataFrame(
        {"split_file": split_file, "prompt_id": missing[:top_n], "issue": "missing_from_split"}
    )
    extra_preview = pd.DataFrame(
        {"split_file": split_file, "prompt_id": extra[:top_n], "issue": "extra_not_in_all"}
    )
    return summary, missing_preview, extra_preview


def cluster_distribution(
    merged: pd.DataFrame,
    global_cluster_share: pd.Series,
    *,
    split_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_counts = (
        merged.groupby(["split", "cluster"], dropna=False)
        .agg(
            row_count=("prompt_id", "size"),
            pass_count=("checker_pass", "sum"),
            prompt_tokens_median=("prompt_tokens", "median"),
            prompt_tokens_p90=("prompt_tokens", lambda s: float(np.percentile(s, 90))),
        )
        .reset_index()
    )
    split_totals = merged.groupby("split", dropna=False).size().rename("split_row_count")
    cluster_counts = cluster_counts.merge(
        split_totals.reset_index(),
        on="split",
        how="left",
        validate="many_to_one",
    )
    cluster_counts["share"] = cluster_counts["row_count"] / cluster_counts["split_row_count"]
    cluster_counts["pass_rate"] = cluster_counts["pass_count"] / cluster_counts["row_count"]
    cluster_counts["global_share"] = cluster_counts["cluster"].map(global_cluster_share).fillna(0.0)
    cluster_counts["share_diff_from_all"] = cluster_counts["share"] - cluster_counts["global_share"]
    cluster_counts.insert(0, "split_file", split_file)
    cluster_counts = cluster_counts.sort_values(["split", "cluster"]).reset_index(drop=True)

    tvd_rows: list[dict[str, Any]] = []
    all_clusters = sorted(global_cluster_share.index.tolist())
    for split, part in merged.groupby("split", dropna=False):
        split_share = part["cluster"].value_counts(normalize=True, dropna=False)
        abs_diff_sum = 0.0
        for cluster in all_clusters:
            split_value = float(split_share.get(cluster, 0.0))
            global_value = float(global_cluster_share[cluster])
            abs_diff_sum += abs(split_value - global_value)
        tvd_rows.append(
            {
                "split_file": split_file,
                "split": str(split),
                "row_count": int(len(part)),
                "cluster_count": int(part["cluster"].nunique(dropna=False)),
                "cluster_tvd_vs_all": float(0.5 * abs_diff_sum),
            }
        )
    tvd = pd.DataFrame(tvd_rows).sort_values(["split_file", "split"])
    return cluster_counts, tvd


def top_cluster_drifts(cluster_counts: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if cluster_counts.empty:
        return []
    drift = cluster_counts.copy()
    drift["abs_share_diff_from_all"] = drift["share_diff_from_all"].abs()
    columns = [
        "split_file",
        "split",
        "cluster",
        "row_count",
        "share",
        "global_share",
        "share_diff_from_all",
        "pass_rate",
    ]
    return (
        drift.sort_values("abs_share_diff_from_all", ascending=False)
        .head(top_n)[columns]
        .to_dict("records")
    )


def audit_one_split(
    split_path: Path,
    all_df: pd.DataFrame,
    global_cluster_share: pd.Series,
    *,
    output_dir: Path,
    top_n: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    split_df = pd.read_parquet(split_path)
    require_columns(
        split_df,
        ["prompt_id", "base_key", "split", "split_name", "split_type"],
        path=split_path,
    )

    split_file = split_path.name
    merged = split_df.merge(
        all_df[
            [
                "prompt_id",
                "cluster",
                "checker_pass",
                "prompt_tokens",
                "output_tokens",
                "output_truncated",
            ]
        ],
        on="prompt_id",
        how="left",
        validate="one_to_one",
    )
    if merged["cluster"].isna().any():
        missing_count = int(merged["cluster"].isna().sum())
        raise ValueError(f"{split_file} has {missing_count} prompt_id values missing from all set")

    split_counts = (
        split_df["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="row_count")
        .sort_values("split")
        .to_dict("records")
    )
    pass_by_split = (
        merged.groupby("split", dropna=False)
        .agg(
            row_count=("checker_pass", "size"),
            pass_count=("checker_pass", "sum"),
            truncated_rows=("output_truncated", "sum"),
        )
        .reset_index()
    )
    pass_by_split["pass_rate"] = pass_by_split["pass_count"] / pass_by_split["row_count"]

    prompt_id_summary = duplicate_summary(split_df, "prompt_id")
    base_key_summary = duplicate_summary(split_df, "base_key")
    prompt_leakage, prompt_leak_preview = cross_partition_leakage(
        split_df,
        "prompt_id",
        split_file=split_file,
        top_n=top_n,
    )
    key_leakage, key_leak_preview = cross_partition_leakage(
        split_df,
        "base_key",
        split_file=split_file,
        top_n=top_n,
    )
    coverage, missing_preview, extra_preview = coverage_summary(
        split_df["prompt_id"],
        all_df["prompt_id"],
        split_file=split_file,
        top_n=top_n,
    )
    cluster_counts, cluster_tvd = cluster_distribution(
        merged,
        global_cluster_share,
        split_file=split_file,
    )

    prompt_dup_preview = duplicate_preview(
        split_df,
        "prompt_id",
        split_file=split_file,
        top_n=top_n,
    )
    key_dup_preview = duplicate_preview(
        split_df,
        "base_key",
        split_file=split_file,
        top_n=top_n,
    )

    issue_flags = {
        "has_prompt_id_duplicate_rows": prompt_id_summary["duplicate_row_count"] > 0,
        "has_prompt_id_cross_split_leakage": prompt_leakage["leaked_value_count"] > 0,
        "has_missing_or_extra_prompt_ids": (
            coverage["missing_prompt_id_count"] > 0 or coverage["extra_prompt_id_count"] > 0
        ),
        "has_base_key_cross_split_leakage": key_leakage["leaked_value_count"] > 0,
        "has_group_split_base_key_leakage": (
            split_path.stem == "group_key_seed42" and key_leakage["leaked_value_count"] > 0
        ),
    }
    critical_issues = [
        name
        for name in [
            "has_prompt_id_duplicate_rows",
            "has_prompt_id_cross_split_leakage",
            "has_missing_or_extra_prompt_ids",
            "has_group_split_base_key_leakage",
        ]
        if issue_flags[name]
    ]

    summary = {
        "split_file": split_file,
        "split_stem": split_path.stem,
        "split_name_values": sorted(map(str, split_df["split_name"].dropna().unique())),
        "split_type_values": sorted(map(str, split_df["split_type"].dropna().unique())),
        "row_count": int(len(split_df)),
        "split_counts": split_counts,
        "pass_by_split": pass_by_split.to_dict("records"),
        "prompt_id_hash_summary": prompt_id_summary,
        "prompt_id_hash_duplicate_by_split": duplicate_by_partition(split_df, "prompt_id"),
        "prompt_id_hash_cross_split_leakage": prompt_leakage,
        "base_key_summary": base_key_summary,
        "base_key_duplicate_by_split": duplicate_by_partition(split_df, "base_key"),
        "base_key_cross_split_leakage": key_leakage,
        "coverage_vs_all": coverage,
        "cluster_tvd_by_split": cluster_tvd.to_dict("records"),
        "top_cluster_share_drifts": top_cluster_drifts(cluster_counts, top_n),
        "issue_flags": issue_flags,
        "critical_issues": critical_issues,
    }

    frames = {
        "pass_by_split": pass_by_split.assign(split_file=split_file),
        "cluster_distribution": cluster_counts,
        "cluster_tvd": cluster_tvd,
        "prompt_id_duplicate_preview": prompt_dup_preview,
        "base_key_duplicate_preview": key_dup_preview,
        "prompt_id_leakage_preview": prompt_leak_preview,
        "base_key_leakage_preview": key_leak_preview,
        "coverage_missing_preview": missing_preview,
        "coverage_extra_preview": extra_preview,
    }

    split_summary_path = output_dir / f"{split_path.stem}.summary.json"
    write_json(summary, split_summary_path)
    summary["summary_path"] = str(split_summary_path)
    return summary, frames


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def main() -> None:
    args = parse_args()
    all_path = resolve_path(args.all_path)
    splits_dir = resolve_path(args.splits_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else splits_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.read_parquet(all_path)
    require_columns(
        all_df,
        [
            "prompt_id",
            "base_key",
            "cluster",
            "checker_pass",
            "prompt_tokens",
            "output_tokens",
            "output_truncated",
        ],
        path=all_path,
    )
    global_cluster_share = all_df["cluster"].value_counts(normalize=True, dropna=False).sort_index()
    split_paths = sorted(splits_dir.glob("*.parquet"))
    if not split_paths:
        raise FileNotFoundError(f"No split parquet files found under {splits_dir}")

    summaries: list[dict[str, Any]] = []
    frame_groups: dict[str, list[pd.DataFrame]] = {
        "pass_by_split": [],
        "cluster_distribution": [],
        "cluster_tvd": [],
        "prompt_id_duplicate_preview": [],
        "base_key_duplicate_preview": [],
        "prompt_id_leakage_preview": [],
        "base_key_leakage_preview": [],
        "coverage_missing_preview": [],
        "coverage_extra_preview": [],
    }
    for split_path in split_paths:
        summary, frames = audit_one_split(
            split_path,
            all_df,
            global_cluster_share,
            output_dir=output_dir,
            top_n=args.top_n,
        )
        summaries.append(summary)
        for name, frame in frames.items():
            frame_groups[name].append(frame)

    aggregate_frames = {name: concat_frames(frames) for name, frames in frame_groups.items()}
    output_paths = {
        "pass_by_split": write_csv(
            aggregate_frames["pass_by_split"],
            output_dir / "pass_by_split.csv",
        ),
        "cluster_distribution_by_split": write_csv(
            aggregate_frames["cluster_distribution"],
            output_dir / "cluster_distribution_by_split.csv",
        ),
        "cluster_tvd_by_split": write_csv(
            aggregate_frames["cluster_tvd"],
            output_dir / "cluster_tvd_by_split.csv",
        ),
        "prompt_id_hash_duplicate_preview": write_csv(
            aggregate_frames["prompt_id_duplicate_preview"],
            output_dir / "prompt_id_hash_duplicate_preview.csv",
        ),
        "base_key_duplicate_preview": write_csv(
            aggregate_frames["base_key_duplicate_preview"],
            output_dir / "base_key_duplicate_preview.csv",
        ),
        "prompt_id_hash_cross_split_leakage_preview": write_csv(
            aggregate_frames["prompt_id_leakage_preview"],
            output_dir / "prompt_id_hash_cross_split_leakage_preview.csv",
        ),
        "base_key_cross_split_leakage_preview": write_csv(
            aggregate_frames["base_key_leakage_preview"],
            output_dir / "base_key_cross_split_leakage_preview.csv",
        ),
        "coverage_missing_prompt_id_preview": write_csv(
            aggregate_frames["coverage_missing_preview"],
            output_dir / "coverage_missing_prompt_id_preview.csv",
        ),
        "coverage_extra_prompt_id_preview": write_csv(
            aggregate_frames["coverage_extra_preview"],
            output_dir / "coverage_extra_prompt_id_preview.csv",
        ),
    }

    critical_issue_count = sum(len(summary["critical_issues"]) for summary in summaries)
    split_overview = pd.DataFrame(
        [
            {
                "split_file": summary["split_file"],
                "row_count": summary["row_count"],
                "prompt_id_duplicate_rows": summary["prompt_id_hash_summary"][
                    "duplicate_row_count"
                ],
                "prompt_id_cross_split_leaked_values": summary[
                    "prompt_id_hash_cross_split_leakage"
                ]["leaked_value_count"],
                "base_key_duplicate_rows": summary["base_key_summary"][
                    "duplicate_row_count"
                ],
                "base_key_cross_split_leaked_values": summary[
                    "base_key_cross_split_leakage"
                ]["leaked_value_count"],
                "missing_prompt_ids": summary["coverage_vs_all"]["missing_prompt_id_count"],
                "extra_prompt_ids": summary["coverage_vs_all"]["extra_prompt_id_count"],
                "critical_issue_count": len(summary["critical_issues"]),
            }
            for summary in summaries
        ]
    )
    output_paths["split_duplicate_overview"] = write_csv(
        split_overview,
        output_dir / "split_duplicate_overview.csv",
    )

    all_prompt_id_summary = duplicate_summary(all_df, "prompt_id")
    all_base_key_summary = duplicate_summary(all_df, "base_key")
    global_summary = {
        "all_path": str(all_path),
        "splits_dir": str(splits_dir),
        "output_dir": str(output_dir),
        "all_row_count": int(len(all_df)),
        "all_prompt_id_hash_summary": all_prompt_id_summary,
        "all_base_key_summary": all_base_key_summary,
        "all_cluster_count": int(all_df["cluster"].nunique(dropna=False)),
        "all_cluster_distribution": (
            all_df["cluster"]
            .value_counts(dropna=False)
            .rename_axis("cluster")
            .reset_index(name="row_count")
            .sort_values("cluster")
            .to_dict("records")
        ),
        "split_files": [path.name for path in split_paths],
        "split_summaries": summaries,
        "critical_issue_count": int(critical_issue_count),
        "output_paths": output_paths,
    }
    summary_path = output_dir / "split_duplicate_cluster_audit.summary.json"
    output_paths["summary"] = write_json(global_summary, summary_path)

    print(json.dumps(global_summary, indent=2, ensure_ascii=False, default=json_default))
    if args.fail_on_critical and critical_issue_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
