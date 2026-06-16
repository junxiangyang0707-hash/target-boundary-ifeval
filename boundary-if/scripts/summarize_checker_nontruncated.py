from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize checker results excluding truncated outputs."
    )
    parser.add_argument(
        "--checks-dir",
        default="data/checks/qwen3_4b_instruct_2507_under2048_ifevalg_deterministic",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def rate_table(
    df: pd.DataFrame,
    group_columns: list[str],
    pass_column: str = "checker_pass",
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_columns, dropna=False)
        .agg(
            row_count=(pass_column, "size"),
            pass_count=(pass_column, "sum"),
            checker_error_rows=("checker_error_count", lambda s: int((s > 0).sum())),
            prompt_tokens_p50=("prompt_tokens", "median"),
            output_tokens_p50=("output_tokens", "median"),
            output_tokens_p90=("output_tokens", lambda s: float(np.percentile(s, 90))),
            num_constraints_mean=("num_constraints", "mean"),
        )
        .reset_index()
    )
    grouped["pass_rate"] = grouped["pass_count"] / grouped["row_count"]
    return grouped.sort_values(["pass_rate", "row_count"], ascending=[False, False])


def constraint_rate_table(constraints: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        constraints.groupby("instruction_id", dropna=False)
        .agg(
            row_count=("followed", "size"),
            pass_count=("followed", "sum"),
            checker_error_count=("checker_error", "sum"),
        )
        .reset_index()
    )
    grouped["pass_rate"] = grouped["pass_count"] / grouped["row_count"]
    grouped["checker_error_rate"] = grouped["checker_error_count"] / grouped["row_count"]
    return grouped.sort_values(["pass_rate", "row_count"], ascending=[False, False])


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False, encoding="utf-8")
    return str(path)


def records(frame: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    return frame.head(n).to_dict("records")


def main() -> None:
    args = parse_args()
    checks_dir = resolve_path(args.checks_dir)
    output_dir = (
        resolve_path(args.output_dir) if args.output_dir else checks_dir / "nontruncated_stats"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = pd.read_parquet(checks_dir / "checks_with_clusters.parquet")
    constraints = pd.read_parquet(checks_dir / "per_constraint.parquet")
    subset = checks[~checks["output_truncated"]].copy()
    constraints_subset = constraints.merge(
        subset[["under2048_index", "prompt_id"]],
        on=["under2048_index", "prompt_id"],
        how="inner",
        validate="many_to_one",
    )

    cluster_rates = rate_table(subset, ["cluster"])
    num_constraint_rates = rate_table(subset, ["num_constraints"])
    length_rates = rate_table(subset, ["length_bin"])
    family_rates = rate_table(subset, ["constraint_family_signature"])
    instruction_rates = constraint_rate_table(constraints_subset)

    output_paths = {
        "pass_rate_by_cluster_nontruncated": write_csv(
            cluster_rates,
            output_dir / "pass_rate_by_cluster_nontruncated.csv",
        ),
        "pass_rate_by_num_constraints_nontruncated": write_csv(
            num_constraint_rates,
            output_dir / "pass_rate_by_num_constraints_nontruncated.csv",
        ),
        "pass_rate_by_length_bin_nontruncated": write_csv(
            length_rates,
            output_dir / "pass_rate_by_length_bin_nontruncated.csv",
        ),
        "pass_rate_by_constraint_family_signature_nontruncated": write_csv(
            family_rates,
            output_dir / "pass_rate_by_constraint_family_signature_nontruncated.csv",
        ),
        "pass_rate_by_instruction_id_nontruncated": write_csv(
            instruction_rates,
            output_dir / "pass_rate_by_instruction_id_nontruncated.csv",
        ),
    }

    pass_count = int(subset["checker_pass"].sum())
    row_count = int(len(subset))
    constraint_pass_count = int(constraints_subset["followed"].sum())
    constraint_count = int(len(constraints_subset))
    high_risk_clusters = cluster_rates[cluster_rates["cluster"].isin([19, 6, 22, 15])]
    summary = {
        "source_checks_dir": str(checks_dir),
        "filter": "output_truncated == False",
        "checker_name": sorted(subset["checker_name"].dropna().unique().tolist()),
        "row_count": row_count,
        "excluded_truncated_count": int(checks["output_truncated"].sum()),
        "pass_count": pass_count,
        "fail_count": row_count - pass_count,
        "pass_rate": round(pass_count / row_count, 8),
        "checker_error_rows": int((subset["checker_error_count"] > 0).sum()),
        "constraint_check_count": constraint_count,
        "constraint_followed_count": constraint_pass_count,
        "constraint_pass_rate": round(constraint_pass_count / constraint_count, 8),
        "cluster_count": int(subset["cluster"].nunique()),
        "top_pass_rate_clusters": records(cluster_rates, 10),
        "lowest_pass_rate_clusters": records(
            cluster_rates.sort_values(["pass_rate", "row_count"], ascending=[True, False]),
            10,
        ),
        "high_truncation_risk_cluster_rates": high_risk_clusters.sort_values(
            "cluster"
        ).to_dict("records"),
        "num_constraints_rates": num_constraint_rates.sort_values("num_constraints").to_dict(
            "records"
        ),
        "length_bin_rates": length_rates.to_dict("records"),
        "lowest_instruction_pass_rates": records(
            instruction_rates.sort_values(["pass_rate", "row_count"], ascending=[True, False]),
            15,
        ),
        "top_instruction_pass_rates": records(instruction_rates, 15),
        "outputs": output_paths,
    }
    summary_path = output_dir / "nontruncated_summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
