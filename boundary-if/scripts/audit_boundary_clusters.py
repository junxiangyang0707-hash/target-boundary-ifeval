from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_GENERATION_DIRS = [
    "qwen3_4b_instruct_2507_under2048_0_10000_max2048",
    "qwen3_4b_instruct_2507_under2048_10000_20000_max2048",
    "qwen3_4b_instruct_2507_under2048_20000_30000_max2048",
    "qwen3_4b_instruct_2507_under2048_30000_40000_max2048",
    "qwen3_4b_instruct_2507_under2048_40000_50000_max2048",
    "qwen3_4b_instruct_2507_under2048_50000_60000_max2048",
    "qwen3_4b_instruct_2507_under2048_60000_70000_max2048",
    "qwen3_4b_instruct_2507_under2048_70000_80000_max2048",
    "qwen3_4b_instruct_2507_under2048_80000_94390_max2048",
]

TARGET_CLUSTERS = [7, 5, 9, 20, 18]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample boundary clusters and summarize checker failures."
    )
    parser.add_argument("--generation-root", default="data/generations")
    parser.add_argument("--generation-dirs", nargs="*", default=DEFAULT_GENERATION_DIRS)
    parser.add_argument(
        "--checks-dir",
        default="data/checks/qwen3_4b_instruct_2507_under2048_ifevalg_deterministic",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/checks/qwen3_4b_instruct_2507_under2048_ifevalg_deterministic/"
            "cluster_boundary_audit"
        ),
    )
    parser.add_argument("--clusters", nargs="*", type=int, default=TARGET_CLUSTERS)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-truncated", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def read_generation_outputs(root: Path, names: list[str]) -> pd.DataFrame:
    columns = [
        "under2048_index",
        "prompt_id",
        "user_prompt",
        "constraint_text",
        "ground_truth_spec",
        "instruction_ids_json",
        "response_text",
    ]
    frames = []
    for name in names:
        path = root / name / "outputs.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["generation_dir"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_kwargs_json(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    return parsed or {}


def constraint_target_summary(constraints: pd.DataFrame) -> str:
    parts = []
    for row in constraints.sort_values("constraint_index").itertuples(index=False):
        status = "PASS" if bool(row.followed) else "FAIL"
        parts.append(
            f"{row.constraint_index}:{status}:{row.instruction_id} kwargs={row.kwargs_json}"
        )
    return " | ".join(parts)


def failed_summary(constraints: pd.DataFrame) -> str:
    failed = constraints[~constraints["followed"]].sort_values("constraint_index")
    if failed.empty:
        return "none"
    return " | ".join(
        f"{row.constraint_index}:{row.instruction_id} kwargs={row.kwargs_json}"
        for row in failed.itertuples(index=False)
    )


def is_invalid_nth(row: pd.Series) -> bool:
    if row["instruction_id"] != "length_constraints:nth_paragraph_first_word":
        return False
    kwargs = parse_kwargs_json(row["kwargs_json"])
    num_paragraphs = kwargs.get("num_paragraphs")
    nth_paragraph = kwargs.get("nth_paragraph")
    return (
        num_paragraphs is not None
        and nth_paragraph is not None
        and int(nth_paragraph) > int(num_paragraphs)
    )


def choose_cluster_sample(
    cluster_rows: pd.DataFrame,
    cluster: int,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    pass_rows = cluster_rows[cluster_rows["checker_pass"]]
    fail_rows = cluster_rows[~cluster_rows["checker_pass"]]
    pass_rate = float(cluster_rows["checker_pass"].mean())
    pass_target = 20 if pass_rate >= 0.2 else 10
    pass_take = min(pass_target, len(pass_rows), sample_size // 2)
    fail_take = min(sample_size - pass_take, len(fail_rows))
    extra_pass_take = min(sample_size - pass_take - fail_take, len(pass_rows) - pass_take)

    pieces = []
    if pass_take:
        pieces.append(pass_rows.sample(n=pass_take, random_state=seed + cluster * 10))
    if fail_take:
        pieces.append(fail_rows.sample(n=fail_take, random_state=seed + cluster * 10 + 1))
    if extra_pass_take:
        remaining = pass_rows.drop(pieces[0].index if pieces else [])
        pieces.append(remaining.sample(n=extra_pass_take, random_state=seed + cluster * 10 + 2))
    sample = pd.concat(pieces, ignore_index=False)
    return sample.sort_values(["checker_pass", "under2048_index"], ascending=[False, True])


def truncate_text(value: Any, limit: int = 1800) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n...[truncated in audit, {omitted} chars omitted]"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            text = str(value).replace("\n", " ")
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_cluster_summary(
    checks: pd.DataFrame,
    constraints: pd.DataFrame,
    clusters: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []
    issues = []
    for cluster in clusters:
        cluster_checks = checks[checks["cluster"].eq(cluster)]
        cluster_constraints = constraints.merge(
            cluster_checks[["under2048_index", "prompt_id", "cluster"]],
            on=["under2048_index", "prompt_id"],
            how="inner",
            validate="many_to_one",
        )
        failed = cluster_constraints[~cluster_constraints["followed"]]
        invalid_nth = cluster_constraints.apply(is_invalid_nth, axis=1)
        rows.append(
            {
                "cluster": cluster,
                "row_count": int(len(cluster_checks)),
                "pass_count": int(cluster_checks["checker_pass"].sum()),
                "pass_rate": float(cluster_checks["checker_pass"].mean()),
                "constraint_count": int(len(cluster_constraints)),
                "failed_constraint_count": int(len(failed)),
                "num_constraints_mean": float(cluster_checks["num_constraints"].mean()),
                "prompt_tokens_p50": float(cluster_checks["prompt_tokens"].median()),
                "output_tokens_p50": float(cluster_checks["output_tokens"].median()),
                "output_tokens_p90": float(cluster_checks["output_tokens"].quantile(0.9)),
                "invalid_nth_constraint_count": int(invalid_nth.sum()),
            }
        )
        failure_counts = (
            failed.groupby("instruction_id")
            .size()
            .reset_index(name="fail_count")
            .sort_values("fail_count", ascending=False)
        )
        failure_total = max(1, int(len(failed)))
        for row in failure_counts.head(20).itertuples(index=False):
            failures.append(
                {
                    "cluster": cluster,
                    "instruction_id": row.instruction_id,
                    "fail_count": int(row.fail_count),
                    "failed_constraint_share": int(row.fail_count) / failure_total,
                }
            )
        issues.append(
            {
                "cluster": cluster,
                "checker_error_rows": int((cluster_checks["checker_error_count"] > 0).sum()),
                "invalid_nth_constraint_count": int(invalid_nth.sum()),
                "num_constraints_mismatch_rows": int(
                    (
                        cluster_checks["num_constraints"]
                        != cluster_checks["num_constraints_checked"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(failures), pd.DataFrame(issues)


def write_markdown_report(
    path: Path,
    sample: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    issue_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Boundary Cluster Sample Audit",
        "",
        "Scope: deterministic checker results, `output_truncated=False` unless noted otherwise.",
        "",
        "## Cluster Summary",
        "",
        markdown_table(cluster_summary),
        "",
        "## Top Failed Instructions",
        "",
    ]
    for cluster in sorted(sample["cluster"].unique()):
        lines.extend(
            [
                f"### C{cluster}",
                "",
                markdown_table(failure_summary[failure_summary["cluster"].eq(cluster)].head(12)),
                "",
            ]
        )
    lines.extend(
        [
            "## Data Construction Signals",
            "",
            markdown_table(issue_summary),
            "",
            "## Samples",
            "",
        ]
    )
    for cluster in sorted(sample["cluster"].unique()):
        cluster_sample = sample[sample["cluster"].eq(cluster)]
        pass_count = int(cluster_sample["checker_pass"].sum())
        lines.extend(
            [
                f"## C{cluster} Samples",
                "",
                f"Sample rows: {len(cluster_sample)}, pass rows: {pass_count}",
                "",
            ]
        )
        for row in cluster_sample.itertuples(index=False):
            lines.extend(
                [
                    f"### C{cluster} idx={row.under2048_index} pass={row.checker_pass}",
                    "",
                    f"- prompt_id: `{row.prompt_id}`",
                    f"- prompt_tokens: `{row.prompt_tokens}`",
                    f"- output_tokens: `{row.output_tokens}`",
                    f"- num_constraints: `{row.num_constraints}`",
                    f"- followed/checked: `{row.followed_count}/{row.num_constraints_checked}`",
                    f"- failure summary: {row.failed_summary}",
                    "",
                    "Target output constraints:",
                    "",
                    "```text",
                    truncate_text(row.constraint_text, limit=1500),
                    "```",
                    "",
                    "Parsed target spec:",
                    "",
                    "```text",
                    truncate_text(row.target_summary, limit=1500),
                    "```",
                    "",
                    "Prompt:",
                    "",
                    "```text",
                    truncate_text(row.user_prompt, limit=1800),
                    "```",
                    "",
                    "Model output excerpt:",
                    "",
                    "```text",
                    truncate_text(row.response_text, limit=1800),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    checks_dir = resolve_path(args.checks_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = pd.read_parquet(checks_dir / "checks_with_clusters.parquet")
    if not args.include_truncated:
        checks = checks[~checks["output_truncated"]].copy()
    checks = checks[checks["cluster"].isin(args.clusters)].copy()
    constraints = pd.read_parquet(checks_dir / "per_constraint.parquet")
    outputs = read_generation_outputs(resolve_path(args.generation_root), args.generation_dirs)

    cluster_summary, failure_summary, issue_summary = build_cluster_summary(
        checks,
        constraints,
        args.clusters,
    )

    sample_pieces = []
    for cluster in args.clusters:
        cluster_rows = checks[checks["cluster"].eq(cluster)]
        sample_pieces.append(
            choose_cluster_sample(cluster_rows, cluster, args.sample_size, args.seed)
        )
    sample = pd.concat(sample_pieces, ignore_index=True)
    sample_constraints = constraints.merge(
        sample[["under2048_index", "prompt_id"]],
        on=["under2048_index", "prompt_id"],
        how="inner",
        validate="many_to_one",
    )
    target_by_prompt = {
        prompt_id: constraint_target_summary(frame)
        for prompt_id, frame in sample_constraints.groupby("prompt_id")
    }
    failure_by_prompt = {
        prompt_id: failed_summary(frame)
        for prompt_id, frame in sample_constraints.groupby("prompt_id")
    }

    sample = sample.merge(
        outputs,
        on=["under2048_index", "prompt_id"],
        how="left",
        validate="one_to_one",
    )
    sample["target_summary"] = sample["prompt_id"].map(target_by_prompt)
    sample["failed_summary"] = sample["prompt_id"].map(failure_by_prompt)
    sample["sample_reason"] = sample["checker_pass"].map(
        {True: "pass-stratified", False: "fail-stratified"}
    )
    sample = sample.sort_values(
        ["cluster", "checker_pass", "under2048_index"],
        ascending=[True, False, True],
    )

    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False, encoding="utf-8")
    failure_summary.to_csv(output_dir / "failure_by_instruction.csv", index=False, encoding="utf-8")
    issue_summary.to_csv(output_dir / "data_issue_signals.csv", index=False, encoding="utf-8")
    sample.to_csv(output_dir / "cluster_boundary_samples.csv", index=False, encoding="utf-8")
    sample_constraints.to_csv(
        output_dir / "cluster_boundary_sample_constraints.csv",
        index=False,
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "cluster_boundary_samples.md",
        sample,
        cluster_summary,
        failure_summary,
        issue_summary,
    )

    summary = {
        "checks_dir": str(checks_dir),
        "filter": "all rows" if args.include_truncated else "output_truncated == False",
        "clusters": args.clusters,
        "sample_size_per_cluster_requested": args.sample_size,
        "sample_row_count": int(len(sample)),
        "cluster_summary": cluster_summary.to_dict("records"),
        "top_failures": failure_summary.groupby("cluster").head(5).to_dict("records"),
        "data_issue_signals": issue_summary.to_dict("records"),
        "outputs": {
            "cluster_summary": str(output_dir / "cluster_summary.csv"),
            "failure_by_instruction": str(output_dir / "failure_by_instruction.csv"),
            "data_issue_signals": str(output_dir / "data_issue_signals.csv"),
            "samples_csv": str(output_dir / "cluster_boundary_samples.csv"),
            "sample_constraints": str(output_dir / "cluster_boundary_sample_constraints.csv"),
            "samples_markdown": str(output_dir / "cluster_boundary_samples.md"),
            "summary": str(output_dir / "cluster_boundary_audit.summary.json"),
        },
    }
    (output_dir / "cluster_boundary_audit.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
