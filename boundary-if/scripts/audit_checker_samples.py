from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from boundary_if.checkers.ifevalg import ensure_nltk_data, run_checker

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

TARGET_CLUSTERS = [7, 5, 9, 19, 6, 22, 15]
SHADOW_INSTRUCTION_IDS = [
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "detectable_format:json_format",
    "detectable_format:title",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "punctuation:punctuation_dot",
    "punctuation:punctuation_exclamation",
    "startend:end_checker",
    "startend:quotation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stored checker decisions by sampling.")
    parser.add_argument("--generation-root", default="data/generations")
    parser.add_argument("--generation-dirs", nargs="*", default=DEFAULT_GENERATION_DIRS)
    parser.add_argument(
        "--checks-dir",
        default="data/checks/qwen3_4b_instruct_2507_under2048_ifevalg",
    )
    parser.add_argument(
        "--output-dir",
        default="data/checks/qwen3_4b_instruct_2507_under2048_ifevalg/audit",
    )
    parser.add_argument("--nltk-cache-dir", default=".cache/nltk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-samples-per-group", type=int, default=6)
    parser.add_argument("--cluster-samples-per-outcome", type=int, default=2)
    parser.add_argument("--shadow-samples-per-id", type=int, default=250)
    parser.add_argument("--recheck-sample-size", type=int, default=1000)
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
        "output_tokens",
        "output_truncated",
        "finish_reason",
    ]
    frames = []
    for name in names:
        path = root / name / "outputs.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["generation_dir"] = name
        frames.append(frame)
    outputs = pd.concat(frames, ignore_index=True)
    return outputs.sort_values("under2048_index", kind="mergesort").reset_index(drop=True)


def sample_group(
    df: pd.DataFrame,
    mask: pd.Series,
    n: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    candidates = df.loc[mask].copy()
    if candidates.empty:
        return candidates
    sampled = candidates.sample(n=min(n, len(candidates)), random_state=seed)
    sampled["audit_group"] = label
    return sampled


def choose_row_samples(
    checks: pd.DataFrame,
    seed: int,
    per_group: int,
    per_cluster: int,
) -> pd.DataFrame:
    samples = [
        sample_group(
            checks,
            checks["checker_pass"] & ~checks["output_truncated"],
            per_group,
            seed,
            "pass_not_truncated",
        ),
        sample_group(
            checks,
            ~checks["checker_pass"] & ~checks["output_truncated"],
            per_group,
            seed + 1,
            "fail_not_truncated",
        ),
        sample_group(
            checks,
            checks["checker_pass"] & checks["output_truncated"],
            per_group,
            seed + 2,
            "pass_truncated",
        ),
        sample_group(
            checks,
            ~checks["checker_pass"] & checks["output_truncated"],
            per_group,
            seed + 3,
            "fail_truncated",
        ),
    ]
    for cluster in TARGET_CLUSTERS:
        in_cluster = checks["cluster"].eq(cluster)
        samples.append(
            sample_group(
                checks,
                in_cluster & checks["checker_pass"],
                per_cluster,
                seed + cluster * 10,
                f"cluster_{cluster}_pass",
            )
        )
        samples.append(
            sample_group(
                checks,
                in_cluster & ~checks["checker_pass"],
                per_cluster,
                seed + cluster * 10 + 1,
                f"cluster_{cluster}_fail",
            )
        )
    rows = pd.concat([frame for frame in samples if not frame.empty], ignore_index=True)
    rows = rows.drop_duplicates("under2048_index", keep="first")
    return rows.sort_values(["audit_group", "under2048_index"], kind="mergesort")


def parse_kwargs(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected kwargs object, got {type(parsed).__name__}")
    return parsed


def shadow_keyword_existence(response: str, kwargs: dict[str, Any]) -> bool:
    return all(
        re.search(str(keyword), response, flags=re.IGNORECASE)
        for keyword in kwargs["keywords"]
    )


def shadow_keyword_frequency(response: str, kwargs: dict[str, Any]) -> bool:
    count = len(re.findall(str(kwargs["keyword"]), response, flags=re.IGNORECASE))
    frequency = int(kwargs["frequency"])
    relation = kwargs["relation"]
    if relation == "less than":
        return count < frequency
    if relation == "at least":
        return count >= frequency
    raise ValueError(f"Unsupported relation: {relation}")


def shadow_forbidden_words(response: str, kwargs: dict[str, Any]) -> bool:
    return all(
        not re.search(r"\b" + str(word) + r"\b", response, flags=re.IGNORECASE)
        for word in kwargs["forbidden_words"]
    )


def shadow_json_format(response: str, _kwargs: dict[str, Any]) -> bool:
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def shadow_title(response: str, _kwargs: dict[str, Any]) -> bool:
    titles = re.findall(r"<<[^\n]+>>", response)
    return any(title.lstrip("<").rstrip(">").strip() for title in titles)


def shadow_number_words(response: str, kwargs: dict[str, Any]) -> bool:
    count = len(re.findall(r"\w+", response))
    num_words = int(kwargs["num_words"])
    relation = kwargs["relation"]
    if relation == "less than":
        return count < num_words
    if relation == "at least":
        return count >= num_words
    raise ValueError(f"Unsupported relation: {relation}")


def shadow_no_comma(response: str, _kwargs: dict[str, Any]) -> bool:
    return "," not in response


def shadow_no_dot(response: str, _kwargs: dict[str, Any]) -> bool:
    return "." not in response


def shadow_no_exclamation(response: str, _kwargs: dict[str, Any]) -> bool:
    return "!" not in response


def shadow_end_checker(response: str, kwargs: dict[str, Any]) -> bool:
    value = response.strip().strip('"').lower()
    return value.endswith(str(kwargs["end_phrase"]).strip().lower())


def shadow_quotation(response: str, _kwargs: dict[str, Any]) -> bool:
    value = response.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


SHADOW_CHECKERS: dict[str, Callable[[str, dict[str, Any]], bool]] = {
    "keywords:existence": shadow_keyword_existence,
    "keywords:frequency": shadow_keyword_frequency,
    "keywords:forbidden_words": shadow_forbidden_words,
    "detectable_format:json_format": shadow_json_format,
    "detectable_format:title": shadow_title,
    "length_constraints:number_words": shadow_number_words,
    "punctuation:no_comma": shadow_no_comma,
    "punctuation:punctuation_dot": shadow_no_dot,
    "punctuation:punctuation_exclamation": shadow_no_exclamation,
    "startend:end_checker": shadow_end_checker,
    "startend:quotation": shadow_quotation,
}


def build_shadow_sample(
    constraints: pd.DataFrame,
    outputs: pd.DataFrame,
    seed: int,
    per_id: int,
) -> pd.DataFrame:
    pieces = []
    eligible = constraints[constraints["instruction_id"].isin(SHADOW_INSTRUCTION_IDS)]
    for offset, instruction_id in enumerate(SHADOW_INSTRUCTION_IDS):
        candidates = eligible[eligible["instruction_id"].eq(instruction_id)]
        if candidates.empty:
            continue
        pieces.append(candidates.sample(n=min(per_id, len(candidates)), random_state=seed + offset))
    sample = pd.concat(pieces, ignore_index=True)
    sample = sample.merge(
        outputs[["under2048_index", "prompt_id", "response_text"]],
        on=["under2048_index", "prompt_id"],
        how="left",
        validate="many_to_one",
    )
    records = []
    for row in sample.itertuples(index=False):
        kwargs = parse_kwargs(row.kwargs_json)
        response = "" if pd.isna(row.response_text) else str(row.response_text)
        shadow_followed = SHADOW_CHECKERS[row.instruction_id](response, kwargs)
        records.append(
            {
                "under2048_index": int(row.under2048_index),
                "prompt_id": str(row.prompt_id),
                "constraint_index": int(row.constraint_index),
                "instruction_id": str(row.instruction_id),
                "kwargs_json": str(row.kwargs_json),
                "official_followed": bool(row.followed),
                "shadow_followed": bool(shadow_followed),
                "agrees": bool(row.followed) == bool(shadow_followed),
            }
        )
    return pd.DataFrame(records)


def probe_case(
    instruction_id: str,
    kwargs: dict[str, Any],
    response: str,
    expected: bool,
) -> dict[str, Any]:
    spec = json.dumps(
        [{"instruction_id": [instruction_id], "kwargs": [kwargs]}],
        ensure_ascii=False,
    )
    result = run_checker(spec, response)
    return {
        "instruction_id": instruction_id,
        "kwargs": kwargs,
        "response": response,
        "expected": expected,
        "actual": bool(result["strict_pass"]),
        "passed": bool(result["strict_pass"]) == expected,
        "checker_error_count": int(result["checker_error_count"]),
        "per_constraint": result["per_constraint"],
    }


def run_probe_cases() -> pd.DataFrame:
    cases = [
        ("keywords:forbidden_words", {"forbidden_words": ["apple"]}, "banana only", True),
        ("keywords:forbidden_words", {"forbidden_words": ["apple"]}, "an apple appears", False),
        ("keywords:existence", {"keywords": ["alpha", "beta"]}, "alpha and beta", True),
        ("keywords:existence", {"keywords": ["alpha", "beta"]}, "alpha only", False),
        (
            "keywords:frequency",
            {"keyword": "ha", "frequency": 2, "relation": "at least"},
            "ha ha",
            True,
        ),
        (
            "keywords:frequency",
            {"keyword": "ha", "frequency": 2, "relation": "at least"},
            "ha",
            False,
        ),
        ("detectable_format:json_format", {}, '{"answer": 1}', True),
        ("detectable_format:json_format", {}, "answer: 1", False),
        ("detectable_format:title", {}, "<<A Title>>\nbody", True),
        ("detectable_format:title", {}, "A Title\nbody", False),
        (
            "length_constraints:number_words",
            {"num_words": 3, "relation": "at least"},
            "one two three",
            True,
        ),
        (
            "length_constraints:number_words",
            {"num_words": 3, "relation": "at least"},
            "one two",
            False,
        ),
        ("punctuation:no_comma", {}, "no comma here", True),
        ("punctuation:no_comma", {}, "comma, here", False),
        ("punctuation:punctuation_dot", {}, "no dot here", True),
        ("punctuation:punctuation_dot", {}, "dot.", False),
        ("punctuation:punctuation_exclamation", {}, "calm", True),
        ("punctuation:punctuation_exclamation", {}, "loud!", False),
        ("startend:end_checker", {"end_phrase": "Done."}, "This is Done.", True),
        ("startend:end_checker", {"end_phrase": "Done."}, "Done. extra", False),
        ("startend:quotation", {}, '"wrapped"', True),
        ("startend:quotation", {}, "not wrapped", False),
    ]
    return pd.DataFrame([probe_case(*case) for case in cases])


def recheck_rows(sample: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    merged = sample.merge(
        outputs[["under2048_index", "prompt_id", "ground_truth_spec", "response_text"]],
        on=["under2048_index", "prompt_id"],
        how="left",
        validate="one_to_one",
    )
    records = []
    for row in merged.itertuples(index=False):
        response = "" if pd.isna(row.response_text) else str(row.response_text)
        result = run_checker(str(row.ground_truth_spec), response)
        records.append(
            {
                "under2048_index": int(row.under2048_index),
                "prompt_id": str(row.prompt_id),
                "stored_checker_pass": bool(row.checker_pass),
                "rechecked_checker_pass": bool(result["strict_pass"]),
                "stored_followed_count": int(row.followed_count),
                "rechecked_followed_count": int(result["followed_count"]),
                "stored_failed_count": int(row.failed_count),
                "rechecked_failed_count": int(result["failed_count"]),
                "agrees": bool(row.checker_pass) == bool(result["strict_pass"])
                and int(row.followed_count) == int(result["followed_count"])
                and int(row.failed_count) == int(result["failed_count"]),
            }
        )
    return pd.DataFrame(records)


def truncate_text(value: Any, limit: int = 2200) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[truncated in audit, {len(text) - limit} chars omitted]"


def write_markdown_report(
    path: Path,
    row_samples: pd.DataFrame,
    constraints: pd.DataFrame,
    outputs: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    sample = row_samples.merge(
        outputs[
            [
                "under2048_index",
                "prompt_id",
                "user_prompt",
                "constraint_text",
                "ground_truth_spec",
                "response_text",
            ]
        ],
        on=["under2048_index", "prompt_id"],
        how="left",
        validate="one_to_one",
    )
    sample_constraints = constraints[constraints["prompt_id"].isin(sample["prompt_id"])]
    constraints_by_prompt = {
        prompt_id: frame.sort_values("constraint_index")
        for prompt_id, frame in sample_constraints.groupby("prompt_id")
    }

    lines = [
        "# Checker Sample Audit",
        "",
        "## Summary",
        "",
        f"- row_sample_count: {summary['row_sample_count']}",
        f"- shadow_sample_count: {summary['shadow_sample_count']}",
        f"- shadow_mismatch_count: {summary['shadow_mismatch_count']}",
        f"- recheck_sample_count: {summary['recheck_sample_count']}",
        f"- recheck_mismatch_count: {summary['recheck_mismatch_count']}",
        f"- probe_case_count: {summary['probe_case_count']}",
        f"- probe_failed_count: {summary['probe_failed_count']}",
        "",
        "## Row Samples",
        "",
    ]

    for row in sample.itertuples(index=False):
        lines.extend(
            [
                f"### {row.audit_group} | idx={row.under2048_index} | pass={row.checker_pass}",
                "",
                f"- prompt_id: `{row.prompt_id}`",
                f"- cluster: `{row.cluster}`",
                f"- num_constraints: `{row.num_constraints}`",
                f"- output_truncated: `{row.output_truncated}`",
                f"- output_tokens: `{row.output_tokens}`",
                f"- followed_count/checked: `{row.followed_count}/{row.num_constraints_checked}`",
                "",
                "Constraints:",
                "",
            ]
        )
        prompt_constraints = constraints_by_prompt.get(row.prompt_id, pd.DataFrame())
        for constraint in prompt_constraints.itertuples(index=False):
            lines.append(
                "- "
                f"{constraint.constraint_index}: `{constraint.instruction_id}`, "
                f"followed={constraint.followed}, kwargs=`{constraint.kwargs_json}`"
            )
        lines.extend(
            [
                "",
                "User prompt:",
                "",
                "```text",
                truncate_text(row.user_prompt),
                "```",
                "",
                "Constraint text:",
                "",
                "```text",
                truncate_text(row.constraint_text),
                "```",
                "",
                "Answer:",
                "",
                "```text",
                truncate_text(row.response_text),
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_nltk_data(resolve_path(args.nltk_cache_dir))
    checks_dir = resolve_path(args.checks_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = read_generation_outputs(resolve_path(args.generation_root), args.generation_dirs)
    checks = pd.read_parquet(checks_dir / "checks_with_clusters.parquet")
    constraints = pd.read_parquet(checks_dir / "per_constraint.parquet")

    row_samples = choose_row_samples(
        checks,
        seed=args.seed,
        per_group=args.row_samples_per_group,
        per_cluster=args.cluster_samples_per_outcome,
    )
    row_samples.to_csv(output_dir / "row_samples.csv", index=False, encoding="utf-8")

    shadow = build_shadow_sample(
        constraints,
        outputs,
        seed=args.seed,
        per_id=args.shadow_samples_per_id,
    )
    shadow.to_csv(output_dir / "shadow_check_sample.csv", index=False, encoding="utf-8")
    shadow_summary = (
        shadow.groupby("instruction_id", dropna=False)
        .agg(
            sample_count=("agrees", "size"),
            agreement_count=("agrees", "sum"),
            mismatch_count=("agrees", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    shadow_summary["agreement_rate"] = (
        shadow_summary["agreement_count"] / shadow_summary["sample_count"]
    )
    shadow_summary.to_csv(output_dir / "shadow_check_summary.csv", index=False, encoding="utf-8")
    shadow[~shadow["agrees"]].to_csv(
        output_dir / "shadow_check_mismatches.csv",
        index=False,
        encoding="utf-8",
    )

    probes = run_probe_cases()
    probes.to_json(output_dir / "probe_cases.json", orient="records", force_ascii=False, indent=2)

    recheck_source = checks.sample(
        n=min(args.recheck_sample_size, len(checks)),
        random_state=args.seed,
    )
    rechecked = recheck_rows(recheck_source, outputs)
    rechecked.to_csv(output_dir / "recheck_sample.csv", index=False, encoding="utf-8")

    summary = {
        "seed": args.seed,
        "row_sample_count": int(len(row_samples)),
        "shadow_instruction_ids": SHADOW_INSTRUCTION_IDS,
        "shadow_sample_count": int(len(shadow)),
        "shadow_mismatch_count": int((~shadow["agrees"]).sum()),
        "shadow_summary": shadow_summary.to_dict("records"),
        "probe_case_count": int(len(probes)),
        "probe_failed_count": int((~probes["passed"]).sum()),
        "recheck_sample_count": int(len(rechecked)),
        "recheck_mismatch_count": int((~rechecked["agrees"]).sum()),
        "outputs": {
            "row_samples": str(output_dir / "row_samples.csv"),
            "markdown_report": str(output_dir / "checker_sample_audit.md"),
            "shadow_check_sample": str(output_dir / "shadow_check_sample.csv"),
            "shadow_check_summary": str(output_dir / "shadow_check_summary.csv"),
            "shadow_check_mismatches": str(output_dir / "shadow_check_mismatches.csv"),
            "probe_cases": str(output_dir / "probe_cases.json"),
            "recheck_sample": str(output_dir / "recheck_sample.csv"),
            "summary": str(output_dir / "checker_sample_audit.summary.json"),
        },
    }
    write_markdown_report(
        output_dir / "checker_sample_audit.md",
        row_samples,
        constraints,
        outputs,
        summary,
    )
    (output_dir / "checker_sample_audit.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
