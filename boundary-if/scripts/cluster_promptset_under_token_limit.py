from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster prompt rows under a token limit and write cluster audit artifacts."
    )
    parser.add_argument(
        "--promptset-file",
        default="data/promptsets/if_multi_constraints_upto5.normalized.parquet",
    )
    parser.add_argument(
        "--tokens-file",
        default=(
            "data/promptsets/"
            "if_multi_constraints_upto5.qwen3_4b_instruct_2507.tokens.parquet"
        ),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--n-clusters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument(
        "--assignments-file",
        default="data/reports/prompt_clusters_under2048_seed42.parquet",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/reports/prompt_clusters_under2048_seed42.summary.csv",
    )
    parser.add_argument(
        "--audit-file",
        default="data/reports/prompt_clusters_under2048_seed42.audit.json",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def length_bin_series(tokens: pd.Series) -> pd.Series:
    return pd.cut(
        tokens,
        bins=[0, 128, 256, 512, 1024, 2048],
        labels=["001-128", "129-256", "257-512", "513-1024", "1025-2047"],
        right=False,
        include_lowest=True,
    ).astype(str)


def percentile(values: pd.Series, q: float) -> float:
    return round(float(np.percentile(values.to_numpy(), q)), 2)


def top_values(values: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    counts = values.fillna("<missing>").astype(str).value_counts().head(limit)
    return [{"value": value, "count": int(count)} for value, count in counts.items()]


def make_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in df.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": int(cluster_id),
                "row_count": int(len(group)),
                "share": round(len(group) / len(df), 6),
                "prompt_tokens_min": int(group["prompt_tokens"].min()),
                "prompt_tokens_p50": percentile(group["prompt_tokens"], 50),
                "prompt_tokens_p90": percentile(group["prompt_tokens"], 90),
                "prompt_tokens_max": int(group["prompt_tokens"].max()),
                "num_constraints_mean": round(float(group["num_constraints"].mean()), 3),
                "top_family_signatures": json.dumps(
                    top_values(group["constraint_family_signature"], 3),
                    ensure_ascii=False,
                ),
                "top_constraint_signatures": json.dumps(
                    top_values(group["constraint_signature"], 3),
                    ensure_ascii=False,
                ),
                "length_bin_counts": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in group["length_bin"].value_counts().sort_index().items()
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("row_count", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    total_start = now()
    promptset_file = resolve_path(args.promptset_file)
    tokens_file = resolve_path(args.tokens_file)
    assignments_file = resolve_path(args.assignments_file)
    summary_csv = resolve_path(args.summary_csv)
    audit_file = resolve_path(args.audit_file)
    assignments_file.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    load_start = now()
    promptset = pd.read_parquet(
        promptset_file,
        columns=[
            "prompt_id",
            "user_prompt",
            "num_constraints",
            "constraint_signature",
            "constraint_family_signature",
        ],
    )
    tokens = pd.read_parquet(tokens_file, columns=["prompt_id", "prompt_tokens"])
    df = promptset.merge(tokens, on="prompt_id", how="inner", validate="one_to_one")
    df = df[df["prompt_tokens"] < args.max_prompt_tokens].copy()
    df["length_bin"] = length_bin_series(df["prompt_tokens"])
    load_seconds = elapsed_since(load_start)

    vector_start = now()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.9,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    features = vectorizer.fit_transform(df["user_prompt"].astype(str))
    vector_seconds = elapsed_since(vector_start)

    cluster_start = now()
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=4096,
        n_init=5,
        reassignment_ratio=0.01,
    )
    df["cluster"] = kmeans.fit_predict(features).astype(np.int16)
    cluster_seconds = elapsed_since(cluster_start)

    silhouette = None
    if args.silhouette_sample_size > 0 and len(df) > args.n_clusters:
        silhouette_start = now()
        sample_size = min(args.silhouette_sample_size, len(df))
        try:
            silhouette = {
                "sample_size": int(sample_size),
                "score": round(
                    float(
                        silhouette_score(
                            features,
                            df["cluster"].to_numpy(),
                            metric="cosine",
                            sample_size=sample_size,
                            random_state=args.seed,
                        )
                    ),
                    6,
                ),
                "seconds": elapsed_since(silhouette_start),
            }
        except Exception as exc:
            silhouette = {
                "sample_size": int(sample_size),
                "score": None,
                "error": str(exc),
                "seconds": elapsed_since(silhouette_start),
            }

    assignments = df[
        [
            "prompt_id",
            "prompt_tokens",
            "length_bin",
            "cluster",
            "num_constraints",
            "constraint_signature",
            "constraint_family_signature",
        ]
    ].copy()
    assignments.to_parquet(assignments_file, index=False)

    cluster_summary = make_cluster_summary(df)
    cluster_summary.to_csv(summary_csv, index=False, encoding="utf-8")

    audit = {
        "promptset_file": str(promptset_file),
        "tokens_file": str(tokens_file),
        "assignments_file": str(assignments_file),
        "summary_csv": str(summary_csv),
        "max_prompt_tokens": args.max_prompt_tokens,
        "row_count": int(len(df)),
        "n_clusters": args.n_clusters,
        "seed": args.seed,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "max_features": args.max_features,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "kmeans": {
            "type": "MiniBatchKMeans",
            "inertia": round(float(kmeans.inertia_), 4),
            "n_iter": int(kmeans.n_iter_),
        },
        "silhouette": silhouette,
        "length_distribution": {
            "min": int(df["prompt_tokens"].min()),
            "p50": percentile(df["prompt_tokens"], 50),
            "p90": percentile(df["prompt_tokens"], 90),
            "p95": percentile(df["prompt_tokens"], 95),
            "p99": percentile(df["prompt_tokens"], 99),
            "max": int(df["prompt_tokens"].max()),
        },
        "cluster_size": {
            "min": int(cluster_summary["row_count"].min()),
            "p50": percentile(cluster_summary["row_count"], 50),
            "max": int(cluster_summary["row_count"].max()),
        },
        "timing_seconds": {
            "load": load_seconds,
            "vectorize": vector_seconds,
            "cluster": cluster_seconds,
            "total": elapsed_since(total_start),
        },
    }
    audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
