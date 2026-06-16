from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("runs/model_comparisons/strict_atomic_tokenizer_key_runs")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_neural_run(
    rows: list[dict[str, Any]],
    *,
    tokenizer_scope: str,
    model: str,
    curve_label: str,
    metrics_path: str,
    manifest_path: str,
) -> None:
    metrics = read_json(metrics_path)
    manifest = read_json(manifest_path)
    test = metrics["by_split"]["test"]
    val = metrics["by_split"]["val"]
    rows.append(
        {
            "tokenizer_scope": tokenizer_scope,
            "model": model,
            "curve_label": curve_label,
            "seed": 42,
            "actual_train_rows": int(manifest["data_counts"]["train_rows"]),
            "best_epoch": int(manifest["best"]["epoch"]),
            "val_auprc": float(val["auprc"]),
            "test_auroc": float(test["auroc"]),
            "test_auprc": float(test["auprc"]),
            "test_positive_rate": float(test["positive_rate"]),
            "test_auprc_lift": float(test["auprc"]) / float(test["positive_rate"]),
            "total_seconds": float(manifest["timing_seconds"]["total"]),
            "output_dir": manifest["output_files"]["metrics"].replace("/metrics.json", ""),
        }
    )


def build_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    strict_m1 = pd.read_csv(
        "runs/m1_strict_atomic_tokenizer_atomic_constraint_heldout_seed42_max2048/"
        "learning_curve_summary.csv"
    ).iloc[0]
    rows.append(
        {
            "tokenizer_scope": "atomic_train_only",
            "model": "M1 TF-IDF",
            "curve_label": "full",
            "seed": 42,
            "actual_train_rows": int(strict_m1.actual_train_rows),
            "best_epoch": None,
            "val_auprc": float(strict_m1.val_auprc),
            "test_auroc": float(strict_m1.test_auroc),
            "test_auprc": float(strict_m1.test_auprc),
            "test_positive_rate": float(strict_m1.test_positive_rate),
            "test_auprc_lift": float(strict_m1.test_auprc_lift),
            "total_seconds": float(strict_m1.total_seconds),
            "output_dir": str(strict_m1.output_dir),
        }
    )
    append_neural_run(
        rows,
        tokenizer_scope="atomic_train_only",
        model="M3 mean",
        curve_label="40k",
        metrics_path="runs/strict_atomic_tokenizer_key_runs/m3_mean_40k_seed42/metrics.json",
        manifest_path="runs/strict_atomic_tokenizer_key_runs/m3_mean_40k_seed42/manifest.json",
    )
    append_neural_run(
        rows,
        tokenizer_scope="atomic_train_only",
        model="M3 mean",
        curve_label="full",
        metrics_path="runs/strict_atomic_tokenizer_key_runs/m3_mean_full_seed42/metrics.json",
        manifest_path="runs/strict_atomic_tokenizer_key_runs/m3_mean_full_seed42/manifest.json",
    )
    append_neural_run(
        rows,
        tokenizer_scope="atomic_train_only",
        model="M4 frozen",
        curve_label="full",
        metrics_path="runs/strict_atomic_tokenizer_key_runs/m4_frozen_full_seed42/metrics.json",
        manifest_path="runs/strict_atomic_tokenizer_key_runs/m4_frozen_full_seed42/manifest.json",
    )

    old_m1 = pd.read_csv(
        "runs/m1_learning_curve_atomic_constraint_heldout_seed42_max2048/"
        "learning_curve_summary.csv"
    )
    old_m1_full = old_m1[old_m1["curve_label"].eq("80k/full")].iloc[0]
    rows.append(
        {
            "tokenizer_scope": "group_key_train_tokenizer",
            "model": "M1 TF-IDF",
            "curve_label": "full",
            "seed": 42,
            "actual_train_rows": int(old_m1_full.actual_train_rows),
            "best_epoch": None,
            "val_auprc": float(old_m1_full.val_auprc),
            "test_auroc": float(old_m1_full.test_auroc),
            "test_auprc": float(old_m1_full.test_auprc),
            "test_positive_rate": float(old_m1_full.test_positive_rate),
            "test_auprc_lift": float(old_m1_full.test_auprc_lift),
            "total_seconds": float(old_m1_full.total_seconds),
            "output_dir": str(old_m1_full.output_dir),
        }
    )

    old_m3 = pd.read_csv(
        "runs/m3_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
        "multiseed_learning_curve_summary.csv"
    )
    for _, row in old_m3[
        old_m3["train_seed"].eq(42) & old_m3["curve_label"].isin(["40k", "full"])
    ].iterrows():
        rows.append(
            {
                "tokenizer_scope": "group_key_train_tokenizer",
                "model": "M3 mean",
                "curve_label": str(row.curve_label),
                "seed": 42,
                "actual_train_rows": int(row.actual_train_rows),
                "best_epoch": int(row.best_epoch),
                "val_auprc": float(row.val_auprc),
                "test_auroc": float(row.test_auroc),
                "test_auprc": float(row.test_auprc),
                "test_positive_rate": float(row.test_positive_rate),
                "test_auprc_lift": float(row.test_auprc_lift),
                "total_seconds": float(row.total_seconds),
                "output_dir": str(row.output_dir),
            }
        )

    old_m4 = pd.read_csv(
        "runs/m4_initial_mean_multiseed_learning_curve_atomic_constraint_heldout_max2048/"
        "multiseed_learning_curve_summary.csv"
    )
    old_m4_full = old_m4[old_m4["train_seed"].eq(42) & old_m4["curve_label"].eq("full")].iloc[0]
    rows.append(
        {
            "tokenizer_scope": "group_key_train_tokenizer",
            "model": "M4 frozen",
            "curve_label": "full",
            "seed": 42,
            "actual_train_rows": int(old_m4_full.actual_train_rows),
            "best_epoch": int(old_m4_full.best_epoch),
            "val_auprc": float(old_m4_full.val_auprc),
            "test_auroc": float(old_m4_full.test_auroc),
            "test_auprc": float(old_m4_full.test_auprc),
            "test_positive_rate": float(old_m4_full.test_positive_rate),
            "test_auprc_lift": float(old_m4_full.test_auprc_lift),
            "total_seconds": float(old_m4_full.total_seconds),
            "output_dir": str(old_m4_full.output_dir),
        }
    )

    summary = pd.DataFrame(rows)
    summary["run_key"] = summary["model"] + " " + summary["curve_label"]
    return summary.sort_values(["run_key", "tokenizer_scope"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame) -> None:
    order = ["M1 TF-IDF full", "M3 mean 40k", "M3 mean full", "M4 frozen full"]
    plot_df = summary[summary["run_key"].isin(order)].copy()
    plot_df["run_key"] = pd.Categorical(plot_df["run_key"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["run_key", "tokenizer_scope"])
    colors = {
        "group_key_train_tokenizer": "#7c8db5",
        "atomic_train_only": "#2a9d8f",
    }
    labels = {
        "group_key_train_tokenizer": "old group-tokenizer",
        "atomic_train_only": "strict atomic-tokenizer",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["test_auroc", "test_auprc"],
        ["Test AUROC", "Test AUPRC"],
        strict=True,
    ):
        x_values = range(len(order))
        width = 0.36
        for offset, scope in [(-width / 2, "group_key_train_tokenizer"), (width / 2, "atomic_train_only")]:
            values: list[float] = []
            for key in order:
                selected = plot_df[
                    (plot_df["run_key"].astype(str) == key)
                    & (plot_df["tokenizer_scope"] == scope)
                ]
                values.append(float(selected[metric].iloc[0]) if len(selected) else float("nan"))
            ax.bar(
                [idx + offset for idx in x_values],
                values,
                width=width,
                label=labels[scope],
                color=colors[scope],
                alpha=0.92,
            )
            for idx, value in enumerate(values):
                if pd.notna(value):
                    ax.text(
                        idx + offset,
                        value + 0.006,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
        ax.set_title(title)
        ax.set_ylim(0.25 if metric == "test_auprc" else 0.75, 0.88 if metric == "test_auroc" else 0.56)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(list(x_values))
        ax.set_xticklabels(order, rotation=20, ha="right")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Strict atomic-train tokenizer vs old group-key tokenizer (seed 42)")
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(OUTPUT_DIR / "strict_vs_group_tokenizer_seed42.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "strict_vs_group_tokenizer_seed42.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary.to_csv(OUTPUT_DIR / "strict_atomic_tokenizer_key_results.csv", index=False)
    summary.to_json(
        OUTPUT_DIR / "strict_atomic_tokenizer_key_results.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    plot_summary(summary)
    print(
        summary[
            [
                "tokenizer_scope",
                "model",
                "curve_label",
                "actual_train_rows",
                "best_epoch",
                "val_auprc",
                "test_auroc",
                "test_auprc",
                "test_auprc_lift",
                "total_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"wrote {OUTPUT_DIR / 'strict_atomic_tokenizer_key_results.csv'}")
    print(f"wrote {OUTPUT_DIR / 'strict_vs_group_tokenizer_seed42.png'}")


if __name__ == "__main__":
    main()
