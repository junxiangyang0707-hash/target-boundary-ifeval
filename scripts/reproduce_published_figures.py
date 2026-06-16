from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
DEFAULT_OUTPUT = ROOT / "reproduced_figures"

COLORS = {
    "M1_TFIDF": "#F0986E",
    "M3_mean": "#A3BEFA",
    "M4_frozen": "#A3D576",
}
LABELS = {
    ("M1_TFIDF", "baseline", "full"): "M1 TF-IDF full",
    ("M3_mean", "mean_pooling", "40k"): "M3 mean 40k",
    ("M3_mean", "mean_pooling", "full"): "M3 mean full",
    ("M4_frozen", "frozen_encoder", "20k"): "M4 frozen 20k",
    ("M4_frozen", "frozen_encoder", "40k"): "M4 frozen 40k",
    ("M4_frozen", "frozen_encoder", "full"): "M4 frozen full",
}


def label_for(row: pd.Series) -> str:
    return LABELS.get(
        (row["model_family"], row["model_variant"], row["train_size_label"]),
        f"{row['model_family']} {row['train_size_label']}",
    )


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E8F0",
            "font.size": 10,
        }
    )


def plot_main(output: Path) -> None:
    df = pd.read_csv(TABLES / "run_metrics_by_config.csv")
    test = df[df["split"] == "test"].copy()
    test["display"] = test.apply(label_for, axis=1)
    order = list(LABELS.values())
    test["order"] = test["display"].map({name: i for i, name in enumerate(order)})
    test = test.sort_values("order")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        test["display"],
        test["AUPRC_mean"],
        xerr=test["AUPRC_std"].fillna(0),
        color=[COLORS.get(row["model_family"], "#999999") for _, row in test.iterrows()],
        edgecolor="#1F2430",
        capsize=3,
    )
    baseline = test["baseline_AUPRC_mean"].dropna().iloc[0]
    ax.axvline(baseline, color="#1F2430", linestyle=":", label=f"baseline {baseline:.1%}")
    ax.set_title("Strict atomic-tokenizer test AUPRC")
    ax.set_xlabel("Test AUPRC")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "figure4_strict_main_result_reproduced.png", dpi=180)
    plt.close(fig)


def plot_selective(output: Path) -> None:
    df = pd.read_csv(TABLES / "selective_metrics.csv")
    keep = df.apply(lambda r: (r["model_family"], r["model_variant"], r["train_size_label"]) in LABELS, axis=1)
    df = df[keep].copy()
    df["display"] = df.apply(label_for, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for display, part in df.groupby("display"):
        part = part.sort_values("coverage")
        ax.plot(part["coverage"], part["selective_accuracy_mean"], marker="o", label=display)
    ax.set_title("Selective accuracy")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure8_selective_accuracy_reproduced.png", dpi=180)
    plt.close(fig)


def plot_topk(output: Path) -> None:
    df = pd.read_csv(TABLES / "topk_metrics.csv")
    keep = df.apply(lambda r: (r["model_family"], r["model_variant"], r["train_size_label"]) in LABELS, axis=1)
    df = df[keep].copy()
    df["display"] = df.apply(label_for, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for display, part in df.groupby("display"):
        part = part.sort_values("k")
        ax.plot(part["k"], part["precision_at_k_mean"], marker="o", label=display)
    ax.set_title("Top-k precision")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure9_topk_precision_reproduced.png", dpi=180)
    plt.close(fig)


def plot_calibration(output: Path) -> None:
    df = pd.read_csv(TABLES / "calibration_bins.csv")
    df = df[df["split"] == "test"].copy()
    keep = df.apply(lambda r: (r["model_family"], r["model_variant"], r["train_size_label"]) in LABELS, axis=1)
    df = df[keep].copy()
    df["display"] = df.apply(label_for, axis=1)
    avg = (
        df.groupby(["display", "bin_midpoint"], as_index=False)
        .agg(avg_pred=("avg_pred", "mean"), empirical_pass_rate=("empirical_pass_rate", "mean"))
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle=":", color="#1F2430", label="perfect calibration")
    for display, part in avg.groupby("display"):
        part = part.sort_values("bin_midpoint")
        ax.plot(part["avg_pred"], part["empirical_pass_rate"], marker="o", label=display)
    ax.set_title("Calibration on strict atomic test")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical pass rate")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure10_calibration_reproduced.png", dpi=180)
    plt.close(fig)


def plot_features(output: Path) -> None:
    df = pd.read_csv(TABLES / "feature_coefficients.csv")
    pos = df[df["direction"] == "positive"].head(10)
    neg = df[df["direction"] == "negative"].head(10)
    plot = pd.concat([neg, pos], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#A3BEFA" if v < 0 else "#F0986E" for v in plot["coefficient"]]
    ax.barh(plot["decoded_feature"], plot["coefficient"], color=colors, edgecolor="#1F2430")
    ax.axvline(0, color="#1F2430", linewidth=1)
    ax.set_title("M1 TF-IDF feature coefficients")
    ax.set_xlabel("Logistic-regression coefficient")
    fig.tight_layout()
    fig.savefig(output / "figure11_m1_features_reproduced.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce core published figures from v0.2 tables.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    style()
    plot_main(output)
    plot_selective(output)
    plot_topk(output)
    plot_calibration(output)
    plot_features(output)
    print(f"Wrote reproduced figures to {output}")


if __name__ == "__main__":
    main()
