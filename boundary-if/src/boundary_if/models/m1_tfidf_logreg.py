from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

MODEL_ID = "M1"
MODEL_NAME = "M1_raw_prompt_bpe_token_ngram_tfidf_logreg"


@dataclass(frozen=True)
class M1TfidfLogRegConfig:
    ngram_min: int = 1
    ngram_max: int = 3
    max_features: int | None = 500_000
    min_df: int = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    norm: str = "l2"
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "saga"
    max_iter: int = 2000
    class_weight: str | None = "balanced"
    random_state: int = 42
    n_jobs: int = -1
    threshold: float = 0.5
    drop_truncated: bool = False

    @property
    def ngram_range(self) -> tuple[int, int]:
        return (self.ngram_min, self.ngram_max)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ngram_range"] = list(self.ngram_range)
        return payload


def split_token_string(text: str) -> list[str]:
    return text.split()


def token_ids_to_ngram_text(token_ids: Any) -> str:
    if token_ids is None:
        return ""
    if isinstance(token_ids, np.ndarray):
        values = token_ids.tolist()
    else:
        values = list(token_ids)
    return " ".join(str(int(token_id)) for token_id in values)


def token_ids_column_to_texts(token_ids: pd.Series) -> list[str]:
    return [token_ids_to_ngram_text(ids) for ids in token_ids]


def build_m1_pipeline(config: M1TfidfLogRegConfig) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=split_token_string,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        norm=config.norm,
        dtype=np.float32,
    )
    classifier_kwargs: dict[str, Any] = {
        "C": config.C,
        "solver": config.solver,
        "max_iter": config.max_iter,
        "class_weight": config.class_weight,
        "random_state": config.random_state,
    }
    if config.penalty not in ("l2", "default", None):
        classifier_kwargs["penalty"] = config.penalty
    classifier = LogisticRegression(**classifier_kwargs)
    return Pipeline([("tfidf", vectorizer), ("logreg", classifier)])


def fit_m1_pipeline(train_frame: pd.DataFrame, config: M1TfidfLogRegConfig) -> Pipeline:
    pipeline = build_m1_pipeline(config)
    texts = token_ids_column_to_texts(train_frame["input_ids"])
    labels = train_frame["label"].astype(int).to_numpy()
    pipeline.fit(texts, labels)
    return pipeline


def predict_m1(pipeline: Pipeline, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    texts = token_ids_column_to_texts(frame["input_ids"])
    probabilities = pipeline.predict_proba(texts)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    output = frame[
        [
            "prompt_id",
            "split",
            "label",
            "raw_prompt_bpe_token_count_full",
            "raw_prompt_bpe_token_count",
            "raw_prompt_bpe_truncated",
            "num_constraints",
            "cluster",
            "length_bin",
        ]
    ].copy()
    output["m1_pred_proba"] = probabilities.astype(np.float32)
    output["m1_pred_label"] = predictions
    return output


def binary_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = labels.astype(int)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "row_count": int(labels.size),
        "positive_count": int(labels.sum()),
        "positive_rate": round(float(labels.mean()), 8),
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(labels, predictions)), 8),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 8),
        "precision": round(
            float(precision_score(labels, predictions, zero_division=0)),
            8,
        ),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 8),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 8),
        "brier": round(float(brier_score_loss(labels, probabilities)), 8),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1])), 8),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    if len(np.unique(labels)) == 2:
        metrics["auroc"] = round(float(roc_auc_score(labels, probabilities)), 8)
        metrics["auprc"] = round(float(average_precision_score(labels, probabilities)), 8)
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return metrics


def evaluate_predictions(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = predictions["label"].astype(int).to_numpy()
    probabilities = predictions["m1_pred_proba"].astype(float).to_numpy()
    return binary_classification_metrics(labels, probabilities, threshold=threshold)


def evaluate_by_split(predictions: pd.DataFrame, threshold: float) -> dict[str, dict[str, Any]]:
    split_metrics: dict[str, dict[str, Any]] = {}
    for split, split_frame in predictions.groupby("split", sort=True):
        split_metrics[str(split)] = evaluate_predictions(split_frame, threshold)
    return split_metrics


def make_raw_bpe_length_bucket(values: pd.Series) -> pd.Series:
    bins = [0, 64, 128, 256, 512, 1024, 2048, np.inf]
    labels = ["0001-0064", "0065-0128", "0129-0256", "0257-0512", "0513-1024", "1025-2048", "2049+"]
    return pd.cut(values.astype(int), bins=bins, labels=labels, right=True, include_lowest=True)


def feature_summary(pipeline: Pipeline, top_n: int = 50) -> pd.DataFrame:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["logreg"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    top_positive = np.argsort(coefficients)[-top_n:][::-1]
    top_negative = np.argsort(coefficients)[:top_n]
    rows = []
    for rank, index in enumerate(top_positive, start=1):
        rows.append(
            {
                "direction": "positive",
                "rank": rank,
                "feature": feature_names[index],
                "coefficient": float(coefficients[index]),
            }
        )
    for rank, index in enumerate(top_negative, start=1):
        rows.append(
            {
                "direction": "negative",
                "rank": rank,
                "feature": feature_names[index],
                "coefficient": float(coefficients[index]),
            }
        )
    return pd.DataFrame(rows)


def pipeline_size_summary(pipeline: Pipeline) -> dict[str, Any]:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["logreg"]
    vocabulary_size = len(vectorizer.vocabulary_)
    coef = classifier.coef_
    return {
        "vocabulary_size": int(vocabulary_size),
        "coefficient_shape": list(coef.shape),
        "coefficient_nonzero_count": int(np.count_nonzero(coef)),
        "idf_dtype": str(vectorizer.idf_.dtype),
        "coef_dtype": str(coef.dtype),
        "sparse_supported": sparse.issparse(vectorizer.transform(["1 2 3"])),
    }
