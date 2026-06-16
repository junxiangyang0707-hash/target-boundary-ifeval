from __future__ import annotations

import numpy as np
import pandas as pd

from boundary_if.models.m1_tfidf_logreg import (
    M1TfidfLogRegConfig,
    evaluate_predictions,
    fit_m1_pipeline,
    predict_m1,
    token_ids_to_ngram_text,
)


def test_token_ids_to_ngram_text_handles_arrays_and_lists():
    assert token_ids_to_ngram_text([1, 22, 333]) == "1 22 333"
    assert token_ids_to_ngram_text(np.array([4, 5, 6], dtype=np.int32)) == "4 5 6"


def test_m1_tfidf_logreg_smoke_fit_predict():
    frame = pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(8)],
            "split": ["train"] * 8,
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "input_ids": [
                [10, 11, 12, 13],
                [10, 11, 12, 14],
                [10, 15, 12, 13],
                [10, 11, 16, 13],
                [90, 91, 92, 93],
                [90, 91, 92, 94],
                [90, 95, 92, 93],
                [90, 91, 96, 93],
            ],
            "raw_prompt_bpe_token_count_full": [4] * 8,
            "raw_prompt_bpe_token_count": [4] * 8,
            "raw_prompt_bpe_truncated": [False] * 8,
            "num_constraints": [1] * 8,
            "cluster": [0] * 8,
            "length_bin": ["001-128"] * 8,
        }
    )
    config = M1TfidfLogRegConfig(
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        max_features=100,
        solver="liblinear",
        max_iter=200,
        class_weight=None,
    )

    pipeline = fit_m1_pipeline(frame, config)
    predictions = predict_m1(pipeline, frame, threshold=0.5)
    metrics = evaluate_predictions(predictions, threshold=0.5)

    assert set(predictions["m1_pred_label"].unique()) <= {0, 1}
    assert predictions["m1_pred_proba"].between(0, 1).all()
    assert metrics["row_count"] == 8
    assert metrics["auroc"] is not None
    assert metrics["auprc"] is not None
