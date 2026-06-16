from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from boundary_if.models.tiny_transformer import (
    M3TinyTransformer,
    M3TinyTransformerConfig,
    evaluate_m3_predictions,
    make_m3_dataloader,
    predict_m3,
)
from boundary_if.training.sampling import stable_sample_frame


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(6)],
            "split": ["train"] * 6,
            "label": [0, 0, 0, 1, 1, 1],
            "input_ids": [
                [1, 2, 3],
                np.array([4, 5, 6, 7], dtype=np.int32),
                [8, 9],
                [20, 21, 22],
                [23, 24, 25, 26],
                [27, 28],
            ],
            "raw_prompt_bpe_token_count_full": [3, 4, 2, 3, 4, 2],
            "raw_prompt_bpe_token_count": [3, 4, 2, 3, 4, 2],
            "raw_prompt_bpe_truncated": [False] * 6,
            "num_constraints": [1, 2, 1, 2, 3, 3],
            "cluster": [0, 0, 1, 1, 2, 2],
            "length_bin": ["001-128"] * 6,
        }
    )


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_m3_forward_and_predict_smoke(pooling: str):
    config = M3TinyTransformerConfig(
        vocab_size=64,
        max_length=8,
        hidden_size=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        pooling=pooling,
        classifier_hidden_size=16,
    )
    frame = make_frame()
    dataloader = make_m3_dataloader(
        frame,
        config,
        batch_size=3,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape == (3, 4)
    assert batch["attention_mask"].sum().item() == 9
    assert int(batch["input_ids"][2, 2]) == config.resolved_pad_token_id

    model = M3TinyTransformer(config)
    cls_state, token_states, token_mask = model.encode_prompt(
        batch["input_ids"],
        batch["attention_mask"],
    )
    assert cls_state.shape == (3, config.hidden_size)
    assert token_states.shape == (3, 4, config.hidden_size)
    assert token_mask.shape == batch["attention_mask"].shape

    pooled = model.pool_prompt(batch["input_ids"], batch["attention_mask"])
    assert pooled.shape == (3, config.hidden_size)

    logits = model(batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (3,)
    pass_probabilities = model.predict_pass_probability(
        batch["input_ids"],
        batch["attention_mask"],
    )
    assert pass_probabilities.shape == (3,)
    assert bool(((pass_probabilities >= 0) & (pass_probabilities <= 1)).all())

    predictions = predict_m3(model, dataloader, device=torch.device("cpu"))
    metrics = evaluate_m3_predictions(predictions, threshold=0.5)
    assert predictions["m3_pred_proba"].between(0, 1).all()
    assert set(predictions["m3_pred_label"].unique()) <= {0, 1}
    assert metrics["row_count"] == 6
    assert metrics["auroc"] is not None
    assert metrics["auprc"] is not None


def test_stable_sample_frame_is_deterministic_and_nested():
    frame = pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(10)],
            "value": list(range(10)),
        }
    )

    small = stable_sample_frame(frame, sample_size=3, sample_seed=42)
    large = stable_sample_frame(frame, sample_size=6, sample_seed=42)
    repeated = stable_sample_frame(frame, sample_size=3, sample_seed=42)
    different_seed = stable_sample_frame(frame, sample_size=3, sample_seed=43)

    assert small["prompt_id"].tolist() == repeated["prompt_id"].tolist()
    assert set(small["prompt_id"]) <= set(large["prompt_id"])
    assert small["prompt_id"].tolist() != different_seed["prompt_id"].tolist()
