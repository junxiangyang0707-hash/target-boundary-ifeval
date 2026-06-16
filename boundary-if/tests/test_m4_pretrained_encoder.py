from __future__ import annotations

import pandas as pd
import torch

from boundary_if.models.m4_pretrained_encoder import (
    M4EncoderConfig,
    M4FrozenEncoderClassifier,
    M4PretrainingModel,
    compute_token_mask_categories,
    derive_structural_targets,
    make_mlm_batch,
    make_m4_classifier_dataloader,
    quote_inside_mask,
)


class DummyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        return_offsets_mapping: bool,
        verbose: bool,
    ):
        del add_special_tokens, truncation, return_offsets_mapping, verbose
        offsets = [(index, index + 1) for index in range(min(len(text), max_length))]
        return {"offset_mapping": offsets}


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(4)],
            "split": ["train"] * 4,
            "label": [0, 1, 0, 1],
            "input_ids": [[5, 6, 7], [8, 9, 10, 11], [12, 13], [14, 15, 16]],
            "raw_prompt_bpe_token_count_full": [3, 4, 2, 3],
            "raw_prompt_bpe_token_count": [3, 4, 2, 3],
            "raw_prompt_bpe_truncated": [False] * 4,
            "num_constraints": [1, 2, 3, 4],
            "cluster": [0, 0, 1, 1],
            "length_bin": ["001-128"] * 4,
        }
    )


def test_structural_targets_cover_requested_groups():
    labels, bucket = derive_structural_targets(
        [
            "length_constraints:number_words",
            "keywords:existence",
            "keywords:forbidden_words",
            "punctuation:no_comma",
            "detectable_format:json_format",
            "change_case:english_capital",
            "startend:end_checker",
            "detectable_format:number_bullet_lists",
        ],
        4,
    )

    assert labels == [1.0] * 8
    assert bucket == 3


def test_quote_and_digit_mask_categories():
    text = 'Say "abc 123" now'
    quoted_a = text.index("a", text.index('"'))
    inside = quote_inside_mask(text)
    assert any(inside[quoted_a : text.index("3") + 1])
    categories = compute_token_mask_categories(
        text=text,
        input_ids=list(range(len(text))),
        tokenizer=DummyTokenizer(),
        max_length=64,
    )
    assert categories[quoted_a] == 1
    assert categories[text.index("1")] == 1
    assert categories[text.index("S")] == 0


def test_mlm_batch_masks_digit_quote_category_at_higher_rate():
    input_ids = torch.tensor([[5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    mask_categories = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    masked_input_ids, mlm_labels, selected = make_mlm_batch(
        input_ids,
        attention_mask,
        mask_categories,
        base_mask_rate=0.01,
        high_mask_rate=0.99,
        mask_token_id=4,
        special_token_ids={0, 1},
    )

    assert bool(selected[0, 1])
    assert int(masked_input_ids[0, 1]) == 4
    assert int(mlm_labels[0, 1]) == 6


def test_m4_pretraining_and_classifier_forward_smoke():
    config = M4EncoderConfig(
        vocab_size=64,
        max_length=8,
        hidden_size=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        classifier_hidden_size=16,
    )
    input_ids = torch.tensor([[5, 6, 7], [8, 9, config.resolved_pad_token_id]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    pretraining_model = M4PretrainingModel(config)
    outputs = pretraining_model(input_ids, attention_mask)
    assert outputs["mlm_logits"].shape == (2, 3, config.vocab_size)
    assert outputs["structural_logits"].shape == (2, 8)
    assert outputs["num_constraint_logits"].shape == (2, 4)

    classifier = M4FrozenEncoderClassifier(config)
    classifier.load_pretrained_encoder(pretraining_model.prompt_encoder.state_dict())
    logits = classifier(input_ids, attention_mask)
    assert logits.shape == (2,)
    assert not any(parameter.requires_grad for parameter in classifier.prompt_encoder.parameters())

    dataloader = make_m4_classifier_dataloader(
        make_frame(),
        config,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].sum().item() == 7
