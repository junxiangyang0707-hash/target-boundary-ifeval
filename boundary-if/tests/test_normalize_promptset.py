from __future__ import annotations

import json

from boundary_if.data.normalize_promptset import normalize_row, parse_ground_truth


def test_parse_ground_truth_extracts_instruction_ids() -> None:
    raw = (
        "[{'instruction_id': ['detectable_format:json_format', "
        "'keywords:forbidden_words'], 'kwargs': [None, {'forbidden_words': ['x']}]}]"
    )

    ground_truth_spec, instruction_ids, parse_ok, parse_error = parse_ground_truth(raw)

    assert parse_ok is True
    assert parse_error is None
    assert instruction_ids == ["detectable_format:json_format", "keywords:forbidden_words"]
    assert json.loads(ground_truth_spec)[0]["kwargs"][1]["forbidden_words"] == ["x"]


def test_normalize_row_generates_standard_fields() -> None:
    row = {
        "key": "sample-key",
        "messages": [{"role": "user", "content": "Answer in JSON."}],
        "ground_truth": "[{'instruction_id': ['keywords:forbidden_words'], 'kwargs': [None]}]",
        "dataset": "ifeval",
        "constraint_type": "multi",
        "constraint": "Do not use a forbidden word.",
    }

    normalized, parse_ok, parse_error = normalize_row(row)

    assert parse_ok is True
    assert parse_error is None
    assert len(normalized["prompt_id"]) == 64
    assert normalized["base_key"] == "sample-key"
    assert normalized["raw_messages"] == row["messages"]
    assert normalized["user_prompt"] == "Answer in JSON."
    assert normalized["constraint_text"] == "Do not use a forbidden word."
    assert normalized["instruction_ids"] == ["keywords:forbidden_words"]
    assert normalized["constraint_signature"] == "keywords:forbidden_words"
    assert normalized["constraint_family_signature"] == "keywords"
    assert normalized["num_constraints"] == 1
    assert normalized["source_dataset"] == "ifeval"
    assert normalized["constraint_type"] == "multi"
