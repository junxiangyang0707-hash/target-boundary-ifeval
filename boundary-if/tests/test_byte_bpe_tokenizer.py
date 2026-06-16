from __future__ import annotations

import pandas as pd

from boundary_if.tokenization.byte_bpe import (
    ByteBpeTrainingConfig,
    save_tokenizer_artifacts,
    train_tokenizer,
)


def test_train_raw_prompt_byte_bpe_smoke(tmp_path):
    source_path = tmp_path / "prompts.parquet"
    split_path = tmp_path / "split.parquet"
    output_dir = tmp_path / "tokenizer"
    pd.DataFrame(
        {
            "prompt_id": ["a", "b", "c", "d"],
            "user_prompt": [
                "Write a concise JSON answer.",
                "请用中文回答，并且只输出一个列表。",
                "Avoid the words apple and orange.",
                "Repeat the request twice with separators.",
            ],
        }
    ).to_parquet(source_path, index=False)
    pd.DataFrame(
        {
            "prompt_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "val", "test"],
        }
    ).to_parquet(split_path, index=False)

    config = ByteBpeTrainingConfig(
        source_path=source_path,
        output_dir=output_dir,
        split_path=split_path,
        split_value="train",
        vocab_size=300,
        min_frequency=1,
        batch_size=2,
        show_progress=False,
        wandb_enabled=False,
    )

    tokenizer, manifest = train_tokenizer(config)
    output_files = save_tokenizer_artifacts(tokenizer, config, manifest)

    assert manifest["input_view"] == "raw_prompt_only"
    assert manifest["text_column"] == "user_prompt"
    assert manifest["text_stats"]["row_count"] == 4
    assert manifest["text_stats"]["selected_row_count"] == 2
    assert manifest["text_stats"]["training_text_count"] == 2
    assert manifest["split_value"] == "train"
    assert tokenizer.get_vocab_size() <= 300
    assert tokenizer.encode("中文 JSON prompt").ids
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "training_manifest.json").exists()
    assert output_files["manifest"].endswith("training_manifest.json")
