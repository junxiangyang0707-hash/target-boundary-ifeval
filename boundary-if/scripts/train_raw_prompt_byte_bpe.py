from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from boundary_if.tokenization.byte_bpe import (
    ByteBpeTrainingConfig,
    init_wandb_if_enabled,
    save_tokenizer_artifacts,
    train_tokenizer,
)

DEFAULT_SOURCE_PATH = (
    "data/promptsets/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.all.parquet"
)
DEFAULT_OUTPUT_DIR = "data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k"
DEFAULT_SPLIT_FILE = (
    "data/splits/qwen3_4b_instruct_2507_under2048_nontruncated/group_key_seed42.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on raw user_prompt text only."
    )
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split-value", default="train")
    parser.add_argument("--text-column", default="user_prompt")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", default="raw_prompt_byte_bpe_v8k")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "runs/wandb"))
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "online")

    config = ByteBpeTrainingConfig(
        source_path=resolve_path(args.source_path),
        output_dir=resolve_path(args.output_dir),
        split_path=resolve_path(args.split_file) if args.split_file else None,
        split_value=args.split_value if args.split_file else None,
        text_column=args.text_column,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        batch_size=args.batch_size,
        limit=args.limit,
        deduplicate=args.deduplicate,
        model_max_length=args.model_max_length,
        show_progress=not args.no_progress,
        wandb_enabled=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_dir=resolve_path(args.wandb_dir),
        wandb_name=args.wandb_name,
    )

    tokenizer, manifest = train_tokenizer(config)
    run = init_wandb_if_enabled(config, manifest)
    output_files = save_tokenizer_artifacts(tokenizer, config, manifest)
    if run is not None:
        run.summary.update(
            {
                "vocab_size_actual": manifest["vocab_size_actual"],
                "training_text_count": manifest["text_stats"]["training_text_count"],
                "output_dir": output_files["output_dir"],
            }
        )
        run.finish()

    print(
        json.dumps(
            {"manifest": manifest, "output_files": output_files},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
