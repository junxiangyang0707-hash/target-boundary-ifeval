from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from boundary_if.common.data_io import write_json


@dataclass(frozen=True)
class ByteBpeTrainingConfig:
    source_path: Path
    output_dir: Path
    split_path: Path | None = None
    split_value: str | None = None
    text_column: str = "user_prompt"
    input_view: str = "raw_prompt_only"
    vocab_size: int = 8000
    min_frequency: int = 2
    batch_size: int = 8192
    limit: int | None = None
    deduplicate: bool = False
    model_max_length: int = 2048
    add_prefix_space: bool = False
    trim_offsets: bool = False
    normalizer: str = "nfc"
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    mask_token: str = "<mask>"
    additional_special_tokens: tuple[str, ...] = field(default_factory=tuple)
    show_progress: bool = True
    wandb_enabled: bool = True
    wandb_project: str = "boundary-if"
    wandb_dir: Path = Path("runs/wandb")
    wandb_name: str | None = None
    wandb_job_type: str = "train_tokenizer"

    @property
    def special_tokens(self) -> list[str]:
        tokens = [
            self.unk_token,
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.mask_token,
            *self.additional_special_tokens,
        ]
        return list(dict.fromkeys(tokens))


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def load_allowed_prompt_ids(config: ByteBpeTrainingConfig) -> set[str] | None:
    if config.split_path is None:
        return None
    if config.split_value is None:
        raise ValueError("split_value must be set when split_path is provided.")

    split_df = pd.read_parquet(config.split_path, columns=["prompt_id", "split"])
    selected = split_df.loc[split_df["split"].astype(str) == config.split_value, "prompt_id"]
    if selected.empty:
        raise ValueError(
            f"No prompt_id rows found in {config.split_path} for split={config.split_value!r}."
        )
    return set(selected.astype(str).tolist())


def scan_prompt_texts(
    config: ByteBpeTrainingConfig,
    allowed_prompt_ids: set[str] | None = None,
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(config.source_path)
    row_count = 0
    selected_row_count = 0
    non_empty_count = 0
    empty_count = 0
    char_count = 0
    max_chars = 0
    unique_texts: set[str] | None = set() if config.deduplicate else None
    columns = (
        ["prompt_id", config.text_column]
        if allowed_prompt_ids is not None
        else [config.text_column]
    )

    for batch in parquet_file.iter_batches(
        batch_size=config.batch_size,
        columns=columns,
    ):
        prompt_ids = batch.column(0).to_pylist() if allowed_prompt_ids is not None else None
        text_values = (
            batch.column(1).to_pylist()
            if allowed_prompt_ids is not None
            else batch.column(0).to_pylist()
        )
        for index, value in enumerate(text_values):
            row_count += 1
            if allowed_prompt_ids is not None:
                prompt_id = str(prompt_ids[index])
                if prompt_id not in allowed_prompt_ids:
                    continue
            if config.limit is not None and selected_row_count >= config.limit:
                break
            text = _coerce_text(value)
            selected_row_count += 1
            if text:
                non_empty_count += 1
                char_count += len(text)
                max_chars = max(max_chars, len(text))
                if unique_texts is not None:
                    unique_texts.add(text)
            else:
                empty_count += 1
        if config.limit is not None and selected_row_count >= config.limit:
            break

    training_text_count = len(unique_texts) if unique_texts is not None else non_empty_count
    return {
        "row_count": row_count,
        "selected_row_count": selected_row_count,
        "non_empty_text_count": non_empty_count,
        "empty_text_count": empty_count,
        "training_text_count": training_text_count,
        "deduplicate": config.deduplicate,
        "unique_text_count": len(unique_texts) if unique_texts is not None else None,
        "total_chars": char_count,
        "mean_chars_non_empty": round(char_count / non_empty_count, 4)
        if non_empty_count
        else 0.0,
        "max_chars": max_chars,
    }


def iter_prompt_texts(
    config: ByteBpeTrainingConfig,
    allowed_prompt_ids: set[str] | None = None,
) -> Iterator[str]:
    parquet_file = pq.ParquetFile(config.source_path)
    row_count = 0
    selected_row_count = 0
    seen: set[str] | None = set() if config.deduplicate else None
    columns = (
        ["prompt_id", config.text_column]
        if allowed_prompt_ids is not None
        else [config.text_column]
    )

    for batch in parquet_file.iter_batches(
        batch_size=config.batch_size,
        columns=columns,
    ):
        prompt_ids = batch.column(0).to_pylist() if allowed_prompt_ids is not None else None
        text_values = (
            batch.column(1).to_pylist()
            if allowed_prompt_ids is not None
            else batch.column(0).to_pylist()
        )
        for index, value in enumerate(text_values):
            row_count += 1
            if allowed_prompt_ids is not None:
                prompt_id = str(prompt_ids[index])
                if prompt_id not in allowed_prompt_ids:
                    continue
            if config.limit is not None and selected_row_count >= config.limit:
                return
            selected_row_count += 1
            text = _coerce_text(value)
            if not text:
                continue
            if seen is not None:
                if text in seen:
                    continue
                seen.add(text)
            yield text


def build_byte_bpe_tokenizer(config: ByteBpeTrainingConfig) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token, fuse_unk=True))
    if config.normalizer == "nfc":
        tokenizer.normalizer = normalizers.NFC()
    elif config.normalizer in ("none", "null", ""):
        tokenizer.normalizer = None
    else:
        raise ValueError(f"Unsupported normalizer: {config.normalizer}")

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=config.add_prefix_space,
        trim_offsets=config.trim_offsets,
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(config: ByteBpeTrainingConfig) -> tuple[Tokenizer, dict[str, Any]]:
    if config.input_view != "raw_prompt_only":
        raise ValueError(f"Unsupported input_view for this trainer: {config.input_view}")

    allowed_prompt_ids = load_allowed_prompt_ids(config)

    scan_start = now()
    text_stats = scan_prompt_texts(config, allowed_prompt_ids=allowed_prompt_ids)
    scan_seconds = elapsed_since(scan_start)
    if text_stats["training_text_count"] <= 0:
        raise ValueError("No non-empty raw prompts found for tokenizer training.")

    tokenizer = build_byte_bpe_tokenizer(config)
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=config.show_progress,
    )

    train_start = now()
    tokenizer.train_from_iterator(
        iter_prompt_texts(config, allowed_prompt_ids=allowed_prompt_ids),
        trainer=trainer,
        length=text_stats["training_text_count"],
    )
    train_seconds = elapsed_since(train_start)

    pad_id = tokenizer.token_to_id(config.pad_token)
    if pad_id is not None:
        tokenizer.enable_padding(pad_id=pad_id, pad_token=config.pad_token)

    manifest = {
        "source_path": str(config.source_path),
        "split_path": str(config.split_path) if config.split_path is not None else None,
        "split_value": config.split_value,
        "text_column": config.text_column,
        "input_view": config.input_view,
        "vocab_size_requested": config.vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "min_frequency": config.min_frequency,
        "model_max_length": config.model_max_length,
        "byte_level": {
            "add_prefix_space": config.add_prefix_space,
            "trim_offsets": config.trim_offsets,
            "initial_alphabet": "ByteLevel.alphabet()",
        },
        "normalizer": config.normalizer,
        "special_tokens": config.special_tokens,
        "text_stats": text_stats,
        "timing_seconds": {
            "scan_texts": scan_seconds,
            "train_tokenizer": train_seconds,
        },
        "wandb": {
            "mode": os.environ.get("WANDB_MODE", "online"),
            "project": config.wandb_project,
            "dir": str(config.wandb_dir),
            "enabled": config.wandb_enabled,
        },
    }
    return tokenizer, manifest


def save_tokenizer_artifacts(
    tokenizer: Tokenizer,
    config: ByteBpeTrainingConfig,
    manifest: dict[str, Any],
) -> dict[str, str]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = config.output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    model_files = tokenizer.model.save(str(config.output_dir))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token=config.unk_token,
        pad_token=config.pad_token,
        bos_token=config.bos_token,
        eos_token=config.eos_token,
        mask_token=config.mask_token,
        additional_special_tokens=list(config.additional_special_tokens),
        model_max_length=config.model_max_length,
    )
    fast_tokenizer.save_pretrained(str(config.output_dir))

    manifest_path = config.output_dir / "training_manifest.json"
    config_path = config.output_dir / "training_config.json"
    output_files = {
        "output_dir": str(config.output_dir),
        "tokenizer_json": str(tokenizer_json),
        "manifest": str(manifest_path),
        "training_config": str(config_path),
        "model_files": [str(path) for path in model_files],
    }
    manifest = {
        **manifest,
        "output_files": output_files,
        "sample_encodings": sample_encodings(tokenizer, config),
    }
    write_json(manifest, manifest_path)
    write_json(config_to_jsonable(config), config_path)
    return output_files


def sample_encodings(
    tokenizer: Tokenizer,
    config: ByteBpeTrainingConfig,
    sample_count: int = 3,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    allowed_prompt_ids = load_allowed_prompt_ids(config)
    for text in iter_prompt_texts(config, allowed_prompt_ids=allowed_prompt_ids):
        encoded = tokenizer.encode(text)
        samples.append(
            {
                "text_preview": text[:200],
                "char_count": len(text),
                "token_count": len(encoded.ids),
                "first_token_ids": encoded.ids[:32],
                "first_tokens": encoded.tokens[:32],
            }
        )
        if len(samples) >= sample_count:
            break
    return samples


def config_to_jsonable(config: ByteBpeTrainingConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key in ("source_path", "output_dir", "wandb_dir"):
        raw[key] = str(raw[key])
    raw["split_path"] = str(raw["split_path"]) if raw["split_path"] is not None else None
    raw["additional_special_tokens"] = list(raw["additional_special_tokens"])
    return raw


def init_wandb_if_enabled(
    config: ByteBpeTrainingConfig,
    manifest: dict[str, Any],
):
    if not config.wandb_enabled:
        return None
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    os.environ.setdefault("WANDB_DIR", str(config.wandb_dir))

    import wandb

    config.wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.wandb_project,
        mode=os.environ.get("WANDB_MODE", "online"),
        dir=str(config.wandb_dir),
        name=config.wandb_name,
        job_type=config.wandb_job_type,
        tags=["tokenizer", "byte-level-bpe", "raw-prompt-only"],
        config={
            "tokenizer_training": config_to_jsonable(config),
            "tokenizer_manifest": manifest,
        },
    )
    return run
