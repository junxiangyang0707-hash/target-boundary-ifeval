from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from boundary_if.models.m4_pretrained_encoder import (
    MODEL_ID,
    MODEL_NAME,
    M4EncoderConfig,
    M4PretrainingModel,
    make_m4_pretraining_dataloader,
    set_torch_seed,
    train_m4_pretraining_epoch,
)

DEFAULT_INPUT_FILE = (
    "data/tokenized/"
    "if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated."
    "atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k.max2048.parquet"
)
DEFAULT_PROMPTSET_FILE = (
    "data/promptsets/if_multi_constraints_upto5.qwen3_4b_instruct_2507."
    "under2048_nontruncated.all.parquet"
)
DEFAULT_TOKENIZER_DIR = "data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_group_key_train"
DEFAULT_OUTPUT_DIR = "runs/m4_pretraining_atomic_constraint_heldout_seed42_max2048"

TOKENIZED_COLUMNS = [
    "prompt_id",
    "split",
    "input_ids",
    "instruction_ids",
    "num_constraints",
]
PROMPTSET_COLUMNS = ["prompt_id", "user_prompt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain M4 IF-domain encoder with MLM plus structural auxiliary loss."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--promptset-file", default=DEFAULT_PROMPTSET_FILE)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--digit-quote-mask-rate", type=float, default=0.30)
    parser.add_argument("--lambda-struct", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-wandb-save-model", action="store_true")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "boundary-if"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "runs/wandb"))
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def resolve_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return Path.cwd() / raw_path


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(raw_device)


def make_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        remaining = num_training_steps - current_step
        decay_steps = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_pretraining_frame(
    tokenized_path: Path,
    promptset_path: Path,
    *,
    train_split: str,
    limit_train: int | None,
) -> pd.DataFrame:
    tokenized = pd.read_parquet(tokenized_path, columns=TOKENIZED_COLUMNS)
    tokenized = tokenized[tokenized["split"].astype(str) == train_split].copy()
    if tokenized.empty:
        raise ValueError(f"No rows found for split={train_split!r} in {tokenized_path}.")
    promptset = pd.read_parquet(promptset_path, columns=PROMPTSET_COLUMNS)
    if promptset["prompt_id"].duplicated().any():
        raise ValueError(f"{promptset_path} has duplicate prompt_id values.")
    merged = tokenized.merge(promptset, on="prompt_id", how="left", validate="one_to_one")
    missing_prompt_count = int(merged["user_prompt"].isna().sum())
    if missing_prompt_count:
        raise ValueError(f"Missing user_prompt for {missing_prompt_count} training rows.")
    if limit_train is not None:
        merged = merged.head(limit_train).copy()
    return merged


def init_wandb(args: argparse.Namespace, config: M4EncoderConfig):
    if args.no_wandb:
        return None

    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    wandb_dir = resolve_path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        name=args.wandb_name or f"M4_pretrain_encoder_seed{args.seed}",
        group=args.wandb_group,
        job_type="m4_pretrain_encoder",
        tags=["M4", "pretraining", "MLM", "structural-aux", "raw-prompt-bpe"],
        config={
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "input_file": args.input_file,
            "promptset_file": args.promptset_file,
            "tokenizer_dir": args.tokenizer_dir,
            "output_dir": args.output_dir,
            "m4_config": config.to_dict(),
            "pretraining": {
                "train_split": args.train_split,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "grad_clip_norm": args.grad_clip_norm,
                "mask_rate": args.mask_rate,
                "digit_quote_mask_rate": args.digit_quote_mask_rate,
                "lambda_struct": args.lambda_struct,
                "seed": args.seed,
            },
        },
    )


def save_wandb_files(run: Any, paths: list[Path]) -> None:
    if run is None:
        return
    for path in paths:
        if path.exists():
            run.save(str(path), base_path=str(path.parent))


def main() -> None:
    args = parse_args()
    total_start = now()
    set_torch_seed(args.seed)
    input_file = resolve_path(args.input_file)
    promptset_file = resolve_path(args.promptset_file)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(f"Tokenizer at {tokenizer_dir} does not define a mask token.")

    config = M4EncoderConfig(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pooling="mean",
        mask_token_id=int(mask_token_id),
    )
    run = init_wandb(args, config)

    load_start = now()
    train_frame = load_pretraining_frame(
        input_file,
        promptset_file,
        train_split=args.train_split,
        limit_train=args.limit_train,
    )
    load_seconds = elapsed_since(load_start)

    dataloader_start = now()
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = make_m4_pretraining_dataloader(
        train_frame,
        config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    dataloader_seconds = elapsed_since(dataloader_start)

    model = M4PretrainingModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_training_steps = max(1, args.epochs * len(train_loader))
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = make_linear_warmup_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    special_token_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    special_token_ids.add(config.resolved_pad_token_id)

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = now()
        train_stats = train_m4_pretraining_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=args.grad_clip_norm,
            base_mask_rate=args.mask_rate,
            high_mask_rate=args.digit_quote_mask_rate,
            lambda_struct=args.lambda_struct,
            special_token_ids=special_token_ids,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=amp_enabled,
        )
        epoch_record = {
            "epoch": epoch,
            **train_stats,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": elapsed_since(epoch_start),
        }
        history.append(epoch_record)
        if run is not None:
            run.log(
                {
                    "m4_pretrain/epoch": epoch,
                    "m4_pretrain/loss": train_stats["loss"],
                    "m4_pretrain/mlm_loss": train_stats["mlm_loss"],
                    "m4_pretrain/structural_loss": train_stats["structural_loss"],
                    "m4_pretrain/masked_tokens_per_example": train_stats[
                        "masked_tokens_per_example"
                    ],
                    "m4_pretrain/learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    encoder_path = output_dir / "pretrained_encoder.pt"
    history_path = output_dir / "pretraining_history.csv"
    manifest_path = output_dir / "pretraining_manifest.json"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "config": config.to_dict(),
            "prompt_encoder_state_dict": model.prompt_encoder.state_dict(),
            "mlm_head_state_dict": model.mlm_head.state_dict(),
            "structural_binary_head_state_dict": model.structural_binary_head.state_dict(),
            "num_constraints_head_state_dict": model.num_constraints_head.state_dict(),
        },
        encoder_path,
    )
    manifest = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "input_file": str(input_file),
        "promptset_file": str(promptset_file),
        "tokenizer_dir": str(tokenizer_dir),
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "pretraining": {
            "train_split": args.train_split,
            "train_rows": int(len(train_frame)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": num_warmup_steps,
            "grad_clip_norm": args.grad_clip_norm,
            "mask_rate": args.mask_rate,
            "digit_quote_mask_rate": args.digit_quote_mask_rate,
            "lambda_struct": args.lambda_struct,
            "seed": args.seed,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "special_token_ids_excluded_from_mlm": sorted(special_token_ids),
        },
        "output_files": {
            "encoder": str(encoder_path),
            "history": str(history_path),
            "manifest": str(manifest_path),
        },
        "timing_seconds": {
            "load_data": load_seconds,
            "build_dataloader": dataloader_seconds,
            "total": elapsed_since(total_start),
        },
    }
    write_json(manifest, manifest_path)

    if run is not None:
        run.summary.update(
            {
                "model_id": MODEL_ID,
                "train_rows": int(len(train_frame)),
                "epochs": args.epochs,
                "final_loss": history[-1]["loss"] if history else None,
                "total_seconds": manifest["timing_seconds"]["total"],
                "output_dir": str(output_dir),
            }
        )
        save_wandb_files(
            run,
            [
                history_path,
                manifest_path,
                *([] if args.no_wandb_save_model else [encoder_path]),
            ],
        )
        run.finish()
    print(json.dumps({"manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
