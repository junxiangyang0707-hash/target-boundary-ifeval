#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="data/tokenized/if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.atomic_constraint_heldout_seed42.raw_prompt_byte_bpe_v8k_atomic_train.max2048.parquet"
PROMPTSET_FILE="data/promptsets/if_multi_constraints_upto5.qwen3_4b_instruct_2507.under2048_nontruncated.all.parquet"
TOKENIZER_DIR="data/tokenized/tokenizers/raw_prompt_byte_bpe_v8k_atomic_train"
ROOT_DIR="runs/strict_atomic_tokenizer_key_runs"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] START ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  local status=${PIPESTATUS[0]}
  echo "[$(date -Is)] END ${name} status=${status}"
  return "${status}"
}

run_step "m3_mean_40k_seed42" \
  python scripts/train_m3_tiny_transformer.py \
    --input-file "${INPUT_FILE}" \
    --output-dir "${ROOT_DIR}/m3_mean_40k_seed42" \
    --pooling mean \
    --max-length 2048 \
    --batch-size 8 \
    --eval-batch-size 16 \
    --seed 42 \
    --sample-seed 42 \
    --train-sample-size 40000 \
    --wandb-name "M3_strict_atomic_tokenizer_mean_40k_seed42" \
    --wandb-group "strict_atomic_tokenizer_key_runs" \
    --no-wandb-save-model

run_step "m3_mean_full_seed42" \
  python scripts/train_m3_tiny_transformer.py \
    --input-file "${INPUT_FILE}" \
    --output-dir "${ROOT_DIR}/m3_mean_full_seed42" \
    --pooling mean \
    --max-length 2048 \
    --batch-size 8 \
    --eval-batch-size 16 \
    --seed 42 \
    --sample-seed 42 \
    --wandb-name "M3_strict_atomic_tokenizer_mean_full_seed42" \
    --wandb-group "strict_atomic_tokenizer_key_runs" \
    --no-wandb-save-model

run_step "m4_pretrain_seed42" \
  python scripts/train_m4_pretrain_encoder.py \
    --input-file "${INPUT_FILE}" \
    --promptset-file "${PROMPTSET_FILE}" \
    --tokenizer-dir "${TOKENIZER_DIR}" \
    --output-dir "${ROOT_DIR}/m4_pretrain_seed42" \
    --seed 42 \
    --batch-size 8 \
    --wandb-name "M4_strict_atomic_tokenizer_pretrain_seed42" \
    --wandb-group "strict_atomic_tokenizer_key_runs" \
    --no-wandb-save-model

run_step "m4_frozen_full_seed42" \
  python scripts/train_m4_frozen_classifier.py \
    --input-file "${INPUT_FILE}" \
    --encoder-file "${ROOT_DIR}/m4_pretrain_seed42/pretrained_encoder.pt" \
    --output-dir "${ROOT_DIR}/m4_frozen_full_seed42" \
    --max-length 2048 \
    --batch-size 8 \
    --eval-batch-size 16 \
    --seed 42 \
    --sample-seed 42 \
    --wandb-name "M4_strict_atomic_tokenizer_full_seed42" \
    --wandb-group "strict_atomic_tokenizer_key_runs" \
    --no-wandb-save-model

echo "[$(date -Is)] strict atomic tokenizer key runs complete"
