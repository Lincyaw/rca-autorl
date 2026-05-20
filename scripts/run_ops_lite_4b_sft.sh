#!/usr/bin/env bash
set -euo pipefail

# Run RCA trajectory SFT on ops-lite data with the final working 4B setup.
# Override env vars instead of editing this script, e.g. EPOCHS=20 OUT_ROOT=.runs/foo ./scripts/run_ops_lite_4b_sft.sh
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
DATA_PATH=${DATA_PATH:-.runs/data/ops-lite-trajectories/extractor.jsonl}
OUT_ROOT=${OUT_ROOT:-.runs/ops-lite-trajectory-sft-4b-2gpu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-autorl-ops-lite-trajectory-sft-4b}
TRIAL_NAME=${TRIAL_NAME:-qwen3-4b-instruct-2gpu}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_LENGTH=${MAX_LENGTH:-32768}
MAX_TOKENS_PER_MB=${MAX_TOKENS_PER_MB:-32768}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
LR=${LR:-1e-5}
SAVE_FREQ_EPOCHS=${SAVE_FREQ_EPOCHS:-1}
EVAL_FREQ_EPOCHS=${EVAL_FREQ_EPOCHS:-1}
TIMEOUT=${TIMEOUT:-8h}

mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT/name_resolve"

PATH="$PWD/.venv/bin:$PATH" \
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  HF_ENDPOINT="$HF_ENDPOINT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  timeout "$TIMEOUT" python3 -m autorl.experiments.agent_sft.train \
  --config configs/sft/agentm_rca_sft_smoke.yaml \
  actor.path="$MODEL_PATH" \
  tokenizer_path="$TOKENIZER_PATH" \
  actor.backend=fsdp:d2p1t1 \
  total_train_steps=null \
  total_train_epochs="$EPOCHS" \
  experiment_name="$EXPERIMENT_NAME" \
  trial_name="$TRIAL_NAME" \
  train_dataset.path="$DATA_PATH" \
  valid_dataset.path="$DATA_PATH" \
  +train_dataset.scheduling_spec=null \
  +valid_dataset.scheduling_spec=null \
  train_dataset.batch_size="$BATCH_SIZE" \
  valid_dataset.batch_size="$BATCH_SIZE" \
  train_dataset.max_length="$MAX_LENGTH" \
  valid_dataset.max_length="$MAX_LENGTH" \
  actor.mb_spec.max_tokens_per_mb="$MAX_TOKENS_PER_MB" \
  actor.gradient_checkpointing=true \
  actor.optimizer.lr="$LR" \
  saver.freq_epochs="$SAVE_FREQ_EPOCHS" \
  evaluator.freq_epochs="$EVAL_FREQ_EPOCHS" \
  cluster.fileroot="$OUT_ROOT" \
  cluster.name_resolve.nfs_record_root="$OUT_ROOT/name_resolve" \
  2>&1 | tee "$OUT_ROOT/train.log"
