#!/usr/bin/env bash
# Serve an SFT checkpoint on sglang for inspection.
#
# This is for looking at a checkpoint by hand. RL serves its own rollout
# endpoint from the training config (`sglang:` in configs/train/), and
# `RemoteSGLangEngine` is a client of that -- it does not serve anything.
#
# Defaults to the newest checkpoint under .runs/dsh-rca-sft, the fileroot
# configs/sft/dsh_rca_sft.yaml writes to. Override with the first argument:
#   ./scripts/serve_sft.sh /path/to/checkpoint
#
# Environment:
#   PORT        — default 30111.
#   GPU         — CUDA device to serve on, default 0.
#   HF_ENDPOINT — only needed if the tokenizer is not already cached.
#
# Any extra args after the checkpoint are forwarded to launch_server.

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# sglang JIT-compiles kernels through ninja, which it looks up on PATH rather
# than in its own environment. The venv has it; a login shell does not, and the
# failure is a `FileNotFoundError: 'ninja'` from deep inside a child process
# followed by "Received sigquit from a child process".
export PATH="$ROOT_DIR/.venv/bin:$PATH"

# The weights are local. Without this, a hub lookup on a host that cannot reach
# huggingface.co stalls the launch instead of failing it.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CKPT="${1:-}"
if [[ -n "$CKPT" ]]; then
  shift
else
  # Newest by mtime rather than by step number: a re-run of an earlier step is
  # the one being looked at.
  CKPT=$(ls -dt .runs/dsh-rca-sft/checkpoints/*/*/*/default/*/ 2>/dev/null | head -1 || true)
  if [[ -z "$CKPT" ]]; then
    echo "no checkpoint under .runs/dsh-rca-sft — run ./scripts/run_sft.sh first," >&2
    echo "or pass one: ./scripts/serve_sft.sh /path/to/checkpoint" >&2
    exit 2
  fi
fi

if [[ ! -f "$CKPT/config.json" ]]; then
  echo "not a checkpoint (no config.json): $CKPT" >&2
  exit 2
fi

PORT="${PORT:-30111}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "serving $CKPT on 127.0.0.1:$PORT (GPU $CUDA_VISIBLE_DEVICES)" >&2

# `--tool-call-parser qwen` is what turns the model's `<tool_call>{...}` block
# into a structured `tool_calls` entry. Without it the block stays in
# `message.content` as text and `tool_calls` is empty, so a client that reads
# the field sees an episode that called nothing.
#
# `--reasoning-parser qwen3` likewise splits `<think>` into
# `reasoning_content`; the distillation puts reasoning in every assistant turn,
# so without it the thinking is prepended to the answer.
#
# context-length matches configs/sft/dsh_rca_sft.yaml's max_length and the RL
# config's sglang.context_length, so a prompt that trains is a prompt that serves.
exec python3 -m sglang.launch_server \
  --model-path "$CKPT" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --context-length 32768 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen \
  --served-model-name rca-sft \
  "$@"
