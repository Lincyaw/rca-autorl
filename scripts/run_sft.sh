#!/usr/bin/env bash
# Launch the RCA SFT run: one full epoch over the distillation, 8 GPUs.
#
# The smoke script's config stops after one step on one card; this one
# trains to convergence. Defaults:
#   - config: configs/sft/dsh_rca_sft.yaml
#       (Qwen3-4B-Thinking-2507, max_length=32768, fsdp:d8p1t1)
#   - train_dataset.path: data/sft/rca_sessions.jsonl, the checked-in
#     distillation (git-lfs; run `git lfs pull` after a fresh clone).
#
# Overrides:
#   SFT_CONFIG      — a different config file.
#   HF_ENDPOINT     — set this where huggingface.co is unreachable. The
#     tokenizer load contacts the hub even when the weights are already
#     cached, so without a reachable endpoint the run dies before the
#     first step with "Network is unreachable".
#
# Any extra args are forwarded to the train entrypoint, e.g.:
#   ./scripts/run_sft.sh total_train_epochs=3

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# AReaL worker subprocesses call ``python3`` — put the project venv first.
export PATH="$ROOT_DIR/.venv/bin:$PATH"

# A 32k row's activations fragment the allocator badly enough to OOM with
# gigabytes still nominally free.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${SFT_CONFIG:-configs/sft/dsh_rca_sft.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "venv missing — run: UV_HTTP_TIMEOUT=300 uv sync --python 3.12" >&2
  exit 3
fi

# A 132-byte pointer file trains without complaint and teaches nothing.
if [[ $(wc -c <data/sft/rca_sessions.jsonl) -lt 10000 ]]; then
  echo "data/sft/rca_sessions.jsonl is an unresolved git-lfs pointer — run: git lfs pull" >&2
  exit 4
fi

python3 -m autorl.train_sft --config "$CONFIG" "$@"
