#!/usr/bin/env bash
# Launch the RCA SFT smoke run.
#
# Defaults:
#   - config: configs/sft/dsh_rca_sft_smoke.yaml
#       (Qwen3-4B-Thinking-2507, max_length=32768)
#   - train_dataset.path: data/sft/rca_sessions.jsonl, the checked-in
#     distillation (git-lfs; run `git lfs pull` after a fresh clone). Point at
#     a new export with train_dataset.path=.runs/sft/...
#
# Overrides:
#   RCA_DATASET_ROOT — only needed if you switch to a config that
#   feeds RL manifest rows that carry datapack_name instead of an
#   absolute data_dir.
#
# Any extra args are forwarded to the train entrypoint, e.g.:
#   ./scripts/run_sft_smoke.sh actor.path=Qwen/Qwen3-8B-Thinking-2507

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# AReaL worker subprocesses call ``python3`` — put the project venv first.
export PATH="$ROOT_DIR/.venv/bin:$PATH"

CONFIG="${SFT_CONFIG:-configs/sft/dsh_rca_sft_smoke.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "venv missing — run: UV_HTTP_TIMEOUT=300 uv sync --python 3.12" >&2
  exit 3
fi

python3 -m autorl.train_sft --config "$CONFIG" "$@"
