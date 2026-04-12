#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Ensure AReaL child workers inherit the project venv interpreter when they invoke `python3`.
export PATH="$ROOT_DIR/.venv/bin:$PATH"

python3 -m autorl.experiments.agent_workflow.train --config configs/train/smoke.yaml "$@"
