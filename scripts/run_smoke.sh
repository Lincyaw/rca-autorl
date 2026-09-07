#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Ensure AReaL child workers inherit the project venv interpreter when they invoke `python3`.
export PATH="$ROOT_DIR/.venv/bin:$PATH"

# The RCA harness bundle must be in the profile before any rollout starts;
# `dsh plugin` shells out to pnpm and must not race with concurrent rollouts.
# The default matches econfig.dsh_home in the config below.
python3 -m autorl.harness "${DSH_HOME:-$ROOT_DIR/.runs/dsh-rca-smoke/dsh-home}"

python3 -m autorl.train --config configs/train/dsh_rca_smoke.yaml "$@"
