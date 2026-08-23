#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for tool in ruff mypy; do
  if [[ ! -x ".venv/bin/$tool" ]]; then
    printf 'error: .venv/bin/%s is missing; run `uv sync --dev`\n' "$tool" >&2
    exit 7
  fi
done

.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src
