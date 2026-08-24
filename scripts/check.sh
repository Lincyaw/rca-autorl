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

paths=(src)
[[ -d tests ]] && paths+=(tests)

.venv/bin/ruff check "${paths[@]}"
.venv/bin/ruff format --check "${paths[@]}"
.venv/bin/mypy src
