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

# The harness bundle's vocabulary is generated from the fpg profile, and the
# commit hook runs this script and nothing else. A stale copy would let the tool
# accept a submission the verifier rejects, which is the failure this whole
# arrangement exists to prevent, so it is checked here rather than only in tests.
.venv/bin/python -m autorl.fpg --check
