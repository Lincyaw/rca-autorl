#!/usr/bin/env bash
# Background worker tick. Run from cron (e.g. every 30s) or a long-running
# daemon. Calls `tick` for every session that has an inbox file.
#
# Optional env var:
#   AUTORL_HARNESS_ROOT — defaults to $PWD/.harness
set -euo pipefail

ROOT="${AUTORL_HARNESS_ROOT:-${PWD}/.harness}"
PYTHON="${AUTORL_HARNESS_PYTHON:-python3}"

INBOX_DIR="${ROOT}/inbox"
[[ -d "$INBOX_DIR" ]] || exit 0

shopt -s nullglob
for f in "$INBOX_DIR"/*.jsonl; do
  sid="$(basename "$f" .jsonl)"
  "$PYTHON" -m autorl.observability.harness \
    --root "$ROOT" \
    tick --session "$sid" || true
done
