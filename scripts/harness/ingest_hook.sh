#!/usr/bin/env bash
# Claude Code PostToolUse / Stop hook entrypoint.
#
# Pipes the hook payload (JSON on stdin) straight to the harness ingest CLI in
# --from-hook mode. The CLI parses session_id + transcript_path from the
# payload, computes the inbox delta from the transcript, and returns
# immediately. A separate worker (cron / daemon) calls `tick`.
#
# Optional env var:
#   AUTORL_HARNESS_ROOT   — defaults to $PWD/.harness
#   AUTORL_HARNESS_PYTHON — defaults to python3
set -euo pipefail

ROOT="${AUTORL_HARNESS_ROOT:-${PWD}/.harness}"
PYTHON="${AUTORL_HARNESS_PYTHON:-python3}"

"$PYTHON" -m autorl.observability.harness \
  --root "$ROOT" \
  ingest --from-hook >/dev/null
