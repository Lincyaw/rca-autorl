#!/usr/bin/env bash
# Claude Code UserPromptSubmit hook entrypoint.
#
# Reads the hook payload on stdin, prints any pending harness reminder for the
# session to stdout (Claude Code injects stdout as additional context for the
# next turn), and consumes it. Silent when no reminder is pending or when the
# payload is unrecognizable.
#
# Optional env var:
#   AUTORL_HARNESS_ROOT   — defaults to $PWD/.harness
#   AUTORL_HARNESS_PYTHON — defaults to python3
set -euo pipefail

ROOT="${AUTORL_HARNESS_ROOT:-${PWD}/.harness}"
PYTHON="${AUTORL_HARNESS_PYTHON:-python3}"

"$PYTHON" -m autorl.observability.harness \
  --root "$ROOT" \
  inject --from-hook
