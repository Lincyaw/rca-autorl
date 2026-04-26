#!/usr/bin/env bash
# Install the LLM-as-harness P0 hooks into Claude Code's project settings and
# kick off a background worker loop.
#
# Usage:
#   scripts/harness/install.sh             # write .claude/settings.local.json + start worker
#   scripts/harness/install.sh --print     # print the rendered settings JSON to stdout
#   scripts/harness/install.sh --no-worker # only write settings, don't spawn worker
#
# Notes:
#   - We write to .claude/settings.local.json so we don't clobber a checked-in
#     .claude/settings.json. Local settings override and are gitignored by
#     default in Claude Code projects.
#   - The worker is started under nohup; a marker file at
#     .harness/worker.pid records the PID. Re-running install replaces it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTINGS_DIR="${REPO_ROOT}/.claude"
SETTINGS_FILE="${SETTINGS_DIR}/settings.local.json"
HARNESS_ROOT="${REPO_ROOT}/.harness"
WORKER_PID_FILE="${HARNESS_ROOT}/worker.pid"
WORKER_LOG_FILE="${HARNESS_ROOT}/worker.log"

PRINT_ONLY=0
START_WORKER=1
for arg in "$@"; do
  case "$arg" in
    --print) PRINT_ONLY=1 ;;
    --no-worker) START_WORKER=0 ;;
    -h|--help)
      sed -n '2,18p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
  esac
done

render_settings() {
  python3 - "$REPO_ROOT" <<'PY'
import json, sys
root = sys.argv[1]
template = {
    "env": {
        "AUTORL_HARNESS_ROOT": f"{root}/.harness",
        "AUTORL_HARNESS_PYTHON": "python3",
        "PYTHONPATH": f"{root}/src",
    },
    "hooks": {
        "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": f"{root}/scripts/harness/ingest_hook.sh", "timeout": 5}]}],
        "Stop":        [{"matcher": "*", "hooks": [{"type": "command", "command": f"{root}/scripts/harness/ingest_hook.sh", "timeout": 5}]}],
        "UserPromptSubmit": [{"matcher": "*", "hooks": [{"type": "command", "command": f"{root}/scripts/harness/inject_hook.sh", "timeout": 3}]}],
    },
}
print(json.dumps(template, indent=2, ensure_ascii=False))
PY
}

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  render_settings
  exit 0
fi

mkdir -p "$SETTINGS_DIR" "$HARNESS_ROOT"
render_settings > "$SETTINGS_FILE"
echo "wrote $SETTINGS_FILE"

if [[ "$START_WORKER" -eq 1 ]]; then
  if [[ -f "$WORKER_PID_FILE" ]] && kill -0 "$(cat "$WORKER_PID_FILE")" 2>/dev/null; then
    echo "stopping previous worker $(cat "$WORKER_PID_FILE")"
    kill "$(cat "$WORKER_PID_FILE")" || true
    sleep 0.2
  fi
  AUTORL_HARNESS_ROOT="$HARNESS_ROOT" \
    nohup bash -c '
      while true; do
        bash "'"$REPO_ROOT"'/scripts/harness/tick_worker.sh" || true
        sleep 5
      done
    ' >"$WORKER_LOG_FILE" 2>&1 &
  echo $! > "$WORKER_PID_FILE"
  echo "started worker pid $(cat "$WORKER_PID_FILE") (log: $WORKER_LOG_FILE)"
fi

cat <<EOF
done.
- settings: $SETTINGS_FILE
- harness root: $HARNESS_ROOT
- to stop the worker: kill \$(cat $WORKER_PID_FILE)
EOF
