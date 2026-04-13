#!/usr/bin/env bash
# Create a new run directory from template.
# Usage: ./experiments/scripts/new_run.sh <run-id>
# Example: ./experiments/scripts/new_run.sh 20260412-rca-baseline

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run-id>"
  echo "Example: $0 20260412-rca-baseline"
  exit 1
fi

RUN_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(dirname "$SCRIPT_DIR")/runs"
TEMPLATE_DIR="$RUNS_DIR/.template"
TARGET_DIR="$RUNS_DIR/$RUN_ID"

if [ -d "$TARGET_DIR" ]; then
  echo "Error: run directory already exists: $TARGET_DIR"
  exit 1
fi

cp -r "$TEMPLATE_DIR" "$TARGET_DIR"

# Fill in auto-detectable fields
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
DATE=$(date -Iseconds)

if command -v sed &>/dev/null; then
  sed -i "s|run_id: \"run-YYYY-MM-DD-variant\"|run_id: \"$RUN_ID\"|" "$TARGET_DIR/meta.yaml"
  sed -i "s|date: \"YYYY-MM-DDThh:mm:ss\"|date: \"$DATE\"|" "$TARGET_DIR/meta.yaml"
  sed -i "s|commit: \"\"|commit: \"$COMMIT\"|" "$TARGET_DIR/meta.yaml"
  sed -i "s|branch: \"\"|branch: \"$BRANCH\"|" "$TARGET_DIR/meta.yaml"
  sed -i "s|<run-id>|$RUN_ID|" "$TARGET_DIR/notes.md"
fi

echo "Created: $TARGET_DIR"
echo "Next: fill in meta.yaml config section, then run the experiment."
