#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$TASK_DIR/dispatch.py" --config "$TASK_DIR/run.json" "$@"
exec python3 "$TASK_DIR/post_run_verify.py" --config "$TASK_DIR/run.json"
