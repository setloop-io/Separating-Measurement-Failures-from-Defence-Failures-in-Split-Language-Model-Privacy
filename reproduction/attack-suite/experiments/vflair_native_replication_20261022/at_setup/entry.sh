#!/usr/bin/env bash
set -euo pipefail
AT_TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$AT_TASK_DIR/setup.py" --config "$AT_TASK_DIR/run.json" "$@"
