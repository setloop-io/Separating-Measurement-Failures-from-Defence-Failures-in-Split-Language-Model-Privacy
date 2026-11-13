#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
exec python3 experiments/vflair_native_replication_20261022/at_runtime/dispatch.py \
  --config experiments/vflair_native_replication_20261022/at_runtime/run.json
