#!/usr/bin/env bash
# Phase-2 CPU canary: 3 Mode-2 P6 capture cells (one per span), D0 defense.
# Purpose: measure victim-train wall-clock on a 4-core CPU node before grid
# scheduling (anchor: 33.2 s/100-step GPT-2@6 seq32 on the dev host).
set -euo pipefail
CODE="$HOME/llm-attacker"
PY="$CODE/.venv/bin/python"
OUT="$CODE/results/phase2-canary-cpu-v1"
mkdir -p "$OUT"
cd "$CODE"
exec > "$OUT/canary.log 2>&1"

run_cell () {  # span start tag
  local span="$1" start="$2" tag="$3"
  local end=$((start + span))
  echo "=== CELL $tag: span=$span start=$start (trainable $start..$end) ==="
  local t0=$SECONDS
  "$PY" -m attacker.splitbench --model gpt2 --split "$start" \
    --trainable-range "$start,$end" --mode finetune --steps 100 \
    --seq-len 32 --batch-size 1 --lr 5e-5 --device cpu --dtype fp32 \
    --seed 1234 --defense quant --quant-bits 8 --quant-grad \
    --corpus-file results/long-horizon-v1/inputs/pilot-corpus.txt \
    --capture-dir "$OUT/caps-$tag" --save-canonical \
    --output "$OUT/cell-$tag.json" --evidence
  echo "=== CELL $tag elapsed: $((SECONDS - t0)) s ==="
}

run_cell 2 1 s2
run_cell 4 2 s4
run_cell 6 3 s6
echo "CANARY_DONE rc=$?"
