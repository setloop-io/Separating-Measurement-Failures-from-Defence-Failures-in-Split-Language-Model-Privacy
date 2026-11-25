#!/usr/bin/env bash
# Phase-2 GPU canary for koula: capture -> SIP train -> BRE-bf attack.
set -euo pipefail
CODE="$HOME/llm-attacker-pilots/gpt2-wikitext2-v01/code"
PY="$HOME/llm-attacker-pilots/gpt2-wikitext2-v01/.venv/bin/python"
OUT="$CODE/results/phase2-canary-v2"
mkdir -p "$OUT"
cd "$CODE"
exec > "$OUT/canary.log" 2>&1

echo "=== STAGE 1: splitbench capture (gpt2 @6, 2 steps, cuda) ==="
"$PY" -m attacker.splitbench --model gpt2 --split 6 --mode finetune \
  --steps 2 --seq-len 32 --batch-size 1 --lr 5e-5 --device cuda --dtype fp32 \
  --seed 1234 --defense none \
  --corpus-file results/long-horizon-v1/inputs/pilot-corpus.txt \
  --capture-dir "$OUT/caps" --save-canonical \
  --output "$OUT/splitbench.json" --evidence

echo "=== STAGE 2: bisr-baseline SIP train (50 steps) ==="
"$PY" -m attacker --mode training --attack bisr-baseline --phase train \
  --model gpt2 --split 6 --seq-len 32 \
  --aux-file build/phase2/aux-corpus.jsonl \
  --train-steps 50 --inverter-pt "$OUT/bisr.pt" \
  --output "$OUT/sip-train.json" --evidence

echo "=== STAGE 3: bisr-baseline attack, --opt-arm bf (4 samples) ==="
"$PY" -m attacker --mode training --attack bisr-baseline --phase attack \
  --capture-dir "$OUT/caps" --canonical-pt "$OUT/caps/canonical.pt" \
  --inverter-pt "$OUT/bisr.pt" --model gpt2 --split 6 \
  --samples 4 --opt-arm bf \
  --output "$OUT/bisr-bf.json" --evidence

echo "CANARY_DONE rc=$?"
