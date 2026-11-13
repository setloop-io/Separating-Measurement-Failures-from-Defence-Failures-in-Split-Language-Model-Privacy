#!/usr/bin/env bash
# v15b stage (P4): binding demos across the controller's range on the
# weakened defense (noise 0.15, no chaff, no gauges) with the strengthened
# estimator (512/128, 4 updates).
#   v15b.0 weak_b010: B = 0.10 bits/token — tight binding
#   v15b.1 weak_b100: B = 1.00 bits/token — should stay non-binding
#                   (weak boundary leaks ~0.26 bits) or bind lightly
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

EST=(--mine-hidden 512 --mine-embed-dim 128 --mine-estimator-updates 4)

run_cell () {
  local name="$1"; shift
  echo "=== v15b cell $name ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v15_${name}.json" \
    --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --latent-dim 64 --noise-multiplier 0.15 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 0 --seed 42 \
    --cloud-url wss://koula:5025 --adversary-strength 1.0 \
    --attacker-bundle "$BUNDLEDIR/latent_v15_${name}_bundle.pt" \
    "${EST[@]}" "$@" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v15_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v15_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v15_${name}_bundle.pt"
  echo "=== v15b cell $name done ==="
}

run_cell weak_b010 --mine-budget-bits 0.10
run_cell weak_b100 --mine-budget-bits 1.00

echo "V15B_COMPLETE"
