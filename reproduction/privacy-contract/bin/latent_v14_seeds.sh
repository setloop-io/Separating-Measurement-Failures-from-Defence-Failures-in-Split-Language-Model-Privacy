#!/usr/bin/env bash
# v14 seed robustness: re-score the standard tier (v14.1, B=0.50) and the
# unconstrained endpoint (v14.4) on seeds 43 and 44, matching the v9.2
# 3-seed treatment (seed 42 already run in the main sweep).
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; local seed="$2"; shift 2
  local extra=("$@")
  echo "=== v14 seed cell $name (seed=$seed) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v14_${name}.json" \
    --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed "$seed" \
    --cloud-url wss://koula:5025 --adversary-strength 1.0 \
    --attacker-bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    "${extra[@]}" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v14_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v14_${name}_bundle.pt"
  echo "=== v14 seed cell $name done ==="
}

for seed in 43 44; do
  run_cell v141_b050_s${seed} "$seed" --mine-budget-bits 0.50
  run_cell v144_free_s${seed} "$seed" --mine-measure-only
done

echo "V14_SEEDS_COMPLETE"
