#!/usr/bin/env bash
# v15 stage: estimator-power and binding-frontier cells.
# v14 showed the DV/MINE lower bound reads ~0 on the defended boundary at
# every budget tier (lambda never engages).  v15 asks the two follow-ups:
#   v15.0 weak_free:  weakened defense (noise 0.15, no chaff, no gauges),
#                     strengthened estimator, measure-only -> raw leak level
#   v15.1 weak_b025:  same weakened defense + budget 0.25 -> lambda must
#                     engage and hold MI at/under B (the binding cell)
#   v15.2 weak_b050:  same + budget 0.50 -> second binding frontier point
#   v15.3 full_strongest: full v9.2 defense + strengthened estimator,
#                     measure-only -> does a stronger estimator see anything
#                     on the defended boundary?  (honest-limits evidence)
# Strengthened estimator: hidden 512, token embed 128, 4 estimator updates
# per attacker round (v14 used 256 / 64 / 1).
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
STEPS="${STEPS:-2000}"
WARMUP="${WARMUP:-200}"
mkdir -p "$BUNDLEDIR"

EST=(--mine-hidden 512 --mine-embed-dim 128 --mine-estimator-updates 4)
# Weakened defense: low noise, no chaff, no wire rotation/permutation.
WEAK=(--noise-multiplier 0.15 --chaff-tokens 0)

run_cell () {
  local name="$1"; shift
  local extra=("$@")
  echo "=== v15 cell $name (steps=$STEPS) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v15_${name}.json" \
    --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --latent-dim 64 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps "$STEPS" --warmup-steps "$WARMUP" \
    --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --seed 42 \
    --cloud-url wss://koula:5025 --adversary-strength 1.0 \
    --attacker-bundle "$BUNDLEDIR/latent_v15_${name}_bundle.pt" \
    "${EST[@]}" "${extra[@]}" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v15_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v15_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v15_${name}_bundle.pt"
  echo "=== v15 cell $name done ==="
}

run_cell weak_free "${WEAK[@]}" --mine-measure-only
run_cell weak_b025 "${WEAK[@]}" --mine-budget-bits 0.25
run_cell weak_b050 "${WEAK[@]}" --mine-budget-bits 0.50
# Full v9.2 defense + strongest estimator, measurement only.
run_cell full_strongest \
  --secret-wire-rotation --secret-token-permutation \
  --noise-multiplier 0.35 --chaff-tokens 48 --mine-measure-only

echo "V15_COMPLETE"
