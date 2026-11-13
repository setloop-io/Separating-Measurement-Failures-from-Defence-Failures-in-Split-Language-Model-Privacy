#!/usr/bin/env bash
# v14 stage: the certified leakage-budget sweep on the v9.2 winner base
# (D=64, chaff-48, radial MoE E=8, scale gauge off, noise 0.35, wss, K=1,
# adversary-strength 1.0 — the v13 cell convention).
#   v14.0: B = 0.25 bits/token   (tightest certified tier)
#   v14.1: B = 0.50 bits/token   (candidate standard tier)
#   v14.2: B = 1.00 bits/token   (candidate performance tier)
#   v14.3: B = 2.00 bits/token   (relaxed tier)
#   v14.4: unconstrained (no MINE term; frontier's right endpoint = v9.2 base)
# The budget is enforced by the Lagrangian dual controller behind
# --mine-budget-bits (lambda_final reported per cell).
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
STEPS="${STEPS:-2000}"
WARMUP="${WARMUP:-200}"
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local extra=("$@")
  echo "=== v14 cell $name (steps=$STEPS) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v14_${name}.json" \
    --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps "$STEPS" --warmup-steps "$WARMUP" \
    --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    "${extra[@]}" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v14_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v14_${name}_bundle.pt"
  echo "=== v14 cell $name done ==="
}

run_cell v140_b025 --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-budget-bits 0.25
run_cell v141_b050 --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-budget-bits 0.50
run_cell v142_b100 --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-budget-bits 1.00
run_cell v143_b200 --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-budget-bits 2.00
run_cell v144_free --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-measure-only

echo "V14_COMPLETE"
