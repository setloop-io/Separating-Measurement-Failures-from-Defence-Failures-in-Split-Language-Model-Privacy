#!/usr/bin/env bash
# v14 transfer cells (P1): the certified budget on the big backbones.
# 35B hybrid MoE at B = 0.25 / 0.50; 12B Nemo at B = 0.50.
# Big-model operating points from the v13 coverage stage: split 31/36,
# noise 0.40 on 35B (hybrid-MoE margin config), 0.35 on 12B; D=64,
# radial MoE E8 L2, chaff-48, rotation+permutation, K=1, seed 42.
set -uo pipefail

CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1" model="$2" noise="$3"; shift 3
  local extra=("$@")
  echo "=== v14 transfer cell $name ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$model" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v14_${name}.json" \
    --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier "$noise" --clip-norm 1.0 \
    --split-after 31 --resume-after 36 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --cloud-url wss://koula:5025 --adversary-strength 1.0 \
    --attacker-bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    "${extra[@]}" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v14_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v14_${name}_bundle.pt"
  echo "=== v14 transfer cell $name done ==="
}

M35=/workspace/experiments/models/qwen36-35b-a3b
M12=/workspace/experiments/models/mistral-nemo-12b-instruct

run_cell 35b_b025 "$M35" 0.40 --mine-budget-bits 0.25
run_cell 35b_b050 "$M35" 0.40 --mine-budget-bits 0.50
run_cell 12b_b050 "$M12" 0.35 --mine-budget-bits 0.50

echo "V14_TRANSFER_COMPLETE"
