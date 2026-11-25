#!/usr/bin/env bash
# v14 long-horizon cell: the certified standard tier (B=0.50) at 10,000
# steps, matching the v12.2 convergence protocol — blunts the
# "diagnostic scale only" critique for the paper's central cell.
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

echo "=== v14 long-haul cell v141_b050_10k ==="
python3 bin/run_latent_native_v5_06b.py \
  --model "$MODEL" --corpus "$CORPUS" \
  --output "$OUTDIR/latent_v14_v141_b050_10k.json" \
  --cloud-tls-ca "$CA" \
  --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
  --secret-wire-rotation --secret-token-permutation \
  --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
  --split-after 21 --resume-after 26 --seq-len 32 \
  --steps 10000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
  --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
  --lr 3e-4 --remote-grad-clip 1.0 \
  --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
  --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-budget-bits 0.50 \
  --attacker-bundle "$BUNDLEDIR/latent_v14_v141_b050_10k_bundle.pt" \
  || { echo "CELL FAILED"; exit 1; }

python3 -m attacker --attack latent-probe \
  --bundle "$BUNDLEDIR/latent_v14_v141_b050_10k_bundle.pt" \
  --output "$OUTDIR/latent_v14_v141_b050_10k_attacker.json" || echo "PROBE FAILED"
rm -f "$BUNDLEDIR/latent_v14_v141_b050_10k_bundle.pt"
echo "V14_LONGHAUL_COMPLETE"
