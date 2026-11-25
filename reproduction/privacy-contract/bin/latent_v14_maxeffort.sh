#!/usr/bin/env bash
# v14 max-effort battery (P2) + CE cross-check (P3) at the certified
# standard tier (v14.1 config, B=0.50, seed 42).  Reruns the cell keeping
# the trusted bundle, then throws the strongest real attacks at it:
#   - latent-probe at 4x optimization (epochs 60, restarts 8)
#   - latent-matching (VMA / permutation-reversal arms)
#   - latent-sensitivity
#   - ce_probe_mi.py: independent H-CE second-family MI lower bound
# The bundle is trusted-side capture: deleted immediately after scoring.
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLE=$OUTDIR/bundles/latent_v14_v141_b050_kept_bundle.pt
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$OUTDIR/bundles"

echo "=== v14 maxeffort: rerun v14.1 keeping bundle ==="
python3 bin/run_latent_native_v5_06b.py \
  --model "$MODEL" --corpus "$CORPUS" \
  --output "$OUTDIR/latent_v14_v141_b050_kept.json" \
  --cloud-tls-ca "$CA" \
  --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
  --secret-wire-rotation --secret-token-permutation \
  --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
  --split-after 21 --resume-after 26 --seq-len 32 \
  --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
  --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
  --lr 3e-4 --remote-grad-clip 1.0 \
  --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
  --cloud-url wss://koula:5025 --adversary-strength 1.0 \
  --mine-budget-bits 0.50 \
  --attacker-bundle "$BUNDLE" \
  || { echo "CELL FAILED"; exit 1; }

python3 -m attacker --attack latent-probe --epochs 60 --restarts 8 \
  --bundle "$BUNDLE" \
  --output "$OUTDIR/latent_v14_v141_b050_maxeffort_attacker.json" \
  || echo "MAXEFFORT PROBE FAILED"
python3 -m attacker --attack latent-matching \
  --bundle "$BUNDLE" \
  --output "$OUTDIR/latent_v14_v141_b050_matching.json" || echo "MATCHING FAILED"
python3 -m attacker --attack latent-sensitivity \
  --bundle "$BUNDLE" \
  --output "$OUTDIR/latent_v14_v141_b050_sensitivity.json" || echo "SENSITIVITY FAILED"
python3 bin/ce_probe_mi.py --bundle "$BUNDLE" \
  --output "$OUTDIR/latent_v14_v141_b050_ceprobe.json" || echo "CE PROBE FAILED"

rm -f "$BUNDLE"
echo "V14_MAXEFFORT_COMPLETE"
