#!/usr/bin/env bash
# v14 Byzantine stage: integrity replication at the certified budget tier
# (B = 0.50 bits/token, the v14.1 candidate standard tier) on the three
# CPU-only replica nodes soula/voula/sasa (192.168.1.120/121/122), K=3,
# --byzantine-verify.  Server lifecycle (honest vs --malicious-delta 0.5 on
# .122) is orchestrated by the caller; this script only runs the cells.
# CPU replicas are slow (v9.5 measured 9.1-9.3x): STEPS defaults to a
# 400-step diagnostic, override for full-length runs.
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
STEPS="${STEPS:-400}"
WARMUP="${WARMUP:-100}"
URLS="wss://192.168.1.120:5013,wss://192.168.1.121:5013,wss://192.168.1.122:5013"
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  echo "=== v14 byzantine cell $name (steps=$STEPS) ==="
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
    --cloud-urls "$URLS" --cloud-channels 3 --byzantine-verify \
    --adversary-strength 1.0 --mine-budget-bits 0.50 \
    --attacker-bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v14_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v14_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v14_${name}_bundle.pt"
  echo "=== v14 byzantine cell $name done ==="
}

run_cell "$1"

echo "V14_BYZANTINE_COMPLETE"
