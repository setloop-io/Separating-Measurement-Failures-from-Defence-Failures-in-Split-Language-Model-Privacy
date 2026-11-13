# Qwen3-0.6B v4 structured-transform diagnostic

This one-step Toula/Koula arm tested the signed-permuted Hadamard wire
sandwich at four-direction budget B=256. Code commit: `5c1763d`.

## Outcome

The speed and utility gates passed relative to the matched historical plain
arm: step time was 3.563 s versus 2.657 s (1.34x), and held-out loss was 3.963
versus 3.688 (delta +0.275). Evaluation closed with zero pending cloud graphs.

The privacy gate failed. Cross-epoch recovery was 18.75% with 8 solve rows per
epoch (128 total), 21.88% at 16 rows/epoch (256 total), and 21.88% at 32
rows/epoch (512 total), versus label-free bands of 1.17-1.76%. The structured
mode is therefore rejected and remains experimental. It must not replace the
dense transform.

No boundary DP or local DP-SGD was enabled in this rotation-only ablation.
Raw captures and seeds were deleted after aggregate attacker evaluation.

