# Qwen3-0.6B v4 dense-sandwich diagnostic

This rotation-only arm tested the privacy-tested dense orthogonal family in a
wire sandwich, avoiding cloud-weight refolding. Code commit: `92aca04`.

Utility passed: held-out loss was 3.772 versus 3.688 plain (delta +0.084), and
the cloud closed with zero pending graphs. Cost improved from the v3 full
stack's 60.74 s to 21.31 s, but remained 8.02x the 2.657 s plain step and
failed the 3x cost gate because each peer independently derived 17 portable
dense QR transforms.

Privacy also failed without boundary noise. Pooled recovery was 21.88%,
21.88%, and 18.75% for 8/16/32 solve rows per epoch (128/256/512 total),
versus 5.47-6.25% label-free bands. Therefore neither increasing the ratchet
budget nor promoting dense sandwich alone is justified. The result shows the
previous v3 privacy pass depended materially on its destructive boundary DP.

No boundary DP or DP-SGD was enabled. Raw captures and seeds were deleted.

