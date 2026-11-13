# Production-v4 candidate sweep

This directory defines the next fail-loud 0.6B training ablation. It is not a
production approval. Toula is trusted; Koula is compromised. The fast
dense sandwich protects saved wire captures only and is deliberately retested
by the adaptive attacker before it may replace dense folding. The structured
Hadamard candidate was rejected after its first adaptive attack.

Run arms in `sweep.json` in order. A candidate advances only if
`bin/validate_v4_candidate.py` passes privacy, utility, cost, and corrected
token-level DP gates. Private norm telemetry is diagnostic and may not be used
to claim formal DP; formal clipping parameters must be fixed using public or
separately DP-calibrated data.
