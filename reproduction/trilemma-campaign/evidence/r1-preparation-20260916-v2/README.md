# R1 workload preparation v2 — 2026-09-16 (owned-node engineering)

Prepared from the clean master checkout at `fcd0922` (source inventory seal
`c96d16ecde0c8982eb0cf1761310b349ba4f1611e92e73c54be4db0c30f097e0`, 97 bound files).
v2 exists because Stage A execution `engA001` failed on the defended-gradient cell:
`remote_service._defend_outbound` moved defense noise to the gradient dtype but not
its device (`cuda:0` vs `cpu`). The one-line fix is commit `fcd0922`. The v1 bundle
(`45d7682e…`), its consumed admission and the preserved engA001 failure records stay
untouched as failure evidence. **Engineering only: `scientific_execution_admitted: false`.**

- `workload.tar.gz` — the complete prepared bundle (183 files), archive SHA-256
  `54b21ae2415756c4d8bf518b2d5c5f4bc48fc9df5d34b713d01aea172219d9e1` (xattr-free).
- `r1_bundle.json` — bundle inventory; bundle SHA-256
  `e123089f1955e9fb7c9d22b0035e448a57a7b5960c4609a30eabcab3e6f03a3d`.
- `plan.json` — the four planned workloads (all engineering/blocked-draft).
- `validation.json` — archive/restore/verify record and the failure→fix→new-packet chain.

Restoration check: extract `workload.tar.gz` to a new directory and run
`python <restored>/r1-bundle-eng-20260916-v2/scripts/run_r1.py verify-bundle` with a
Python environment carrying the project's runtime dependencies.
