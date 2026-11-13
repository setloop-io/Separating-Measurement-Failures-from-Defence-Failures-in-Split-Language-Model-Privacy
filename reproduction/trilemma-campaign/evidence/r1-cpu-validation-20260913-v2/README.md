# R1 CPU systems evidence package

This package preserves the final local CPU validation of the prospective systems
implementation. It is **engineering evidence only** and cannot support an R1
scientific claim.

- Archive: `r1-cpu-validation-20260913-v2.tar.gz`
- SHA-256: `7c76754f19b8479a2536c8cf2b362a28769f8ce6f7f004aab1d9fb0ab3e60a59`
- Indexed files: 528
- Extracted bytes: 33,707,654
- Compressed bytes: 24,286,417
- Restoration verification: 528/528 files, zero failures

`index.json` binds every archived path, size, mode and SHA-256.
`verification.json` records a fresh extraction and full byte check.

The run used real Tiny Shakespeare bytes, real model training, separate trusted and
corpus-free remote processes, authenticated framed tensors, coordinated checkpoints,
semantic artifact validation and draft attacks over all 12 target/channel combinations.
It did not use a GPU, a second host, paid infrastructure, admitted scientific thresholds,
calibration, exploration, confirmation or reserved seeds.

The source was a then-uncommitted working revision over `fd4b014`;
every run retains the same exact source inventory, environment and patch (inventory
digest `9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0`).

Updated 2026-09-13: that revision is now committed as `6ebc2bf..5f9fcb0`, and the
source inventory still recomputes to the same digest over 97 bound files, so the seal
survived the commit unchanged. The newly prepared workload was produced from clean
commit `5f9fcb0` and is retained at `evidence/r1-preparation-20260913/`. Both of those
GPU-authorization prerequisites are met. The remaining blockers are the signed
budget/environment review and the separate Stage A and Stage B admissions.

Tensor payloads under `outputs/` are excluded by `.gitignore`, which admits only
`.json`, `.jsonl`, `.md`, `.txt`, `.py`, `.csv`, `.log` and `.png`. The archive above
is the complete self-verifying copy of each run tree, and `index.json` binds every
archived path, size, mode and SHA-256.
