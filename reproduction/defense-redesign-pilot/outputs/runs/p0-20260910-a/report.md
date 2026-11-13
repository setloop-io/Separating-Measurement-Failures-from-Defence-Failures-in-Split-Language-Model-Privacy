# P0 engineering report: p0-20260910-a

Status: engineering diagnostics only. No scientific hypothesis, instrument
qualification, convergence, or defense-success claim follows from these runs.

- Completed: 10/10; failed: 0.
- Every recorded artifact hash and every archived source file verified.
- Completed runs have finite losses and the expected remote-update behavior.
- Split runs preserve full-precision forward/returned values, both gradients, and quantizer metadata.
- Probe fitting and evaluation use separate text regions; all diagnostic probes remain unqualified.
- Privacy outcome: **undetermined**; active attacks were not evaluated.

Source tree: `e143a79dc0874b146a62b1858df1fe20673b0ef3ccea77dccc132edf13b58aa0`. See [source record](source.json),
[matrix summary](summary.json), and [verification details](verification.json).

| Cell | Mode | Status | Remote changed | Capture panels |
|---|---|---|---|---|
| all-local | all_local | completed | False | 0 |
| d0-q8 | frozen_random | completed | False | 2 |
| d0-q8 | removed_remote | completed | False | 2 |
| d0-q8 | trained | completed | True | 2 |
| d1-s01-q8 | frozen_random | completed | False | 2 |
| d1-s01-q8 | removed_remote | completed | False | 2 |
| d1-s01-q8 | trained | completed | True | 2 |
| d3-q2 | frozen_random | completed | False | 2 |
| d3-q2 | removed_remote | completed | False | 2 |
| d3-q2 | trained | completed | True | 2 |

Omitted attacks: timing; optimizer state; full evolving remote state; active attackers; adaptive attackers; within-run joint tensors; cross-run joint attacks; history accumulation; canary membership; natural-sequence membership; unrestricted semantic recovery; quantizer metadata attacks; returned values attacks; full-precision inbound/outbound gradient attacks.

Next: specify and qualify the calibration controls, resolve the D1 basis/checkpoint
policy, and freeze the draft pilot. The three-step losses and two-epoch probe scores
are retained in per-run metrics solely for debugging; they cannot select a defense.
