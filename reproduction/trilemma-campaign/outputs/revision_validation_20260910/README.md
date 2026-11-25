# Post-hoc verification of retained Paper 3 analyses and probes

Verified on 2026-09-10. This is a **new verification record**, not a new
confirmatory cohort, reconstructed preregistration, or reproduction of historical
model training. No original result, capture, manifest, or scientific scorer was
overwritten. The historical execution-source and freeze-timing limitations remain.

## Findings

- The known-plaintext ridge probe was independently refitted. All **16 seed
  records and 128 fold accuracies match exactly**, including null accuracies,
  majority baselines, and original capture hashes. The mean byte accuracies are
  **96.71630859375% at q4** and **97.802734375% at q32**. Mean shuffled-label null
  accuracies are 7.40966796875% and 9.2529296875%; the majority baseline is
  13.76953125% at both precisions.
- P5's forward property probes were refitted from calibration-s919 using the
  unchanged scorer, seed 919, original q8/q4 defense mapping, and original
  `matched_test_null=True` rule. All **112 per-seed maxima over the three targets
  match the archived aggregation exactly** (maximum absolute difference 0).
  The new output also retains the previously discarded individual targets.
- Punctuation alone exceeds P5's 0.02 worst-seed gate in all 14 cells in this
  post-hoc recomputation. This resolves which target rejects each cell under the
  reused qualification; it does not change the historical aggregation record or
  supply a new qualification study.
- P5 utility, causal-gain contrasts, and analytical share were recomputed from
  retained curves with the original analyzer. All selected fields, including
  per-seed contrasts and utility confidence intervals, match exactly in all
  14 cells; no operational failure was reported.
- A final read-only preservation check verified **all 5,308 original indexed
  files (52,148,895,660 bytes)**, with zero missing or mismatched files. The
  original release index remains byte-identical to its committed version.
  See [`original_evidence_preservation.json`](original_evidence_preservation.json)
  and [`preservation_execution.json`](preservation_execution.json). This establishes
  preserved file identity, not independent validation of every scientific claim.

The known-plaintext attack fits seven fixed 128-byte windows (896 labeled
byte/activation pairs) and tests the eighth. Reconstructed byte offsets verify
that these eight windows do not overlap; they are from the same held-out region
and are reused across all model seeds. This is not fresh-document recovery,
reconstruction without known plaintext, or an independent new-document sample.
The original JSON's additional `verification` object is not emitted by the
standalone script and was not invented in the new script output.

## Qualification and target-resolved P5 results

Qualification was **reused without tuning or refitting** from
[`privacy_extended_p4b.json`](../privacy_extended_p4b/privacy_extended_p4b.json).
The original positive-signal floor is 0.10 and absolute null band is 0.02.

| Target | Stored positive-control r advantage | Stored null-pipeline advantage | Individually qualified |
|---|---:|---:|---|
| Punctuation fraction | 0.714849 | 0.002866 | Yes |
| Vowel fraction | 0.754259 | 0.027804 | No: null band exceeded |
| Newline count | 0.544007 | 0.089633 | No: null band exceeded |

The following are newly retained **worst-seed punctuation r advantages**. They
are differences of correlations and can exceed one; they are not probabilities.

| Span (start) | D0, q8 | D1, q4 |
|---|---:|---:|
| 1 (3) | 0.969676 | 1.094183 |
| 2 (3) | 1.241990 | 1.090166 |
| 3 (2) | 0.480547 | 0.662555 |
| 4 (2) | 0.866115 | 1.042183 |
| 5 (1) | 0.449698 | 0.748680 |
| 6 (1) | 1.014506 | 1.055428 |
| 7 (0) | 0.803087 | 0.670588 |

This property refit used CPU, Python 3.12.12, PyTorch 2.14.0, and NumPy 2.5.2.
The 116 retained inputs were checked against the release index before use and
rehashed after use; all remained identical. Imported source files and the
original P5 scorer were hashed before and after the run and did not change
during verification. Exact environment, input hashes, source hashes, and all
per-target/per-seed values are retained in the JSON output.

**Structure and gradient probes were not independently refitted.** A single
CPU training epoch on the 131,072 calibration examples took 1.323 seconds,
projecting roughly 17.6 minutes for the four frozen 200-epoch forward structure
fits. That benchmark model was discarded and supplied no scientific score.
Gradient qualification, cross-model token inversion, and structure results
remain retained evidence rather than independently refitted results here.

Timing, optimizer state beyond the remote module, full evolving remote state,
returned values, full-precision inbound/outbound gradients, active/adaptive
attackers, within-run joint tensor attacks, cross-run joint attacks,
history-accumulation, canary/natural-sequence membership, and unrestricted
semantics are not evaluated by this forward-only verification. Previously
unqualified instruments remain unqualified. Reusing the original calibration
does not establish complete-transcript privacy or remove latent-transfer and
fixed-text-panel limitations.

## Outputs and reproduction

- [`comparison.json`](comparison.json): exact comparisons and input checks;
  [`compare_validation.py`](compare_validation.py) regenerates this comparison.
- [`within_run_probe.json`](within_run_probe.json): independently refitted
  known-plaintext records; [`within_run_execution.json`](within_run_execution.json)
  and [`within_run_probe_path_change.diff`](within_run_probe_path_change.diff)
  record the sole script change: make the original repository root explicit so
  the copied script writes its result next to itself. Mathematical operations,
  labels, seeds, folds, and regularization were unchanged.
- [`p5_property_refit.json`](p5_property_refit.json): target-resolved results and
  provenance; [`refit_p5_property.py`](refit_p5_property.py) regenerates them.
- [`p5_nonprivacy_analysis.json`](p5_nonprivacy_analysis.json): utility, gain,
  and share recomputation. **Ignore this file's privacy flags, `holds_all`, and
  triangle verdict:** privacy scoring was disabled, so those fields are
  missing-score placeholders, not observed attack failures or a new joint-gate
  result. The raw generated summary carries the same warning. Use
  `comparison.json` for the fields actually validated.

Commands from the Paper 3 repository root (create an isolated environment with
the listed dependencies; no SciPy is needed for these probes):

```bash
VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python outputs/revision_validation_20260910/within_run_probe.py
VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python outputs/revision_validation_20260910/refit_p5_property.py --benchmark-structure
.venv/bin/python scripts/analyze_delegation.py --reuse-qualification --no-privacy-scoring --no-figure --analysis-path outputs/revision_validation_20260910/p5_nonprivacy_analysis.json --summary-path outputs/revision_validation_20260910/p5_nonprivacy_raw_summary.md
.venv/bin/python outputs/revision_validation_20260910/compare_validation.py
```

Run these in a disposable evidence checkout. The copied known-plaintext script's
explicit `ROOT` must be changed to that checkout's root (the only path-specific
line); do not point its output at the historical review directory. The P5 driver
accepts `--root` and `--output` for alternate paths. These commands overwrite
only the **new verification outputs**, so preserve this dated directory before
performing another comparison. The P5 analyzer's raw Markdown retains its old
template wording; a leading verification warning was added here after the run,
without changing any original summary.
