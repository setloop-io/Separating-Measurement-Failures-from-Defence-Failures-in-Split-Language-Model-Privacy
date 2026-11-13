# Paper A closure record — 17 September 2026

Updated 20 September: the manuscript correction and **complete corrected P6
raw-evidence release are verified**. The broader closure queue still includes
the separately amended Qwen experiment and final preprint integration.

## Revised scope

Paper A now reports the canonical v3 reconstruction campaign and the corrected
GPT-2 P6 Mode-2 follow-up. The title is *Beyond Layer Count: Activation and
Gradient Leakage in Split Language Models*.

- Withdrawn as active claims: the historical 738-run programme, its causal
  surrogate conclusion, corpus-dependent frontier and empty three-gate set.
  Their owner retired the results for missing execution-source provenance.
- Preserved: the verified v3 reconstruction endpoint and descriptive controls,
  with fixed-state observation, shared Paper B evidence and separate utility
  scheduling stated explicitly.
- Added: 192 corrected GPT-2 P6 trained cells, eight controls, four endpoint
  tests, fixed held-out-seed prediction checks and all 24 crossed cell means.
  The D1 leakage test fails corrected alpha; its positive bootstrap interval
  is not presented as a significance test. The interval level is 98.75%.
- Clarified: P6's trainable range is [start,start+span), but the observed seam
  is max(1,start). Start therefore changes the observation depth as well as
  the trainable range for starts above one. It is not an isolated training-
  location effect with observation depth fixed.
- Corrected historical appendix: P0 engineering checks, Pearson-r advantage,
  eight P2 gains, correctly centred t and exact Wilcoxon statistics.
- Retained limitations: original-setting BiSR fidelity is unestablished.
  Paper B now reports qualified single-seam conformance and a separate adapted
  comparison at distinct budgets. The original FP32 Qwen run failed; a fresh
  FP64 amendment is running and remains excluded until qualification. External
  R1 exploratory executions are not held-out confirmation.

The original broad feasibility question requires fresh admissible experiments
with jointly measured gates. It is outside the revised paper's claims. The
separately authorised Phase 2 and PR #35 jobs remain independent workstreams.

## Validation

- `python -m pytest tests/ -q`: **2,563 passed, 36 skipped**, 343.67 seconds.
  The skipped checks are not represented as passed.
- PDF: 12 pages; pdflatex/bibtex build completed. Final log has no undefined
  references/citations, overfull boxes or LaTeX warnings. All 12 rendered pages
  inspected; figures, tables, captions and page numbering are legible.
- `git diff --check`: passed.
- Publication helper: command-line smoke check and split-archive round-trip
  passed. Manuscript upload and pinned Git/LFS digest verification passed.

## Immutable release anchors

Repository: `Setloop/llm-attacker-paper-a` (public when checked on 20 September).
No visibility change was made by this closure task.

1. Baseline v3 and historical audit:
   `a02fb09280dae643da4f5a11abf08ca8f62d8337`.
2. Revised manuscript, figures, P6 analysis, protocol, input, frozen source and
   reproduction instructions: **19 files verified** at
   `f7b2aef7649588e9021805f2edc25428b6f26c93`, under
   `revision-20260917/manuscript/`.
   [Pinned revision index](https://huggingface.co/datasets/Setloop/llm-attacker-paper-a/blob/f7b2aef7649588e9021805f2edc25428b6f26c93/revision-20260917/manuscript/README.md).
3. Corrected P6 full raw tier: **11 files, 34,703,146,309 bytes verified** at
   `0bec5c37b960fdb5be5b2bd8fab1cdd05a4bbe4c`, under
   `revision-20260917/p6-raw/`. This includes captures, checkpoints, predictions
   and evidence journals for all 200 qualified cells.
4. Current manuscript and additive release index: **14 files, 573,793 bytes
   verified** at `2540f5904e5aaeb019179f9f528209a10f746ea1`, under
   `closure-20260920/manuscript/`.
   [Pinned closure index](https://huggingface.co/datasets/Setloop/llm-attacker-paper-a/blob/2540f5904e5aaeb019179f9f528209a10f746ea1/closure-20260920/manuscript/README.md).

Local verification:
`build/paper-a-revision-20260917/manuscript-status-verification.json`.
Manifest SHA-256:
`6720b92430680da6dd47aa2a50ca6b464cbe696263e04db5f3bd518f78cea95a`.
The old baseline manuscript and INDEX remain immutable historical artefacts;
this revision index supersedes their scientific claim mapping.

## Completed full-evidence publication — 20 September

The publisher on koula verified all 200 source cells against the frozen
protocol, analysis anchors, file digests and journals, archived the evidence
into bounded parts and completed immutable HF verification at 19:00:18 UTC.
Its PID 2877565 is no longer running; no replacement publisher is needed.
Source evidence remains at
`koula:~/llm-attacker-p6-v2-d5ec03e/results/p6-corrected-gpt2-v2/`.

The separate local check verified the exact pinned HF file set, downloaded
small-file SHA-256 hashes and each archive part's pinned LFS SHA-256 digest.
It did not perform another full 34.7 GB download and extraction. Reports are
preserved in `build/paper-a-raw-closure-20260920/` and the current HF manuscript
tier. The durable combined record is
[paper-a-release-verification-2026-09-20.json](paper-a-release-verification-2026-09-20.json).
Raw manifest SHA-256:
`0daaabcefa94de9631f8e7f9b448a8383b0b20245d73e4ab105601b2e57e4f17`.

The updated 12-page PDF passed compilation and visual review. Both provisional
arXiv packages were rebuilt at
`build/arxiv-preprints-20260920-paper-a-raw/INDEX.json`; all ZIP payload and PDF
hashes matched their metadata. Existing figures and scientific analyses are
unchanged. The earlier test suite result above belongs to the 17 September
revision; it was not rerun for this release-link and scope-wording update.
See the [current release index](paper-a-release-2026-09-20.md) for pinned
reproduction instructions. The old pending/private status statements remain
only in immutable historical HF copies and are superseded by this index.
