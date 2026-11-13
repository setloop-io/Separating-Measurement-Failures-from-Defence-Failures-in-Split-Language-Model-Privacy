# Paper 3 reviewer revision — 2026-09-10

> **R1 reset (2026-09-10): historical reference only.** This document describes
> pre-reset work. All associated observations and analyses are inadmissible for
> active scientific claims. The current study has no results; follow
> [the R1 plan](R1_RESEARCH_PLAN.md) and [active ledger](CLAIM_EVIDENCE_LEDGER.md).
> The reference tag preserves the original version of this document.

The revision qualifies historical provenance, corrects control and privacy-gate
interpretation, and makes the retained evidence inspectable. It preserves the
original protocols, manifests, raw observations, and analysis summaries.

## Manuscript

- Replaced unqualified preregistration claims with documented cohort-specific
  protocols and an explicit historical source/timing limitation in the abstract
  and main text. Same-team cross-setting replication is identified as such.
- Disclosed the P2 width-64 bypass versus the headline width-256 remote model,
  and the P5 comparison of D1 with the D0 bypass.
- Retained the joint finite-set finding through all D0 qualified structure
  failures and all D1 utility failures. The P5 table now shows structure-gate
  outcomes, including the three D1 probe passes, with full coverage limitations.
- Added the recovered known-plaintext byte attack as explicitly exploratory and
  the new target-resolved P5 property refit as retrospective verification.
  Original summaries and their family-rule classifications remain intact.

## Verification

The [verification package](../outputs/revision_validation_20260910/README.md)
retains commands, input/source hashes, individual scores, and comparisons.
The known-plaintext refit matches all sixteen seed records and 128 fold scores.
The P5 property refit matches all 112 aggregate scores and retains qualified
punctuation scores above the gate in every cell. All fourteen cells' utility,
gain, and analytic-share fields also match their original values.

Qualification was reused. Full structure and gradient probes were not independently
refitted; their stored results retain that limitation. No historical model-training
cohort was rerun. The preservation check verifies all 5,308 original indexed files
(52,148,895,660 bytes), with no missing or changed files.

## Prospective software repairs

Completion checks now validate schema, required artifacts and hashes, configuration,
protocol/run/output identities, and retained source before skipping an execution.
Unverifiable or occupied runs stop explicitly. New executions retain a bounded
source archive, file inventory, configuration, Git patch, available environment
listing, invocation, and run-start metadata. Source, patch, environment, and config
digests are checked consistently, including symlink and missing-information cases.

All nine cohort drivers and the single-cell entry point support a separate
`--manifest-dir`; the pilot also exposes `--output-root`. Dry-runs enumerate blocked,
pending, and verified cells without training. Existing failed attempts cannot be
overwritten by an automatic retry. Legacy manifests remain readable; missing
historical snapshots cannot become source-verified through this repair.

The full suite passed 224 tests and 29 subtests. A subsequent availability-fallback
regression was added and all 34 provenance/resume tests passed. The updated
15-page PDF builds without LaTeX warnings and was visually inspected.
Exact commands and file hashes are in
[revision_checks.json](../outputs/revision_validation_20260910/revision_checks.json).
Tiny synthetic unit executions are software checks, not scientific cohorts.

## Remaining boundaries

Paper 3's exact historical execution snapshots and independent protocol-freeze
timing remain unrecovered, including after the bounded Paper 1/Paper 2 search.
The [provenance supplement](HISTORICAL_PROVENANCE.md) records that limitation.
Deployment measurements, independent-team replication, and broader attack coverage
remain future research. No new universal impossibility or privacy guarantee is claimed.

Both repositories remain private. The v1 evidence release retains its original
source target and index; these subsequent repairs and verification records are
available in Git. Completion of the large evidence upload is recorded in the
release's `VERIFICATION.json`; this revision does not label an incomplete upload
as fully available to reviewers. Follow [REVIEWER_EVIDENCE.md](REVIEWER_EVIDENCE.md).
