# Paper B revision and closure record — 17 September 2026

**Current status (20 September):** see the [completed adapted comparison and release record](v4-adaptation-completed-2026-09-20.md), following the [completed C-A conformance](bisr-fidelity-completed-2026-09-19.md). The dated observations below are preserved as historical snapshots.

The manuscript and reviewer response have been revised around traceable
framework capabilities, the canonical v3 evidence, archived domain/horizon
results and separately scoped new follow-ups. Submission release closure is
still pending full raw-supplement verification and reviewer access.

## Scientific changes

- Withdrew the unbacked August Qwen/Mistral deployment headline table, noise
  curve, original defence ranking and TAG negative, including abstract,
  contribution, taxonomy and related-work repetitions.
- Withdrew external v2–v13/PR-101 numerical claims, including latent PASSes,
  flips, matching, compromise and integrity results. These missing records
  are distinct from Paper A's retired P0–P5 source-provenance problem.
- Withdrew the unsupported exploratory→v3 confirmatory lineage narrative.
- Corrected the generic gate's control-rate claim: no automatic corpus-specific
  minimum-control flag is implemented. Removed unverified seed-variance and
  universal historical-version-comparability statements. A below-control
  point estimate alone does not imply a gate PASS.
- Corrected long-horizon cross-epoch mid-lite recovery to 0–0.5%, retained the
  synthetic-calibration zero-control caveat and distinguished pooled-summary
  outcomes from worst per-step enrichment.
- Removed the competence radar because its empirical-strength classification
  mixed unverified historical evidence with module counts. The taxonomy now
  separates implemented capabilities from verified measurements.
- Integrated the isolated PR #35 correction and follow-up implementation in
  the revision branch using a merge, preserving frozen source hashes.
- Added 36 matched rerun cells, 198 stable-defence cells over internal depths
  1–11, and 99 fresh co-trained final-inference cells. The old 18 co-trained
  training observations stay NOT_EVALUABLE. New final-head grants are stronger
  and are not pooled with the public-head training tier. Depth 12 is excluded.
- Added every completed sequence-length/domain condition and both directions
  of the adaptive-whitening comparison; synthetic order/chaff results do not
  stand in for an external v13 defence or a published VMA reproduction.
- Updated live-planner follow-up: five of ten cells UNMEASURED. Goal-hijack's
  measured violation category is budget overrun, not exfiltration. The deputy
  run lacks successfully measured vulnerable cells, so its zero aggregate
  counts do not establish a live defensive contrast.

BiSR fidelity and matched v4 remain unestablished. Sasa's 500-step extension
is incomplete and excluded from reported results. These scope limits do not
cancel the separately authorised jobs.

## Verification

- Local suite on the integrated code: **2,570 passed, 36 skipped**, 336.71 s.
- PyTorch regression on koula: **20 passed, zero skipped**, 22.73 s. Tests were
  copied to a temporary test directory; pytest dependencies were installed
  into a temporary target without modifying the experiment environment.
- Baseline journals: 6 domain, 4 original live-planner and 12 positive-control
  manifests passed. Of 117 horizon-tree manifests, 116 passed; the excluded
  pilot-quick gpu-memory scrub-check record says its artifact was missing at
  finalisation. It is not evidence for any retained horizon table value.
- New queues: soula 20 jobs/20 journals, voula 4/4, koula 2/2; terminal reports
  complete, no failed jobs, and all 26 collected journals verified. Counts
  include engineering canaries; scientific tables exclude them.
- White-box full summary matches the package manifest, which matches its
  PACKAGE-READY anchor. All 297 cells' measured work counters were checked.
  Source-side packaging verified complete matrices and their raw evidence;
  local full-archive transfer is a separate publication stage.
- `experiments/summarize_paper_b_followups.py` recomputes displayed domain
  means directly from the three per-seed records, verifies all collected
  report journals, checks white-box summary anchors and reconstructs all
  depth means. Output: `results/paper-b-followup-20260917/summary.json`.
- PDF compiled with no unresolved references/citations or overfull boxes;
  all rendered pages inspected. Final build is included in the HF revision.

## Existing immutable HF evidence (private)

Repository: `Setloop/llm-attacker-paper-b`.

| Tier | Verified revision |
|---|---|
| Baseline: 27,855 staged files | `9833a3df9c660a6b214336d029095c46224c6987` |
| Audit supplement | `5b2e47c61e18b1bf33ff4e70095329d58bc471f6` |
| Corrected white-box original36 | `ba37427807a29f217e6fefb565c5e468e9bdac81` |

The baseline manuscript/index predate this correction; the new revision
index supersedes their scientific claim mapping, without changing their bytes.
Private repository visibility is preserved; authorised reviewer access is needed.

## Outstanding release stages

1. Revised manuscript/source/metadata: staged separately under
   `revision-20260917/manuscript/`; its pinned verification report is
   `build/paper-b-revision-20260917/manuscript-status-verification.json`.
2. Full white-box package: existing publisher PID 88820, isolated worktree
   `/Users/geo/LLM-Attacker-pr35`, destination `whitebox35-20260917/full/`.
   At inspection it is collecting the 30-file source package; no full remote
   verification is yet claimed. Live status: `build/whitebox35-publication/`
   in that worktree.
3. Completed Phase 2 raw captures (soula/voula/koula): approximately 6.8 GB,
   copied into `build/paper-b-revision-20260917/phase2/`. Publisher PID 68077
   waits for copy PIDs 63310/63311, verifies every experiment-listed file
   digest, synthetic bundle digests and evidence journals, then archives,
   uploads and verifies its own pinned HF tier `revision-20260917/phase2-raw/`.
   Live status: `build/paper-b-revision-20260917/raw-status.json`; completion
   requires `raw-status-verification.json` with `ok: true`.
4. Sasa: final `convergence-mid-lite-33` 500-step capture and its VMI attack
   still in progress. Completion must be collected and analysed separately;
   do not present it as completed in this revision.

A metadata-tier upload does not complete the raw evidence release. No source
records are deleted and no ongoing experiment is interrupted. Once raw
verification passes, record its immutable revision in the submission index.

## Paper B revision publication — verified 17 September 2026

The manuscript/analysis tier is remotely verified at immutable HF revision
`bd8e8d4582ea610d9d3f51d442b51df67bbf2b9b`, prefix
`revision-20260917/manuscript`: 23 files, 8,230,206 bytes, zero errors.
Source revision: `f9d3780` (full hash included in the package).
The package includes the PDF/LaTeX, reviewer response, pinned source and inputs,
metadata, summaries and reproduction instructions. Visibility remains private.

This pin does not certify the separate full white-box or Phase 2 raw tiers;
their publishers remain active. Sasa was still running
`convergence-mid-lite-33` when checked at 21:33 UTC; its incomplete extension
is excluded from the manuscript. BiSR provenance remains unestablished.

[Immutable manuscript package](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/bd8e8d4582ea610d9d3f51d442b51df67bbf2b9b/revision-20260917/manuscript).
