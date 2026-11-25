# Completed BiSR C-A conformance and evidence release

**20 September update:** the separately approved adaptation amendment has completed predictions, statistical analysis and archive-based CPU reproduction. See its [completion and release record](v4-adaptation-completed-2026-09-20.md). The launch disposition below records the earlier C-A closure snapshot.

The prospective single-seam C-A campaign completed at **02:29:59 BST on 19 September 2026** with `CONFORMANCE_ESTABLISHED_FOR_DECLARED_C_A`. This closes implementation conformance for the declared configuration only. The original processed SensiReplaced setting and published dual-cut reproduction remain unestablished; the historical audit and frozen v4 protocol are unchanged.

## Verified result and uncertainty

All 24 candidate/reference prediction cells completed their 256 documents: 6,144 document-engine runs, covering seeds 31415/27182/16180, boundaries 3/6 and public/exact head grants. Each retained SIP, b, f and bf outputs. All 12 trained mechanical checks passed. The independent release audit verified all prediction journals, manifest/source/checkpoint hashes, input identity, metric resources, the pre-prediction manifest Git commit and final analysis anchor. All 3,072 paired documents had identical reconstructed arm token sequences.

Every one of eight predeclared endpoints in each of four boundary/grant groups passed the ±0.01 equivalence margin. Paired candidate-minus-reference differences and 90% Student-t intervals across three seed means were all [0,0]. These intervals are conditional on the fixed inputs, implementations and sampled seeds: they do not establish zero population uncertainty or generalisation beyond this experiment. Positional accuracy and ROUGE-L F1 are the primary endpoints; METEOR is descriptive. This is implementation agreement, not an independent claim of attack strength or a privacy guarantee.

The 256 validation articles form an internal holdout from WikiText-103 training shards, not its official test partition. The 112 auxiliary articles yield 15,604 complete windows. Recorded historical documents/titles, WikiText-2 partitions and exact overlapping windows were excluded. These checks do not establish absence from language-model pretraining or semantic near-duplicates.

## Frozen provenance and compute

- Execution source: `8f8a4c4aade436aa31ef79ac294d97f1d392b4d3`.
- Pre-prediction manifest commit: `bda34f0d167f641d4f68fd1635d3ead167cea1e7`.
- Frozen v3 input manifest SHA-256: `4a4f5b76a3e86bf67eba48a35907c9381eddf34ed9e6c1ccf10662320486fe8d`.
- Final analysis SHA-256: `c4d3d5ba590cd066bb9d619761b82c8cf02567496b4bcccc5e63ae397bb7f106`.
- Upstream reference: SplitLLM `c376590227f9718077765824af4088665bb78687`; the retained reference source files and extracted algorithm are hash-pinned.
- Public GPT-2: `607a30d783dfa663caf39e06633721c8d4cfcd7e`; released model/tokenizer hashes match preparation.
- Preparation: 3,383.83 seconds. Preparation through final scoring: 68,123.05 seconds (**18.923 wall-clock hours**), within the approved 36-hour ceiling.

Per-document stage times, peak allocated memory and layer forward/reverse token-position counts are retained, with aggregate measurements in `INTEGRITY.json`. Shared b/bf gradient-matching work is counted once. These are not FLOPs or an equal-compute claim. The elapsed total includes CPU work and orchestration, and is not a measured GPU-utilisation integral. Offline training is separate from online stage measurements.

## Publication and reproduction

Public repository: `Setloop/llm-attacker-paper-b`, additive prefix `fidelity-20260919/evidence`.
Immutable revision: `be59bdca280258ac82ce358cae5f1e0c529869c2`.

[Complete evidence and reproduction instructions](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/be59bdca280258ac82ce358cae5f1e0c529869c2/fidelity-20260919/evidence).

The package retains every original prediction, checkpoint, auxiliary activation cache, observation, truth/input record, source/exclusion record, final metric report, journal, executed source, source archive, model/tokenizer, package inventory and metric resource. Excluded engineering canaries remain explicitly outside the scored set. Original absolute-path manifests are preserved; path relocation instructions do not rewrite their provenance. Both archive-part SHA-256 hashes and an internal per-file manifest are supplied. A fresh archive extraction and CPU re-score verify reproducibility without launching new GPU work. Remote verification must match every expected file at the immutable revision before publication is marked complete.

## Remaining campaign and publication requirements

| Item | Current disposition |
|---|---|
| Declared C-A single-seam implementation conformance | Closed by this qualified result and verified evidence release |
| Original processed SensiReplaced / published dual-cut reproduction | Open; not tested here |
| Frozen v4 B1 prerequisite | Still unsatisfied: its literal requirement is published-setting replication |
| Matched v4 capture evaluation | Not launched or retrospectively qualified; requires the original prerequisite or a separately reviewed prospective scope amendment |
| Paper B baseline, audit, 17 September manuscript | Previously verified; retained unchanged |
| Paper B full white-box package | Verified at `08ed4bd07ecde7613eff62826f6cce6638415d7b`, `whitebox35-20260917/full` |
| Paper B completed soula/voula/koula Phase 2 raw package | Verified at `7aac5e7b1e9db9978ef1ef1366eb9bc5a4637ed4`, `revision-20260917/phase2-raw` |
| Sasa convergence extension | Live terminal status is complete with no failed jobs, dated 17 September 22:22:32 UTC; still excluded pending collection, digest/journal verification and scientific analysis |
| Paper A corrected P6 raw package | Separate transfer/publication problem; not closed by this result |

The new manuscript reports the C-A result with the above limitations. No archived invalid result is relabelled, no frozen v4 protocol is edited, and no new GPU campaign is authorised by this closure record.

The 03:45 local follow-up trigger fired but failed because the conversation already had an active writer. Publication was resumed manually in the existing conversation under the user's explicit instruction; the trigger did not publish evidence or change the scientific result.
