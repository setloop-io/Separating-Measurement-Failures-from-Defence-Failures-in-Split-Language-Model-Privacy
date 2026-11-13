# Paper 3 evidence availability: screenshot fact-check

Audit date: 2026-09-10. This report assesses the supplied screenshot as review
material, not as instructions to rerun or modify experiments. The screenshot's
filenames and counts identify **Paper 3**. Paper 3's local and GitHub master both
resolve to `52191c430110bd8d9dee80538fde22bf1726e56a`; Paper 4's published master
is `365a6d6347ad9da58c0fdbf484a57a98b565ac30`.

## Finding

The screenshot identifies a real evidence-distribution and reproducibility
problem. It does **not** establish that the underlying observations are lost.
All 4,492 manifest-referenced artifacts are present locally and match their
recorded SHA-256 hashes. Their combined size is 52,139,663,688 bytes (52.14 GB).
None is tracked in Paper 3's Git repository. A clone therefore lacks them.

The five analysis JSONs used by the manuscript displays are also present locally.
Every one matches the source hash recorded in the committed
`paper/figures/manuscript_visuals_provenance.json`. None is tracked, and none has
history in the inspected reachable Git history. These findings distinguish file
availability and byte identity from independent validation of the analyses.

Full per-artifact checks and summary hashes are in [inventory.json](inventory.json).

## Claim-by-claim assessment

| Screenshot claim | Checked finding | Assessment |
|---|---|---|
| 768 manifests; 687 confirmatory | Exactly 768 completed manifests: 687 confirmatory, 50 exploratory, 31 calibration | Confirmed |
| 4,492 unique references; zero artifacts present | All 4,492 are absent from tracked Git contents, but all are present locally and hash-match | Correct for a Git-only checkout; incorrect as a claim of local loss |
| Five analysis JSONs absent | All five are local, untracked, absent from inspected Git history, and match committed figure-source hashes | Publication omission, recoverable locally |
| Every manifest points to the initial scaffold and a dirty tree | All reference `be32144570e6fdb6cbdbdf367c9e71ffa685d24a`; all have non-null dirty hashes, with nine distinct values | Confirmed; exact execution source remains unestablished |
| Every manifest carries the exploratory disclaimer | The exact note is hard-coded in `src/pcr/manifest.py:94`, including all 687 confirmatory records | Confirmed metadata inconsistency; the note alone does not classify the actual study design |
| Fresh reproduction silently skips missing runs | The completion helper checks only `status == completed`; drivers then skip without checking artifacts | Real defect; skips are printed, and missing corpus files can cause an earlier failure |
| The 96.72% / 97.80% within-run result cannot be recovered | Script, result JSON, and all sixteen source captures exist locally; captures match review and manifest hashes; stored arithmetic matches the reported values | Exact bundle is missing from Git, but locally recoverable; classifier was not independently refitted in this audit |

The five recovered analysis paths are:

- `outputs/confirmatory/confirmatory_analysis.json`
- `outputs/replication/replication_analysis.json`
- `outputs/rate_sweep/rate_sweep_analysis.json`
- `outputs/segment_stats/segment_stats_analysis.json`
- `outputs/delegation/delegation_analysis.json`

Paper 3's `.gitignore:13` excludes `outputs/*`, except its policy file. That explains
the difference between this machine and a clone. Committing tables and figures
without publishing their inputs does not satisfy the reproducibility expectation
in `outputs/README.md`.

## Historical provenance is a separate unresolved problem

`src/pcr/manifest.py:46` computes a digest of bare `git diff` without retaining
the patch. Bare `git diff` excludes untracked files and staged changes. The
recorded initial commit contains no `src/pcr` implementation; the first committed
implementation is the later `9aac3d0` snapshot. No recoverable historical patch or
source archive was found in the inspected working tree, reachable history,
refs/reflog, or unreachable Git objects. This bounded search does not rule out
external backups. Later code and matching artifact hashes do not establish the
exact historical source or pre-outcome freeze timing.

The screenshot overstates the metadata absence if interpreted across all local
artifacts. Manifest-listed `config.resolved.json` files record corpus digests and
requested devices; `metrics_summary.json` records actual device and wall time.
Separate `run_attempts.jsonl` files contain start/success times. These logs are not
bound as per-run manifest artifacts. Exact command lines and per-run hardware
identity remain absent from the checked manifests. A supplemental recovery index
can report recovered fields with their source and uncertainty; it must not invent
missing execution records or rewrite history as a new preregistration.

The explicit ledger clarification at `docs/CLAIM_EVIDENCE_LEDGER.md:8` already
notes that local generated summaries are ignored and a later snapshot does not
prove historical registration timing. That disclosure is useful, but it does not
replace evidence distribution or complete provenance.

## Resume defect

`completed_manifest_exists()` at `src/pcr/manifest.py:121` checks only whether the
named JSON says completed. It does not verify artifacts, hashes, schema, protocol,
configuration identity, or requested output directory. The confirmatory driver
uses that result at `scripts/run_confirmatory.py:139`; other drivers do likewise.
Changing the output root does not change the manifest lookup.

There are two qualifications to the screenshot's wording: the driver prints its
skip, and config resolution may fail first when ignored corpus files are missing.
Once the corpus is supplied, missing experiment outputs can still be incorrectly
treated as completed. A related run-ID collision is already documented in
`docs/P2C_SEGMENT_STATS_DECISION.md:51`.

The prospective correction should separate **verified complete**, **artifacts
unavailable**, **hash mismatch**, and **identity mismatch**. Missing historical
evidence should trigger retrieval or an explicit error, not silent success or
automatic relabelling of a new rerun as the original experiment.

## Recovered exploratory attack

The local bundle is under `outputs/adversarial_review_2026-09-08/`:
`within_run_probe.py`, `within_run_probe.json`, and `REVIEW.md`. Its sixteen captures
match both the stored review hashes and the committed source manifests. Each
stored seed accuracy equals its eight-fold mean. Across eight seeds, the stored
means are 96.71630859375% at q4 and 97.802734375% at q32, matching the screenshot's
rounding. This audit verified inputs and aggregation; it did not fit the classifier.

The attack is post-hoc known-plaintext input-byte classification from P2b forward
activations: seven 128-byte sequences supply 896 known byte/activation pairs, and
the eighth sequence is held out. The fixed eight text windows are shared across
model seeds. This does not establish fresh-document recovery, recovery without
known plaintext, an active attack, general joint-tensor coverage, or performance
on every P5 cell. Source/capture verification does not reconstruct the review's
full execution provenance: its result JSON contains a verification object that
the standalone script does not emit.

This recovered forward-only probe does not evaluate timing, optimizer or evolving
remote state, returned values, full-precision inbound/outbound gradients,
active/adaptive attackers, within-run joint tensors, cross-run joint attacks,
history accumulation, canary/natural-sequence membership, or unrestricted semantics.

The prose and numeric table are already committed in
`ALL_DOCS_AND_ANALYSIS_REVIEW.md:2475–2492`, introduced in `212de99`; the standalone
code/result bundle is not. The current manuscript correctly limits cross-model
nondetection (`paper/main.tex:419`) and the ledger explicitly avoids promoting
post-hoc results into confirmation (`docs/CLAIM_EVIDENCE_LEDGER.md:40`). Nevertheless,
a clearly labelled exploratory subsection and dedicated artifact entry would make
the positive finding visible without misclassifying it. Historical decision prose
such as “Token-inversion never leaks” should receive an explicit correction notice.

## Consequence for Paper 4

Paper 4 does not have the screenshot's manifest population: it has 80 engineering
manifests, all explicitly `scientific_evidence: false`. Their 173 referenced
artifacts are local; 136 are tracked and 37 tensor files are not. Source archives
are also local and are separate from that artifact count. The newer scenario
loaders verify bound inputs, and the current runners cannot execute scientific
cohorts merely by treating an old manifest as complete.

Publishing Paper 4's tracked code and reports therefore did not publish every
binary input required to reproduce them. The same need for a retrievable artifact
archive applies before external reproducibility claims. Paper 4's inherited
Paper 3 evidence must retain the historical provenance limitation rather than
claim that later vendoring repairs it.

## Recommended repair order

1. Preserve the verified originals and create a versioned artifact archive with
   checksums, retrieval instructions, and a clean-checkout verification path.
   Keep original manifests intact. Publish the small analysis JSONs and exploratory
   attack code/results with their provenance and scope.
2. Fix completion checks prospectively to validate source/configuration identity
   and every required artifact. Test missing, corrupted, and wrong-output-root cases
   without rerunning historical model training.
3. Add a candid historical-provenance supplement: recovered metadata, unrecovered
   source snapshots, inconsistent notes, and limits on substantiating preregistration.
   Do not backdate patches or relabel post-hoc results as confirmation.
4. Give the recovered known-plaintext result an explicitly exploratory manuscript
   and ledger entry. Amend misleading historical statements visibly, preserving
   their original context. Do not extend P2b numbers to P5.
5. Validate a fresh checkout plus downloaded artifacts before declaring the evidence
   package submission-ready. File hashes alone do not verify all scientific claims.

## Audit method and changes

The paper-code-audit skill was applied manually because its `/audit` launcher is
not exposed in this session. Separate researcher and verifier reviews checked the
attack bundle and provenance/resume behavior while the main audit counted and
hashed every referenced artifact. GitHub branch revisions were checked read-only.
No training, probe fitting, analysis regeneration, source fixes, historical edits,
or uploads were performed. Only this report and its inventory were created in
Paper 4's outputs directory; Paper 3 was left unchanged.
