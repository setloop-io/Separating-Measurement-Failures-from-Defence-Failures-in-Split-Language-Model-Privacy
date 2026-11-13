# Reviewer evidence and reproduction

> **R1 reset (2026-09-10): historical reference only.** This document describes
> pre-reset work. All associated observations and analyses are inadmissible for
> active scientific claims. The current study has no results; follow
> [the R1 plan](R1_RESEARCH_PLAN.md) and [active ledger](CLAIM_EVIDENCE_LEDGER.md).
> The reference tag preserves the original version of this document.

The repository and its [evidence release](https://github.com/geopolitis/paper-3-privacy-constrained-remote/releases/tag/evidence-2026-09-10-v1)
remain **private**. Reviewers need access to this repository and an authenticated
GitHub account. Paper 4 has its [own private evidence release](https://github.com/geopolitis/paper-4-defense-redesign/releases/tag/evidence-2026-09-10-v1)
and requires separate repository access. This publication grants no public access.

## What is supplied

All locally retained experiment outputs, both exact corpus files, and the original
run manifests are packaged. Completion of the large upload is established by the
release's `VERIFICATION.json`, which checks all asset sizes and hashes; until then
the full package is not available for remote retrieval. Small JSON results, resolved configurations, curves,
logs, and exploratory attack code are also browsable in Git. The archive includes
all 4,492 manifest-referenced artifacts (52,139,663,688 bytes), plus additional
derived analyses and review material. Nothing has been regenerated to substitute
for a missing historical observation.

The [release index](../evidence/release-20260910-v1.json) binds every file's path,
size, and SHA-256 to a named archive group. Compressed archives are split into parts
of at most 1,500,000,000 bytes to meet GitHub's per-asset size limit. Each part has
its own checksum. The release also supplies `SHA256SUMS` and a copy of the index.
GitHub's automatic source ZIP alone does not include the large evidence files.

## Retrieve and check a fresh checkout

Install Python 3.11 or newer, the GitHub CLI (`gh`), and `zstd`. On macOS these
tools can be installed with Homebrew; on Linux use the platform's packages.
The download uses `gh`'s existing authentication without putting tokens in URLs.

```bash
gh auth login
gh repo clone geopolitis/paper-3-privacy-constrained-remote paper3-review
cd paper3-review
git checkout master
git rev-parse HEAD
python3 scripts/reviewer_evidence.py list
python3 scripts/reviewer_evidence.py fetch
python3 scripts/reviewer_evidence.py verify --report /tmp/paper3-evidence-verification.json
```

Record the printed commit for the review. The revised branch contains the corrected
manuscript, prospective runner safeguards, and subsequent validation records. Its
v1 evidence index is unchanged: fetching those archives restores original observations
without replacing current source or manuscript files. The tag `evidence-2026-09-10-v1`
identifies the older packaging-source snapshot and can be inspected separately;
it should not be used in place of the revised manuscript or as historical execution
source. Its source code also predates the prospective resume repairs.

Allow approximately 110 GB of free space for the full extracted evidence and
download cache. The tool streams decompression and does not create an additional
uncompressed tar file. Repeating `fetch` verifies existing files; it downloads only
groups with missing files. It refuses differing existing evidence rather than
overwriting it. A failed check exits nonzero and identifies missing or mismatched
files. `verify` checks files without running any experiment or loading tensors.

For a smaller inspection, select groups explicitly (repeat `--group`):

```bash
python3 scripts/reviewer_evidence.py fetch --group rate_sweep --group corpora --group adversarial_review_2026-09-08
python3 scripts/reviewer_evidence.py verify --group rate_sweep --group corpora --group adversarial_review_2026-09-08
```

The five manuscript display inputs are directly available in Git:

| Analysis | Original file |
|---|---|
| Causal-gain confirmation | [confirmatory_analysis.json](../outputs/confirmatory/confirmatory_analysis.json) |
| Architecture/corpus replication | [replication_analysis.json](../outputs/replication/replication_analysis.json) |
| Forward-precision sweep | [rate_sweep_analysis.json](../outputs/rate_sweep/rate_sweep_analysis.json) |
| Held-out segment prediction | [segment_stats_analysis.json](../outputs/segment_stats/segment_stats_analysis.json) |
| Delegation/defense sweep | [delegation_analysis.json](../outputs/delegation/delegation_analysis.json) |

Their original bytes match the committed
[display provenance](../paper/figures/manuscript_visuals_provenance.json).
Additional qualified/unqualified privacy results are in
`outputs/privacy_extended_p4b/`, `outputs/privacy_extended/`, and the delegation
qualification files. Qualification remains target-, channel-, and defense-specific.

Earlier P2/P2b reports retain historical statements such as `H5: SUPPORTED` and
`H3: CONFIRMED`, based on the original narrower token probe and precision sweep.
Those are preserved records, **not the final manuscript's privacy or H3 verdicts**.
Read them with the final [claim-evidence ledger](CLAIM_EVIDENCE_LEDGER.md), the P4b
requalification, and the manuscript's explicit omissions and claim boundaries.

## Recompute analyses without replacing the originals

Use a disposable reviewer checkout for recomputation. The current environment is
described in [environment/README.md](../environment/README.md). A dated
[installed-package inventory](../evidence/environment-observed-20260910.json)
is supplied as recovery information; it is not a reconstructed historical lockfile.
Create an isolated environment with Python 3.12 and install the package and the
analysis dependencies (NumPy, PyTorch, SciPy, Matplotlib, PyYAML, and jsonschema).
The observed package versions are available to guide that setup.

After restoring the evidence, these examples recompute stored-curve analyses into
a separate directory. `--no-autoscore` reuses the archived privacy scores; it does
not independently refit privacy probes.

```bash
mkdir -p /tmp/paper3-review-analysis
python scripts/analyze_confirmatory.py --no-autoscore --no-figure --analysis-path /tmp/paper3-review-analysis/confirmatory.json --summary-path /tmp/paper3-review-analysis/confirmatory.md
python scripts/analyze_rate_sweep.py --no-autoscore --no-figure --analysis-path /tmp/paper3-review-analysis/rate.json --summary-path /tmp/paper3-review-analysis/rate.md
python scripts/analyze_replication.py --no-figure --analysis-path /tmp/paper3-review-analysis/replication.json --summary-path /tmp/paper3-review-analysis/replication.md
python scripts/analyze_segment_stats.py --no-figure --analysis-path /tmp/paper3-review-analysis/segment.json --summary-path /tmp/paper3-review-analysis/segment.md
```

`scripts/build_manuscript_visuals.py` renders figures/tables from the five archived
summaries. It writes into `paper/`, so run it only in the disposable checkout.
P5 privacy recomputation is more expensive: `scripts/analyze_delegation.py
--reuse-qualification` reuses archived gradient qualification but refits cohort
privacy probes. Its `--no-privacy-scoring` option cannot reproduce privacy gates.
Use the script's `--analysis-path`, `--summary-path`, and `--no-figure` options to
retain the original summaries. No such probe refitting was performed for this release.

The [known-plaintext probe](../outputs/adversarial_review_2026-09-08/within_run_probe.py)
and its [original results](../outputs/adversarial_review_2026-09-08/within_run_probe.json)
are explicitly **post-hoc exploratory**. Running the original script overwrites
its adjacent result JSON, so use a disposable checkout and retain the release copy
for comparison. It fits seven of eight fixed text windows and tests the eighth;
it does not establish fresh-document recovery or performance on every P5 cell.
The stored JSON has additional verification metadata not emitted by that script.

A separate post-hoc validation on 2026-09-10 refitted the copied known-plaintext
script on the sixteen original captures. All 128 fold accuracies, sixteen seed
records, null means, and majority baselines matched the retained result exactly.
The copied script and comparison records live in `outputs/revision_validation_20260910/`
on the revised branch; they are subsequent validation, not members of the frozen
v1 evidence index. They do not establish fresh-document generalization, independent
training replication, or recovery of the original execution source.

The same validation directory retains a target-resolved P5 property refit: all 112
per-seed aggregate scores exactly match the original results, while the separate
punctuation scores identify a qualified rejecting target in every cell. This is
retrospective analysis verification using the original qualification, not new
confirmation. A P5 nonprivacy recomputation also matches all fourteen cells' utility,
gain, and share fields. Its deliberately skipped privacy/triangle fields are not
validation results. Full structure-probe fitting and gradient requalification were
not repeated; see the validation report for exact commands, inputs, and exclusions.

## Historical limits that publication does not repair

The [dated audit](../evidence/audit-20260910/report.md) and
[inventory](../evidence/audit-20260910/inventory.json) describe the state **before**
this release. Statements there that files were absent from Git refer to that
earlier snapshot. Original manifests are preserved, including their inconsistent
exploratory notes. Their recorded commit is the initial scaffold; the dirty-tree
patches and exact historical source snapshots were not recovered. Later committed
code is available, but cannot establish the exact execution source or pre-outcome
freeze timing. Packaging timestamps must not be interpreted as preregistration.
The [current provenance supplement](HISTORICAL_PROVENANCE.md) records the additional
bounded search in both Paper 1 repositories and Paper 2, which found no matching
Paper 3 source snapshot. Paper 1's E1 source archive was found and belongs to that study.

Corpus digests, resolved configurations, actual devices and timings, and separate
attempt logs are included where they were retained. Missing command lines and
hardware identities have not been invented. The exact byte corpora are
`tinyshakespeare.txt` (SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`)
and Project Gutenberg's *War and Peace*, ebook 2600, including its original notices
(SHA-256 `2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b`).
The byte tokenizer and historical 90/10 split are implemented in `src/pcr/data.py`;
no new preprocessing was applied for distribution.

The training drivers at the immutable v1 evidence-release revision have a status-only
completion check. Their skip messages are not evidence verification; changing only
the output root is insufficient to create a new experiment. The revised branch
contains prospective fail-closed completion checks and retained source snapshots.
It rejects missing/corrupt evidence, mismatched identities, and historical records
without the source needed for a source-verified resume. It preserves occupied run
directories and manifests instead of silently replacing an earlier attempt.
These repairs do not change the original manifests or confer source provenance on them.
Any future training reproduction requires a separately identified run/manifest
directory and explicit source/configuration provenance. The evidence release restores
access to retained evidence; it does not claim an exact historical training
reproduction or independently validate every scientific conclusion.

The revised drivers and single-cell CLI expose `--manifest-dir` to choose a
separate manifest namespace, in addition to a new output directory. A dry run
lists verified, pending, or blocked cells and returns nonzero if any are blocked;
it does not train or interpret an unavailable legacy source as completion. A failed
execution is preserved and requires an explicitly separate attempt; automatic
same-directory retries cannot replace earlier evidence. These controls are not a
new scientific protocol or authorization to reuse historical labels for new results.

Keep the release tag for immutable original evidence. Use the dated revision and
its validation record to inspect subsequent code repairs; do not describe the new
source as the historical implementation that generated those observations.

## Publication checks

The [local extraction and analysis check](../evidence/reproduction-check-20260910.json)
records restoration of all 5,308 indexed files into a fresh GitHub checkout from
the staged release archives. The four stored-curve analyses above ran successfully;
their numeric values matched the archived summaries. Absolute paths and inventories
of unrelated manifests differ in the new checkout. This check reused stored privacy
scores and did not retrain models or independently refit attack probes. The
published release's `VERIFICATION.json` records the completed asset checks.
