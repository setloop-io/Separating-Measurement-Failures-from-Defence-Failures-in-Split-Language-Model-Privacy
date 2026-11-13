# Historical provenance and current reproducibility

> **R1 reset (2026-09-10): historical reference only.** This document describes
> pre-reset work. All associated observations and analyses are inadmissible for
> active scientific claims. The current study has no results; follow
> [the R1 plan](R1_RESEARCH_PLAN.md) and [active ledger](CLAIM_EVIDENCE_LEDGER.md).
> The reference tag preserves the original version of this document.

Prepared 2026-09-10. This disclosure separates retained evidence, retrospective
verification, and missing historical execution records. It is not a replacement
preregistration or a reconstruction of an unavailable source snapshot.

## Evidence that is available

All 4,492 original manifest-referenced artifacts were recovered locally and match
their recorded SHA-256 hashes. They total 52,139,663,688 bytes. The five analysis
JSONs used for manuscript displays also match the hashes recorded in the original
display provenance. Small results are now in Git; large captures and exact corpus
files are distributed through the private evidence release. Publication completion
is recorded separately from local file verification.

The [reviewer guide](REVIEWER_EVIDENCE.md) describes fresh-checkout retrieval,
verification, and analysis. A [dated extraction/analysis check](../evidence/reproduction-check-20260910.json)
recomputed four stored-curve analyses with matching numeric results, reusing archived
privacy scores. Subsequent post-hoc probe validation has its own records under
`outputs/revision_validation_20260910/`. These are specific verification tasks,
not certification of every scientific claim or exact historical training reproduction.

## Evidence that remains missing

All 768 original completed manifests reference scaffold commit
`be32144570e6fdb6cbdbdf367c9e71ffa685d24a`. That commit has no `src/pcr`
implementation. The first committed implementation is the later `9aac3d0` snapshot.
The original manifests contain nine distinct non-null dirty-tree hashes, but the
corresponding patches or complete execution-time source archives were not recovered.
The original hash used bare `git diff`, which omits staged changes and untracked
files; even a recovered matching diff might therefore be incomplete.

The search covered Paper 3's inspected working tree and local Git history, both
local Paper 1 repositories, and Paper 2, including available archives and unreachable
Git objects. Paper 1 has an E1 source archive and original patch matching its own
manifest; neither identifies Paper 3's execution source. No matching Paper 3 patch
was found. External backups and remote execution hosts were not exhaustively searched.
The search records are in [evidence/source-search-20260910](../evidence/source-search-20260910/).

Consequently, later source plus matching artifact hashes do not establish exact
historical execution source or independent pre-outcome protocol-freeze timing.
Repository documents retain their original dates and study labels as historical
claims. The manuscript describes documented cohort protocols and explicitly states
this limitation. Missing timing evidence alone does not prove that protocols were
selected after observing outcomes.

## Recovered metadata and its limits

Original resolved configurations retain corpus digests and requested devices;
metrics retain actual devices and elapsed time. Separate attempt logs retain some
start/success timestamps but are not bound as original per-run manifest artifacts.
Exact command lines and per-run hardware identity were not recovered in those
manifests. Environment digests do not recover an unavailable package list or lockfile.
The dated installed-package inventory describes the recovery environment only.

The boilerplate exploratory disclaimer appears in all original manifests, including
687 labelled confirmatory. It is a metadata inconsistency. Original labels and notes
are preserved; interpretation follows the current claim-evidence ledger and the
available protocol/analysis records, with historical freeze timing explicitly limited.

## Prospective repairs

New completion checks validate the requested run/configuration/output identity and
artifact bytes before treating a run as complete. Source-verified resume additionally
requires a retained source snapshot. A historical manifest without one must fail
explicitly rather than be silently skipped or overwritten by a newly trained run.
New source records cannot repair an old manifest retrospectively. The implementation
and its tests are dated current changes, separate from the evidence release's source
revision and the original training code.

Historical manifests, resolved configurations, captures, analysis JSONs, and protocols
remain intact. Any future training replication needs a distinct run/manifest namespace,
retained source and environment, and an actual pre-execution protocol freeze. Its
outcome would be new evidence, not proof of the missing original execution history.
