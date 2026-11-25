# arXiv preparation and remaining decisions — 20 September 2026

The user selected arXiv preprints, with a later IEEE or USENIX venue to be
decided. Both provisional source archives compile successfully. No submission
has been made. The current packages are indexed by
`build/arxiv-preprints-20260922-freeze-r2-final/INDEX.json`; each includes a PDF, source
ZIP, hashes and metadata draft. Evidence content hashes are verified; manuscript evidence links use immutable HF revisions; the source/PDF candidate is frozen under `submission-candidate-20260922-r2`. The 22 September bibliography/code-citation audit and minor revision are complete; both current packages have been rebuilt and reviewed. The requested repository-history cleanup awaits the evidence-retention decision; the author decisions below also remain.

## Evidence work before final packaging

| Item | Required disposition |
| --- | --- |
| Paper A corrected GPT-2 P6 raw archive | Complete: all 200 cells archived; 11 files / 34,703,146,309 bytes verified against the archive manifest; updated Paper A PDF and release index verified against their manuscript manifest |
| Qwen precision amendment | Complete: all 200 cells and CPU reproduction; 77-file evidence release verified against the release manifest; final 7-file manuscript tier verified against the release manifest; completed-release revision link and rebuilt packages reviewed |
| Sasa 500-step extension | Complete: six trajectories audited, reproduced byte-for-byte on sasa/soula, HF evidence pinned and Paper B updated with calibration overlap and training-only limitations |
| Agentic measurement repair | Complete: ten amended cells and both evidence/manuscript HF tiers verified; historical thinking-mode failures remain separate |
| Original BiSR setting | Explicit exclusion: the processed SensiReplaced version and split identities remain unavailable. Single-seam conformance and adaptation do not establish original-setting dual-cut fidelity |
| Historical external results | Explicit exclusion: the owning project's retirement policy and absent historical execution-source provenance remain in force |

The last two exclusions do not block the scoped preprints. A fresh bounded
five-node review retained these dispositions on 20 September; see
`closure-prerequisites-2026-09-20.json`. Do not repeat unchanged inventories
without a new asset, specified location or evidence-policy change.

## Author decisions before submission

- Decide whether to keep manual evidence gating/private GitHub or prepare an unrestricted code-and-evidence release, and check the selected venue's artifact policy. No visibility setting was changed during this revision.

- Confirm the existing author names, order, affiliations, email and
  corresponding-author designation in both source files. The current draft
  strings have been preserved.
- Confirm the proposed primary category `cs.CR` and whether a `cs.LG` cross-list
  is appropriate.
- Select the arXiv distribution licence. The current repository tree has no LICENSE file; resolve code-distribution licensing as part of any public code release. This is separate from the manuscript licence.
- Review a disclosure of overlap between the two papers: Paper A focuses on
  activation/gradient leakage and the crossed segment study; Paper B focuses on
  the framework and its broader evaluation. The shared frozen GPT-2 campaign,
  joint-versus-activation result and evidence repositories should be identified
  as shared results, not independent replications.
- Decide the later venue and check its submission and overlap requirements
  before preparing venue-specific versions.

## Final package checks

Rebuild both packages into a fresh directory after evidence integration. Check
that every empirical claim resolves to the intended named evidence directory and content manifest,
that exclusions agree across abstract, results, tables and limitations, and that
all tables and figures use the reviewed complete-set analyses. Verify source ZIP
hashes, bibliography and figure inclusion; compile, inspect the rendered PDFs,
and review the arXiv-generated preview as part of a separately authorised
submission. The agentic and Sasa Paper B revision has passed compilation and
visual inspection, including the revised results pages; subsequent edits require
a fresh review.

The 22 September reference audit verified all retained bibliography entries and corrected citation/implementation scope. Both papers generate their bibliographies from their `.bib` files. Current packages passed isolated compilation, ZIP/PDF content-hash checks, citation integrity and visual review (Paper A 14 pages; Paper B 25 pages). Qwen qualification and the other scientific exclusions are unchanged. The updated PDFs are the current GitHub-facing manuscripts; the previous HF manuscript tiers remain intact pending the requested history/publication cleanup. See [the reference audit](reference-audit-2026-09-22.md) and [minor-revision checks](minor-revision-2026-09-22.md). The earlier 13/24-page packages are superseded. HF datasets are publicly listed but manually gated; GitHub is private. These access conditions are now explicit in the manuscript.

The [second-review freeze report](submission-candidate-2026-09-22-r2.md) supersedes the first minor revision's mutable-link policy and package hashes. It records the control and horizon corrections, exact training-control floors, grid overlap and final bibliography/label checks. Submission and artifact-access decisions remain with the author.
