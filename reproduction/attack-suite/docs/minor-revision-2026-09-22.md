# Minor-revision checks — 22 September 2026

This report records the first minor revision. The [second review and frozen candidate](submission-candidate-2026-09-22-r2.md) supersede its current-branch link policy, package paths and page counts; historical checks below remain attributable to that earlier state.

The requested corrections are implemented in both current manuscripts. The primary v3 contrast, corrected GPT-2 P6 results, Qwen qualification and scientific exclusions are unchanged. No model jobs, prediction generation or inferential analyses were rerun.

## Balanced span reporting

Read-only arithmetic checks matched all 480 secret-mixup cells' saved span counts in the merged 3,680-cell file to the released per-cell summary. All 24 condition means agree with the released arm summary. Paper A now presents every class together in Table 3; its abstract and discussion acknowledge the name-recovery reduction. Paper B reports the same class-specific results in its shared v3 case study.

| Class | Documents | Spans | Matched undefended recovery | Secret-mixup recovery |
| --- | ---: | ---: | ---: | ---: |
| Name | 249 | 716 | 67.36–97.44% | 47.85–66.15% |
| Number | 142 | 297 | 97.64–100.00% | 93.94–96.50% |
| Identifier | 2 | 2 | 100.00–100.00% | 90.00–100.00% |

Each cell's rate is exact spans divided by all annotated spans of that class, followed by an equal mean over ten victim seeds and two attack seeds. Ranges cover six arms × two depths × two knowledge grants; they are descriptive ranges, not confidence intervals or paired effect estimates. Classes can share documents. Name annotations were AI-assisted before outcomes, not human gold labels; the two identifier examples do not support population inference. Exact values and source SHA-256 hashes are in the [machine-readable check](minor-revision-2026-09-22.json).

## Bibliography and presentation

- Both abstracts now distinguish 3,200 physical executions from 480 reused measurements. Both results sections explain independently rounded percentages and contrasts; Paper A illustrates why 43.125% is printed as 43.13%.
- The instance-encoding citation now consistently uses the [2021 revised arXiv version](https://arxiv.org/abs/2011.05315), listing all nine authors including Shuang Song. This supersedes the earlier audit's choice of the eight-author IEEE registry record.
- Added and checked [GPT-2](https://cdn.openai.com/better-language-models/language-models.pdf), [WikiText](https://arxiv.org/abs/1609.07843), [Qwen3](https://arxiv.org/abs/2505.09388) and [AdamW](https://arxiv.org/abs/1711.05101) at their uses. The updated [reference/code map](reference-audit-2026-09-22.md) records implementation relationships.
- Defined **vanilla model inversion** from [VFLAIR-LLM](https://arxiv.org/abs/2508.03097). Both papers distinguish the local public-head ridge adaptation from the published optimisation procedure.
- Regenerated both `.bbl` files with BibTeX and rebuilt PDFs. Paper A's Fisher-information entry uses arXiv:2305.04146; Paper B's distinct REFIL preprint uses arXiv:2209.10119. They must not be conflated. All 19 Paper A and 41 Paper B bibliography entries are cited, with no missing, unused or duplicate bibliography keys.

## Completion records and access

The old Qwen commit link was already removed before this revision. An authenticated fetch of the current GitHub `main` qualification report confirms non-null, successful evidence and manuscript verification. This check reads the completed report; it does not repeat a 137 GB download. Current-branch links remain in the manuscripts, with content manifests identifying evaluated evidence. No explicit commit links were reintroduced.

Fresh API checks show both HF datasets are publicly listed (`private=false`) and manually gated (`gated=manual`). Downloads require a request and owner approval. GitHub is private; its report and companion-paper links require repository access. The manuscripts now describe these restrictions. No visibility or gating setting was changed.

Both current review-response letters have been refreshed: Qwen, Sasa, the agentic decoding amendment and the single-seam BiSR adaptation are complete within their stated scopes. Original-setting BiSR and historical-source exclusions remain.

The local failed `build/paper-a-revision-20260917/raw-status.json` has been replaced by a supersession pointer. Its exact original bytes remain in `historical-status/raw-status-failed-20260919.json` under the same directory, with SHA-256 `5f905b5c542260af9906281ccf79a908c7ae27770a3677bc78ec8a480be342d7`. The pointer identifies the completed raw-release verification; it does not fabricate a new publisher outcome.

## Delivery and remaining decisions

Current packages: `build/arxiv-preprints-20260922-minor-revision-reviewed/INDEX.json`. Both isolated source ZIPs compile with no undefined citations/references or overfull boxes. Every ZIP member and PDF matches its recorded SHA-256, and extracted PDF text agrees with the repository PDF. The figure inputs are unchanged. Full-layout contact sheets and enlarged changed pages were reviewed; the corrected P6 appendix table now precedes the bibliography without interrupting its entries.

Paper A is **14 pages**; Paper B is **25 pages**, compared with 13/24 in the earlier reference-audit build. Current `.tex`, `.bib`, `.bbl` and `.pdf` files are consistent. Earlier build paths are retained as prior snapshots. Existing HF manuscript tiers remain historical copies pending the separate history/publication cleanup; this revision updates GitHub-facing files and local preprint packages.

Submission still needs author confirmation of names/order, affiliations and corresponding author; category/cross-list, manuscript licence and shared-evidence overlap disclosure. No arXiv submission or repository-history deletion was performed. The separate history-cleanup task still needs its evidence-retention scope resolved; no scientific evidence was deleted or relabelled.
