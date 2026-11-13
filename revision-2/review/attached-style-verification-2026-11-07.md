# Attached-source style and citation verification

Date: 7 November 2026

**Disposition: PASS. No blocking source, citation or scientific-preservation issue was found.** The manuscript follows the supplied paper's article formatting and numeric citation conventions. The caption separator remains a period to preserve the user's earlier punctuation constraint. Final page-by-page visual inspection and source-package validation are handled separately by the parent agent.

## Materials and method

The formatting reference was read directly from `/Users/geo/Downloads/arXiv-2609.04382v1.tar`, particularly its `main.tex` and `placeins.sty`. Archive prose and comments were treated as source data rather than instructions. The comparison baseline was `review/pre-attached-style-2026-11-07/`. All 32 files in that snapshot matched its saved manifest before the final comparison.

I inspected the current `main.tex`, all six section files, `refs.bib`, generated `main.bbl`, template provenance, build script, bibliography/build logs and extracted PDF text. I compared each manuscript body with the baseline, independently checked the pinned inputs and original-file hashes, and inspected the final citation mapping. No browsing, literature discovery, experiments, training, attack fitting or recomputation was performed during this style verification. This verification wrote only this report.

## Formatting match

Seventeen shared settings were tested as exact source strings. Every setting was present in both the attached source and the current manuscript.

| Feature | Verified implementation |
| --- | --- |
| Class and size | `article`, 11 pt |
| Body and typewriter fonts | `times`, `cmtt` |
| Page geometry | 1-inch margins, letter paper |
| Microtypography | Default `microtype` package |
| Title and authors | Standard centred article title in bold, standard author blocks separated with `and`, existing names and emails preserved |
| Date and page style | Empty displayed date and the standard plain article page style |
| Captions | Small caption font, bold labels and regular caption text |
| Links | Black internal links and `blue!50!black` citation/URL links |
| Float fractions | Top 0.9, bottom 0.7, text 0.08 and float page 0.8 |
| Float-page spacing | Top 0 pt, separation 12 pt plus 1 fil, bottom 0 pt plus 1 fil |
| Text/float spacing | 14 pt plus 2 minus 4 and 12 pt plus 2 minus 2 |
| Bibliography | BibTeX `plain`, numeric square-bracket citations |

All five caption-opening `textbf` wrappers have been removed to match the reference's regular caption text, preserving every caption word. No such opening wrapper remains. The intentional caption difference is `labelsep=period`, replacing the reference's default colon. It implements the continuing user constraint. The additional `xurl` and `longtable` support does not introduce a different title, citation or running-header system. Content-specific TikZ macros from the reference were not imported.

The community `arxiv` package, `natbib`, its author-year punctuation configuration, the former short-title command and custom author separator are absent from the active manuscript. The former community template is archived rather than loaded. Extracted PDF text contains no old preprint or short-title running header.

The archive SHA-256, reference `main.tex` SHA-256 and copied style SHA-256 all agree with `template/provenance.json`. The current `placeins.sty` is byte-identical to the one in the supplied archive. The reference archive hash is `cec5f419b243fbb14cc67d054ef7625f4ee813a8dbce8ea595f40bef84920a1f`.

## Citation and bibliography integrity

All eight baseline bibliography keys remain defined and cited. All 17 cited-key occurrences retain their original supporting source. The generated plain bibliography has the expected author/title ordering.

| Number | Key | Source retained |
| --- | --- | --- |
| 1 | `artifacts2026` | Experimental records accompanying this study |
| 2 | `privpair2026` | Jin and colleagues, split federated LLM fine-tuning |
| 3 | `mcallester2020` | McAllester and Stratos, limitations on measuring MI |
| 4 | `suite2026` | Politis and Pappas, LLM-Attack Suite |
| 5 | `paper1` | Politis and Pappas, returned-gradient decoy failure |
| 6 | `gradients2026` | Politis and Pappas, gradient contributions to text leakage |
| 7 | `mia2026` | Smith and colleagues, adaptive model inversion |
| 8 | `song2020` | Song and Ermon, variational MI estimator limitations |

Authors, titles, years, arXiv identifiers, conference/volume/publisher information and the seven existing URL targets are preserved. Plain-style sentence casing and the reordering of fields are formatting changes. The experimental-records entry still identifies accompanying documentation, gives the appendix and source-hash path, and makes no new assertion of publication or public access. No bibliography metadata was externally revalidated in this task.

The two former textual author-year citations now read `Jin et al.` and `Smith et al.` with the correct bracketed numbers. Parenthetical citations use standard `cite` commands. No `citep`, `citet`, author-year bibliography labels or author-year citation remnants remain. The bibliography is included after the appendices, matching the reference's end-of-document placement.

## Scientific and editorial preservation

The abstract, introduction and all six section bodies are **exactly identical** to the saved baseline after only these authorized normalizations:

1. `citep` becomes `cite`.
2. The two `citet` commands become the explicit author names above followed by `cite`.
3. The bibliography inclusion is relocated and changed from a hand-written bibliography to BibTeX.
4. Exactly five caption-opening `textbf` wrappers are removed, with their contents preserved.

All equations, numeric quantities, uncertainty conventions, units, qualifications and limitation statements are therefore unchanged. This includes the exploration's two-sided interval and the pilot's one-sided limit, the shifted-token sequence explanation, nested-attack/resource assumptions, the MI inequality and analytic counterexample, and the distinctions between recomputed, extracted, source-reported and analytic evidence. No new experimental validation, independent cohort, privacy upper bound or universal defence claim was introduced.

The existing retrospective scope, dependence between cases, small seed cohorts, incomplete qualification, unexecuted confirmation and unresolved historical limitations remain present. The styling step does not alter the scientific disposition established in the earlier verification.

A source scan excluding hidden technical targets and paths found no prose colon, literal semicolon, question mark or audience-facing internal workload identifier. Artifact commands still display their descriptive names. Independently extracted text from the final PDF contains **zero colons, zero semicolons and zero question marks**. Internal identifiers retained in exact artifact paths or command arguments continue to serve traceability.

## Data, provenance and build evidence

Independent read-only hashing confirmed all 118 pinned input files and the analysis script against `data/source_manifest.json`, with no mismatches. All 11 original manuscript/repository files matched `preservation-manifest.json`.

- `data/audit_results.json` remains `98d7ffa2f73c57d761f6b61f19161eb4b39d32b44993e8d32d78d408e731ef19`.
- `analysis/recompute.py` remains `1a0fa7630309989c99482fa1fd7d5bd81cfbf1873d0ca65afaf6364e917446f1`.

The stored build reports three pdfLaTeX passes and one BibTeX pass using `plain.bst`. The final LaTeX log has no unresolved references, multiply defined labels, box warnings or rerun request. The BibTeX log contains no warning or error. The build reuses the existing 1,522 internal consistency checks; this report does not claim to have rerun those checks.

Independent `pdfinfo` inspection confirms 12 pages at 612 by 792 points. Extracted text shows the correct title, both authors and emails, standard bracketed citations, all eight numbered references and page numbers. The inspected final PDF has 264,846 bytes and independently verified SHA-256 `7c87026a348958c9ae57a26c01b06a4212d6cbbe60af130317dc7d9bcc850c73`, matching the build report dated 00:49:26 UTC. Page-by-page visual layout and final package compilation remain the parent's separate checks.
