# Final PDF QA

7 November 2026

The current `main.pdf` has 10 letter-size pages and 315,297 bytes.
Its SHA-256 is `01a3c2793604c35532020e823fb819952005b34a3433b58fa22980e510714c91`. This record supersedes the earlier 13-page layout check,
which is preserved with the pre-edit manuscript under `pre-edit-2026-11-07/`.

The manuscript uses the unchanged, pinned George Kour `arxiv.sty` community
preprint template. Its Times body type, geometry, title block and running
headings follow that style. Latin Modern Mono supplies scalable path/code
text. All fonts are embedded and scalable, with no Type 3 bitmap fonts.
The figures retain vector content at the final 6.5-inch width.

All 10 final pages were rendered with Poppler at 120 dpi and inspected through
two contact sheets. Pages 1, 5, 6 and 10 were also inspected at full rendered
size for the title and abstract, table, figures, equation, proof and source
paths. No clipping, overlap, missing glyphs or unreadable labels were observed.
The title page follows the template's unnumbered style. Pages 2 through 10
carry the running title and page number. The current renders are in
`rendered-2026-11-07/`.

The final PDF text contains zero colons, semicolons, question marks, em dashes
or en dashes. No rhetorical questions or numbered internal-workload references
remain in audience-facing prose, captions, tables or figures. Literal paths
retain their exact identifiers in the reproducibility appendix. Bibliographic
links remain functional with scheme-free display text. Captions use a period
after their number, and multiple citations use commas.

The three-pass build has no unresolved references, citations, or LaTeX
warnings. The extracted text contains no replacement characters or unfinished
placeholders. See `editorial-rendered-check-2026-11-07.json`,
`pdf-fonts.txt` and `build-validation.json`.

The scientific/editorial verification preserved results, metric definitions,
uncertainty conventions, evidential levels and citation keys. The displayed
equations retain their mathematical meaning, with mutual information written
as MI(X,Z) to remove the semicolon notation. Shifted token slices are expressed
as their exact input and label sequences. The approximate source word count
changed by 4.86%, within the supplied instruction's 5% budget.
See `editorial-verification-2026-11-07.md` for the separate verification.

The clean `arxiv-source.zip` contains 14 build, licence and README files.
It compiled successfully for three passes after extraction into a temporary
directory, with identical text and page dimensions on all 10 pages. It excludes
backups, logs, unused template examples and primary experiment records.
`source-package-check.json` binds the package to this PDF. No submission was made.

All 11 protected original files remain byte-identical to the starting snapshot,
and the 118 analysis inputs and analysis-script hash remain pinned. The stored
1,522 checks concern internal consistency of existing evidence. No new training,
attack fitting or independent validation study was run for this editorial update.
