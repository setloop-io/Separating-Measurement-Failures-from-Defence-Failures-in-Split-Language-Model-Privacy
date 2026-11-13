# Final PDF QA

7 November 2026

The current manuscript uses the style from the user-supplied
`arXiv-2609.04382v1.tar`. It has 12 US-letter pages and 264,846
bytes. Its SHA-256 is `7c87026a348958c9ae57a26c01b06a4212d6cbbe60af130317dc7d9bcc850c73`. This record supersedes the community-template
layout check preserved in `pre-attached-style-2026-11-07/`.

The attached paper was compiled separately as a visual reference. Its first
page and bibliography were compared with the current manuscript. The current
source matches its 11-point article class, Times body font, cmtt typewriter,
one-inch margins, standard centred bold title and author blocks, empty date,
plain page numbering, small captions with bold labels and regular body text, float placement and
spacing, and dark-blue linked citations. The prior user punctuation request
is retained through a period after caption numbers. Content-specific diagram
macros and scientific text from the attachment were not imported.

BibTeX's `plain` style generates an alphabetically ordered bibliography with
numeric bracket citations. All eight citation keys resolve. The generated
mapping is Experimental records 1, Jin and colleagues 2, McAllester and Stratos
3, LLM-Attack Suite 4, the decoy-gradient preprint 5, the gradient-contribution
preprint 6, Smith and colleagues 7, and Song and Ermon 8. The original authors,
titles, years, source identifiers and venue information are retained. The
bibliography follows the appendices, as in the attached paper. Author-year
citation commands and the former preprint title bars and running headers are
absent from the active manuscript.

All 12 final pages were rendered at 120 dpi and visually inspected through
two contact sheets. The title page and final bibliography page were also
inspected at full rendered size against the corresponding reference pages.
No clipping, overlap, missing glyphs, unresolved citation markers or unreadable
labels were observed. The final renders are in
`rendered-attached-style-2026-11-07/`. All fonts are embedded and scalable,
with no Type 3 bitmap fonts. The existing figures retain vector content.

The final rendered text contains zero colons, semicolons, question marks,
em dashes or en dashes. It has no audience-facing internal workload names,
replacement characters or unfinished placeholders. Exact source filenames
remain intact in the reproducibility appendix. See
`attached-style-rendered-check-2026-11-07.json` and `pdf-fonts.txt`.

The build completes pdfLaTeX, BibTeX, then two further pdfLaTeX passes with no
warnings, undefined citations or references, or overfull/underfull boxes in
the final manuscript log. See `build-validation.json` and `bibtex.log`.
The clean `arxiv-source.zip` contains 14 build and README files. It was
extracted and compiled independently with the same LaTeX/BibTeX sequence.
Its rendered text, page dimensions and generated bibliography exactly match
the current manuscript. `source-package-check.json` binds it to this PDF.

This is a formatting and citation correction. Scientific prose, equations,
results and evidential qualifications are unchanged except the citation syntax
the explicit author names in the two former textual author-year citations, and
the removal of five caption-opening bold wrappers to match the reference style.
The separate check is in `attached-style-verification-2026-11-07.md`.
The attached archive is a formatting reference and supplies no new behavioural
instructions or empirical claims for this revision. No training, attacks or
new experiment campaign was run. The 11 protected original files and 118
analysis input bindings remain covered by the preservation and input checks.
