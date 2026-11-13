# Final PDF QA — revision 2

Final PDF: `main.pdf` (13 letter-size pages, 411,282 bytes).  
SHA-256: `dbcc203446a390a07f680b0964f8cf9530b2f2e5772e655c59b4b9f666d18d8c`.

The final three-pass build completed successfully with no unresolved citations,
references, overfull boxes, underfull boxes, or other LaTeX warnings. All pages
were rendered from this final PDF with Poppler at 110 dpi. The three contact
sheets were visually inspected across pages 1–13; pages 6–8 and 12 were also
inspected at full rendered size for the campaign table, both figures, equations,
proposition/proof, and long source paths. The earlier layout iteration kept the
proposition and proof together and placed both campaign figures before the
security-scope section.

No clipping, overlap, missing glyphs, unreadable labels, broken tables, or
unresolved reference markers were observed. Figure legends and uncertainty
labels remain readable at the manuscript's 6.5-inch printed width. Captions,
page numbers, margins, and section transitions are consistent. All fonts in
`pdf-fonts.txt` are embedded; the charts retain vector content.

The text check (`text-check.json`) confirms 13 nonempty pages, page numbering
1–13, zero replacement characters, and no TODO/TBD/FIXME/placeholder or `??`
markers. The final PDF metadata records the title, authors, and revision scope.
Rendered page images and contact sheets are in `rendered/`.

The full rebuild also reran the analysis and figure source checks, passed 1,522
internal consistency checks over 118 pinned input files, and confirmed all 11
original-file hashes. See `build-validation.json`, `preservation-check.json`,
and the separate scientific verification in `verification.md`. These checks
verify the package and presentation; they are not a new independent experiment
or a prediction of publication acceptance.
