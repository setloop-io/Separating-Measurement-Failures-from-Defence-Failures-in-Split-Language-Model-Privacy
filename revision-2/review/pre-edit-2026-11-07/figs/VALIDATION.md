# Figure validation

Generated from the final revision-2 data CSVs. The script checked all 19 R1 rows against the original cell CSV; utility means and two-sided 95% t-intervals were independently checked against the four original seeds per cell. It checked all seven P4 gate and privacy declarations against the original generated pilot report. Both style audits are clean; see `figure-audit.log`.

The PDF outputs were rendered using Poppler and visually inspected. Labels, points, intervals, hatch patterns and legends are legible, with no clipped or overlapping text. PDF page widths are 468 points (6.5 inches); fonts are embedded CID TrueType Helvetica Neue, including bold where used. The utility figure is 203.58 points high and the campaign figure 252.72 points high. SVG outputs retain vector shapes and text. Preview PNGs are visual QA artifacts only.

No existing manuscript, evidence, or reproduction file was changed. Every write for this figure task was confined to this new `figs/` directory; only an isolated plotting environment was created under `/tmp`.
