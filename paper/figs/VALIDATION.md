# Figure validation

Both vector figures (`campaign-gates.pdf` and `r1-utility-boundary.pdf`)
were rendered with Poppler and visually inspected at the 6.5-inch
manuscript width. Preview PNGs and vector SVGs accompany the PDFs.

## Typography and layout

- PDF page widths are 468 points (6.5 inches); heights are 252.72 points
  for `campaign-gates.pdf` and 203.58 points for `r1-utility-boundary.pdf`.
- Helvetica Neue and its bold face are embedded as CID TrueType fonts.
- SVG text has a minimum size of 8 points with no embedded raster plot
  elements.
- Panel~(b) of `campaign-gates.pdf` pairs colour fills with hatching and
  explicit `P`, `F`, and `U` glyphs to preserve distinctions in greyscale.
- `r1-utility-boundary.pdf` displays distinct $0.10$-nat mean and
  $0.12$-nat upper-limit thresholds and untruncated four-seed intervals,
  including the negative lower endpoint for `depth-4`.

## Data provenance and cross-checks

- All 12 input SHA-256 hashes in `paper/figs/figure-provenance.json` match
  their current files under `data/`, `evidence/`, and `reproduction/`.
- The seven pilot display labels match the saved run manifests under
  `reproduction/defense-redesign-pilot/experiments/manifests/` (reference,
  three noise scales, and three quantisation bit-widths).
- Overlapping exploration rejection counts are 13 qualified privacy
  breaches, 5 utility-gate failures, and 7 gain-gate failures across all
  19 exploratory cells.
- Utility means and two-sided 95% Student-$t$ intervals ($\mathrm{df}=3$)
  recomputed directly from `evidence/trilemma-campaign/r1_seed_results.csv`
  agree with `data/r1_cells.csv` within $10^{-10}$, confirming that
  `span-6` clears both utility criteria.
- Finite campaign counts carry no sampling intervals; utility intervals
  describe only the four recorded seeds on fixed evaluation text, and a
  utility pass is not a positive privacy result.

## Verified PDF digests

- `paper/figs/campaign-gates.pdf`:
  `4afc4267942963f84f68b67ad1ed58a63c273e7e457f174ccc0e576b3f502cf9`
- `paper/figs/r1-utility-boundary.pdf`:
  `54bee1e81bdf953c5929b8bacd29fd568847d3ded30edca54b4db63dd6c2f187`
