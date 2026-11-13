# Figure captions and reproduction

Both figures are vector PDF/SVG, built at the revision's final 6.5-inch text width, with at least 8-point text. Run `uv run --no-project figs/make_figures.py` from `revision-2/`, or give the absolute script path from any directory. The script reads the independently recomputed CSVs under `data/`, checks them against the original R1 cell table and P4 generated report, and records input SHA-256 hashes and all plotted values in `figure-provenance.json`.

## campaign-gates.pdf

**No tested cell cleared all gates, but rejection and uncertainty have different meanings.** (a) R1 rejection reasons among 19 exploratory cells; reasons overlap, so the counts must not be summed. (b) P4 outcomes for seven exploratory cells: P, pass; F, failed; U, undetermined. All seven pass gain, two fail utility, and privacy fails in four while remaining undetermined in three. Counts enumerate the saved campaigns; no sampling interval is claimed.

LaTeX caption:

```latex
\caption{\textbf{No tested cell cleared all gates, but rejection and uncertainty have different meanings.}
(a) R1 rejection reasons among 19 exploratory cells; reasons overlap and must not be summed.
(b) P4 outcomes for seven exploratory cells: P, pass; F, failed; U, undetermined.
All seven pass gain, two fail utility, and privacy fails in four while remaining undetermined in three.
Counts enumerate the saved campaigns; no sampling interval is claimed.}
```

Sources: `data/r1_cells.csv`, `data/p4_cells.csv`; original declarations in `../evidence/trilemma-campaign/r1_cell_results.csv` and `../reproduction/defense-redesign-pilot/outputs/runs/pilot-20260922-a/report.md`. The underlying campaign remains exploratory; the figure is not a general impossibility statement or a positive privacy certificate.

## r1-utility-boundary.pdf

**Span-6 passes both R1 utility clauses, narrowly.** Points are mean held-out utility penalties, with source two-sided 95% t-intervals over four seeds (3 degrees of freedom). The mean must be at most 0.10 nats and the interval's upper endpoint at most 0.12 nats. Span-6 has mean 0.099971 and upper endpoint 0.119502; P denotes utility pass only. These cells still fail privacy and/or gain.

LaTeX caption:

```latex
\caption{\textbf{Span-6 passes both R1 utility clauses, narrowly.}
Points are mean held-out utility penalties, with source two-sided 95\% $t$-intervals over four seeds
(3 degrees of freedom). The mean must be at most 0.10 nats and the interval's upper endpoint
at most 0.12 nats. Span-6 has mean 0.099971 and upper endpoint 0.119502;
P denotes utility pass only. These cells still fail privacy and/or gain.}
```

Sources: `data/r1_cells.csv`; original `../evidence/trilemma-campaign/r1_cell_results.csv` and `r1_seed_results.csv`. The lower interval for depth-4 is negative because it is the recorded, untruncated t-interval. This figure does not show a paired difference between cells and supports no significance claim about differences between their means. P4 uses a different, one-sided upper-limit construction and is not combined with these intervals.
