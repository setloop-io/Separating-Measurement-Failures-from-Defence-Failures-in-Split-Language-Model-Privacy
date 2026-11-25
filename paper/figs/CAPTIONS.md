# Figure captions and reproduction

Both manuscript figures are rendered at the 6.5-inch text width of the
11pt `article` layout. The authoritative LaTeX captions live in
`paper/sections/results.tex`.

```bash
uv run --no-project paper/figs/make_figures.py
```

## Figure 1 (`campaign-gates.pdf`)

```latex
\caption{Gate outcomes for the \RoneCells-cell exploration and seven-cell
defence pilot. Overlapping bars in panel~(a) tally qualified breaches,
utility losses, and gain deficits across four-seed exploratory candidates
(\artifact{R1}), while panel~(b) decomposes three-seed paired pilot cells
across utility, gain, and composite privacy rules (\artifact{P4}).}
```

## Figure 2 (`r1-utility-boundary.pdf`)

```latex
\caption{Utility loss deltas and descriptive two-sided 95\% Student-$t$
intervals from Eq.~\eqref{eq:interval} across four training seeds on fixed
evaluation text (\artifact{R1}). Separate vertical lines mark the
$0.10$-nat mean cutoff and the $0.12$-nat upper-endpoint cutoff, which
explains why \texttt{span-6} clears both utility criteria even though its
upper interval crosses $0.10$.}
```
