# Figure captions and reproduction

The current manuscript follows the attached paper's caption style, with bold
labels and regular caption text. A period after the label retains the earlier
punctuation request. The authoritative captions are in `sections/results.tex`.

The vector figures remain at the final 6.5-inch text width.

```bash
uv run --no-project figs/make_figures.py
```

## campaigns

```latex
\caption{Qualified breaches remain valid despite incomplete coverage.
The left panel shows overlapping rejection reasons among 19 exploration
configurations evaluated with four seeds each. The right panel gives the
recorded gate outcomes for seven pilot cells with three paired seeds.
The counts describe complete campaigns and carry no population sampling
interval. The source records are
\artifact{R1} and \artifact{P4}.}
```

## utility

```latex
\caption{Span-6 passes both recorded utility criteria.
Means and descriptive two-sided 95\% Student-$t$ intervals across four
training seeds are conditional on fixed evaluation text. The $0.10$ mean
threshold and $0.12$ upper-limit threshold concern different statistics.
An interval endpoint can cross the mean threshold while the utility
criteria still pass. The source is \artifact{R1}. Intervals follow
Eq.~\eqref{eq:interval}.}
```

