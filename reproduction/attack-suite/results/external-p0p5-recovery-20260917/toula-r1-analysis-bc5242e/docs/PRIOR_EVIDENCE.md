# Prior evidence and exclusions

## Source project

The motivating evidence comes from:

`/Users/geo/Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks`

The detailed reassessment is:

`outputs/CAPACITY_CEILING_REASSESSMENT_2026-08-27.md`

These absolute paths are provenance notes for the local research workspace; submission
artifacts must replace them with public, immutable references where appropriate.

## Observations that may motivate this study

- Under a tested frozen-backbone, gauge-constrained surrogate setup, longer replaced spans
  often incurred larger residual loss.
- Increasing nominal boundary width or surrogate size did not reliably remove the residual
  within the tested training budgets.
- One deep-model sweep suggested that segment structure might matter more than nominal layer
  count, but it was single-seed and confounded.
- A never-trained-remote ablation retained nearly all of the reported “cloud contribution,”
  showing that nominal placement did not isolate useful remote learning.

## Claims explicitly excluded

- No information-theoretic capacity ceiling was established.
- No exact 4:1 privacy/utility exchange rate was established.
- No task-specific rate-distortion or communication lower bound was proved.
- No universal privacy–utility–delegation triangle was measured.
- No causal law that depth produces delegability was established.
- The one-seed 47.4% result was marginal on utility and did not have a valid complete-view
  privacy measurement.
- Near-zero MINE estimates are not upper bounds on mutual information and are not privacy
  certificates.

## Permitted use

Prior results may be cited as motivation for hypotheses and for the design of positive
controls. They may not be pooled with new confirmatory runs unless the code, data, protocol,
and sampling design are demonstrated to be exchangeable before outcome inspection.

