# Assurance matrix (adversary-conditioned)

Defense x attacker access x measured leakage x utility cost x evidence
strength. Per-row measurements only: **no universal risk score, no
maturity model** — the current evidence does not validate either.
Rows with missing leakage or utility fields are INCOMPLETE, never
dropped.

## Evidence-strength ladder

| Level | Meaning |
|---|---|
| `fixture` | Fixture/synthetic batteries (e.g. replica-integrity fixtures, view-privacy synthetic de-mixing). Engineering checks on synthetic data; positives do not falsify a deployed defense. |
| `live_channel_engineering` | Live in-repo channels exercised end to end (capture/transport/recovery plumbing). Demonstrates mechanism, not measured leakage against a defended target. |
| `defended_pilot_paired` | Defended development pilot over trained defenders: paired defense-minus-none seed differences with percentile bootstrap CIs, utility deltas, failures-and-nulls accounting. Development tier; conditional on a fixed article sample; not a confirmatory claim. |
| `frozen_crossed_bootstrap` | Frozen crossed-bootstrap study under the v2 protocol (crossed document/run bootstrap with multiplicity correction). Protocol frozen; evidence counts at this level only once executed. |

## Matrix

| Defense | Access (split) | Leakage: fwd recovery − none pp [95% CI] | Utility: ΔPPL vs none [95% CI] | Evidence strength | Status |
|---|---|---:|---:|---|---|
| gaussian | split-3 training-wire | 0.72 [0.2, 1.66] | 0.346 [-1.426, 1.874] | `defended_pilot_paired` | ESTIMATED |
| none | split-3 training-wire | 0 [0, 0] | 0 [0, 0] | `defended_pilot_paired` | ESTIMATED |
| replica-median | split-3 training-wire | -66.7 [-67.68, -64.94] | 49.113 [6.703, 97.342] | `defended_pilot_paired` | ESTIMATED |
| secret-mixup | split-3 training-wire | -5.79 [-6.74, -4.49] | 35.078 [11.014, 67.963] | `defended_pilot_paired` | ESTIMATED |
| trained-bottleneck | split-3 training-wire | -91.6 [-92.09, -91.02] | 6.64e+07 [3.84e+07, 1.01e+08] | `defended_pilot_paired` | ESTIMATED |
| gaussian | split-6 training-wire | 0.2 [-1.66, 1.76] | 0.077 [-1.221, 0.78] | `defended_pilot_paired` | ESTIMATED |
| none | split-6 training-wire | 0 [0, 0] | 0 [0, 0] | `defended_pilot_paired` | ESTIMATED |
| replica-median | split-6 training-wire | -71.22 [-71.68, -70.51] | 45.323 [7.191, 93.578] | `defended_pilot_paired` | ESTIMATED |
| secret-mixup | split-6 training-wire | -7.23 [-8.59, -5.66] | 36.321 [15.961, 71.126] | `defended_pilot_paired` | ESTIMATED |
| trained-bottleneck | split-6 training-wire | -87.6 [-88.57, -86.52] | 1.24e+08 [1.69e+07, 2.6e+08] | `defended_pilot_paired` | ESTIMATED |
