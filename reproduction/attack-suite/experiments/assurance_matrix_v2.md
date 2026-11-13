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
| `strengthened_attacker_development_sweep` | Development sweep over the bottleneck defense family (pca-compress / tb-frozen / tb-cotrain x dims x splits) with a strengthened attacker: base-tier forward/SIPIT attacks plus a defense-aware surrogate arm, and a separately labeled white-box tier that grants the true defense checkpoint. Predefined dual utility gates (no retuning); the sweep selection outcome was NEGATIVE. Development tier — stronger attacker access than the defended pilot, still not a confirmatory claim. |
| `frozen_crossed_bootstrap` | Frozen crossed-bootstrap study under the v2 protocol (crossed document/run bootstrap with multiplicity correction). Protocol frozen; evidence counts at this level only once executed. |

## Matrix

| Defense | Dim | Access (split) | Leakage | Utility: ΔPPL vs none [95% CI] | Evidence strength | Status |
|---|---|---|---|---:|---|---|
| gaussian | — | split-3 training-wire | fwd−none 0.72 pp [0.2, 1.66] | 0.346 [-1.426, 1.874] | `defended_pilot_paired` | ESTIMATED |
| none | — | split-3 training-wire | fwd−none 0 pp [0, 0] | 0 [0, 0] | `defended_pilot_paired` | ESTIMATED |
| replica-median | — | split-3 training-wire | fwd−none -66.7 pp [-67.68, -64.94] | 49.113 [6.703, 97.342] | `defended_pilot_paired` | ESTIMATED |
| secret-mixup | — | split-3 training-wire | fwd−none -5.79 pp [-6.74, -4.49] | 35.078 [11.014, 67.963] | `defended_pilot_paired` | ESTIMATED |
| trained-bottleneck | — | split-3 training-wire | fwd−none -91.6 pp [-92.09, -91.02] | 6.64e+07 [3.84e+07, 1.01e+08] | `defended_pilot_paired` | ESTIMATED |
| pca-compress | 64 | split-3 training-wire | strongest observed 59.28 (base forward) | 144.147 [130.112, 153.759] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 64 | split-3 training-wire | strongest observed 59.24 (base forward) | 140.465 [134.262, 144.075] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-frozen | 64 | split-3 training-wire | strongest observed 59.44 (base forward) | 140.194 [134.879, 143.048] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 64 | split-3 wire + white-box (true checkpoint) | strongest observed 59.83 (white-box inverter forward) | 144.147 [130.112, 153.759] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 64 | split-3 wire + white-box (true checkpoint) | NOT_EVALUABLE (reason in JSON) | — | `strengthened_attacker_development_sweep` | NOT_EVALUABLE |
| tb-frozen | 64 | split-3 wire + white-box (true checkpoint) | strongest observed 59.24 (white-box inverter forward) | 140.194 [134.879, 143.048] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 128 | split-3 training-wire | strongest observed 81.9 (base forward) | 41.955 [39.59, 44.584] (gates 1.5x/2.0x: FAIL/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 128 | split-3 training-wire | strongest observed 80.73 (base forward) | 41.759 [35.12, 48.23] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-frozen | 128 | split-3 training-wire | strongest observed 80.7 (base forward) | 41.819 [35.018, 48.575] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 128 | split-3 wire + white-box (true checkpoint) | strongest observed 81.87 (white-box optimized forward) | 41.955 [39.59, 44.584] (gates 1.5x/2.0x: FAIL/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 128 | split-3 wire + white-box (true checkpoint) | NOT_EVALUABLE (reason in JSON) | — | `strengthened_attacker_development_sweep` | NOT_EVALUABLE |
| tb-frozen | 128 | split-3 wire + white-box (true checkpoint) | strongest observed 81.09 (white-box optimized forward) | 41.819 [35.018, 48.575] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 256 | split-3 training-wire | strongest observed 89.36 (base forward) | 15.476 [14.107, 16.334] (gates 1.5x/2.0x: PASS/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 256 | split-3 training-wire | strongest observed 89.36 (base SIPIT) | 14.604 [12.364, 16] (gates 1.5x/2.0x: PASS/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-frozen | 256 | split-3 training-wire | strongest observed 89.32 (base SIPIT) | 14.604 [12.391, 15.977] (gates 1.5x/2.0x: PASS/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 256 | split-3 wire + white-box (true checkpoint) | strongest observed 89.36 (white-box optimized forward) | 15.476 [14.107, 16.334] (gates 1.5x/2.0x: PASS/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 256 | split-3 wire + white-box (true checkpoint) | NOT_EVALUABLE (reason in JSON) | — | `strengthened_attacker_development_sweep` | NOT_EVALUABLE |
| tb-frozen | 256 | split-3 wire + white-box (true checkpoint) | strongest observed 89.23 (white-box optimized forward) | 14.604 [12.391, 15.977] (gates 1.5x/2.0x: PASS/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| gaussian | — | split-6 training-wire | fwd−none 0.2 pp [-1.66, 1.76] | 0.077 [-1.221, 0.78] | `defended_pilot_paired` | ESTIMATED |
| none | — | split-6 training-wire | fwd−none 0 pp [0, 0] | 0 [0, 0] | `defended_pilot_paired` | ESTIMATED |
| replica-median | — | split-6 training-wire | fwd−none -71.22 pp [-71.68, -70.51] | 45.323 [7.191, 93.578] | `defended_pilot_paired` | ESTIMATED |
| secret-mixup | — | split-6 training-wire | fwd−none -7.23 pp [-8.59, -5.66] | 36.321 [15.961, 71.126] | `defended_pilot_paired` | ESTIMATED |
| trained-bottleneck | — | split-6 training-wire | fwd−none -87.6 pp [-88.57, -86.52] | 1.24e+08 [1.69e+07, 2.6e+08] | `defended_pilot_paired` | ESTIMATED |
| pca-compress | 64 | split-6 training-wire | strongest observed 54.82 (base forward) | 169.538 [156.034, 182.309] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 64 | split-6 training-wire | strongest observed 55.4 (base forward) | 146.74 [138.549, 154.293] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-frozen | 64 | split-6 training-wire | strongest observed 55.34 (base forward) | 146.71 [138.601, 154.299] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 64 | split-6 wire + white-box (true checkpoint) | strongest observed 54.92 (white-box optimized forward) | 169.538 [156.034, 182.309] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 64 | split-6 wire + white-box (true checkpoint) | NOT_EVALUABLE (reason in JSON) | — | `strengthened_attacker_development_sweep` | NOT_EVALUABLE |
| tb-frozen | 64 | split-6 wire + white-box (true checkpoint) | strongest observed 55.53 (white-box optimized forward) | 146.71 [138.601, 154.299] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 128 | split-6 training-wire | strongest observed 75.29 (base forward) | 69.154 [65.155, 72.347] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 128 | split-6 training-wire | strongest observed 75.39 (base forward) | 56.799 [55.777, 57.614] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-frozen | 128 | split-6 training-wire | strongest observed 75.36 (base forward) | 56.87 [56.013, 57.789] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 128 | split-6 wire + white-box (true checkpoint) | strongest observed 75.33 (white-box optimized forward) | 69.154 [65.155, 72.347] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 128 | split-6 wire + white-box (true checkpoint) | NOT_EVALUABLE (reason in JSON) | — | `strengthened_attacker_development_sweep` | NOT_EVALUABLE |
| tb-frozen | 128 | split-6 wire + white-box (true checkpoint) | strongest observed 75.98 (white-box optimized forward) | 56.87 [56.013, 57.789] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 256 | split-6 training-wire | strongest observed 83.85 (base forward) | 35.759 [24.542, 49.334] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 256 | split-6 training-wire | strongest observed 84.24 (base forward) | 22.229 [19.902, 26.261] (gates 1.5x/2.0x: FAIL/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-frozen | 256 | split-6 training-wire | strongest observed 84.21 (base forward) | 22.215 [19.904, 26.24] (gates 1.5x/2.0x: FAIL/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |
| pca-compress | 256 | split-6 wire + white-box (true checkpoint) | strongest observed 83.85 (white-box optimized forward) | 35.759 [24.542, 49.334] (gates 1.5x/2.0x: FAIL/FAIL) | `strengthened_attacker_development_sweep` | ESTIMATED |
| tb-cotrain | 256 | split-6 wire + white-box (true checkpoint) | NOT_EVALUABLE (reason in JSON) | — | `strengthened_attacker_development_sweep` | NOT_EVALUABLE |
| tb-frozen | 256 | split-6 wire + white-box (true checkpoint) | strongest observed 84.18 (white-box optimized forward) | 22.215 [19.904, 26.24] (gates 1.5x/2.0x: FAIL/PASS) | `strengthened_attacker_development_sweep` | ESTIMATED |

## Bottleneck-family sweep selection outcome

- Outcome: **NEGATIVE** — split(s) ['3', '6'] selected NO operating point; the NEGATIVE result is retained and v3 remains UNFROZEN
- Negative splits: 3, 6
- Utility gates (predefined, no retuning): primary 1.5x, secondary 2.0x
- Candidates: tb-frozen, tb-cotrain; control: pca-compress
- v3 status: **UNFROZEN** — a negative outcome retains the NEGATIVE result; v3 remains UNFROZEN — this tool never freezes anything

