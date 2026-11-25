# Mechanical operating-point selection (bottleneck sweep)

Schema `llm_attacker.operating_point_selection.v1`. Gates are PREDEFINED: ppl_after <= 1.5x (primary) / <= 2.0x (secondary) the paired undefended control; retuning is refused. Compute factor <= 2.0x the pca-compress control's train+attack wall seconds.

Leakage statistic: strongest observed across measured arms: per-seed max over the base-tier forward/SIPIT attacks, the defense-aware surrogate arm, the white-box-aware arms, and — when present — the adaptive end-to-end (true transform) arm. A measured statistic over the measured arms only — stronger or better-optimized attacks may recover more.

Eligibility: PCA-superiority requires BOTH the bootstrap condition (point estimate > 0 AND bootstrap CI lower > 0 on the paired per-seed differences) AND the all-seeds-positive condition (per-seed paired difference > 0 for EVERY seed). They are separate mechanical checks, recorded separately per candidate; the bootstrap condition alone does not imply improvement on every paired seed.

## Access and approximation limits

- the defense-aware arm fits a PCA surrogate (stand-in) inverse on public calibration; it is NOT a faithful decoder of the true defense transform and its recovery numbers are an approximation, not a certificate
- white-box-aware rows grant the attacker the TRUE defense checkpoint (encoder weights / PCA basis + provenance) — a strictly stronger access condition than the matched matrix. White-box recovery is a stronger-access measurement; it does not constrain black-box recovery of the same wire, nor the reverse
- adaptive end-to-end rows grant the attacker the TRUE defense checkpoint and optimize each wire's token sequence straight through it — a strictly stronger, adaptive access condition. Adaptive e2e recovery is a stronger-access measurement only: it does not constrain black-box recovery of the same wire, nor the surrogate arm, nor any defense-internals-hidden adversary
- a negative outcome retains the NEGATIVE result; v3 remains UNFROZEN — this tool never freezes anything

## Outcome per split

| Split | Outcome | Selected |
|---:|:---:|---|
| 3 | negative | NONE |
| 6 | negative | NONE |

## Candidate comparisons

| Split | Dim | Method | Gate | Strongest-observed leakage vs PCA (ctrl-cand [CI]) | Bootstrap | All seeds + | Compute ratio | Eligible |
|---:|---:|---|:---:|---|:---:|:---:|---:|:---:|
| 3 | 64 | tb-cotrain | fail | 0.0088 [-0.0078, 0.0176] | FAIL | FAIL | 0.670 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage improvement over pca-compress not established: bootstrap CI [-0.0078125, 0.017578125] includes 0 (ESTIMATED); per-seed paired difference not positive for every seed (seeds [22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 3 | 64 | tb-frozen | fail | 0.0016 [-0.0107, 0.0078] | FAIL | FAIL | 1.030 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage improvement over pca-compress not established: bootstrap CI [-0.0107421875, 0.0078125] includes 0 (ESTIMATED); per-seed paired difference not positive for every seed (seeds [22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 3 | 128 | tb-cotrain | fail | 0.0117 [-0.0117, 0.0254] | FAIL | FAIL | 0.670 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage improvement over pca-compress not established: bootstrap CI [-0.01171875, 0.025390625] includes 0 (ESTIMATED); per-seed paired difference not positive for every seed (seeds [22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 3 | 128 | tb-frozen | fail | 0.0078 [-0.0117, 0.0205] | FAIL | FAIL | 1.038 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage improvement over pca-compress not established: bootstrap CI [-0.01171875, 0.0205078125] includes 0 (ESTIMATED); per-seed paired difference not positive for every seed (seeds [22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 3 | 256 | tb-cotrain | primary | 0.0016 [-0.0088, 0.0078] | FAIL | FAIL | 0.667 | NO: strongest observed leakage improvement over pca-compress not established: bootstrap CI [-0.0087890625, 0.0078125] includes 0 (ESTIMATED); per-seed paired difference not positive for every seed (seeds [22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 3 | 256 | tb-frozen | primary | 0.0016 [-0.0078, 0.0068] | FAIL | FAIL | 1.025 | NO: strongest observed leakage improvement over pca-compress not established: bootstrap CI [-0.0078125, 0.0068359375] includes 0 (ESTIMATED); per-seed paired difference not positive for every seed (seeds [22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 6 | 64 | tb-cotrain | fail | -0.0049 [-0.0107, 0.0029] | FAIL | FAIL | 0.650 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage NOT below pca-compress control (paired point estimate -0.0048828125); per-seed paired difference not positive for every seed (seeds [11, 22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 6 | 64 | tb-frozen | fail | -0.0081 [-0.0127, -0.0029] | FAIL | FAIL | 1.013 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage NOT below pca-compress control (paired point estimate -0.008138020833333334); per-seed paired difference not positive for every seed (seeds [11, 22, 33]); the bootstrap condition alone does not establish improvement on every paired seed |
| 6 | 128 | tb-cotrain | fail | -0.0007 [-0.0234, 0.0273] | FAIL | FAIL | 0.649 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage NOT below pca-compress control (paired point estimate -0.0006510416666666666); per-seed paired difference not positive for every seed (seeds [11, 22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 6 | 128 | tb-frozen | fail | -0.0068 [-0.0264, 0.0156] | FAIL | FAIL | 1.017 | NO: utility gate FAIL at both predefined thresholds (1.5x / 2.0x) on the per-seed rows; strongest observed leakage NOT below pca-compress control (paired point estimate -0.0068359375); per-seed paired difference not positive for every seed (seeds [11, 22]); the bootstrap condition alone does not establish improvement on every paired seed |
| 6 | 256 | tb-cotrain | secondary | -0.0114 [-0.0215, -0.0049] | FAIL | FAIL | 0.641 | NO: strongest observed leakage NOT below pca-compress control (paired point estimate -0.011393229166666666); per-seed paired difference not positive for every seed (seeds [11, 22, 33]); the bootstrap condition alone does not establish improvement on every paired seed |
| 6 | 256 | tb-frozen | secondary | -0.0085 [-0.0215, 0.0029] | FAIL | FAIL | 1.009 | NO: strongest observed leakage NOT below pca-compress control (paired point estimate -0.008463541666666666); per-seed paired difference not positive for every seed (seeds [11, 22]); the bootstrap condition alone does not establish improvement on every paired seed |

**NEGATIVE**: split(s) ['3', '6'] selected NO operating point; the NEGATIVE result is retained and v3 remains UNFROZEN

