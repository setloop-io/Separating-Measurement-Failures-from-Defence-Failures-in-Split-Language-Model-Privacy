# R1 scientific workload preparation — 2026-09-16 (frozen packet)

Prepared from the clean master checkout at `b3c472c` (freeze commit; source
inventory seal `5fdb9d1984d78c542aa961b6cdca44c29dfb05cb01622a6e4d0fbac883fa6cbc`,
101 bound files). Carries the frozen protocol `r1-exploration-1.0` and the
scientific stack committed at `0bcf568` (352 tests + 12 subtests green locally
and on the ARM CPU node Voula). This is the packet the calibration and
exploratory scientific admissions bind. Preparation is not admission:
`scientific_execution_admitted: false`.

- `workload.tar.gz` — the complete prepared bundle (191 files), archive SHA-256
  `48b3b2c520460a4b5721a11d88c2539d04909419126b783835b0682e1e9d45df` (xattr-free).
- `r1_bundle.json` — bundle inventory; bundle SHA-256
  `7987e7f95f03eff012e14f97f76fcfa50ab231e130ddc6994877f8cd067de9e3`.
- `plan.json` — the four planned workloads with the scientific stages enumerated
  (76 calibration / 232 exploratory runs).
- `validation.json` — archive/restore/verify record and stack verification summary.

Restoration check: extract `workload.tar.gz` to a new directory and run
`python <restored>/r1-bundle-sci-20260916/scripts/run_r1.py verify-bundle` with a
Python environment carrying the project's runtime dependencies.
