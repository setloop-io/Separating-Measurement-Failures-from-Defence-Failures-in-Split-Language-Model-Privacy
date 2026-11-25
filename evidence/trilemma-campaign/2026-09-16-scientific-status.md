# Status — 2026-09-16 (evening): scientific stack frozen; calibration complete; exploratory running

Extends `2026-09-16-gpu-engineering-status.md` (engineering stages) into the
scientific programme. Owner authorized owned nodes on 2026-09-16; metered cloud
remains unauthorized.

## Scientific stack (built, tested, frozen today)

- `src/pcr/r1_scientific.py` — admitted calibration/exploratory stage machinery:
  sparse training-frame captures (step 0 / every 200 / 2000, RNG-invariant),
  remote initial+final snapshots bound to initialization and checkpoint,
  scientific admission (`r1-scientific-admission-1.0`) and single-use executor,
  seed-disjointness validation, per-stage limits.
- `src/pcr/r1_scientific_analysis.py` — strict per-seed qualification
  aggregation, frozen-spec proposal with amendment knobs, three-valued cell
  classification (utility/gain gates with Student-t seed intervals, worst-seed
  privacy rule, sensitivity grid, empty-set declaration logic). scipy-free t
  implementation verified against `scripts/stats_utils.py`.
- Attack suites in admitted runs now carry `scientific_evidence: true` /
  `scoped_probe_suite_only` (schema widened; engineering labels untouched).
- Analyzer fix `bc5242e`: bundle-root-relative manifest bindings resolve after
  retrieval/relocation (regression-tested).
- **352 tests + 12 subtests green** on macOS and on ARM Linux (Voula, 697 s).
- Commits: stack `0bcf568` → protocol/matrix freeze `b3c472c`
  (`r1-exploration-1.0`, seal `5fdb9d19…`, 101 files) → scientific packet
  `525de9c` (bundle `7987e7f9…`, verified via Mac restore, both GPU containers,
  Sasa replica) → analyzer fix `bc5242e` → qualification evidence `d221b47`.

## Calibration cal001 (toula) — COMPLETE

76/76 runs (19 cells × 4 seeds, trained, 2000 steps), zero failures, 26,816 s
vs the 28,800 s budget cap; execution root 37 GB retained on toula, retrieval to
the orchestrator in progress. All records `scientific_evidence: true` under the
single-use admission (consumed).

### Qualification outcome (first scientific result of the programme)

- **QUALIFIED (4/24)**: `input_byte@inbound_gradients` and
  `input_byte@outbound_gradients`, both defense settings — positive controls
  clear the floors with margin and matched nulls stay in band.
- **UNQUALIFIED (20/24)**: all `forward_values`/`returned_values` channels and
  all `next_byte_class`/`punctuation_fraction` combinations, predominantly on the
  matched-null criterion (null advantage 0.045–0.56 vs the 0.02 band).
- **Owner freeze decision**: thresholds UNCHANGED; the amendment path
  (`null_band ≥ 0.041`) was rejected because it would admit instruments with
  invalid nulls. Frozen spec `attack-spec-frozen.json` (schema-validated,
  `547b5a93…`) records the verdicts and rationale.
- Protocol consequence: exploratory cells can only be REJECTED (utility/gain
  gates, or a qualified gradient-channel breach) or UNDETERMINED; no scoped PASS
  exists under this instrument suite; unqualified channels are named in every
  report. The trilemma question may remain partially undetermined — a valid
  protocol outcome, not a reporting failure.

## Exploratory exp001 (koula) — INTERRUPTED (co-tenant GPU); exp002 (toula) — RUNNING

**exp001**: 232 runs admitted against the frozen spec; started ~2026-09-16 17:20
UTC on koula. Gracefully interrupted after 9/232 runs (~19:45 UTC) when the
co-tenant workload `zen_albattani` held 88% of koula's GPU: split runs degraded
to 475–720 s (vs 350 s uncontended), projecting 128,400 s against the 86,400 s
budget cap — a certain fail-stop. The stage was SIGINT-stopped; the runner wrote
its execution summary (completed 9, failure `KeyboardInterrupt`); the admission
remains consumed and all records are preserved. No zen_albattani resource was
touched.

**exp002**: freshly reviewed attempt-2 budget/environment admissions (toula,
verified uncontended: 0% GPU util), executed from ~2026-09-16 18:50 UTC on
toula with the same frozen spec (`547b5a93…`); projected ~81,200 s vs cap.
Hourly fail-closed health check armed, now including a co-tenant GPU watch.

## Where the study stands

- Engineering: 32/32 runs complete (CPU validation, Stage A, Stage B).
- Scientific: calibration 76/76 complete + qualification frozen; exploratory
  running; confirmation still undesigned (separate protocol required).
- Claim ledger: still admits no empirical claim; the exploratory classification
  and Phase 4 analysis are what may move rows, within the instrument scope above.
