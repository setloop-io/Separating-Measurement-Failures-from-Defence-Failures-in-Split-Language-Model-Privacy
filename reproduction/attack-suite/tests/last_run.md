# Test run — GREEN

- env: python 3.14.6 (torch present), 2026-10-22T00:09:26Z
- totals: 27 passed, 0 failed, 0 skipped of 27

## Per tier

| tier | passed | failed | skipped |
|---|---|---|---|
| torch-free | 27 | 0 | 0 |

## Slowest 10

- 1.2181s `tests/test_clustered_assessment.py::test_supported_case_needs_real_control_and_utility_without_legacy_changes` (passed)
- 0.0097s `tests/test_clustered_assessment.py::test_domain_loader_selects_the_declared_defense_in_mixed_bundles` (passed)
- 0.0074s `artifacts/paper-b-review-round2-20261021/all-review-closure/cpu-closure-20261022/statistical/reporting/test_temporal_inputs.py::test_domain_training_is_serial_but_posttraining_prefill_is_frozen` (passed)
- 0.0034s `tests/test_clustered_assessment.py::test_case_tables_include_samples_bounds_controls_utility_and_all_mde_rows` (passed)
- 0.0028s `tests/test_clustered_assessment.py::test_horizon_loader_excludes_evidence_sidecars_by_full_filename` (passed)
- 0.0021s `tests/test_clustered_assessment.py::test_malformed_duplicate_and_overflow_utility_fail_safely` (passed)
- 0.0014s `tests/test_clustered_assessment.py::test_matching_labels_do_not_replace_actual_reference_alignment[seeds]` (passed)
- 0.0014s `tests/test_clustered_assessment.py::test_matching_labels_do_not_replace_actual_reference_alignment[documents]` (passed)
- 0.0013s `tests/test_clustered_assessment.py::test_planning_simulation_retains_zero_variance_guard_and_declared_effect` (passed)
- 0.0011s `tests/test_clustered_assessment.py::test_upper_margin_is_not_two_sided_equivalence` (passed)
