# Test run — GREEN

- env: python 3.14.6 (torch present), 2026-09-16T14:13:47Z
- totals: 2555 passed, 0 failed, 36 skipped of 2591

## Per tier

| tier | passed | failed | skipped |
|---|---|---|---|
| bdd | 58 | 0 | 1 |
| torch | 70 | 0 | 33 |
| torch-free | 2427 | 0 | 2 |

## Skipped (by reason)

- **1×** Skipped: privacy_runtime (defense SDK) not importable on this host
  - `tests/test_defenses.py::test_run_scenarios_end_to_end`
- **2×** Skipped: single-cell experiment: covered by the skeleton test
  - `tests/test_coverage.py::test_lane_c_dropping_one_cell_bites[artifact_telemetry_leak]`
  - `tests/test_coverage.py::test_lane_c_dropping_one_cell_bites[seam_inversion]`
- **1×** Skipped: the .pt guard only fires when torch is absent
  - `tests/test_seed_disclosure.py::test_pt_bundle_fails_closed_without_torch`
- **1×** Skipped: torch host: the torch-free guard is covered by the module self-test's torch branch
  - `tests/test_defenses.py::test_torchfree_host_behavior_is_explicit`
- **1×** Skipped: torch installed — the torch checks run instead of SKIPPED
  - `tests/test_ica.py::test_self_test_reports_torch_checks_skipped_and_passes`
- **6×** Skipped: torch installed — torch-absent behavior not applicable
  - `tests/test_captures.py::test_load_tensor_requires_torch`
  - `tests/test_captures.py::test_load_epoch_rows_fails_closed_without_torch`
  - `tests/test_ica.py::test_module_imports_torch_less`
  - `tests/test_synthetic.py::test_module_imports_torch_less`
  - `tests/test_synthetic.py::test_make_toy_world_requires_torch`
  - … and 1 more
- **14×** Skipped: torch present — the numeric toy tier runs instead
  - `tests/test_accumulation.py::test_run_fails_closed_without_torch`
  - `tests/test_accumulation.py::test_cli_fails_closed_without_torch`
  - `tests/test_known_prefix.py::test_run_fails_closed_without_torch`
  - `tests/test_known_prefix.py::test_cli_fails_closed_without_torch`
  - `tests/test_leak_accumulation.py::test_run_fails_closed_without_torch`
  - … and 9 more
- **1×** Skipped: torch present: require_torch passes here
  - `tests/test_body_extraction.py::test_real_tier_requires_torch`
- **7×** Skipped: torch present: the guard does not fire
  - `tests/test_active_cloud.py::test_requires_torch_guard`
  - `tests/test_alignment_search.py::test_requires_torch_guard`
  - `tests/test_ica_bss.py::test_requires_torch_guard`
  - `tests/test_latent_matching.py::test_run_fails_closed_without_torch`
  - `tests/test_latent_sensitivity.py::test_run_fails_closed_without_torch`
  - … and 2 more
- **2×** Skipped: torch-free only edge
  - `tests/test_checkpoint.py::test_pt_checkpoint_without_torch_is_a_descriptive_error`
  - `tests/test_checkpoint.py::test_non_json_state_without_torch_is_a_descriptive_error`

## Slowest 10

- 26.1239s `tests/test_baseline_selftest.py::test_self_test_subprocess_exit_zero` (passed)
- 18.3267s `tests/test_defended_pilot.py::test_quick_end_to_end_cpu` (passed)
- 7.0739s `tests/test_bisr_baseline.py::test_module_self_test_passes` (passed)
- 6.0554s `tests/test_operating_point.py::test_fresh_validation_cli_and_markdown` (passed)
- 4.9752s `tests/test_poisoning_backdoor.py::test_determinism_same_seed` (passed)
- 4.7004s `tests/test_bottleneck_sweep.py::test_analyzer_whitebox_e2e_fail_closed_on_invalid_arm` (passed)
- 4.638s `tests/test_mitm.py::test_self_test_passes` (passed)
- 4.1954s `tests/test_operating_point.py::test_fresh_only_rule` (passed)
- 4.0662s `tests/test_bisr_baseline.py::test_toy_opt_run_journals_artifact_and_evidence` (passed)
- 3.822s `tests/test_operating_point.py::test_fresh_extra_cells_beyond_controls_fail_closed` (passed)
