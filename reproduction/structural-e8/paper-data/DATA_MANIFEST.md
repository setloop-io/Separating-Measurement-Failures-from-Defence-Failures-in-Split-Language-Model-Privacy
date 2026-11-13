# DATA_MANIFEST — committed result artifacts

Inventory of every result JSON tracked in git, with the code commit that
produced it where recorded. Regenerate with the script in the commit that
added this file; do not hand-edit.

Node `h100-1` was lost on 2026-08-07. Everything below survived because it
was committed; anything that lived only on that node did not. That is the
reason this file exists again — it was deleted in the clean slate (`7d4356c`)
and there was no inventory on the day it was needed.

## `collected` — 1,766 artifacts

| artifact | producing commit |
|---|---|
| `h100-1/training/e7_private_ft.json` | c0b3a584 |
| `h100-1/training/e7_private_ft_rep2.json` | c0b3a584 |
| `h100-1/training/e8_robustness_06b_fp32_rep2.json` | c0b3a584 |
| `h100-1/training/seq_inversion_06b_rep2.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_12b.json` | — |
| `h100-1/v2/privacy/e8_8b.json` | — |
| `h100-1/v2/privacy/e8_obfuscation_06b_fp32.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_robustness_06b.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_robustness_06b_fp32_anchorcheck.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_robustness_06b_fp32_kvoid.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_robustness_06b_fp32_kvoid_seed123.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_robustness_06b_fp32_kvoid_seed99.json` | c0b3a584 |
| `h100-1/v2/privacy/e8_robustness_06b_fp32_kvoid_sub100_seed42.json` | c0b3a584 |
| `h100-1/v2/privacy/e8kp_06b.json` | c0b3a584 |
| `h100-1/v2/privacy/e8kp_7b.json` | c0b3a584 |
| `h100-1/v2/privacy/output_inv_06b.json` | c0b3a584 |
| `h100-1/v2/privacy/output_inv_7b.json` | c0b3a584 |
| `h100-1/v2/privacy/sipit_06b.json` | — |
| `h100-1/v2/privacy/ti_12b.json` | — |
| `h100-1/v2/privacy/ti_7b.json` | — |
| `h100-1/v2/privacy/ti_8b.json` | — |
| `h100-2/training/e8_obfuscation_06b_fp32.json` | 54ba1d59 |
| `h100-2/training/e8_robustness_06b_fp32.json` | 54ba1d59 |
| `h100-2/training/e8_d3_06b_20260807T153610Z.json` | 9e82010e |
| `h100-2/issues/issue34_e8kp_corrected_depth1_06b_20260807T155848Z.json` | — |
| `h100-2/issues/issue40_seq_joint_06b_20260807T162239Z.json` | — |
| `h100-2/training/seq_inversion_06b.json` | 54ba1d59 |
| `h100-2/v2/privacy/dlgpp_06b.json` | 54ba1d59 |
| `h100-2/v2/privacy/e8_7b.json` | — |
| `h100-2/v2/privacy/ti_27b.json` | 94bc8421 |
| `h100-2/v2/privacy/ti_27b_d8.json` | 54ba1d59 |
| `invalid/ea4_mislabeled_default_seed42_20260806/koula/ea4_seed44_fedavg_server.json` | — |
| `invalid/ea4_mislabeled_default_seed42_20260806/toula/ea4_seed43_fedavg_client.json` | — |
| `invalid/ea4_mislabeled_default_seed42_20260806/toula/ea4_seed43_membership_property.json` | — |
| `invalid/ea4_mislabeled_default_seed42_20260806/toula/ea4_seed44_fedavg_client.json` | — |
| `invalid/ea4_mislabeled_default_seed42_20260806/toula/ea4_seed44_membership_property.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/ea4_invalid_zero_variance_aggregate.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/koula/ea4_seed43_fedavg_server.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/koula/ea4_seed44_fedavg_server.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/toula/ea4_seed42_membership_property.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/toula/ea4_seed43_fedavg_client.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/toula/ea4_seed43_membership_property.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/toula/ea4_seed44_fedavg_client.json` | — |
| `invalid/ea4_seeded_but_unshuffled_20260807/toula/ea4_seed44_membership_property.json` | — |
| `invalid/p2d1_identity_failure_20260807/p2d1_k_sweep_35b_target_27b_draft_20260807T182544Z.json` | 0e91d50 |
| `invalid/p2d1_exactness_fallback_20260807/p2d1_k4_exactness_confirmation_20260807T203503Z.json` | 33316f1 |
| `invalid/p2d1_exactness_fallback_20260807/p2d1_k4_exactness_confirmation_20260807T203503Z.log` | 33316f1 |
| `invalid/p2d1_exactness_fallback_20260807/p2d1_k4exact_cloud_20260807T203503Z.log` | 33316f1 |
| `h100-2/issues/issue41_sipit_7b_20260807T164125Z.json` | ac96d8d |
| `h100-2/issues/issue41_sipit_7b_20260807T164125Z.log` | ac96d8d |
| `diagnostic/p2d1_k4_20260807/p2d1_k4_correctness_diagnostic_20260807T195140Z.json` | a01c7be |
| `diagnostic/p2d1_k4_20260807/p2d1_k4_correctness_diagnostic_20260807T195140Z.log` | a01c7be |
| `diagnostic/p2d1_k4_20260807/p2d1_k4diag_cloud_20260807T195140Z.log` | a01c7be |
| `koula/ea4_seed42_fedavg_server.json` | — |
| `koula/ea4_seed43_fedavg_server.json` | — |
| `koula/ea4_seed44_fedavg_server.json` | — |
| `koula/issues/issue19_fla_qwen36-27b_fallback_koula_20260807T172444Z.json` | — |
| `toula/ablation_27b/ablation_qwen36-27b_0ms.json` | — |
| `toula/ablation_27b/ablation_qwen36-27b_80ms.json` | — |
| `toula/e9/base_A.json` | — |
| `toula/e9/gen_ids.json` | — |
| `toula/e9/ids_A.json` | — |
| `toula/e9/ids_B.json` | — |
| `toula/e9/obf_A.json` | — |
| `toula/e9/obf_B.json` | — |
| `toula/e9/obf_C.json` | — |
| `toula/e9/session_1/base_A.json` | — |
| `toula/e9/session_1/ids_A.json` | — |
| `toula/e9/session_1/ids_B.json` | — |
| `toula/e9/session_1/obf_B.json` | — |
| `toula/e9/session_1/obf_C.json` | — |
| `toula/e9/session_1/wire_attack_base_sess1.json` | — |
| `toula/e9/session_1/wire_attack_obf_sess1.json` | — |
| `toula/e9/session_2/base_A.json` | — |
| `toula/e9/session_2/ids_A.json` | — |
| `toula/e9/session_2/ids_B.json` | — |
| `toula/e9/session_2/obf_B.json` | — |
| `toula/e9/session_2/obf_C.json` | — |
| `toula/e9/session_2/wire_attack_base_sess2.json` | — |
| `toula/e9/session_2/wire_attack_obf_sess2.json` | — |
| `toula/e9/session_3/base_A.json` | — |
| `toula/e9/session_3/ids_A.json` | — |
| `toula/e9/session_3/ids_B.json` | — |
| `toula/e9/session_3/obf_B.json` | — |
| `toula/e9/session_3/obf_C.json` | — |
| `toula/e9/session_3/wire_attack_base_sess3.json` | — |
| `toula/e9/session_3/wire_attack_obf_sess3.json` | — |
| `toula/e9/wire_attack.json` | — |
| `toula/e9/wire_attack_base.json` | — |
| `toula/e9/wire_attack_obf.json` | — |
| `toula/e9/issue44_fixed_20260807T163240Z/**/*.json` (388 artifacts) | — |
| `toula/e9/issue44_rotated_20260807T163240Z/**/*.json` (388 artifacts) | — |
| `toula/e9/issue44_key_schedule_manifest.json` | — |
| `toula/er_20260811/COLLECTION_MANIFEST.json` | — |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt0_20260811T045925/summary.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_budget_rtt80_20260811T061312/summary.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt0_20260810T213415/summary.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n16_rtt80_20260811T011327/summary.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt0_20260810T235930/summary.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n256_rtt80_20260811T034442/summary.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt0_20260810T224704/summary.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_1/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_1/wire_attack_base_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_1/wire_attack_ratchet_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_1/wire_attack_static_sess1.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_2/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_2/wire_attack_ratchet_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_2/wire_attack_static_sess2.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_3/identity.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_3/wire_attack_ratchet_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/session_3/wire_attack_static_sess3.json` | 11cdb16 |
| `toula/er_20260811/er_n64_rtt80_20260811T023005/summary.json` | 11cdb16 |
| `toula/issues/issue19_fla_qwen36-27b_fallback_toula_20260807T172444Z.json` | — |
| `toula/paper2/p2d0_distributed_draft_smoke_35b_target_27b_draft_20260807T175544Z.json` | 0e91d50 |
| `toula/paper2/p2d4_identity_coverage_35b_target_27b_draft_20260807T181044Z.json` | 0e91d50 |
| `toula/realwan/rtt_mistral-7b_realwan.json` | — |
| `toula/rtt/rtt_qwen36-27b_0ms.json` | — |
| `toula/rtt/rtt_qwen36-27b_20ms.json` | — |
| `toula/rtt/rtt_qwen36-27b_80ms.json` | — |
| `toula/smoke/perplexity_27b_80ms.json` | — |
| `toula/training/ea4/ea4_seed42_fedavg_client.json` | — |
| `toula/training/ea4/ea4_seed42_membership_property.json` | — |
| `toula/training/ea4/ea4_seed43_fedavg_client.json` | — |
| `toula/training/ea4/ea4_seed43_membership_property.json` | — |
| `toula/training/ea4/ea4_seed43_split_ft_train.json` | — |
| `toula/training/ea4/ea4_seed44_fedavg_client.json` | — |
| `toula/training/ea4/ea4_seed44_membership_property.json` | — |
| `toula/training/ea4/ea4_seed44_split_ft_train.json` | — |
| `toula/training/ea4/ea4_training_seed_aggregate.json` | — |
| `toula/training/tmx_7b_overlap_0ms_rep1.json` | — |
| `toula/training/tmx_7b_overlap_0ms_rep2.json` | — |
| `toula/training/tmx_7b_overlap_0ms_rep3.json` | — |
| `toula/training/tmx_7b_overlap_80ms_rep1.json` | — |
| `toula/training/tmx_7b_overlap_80ms_rep2.json` | — |
| `toula/training/tmx_7b_overlap_80ms_rep3.json` | — |
| `toula/training/tmx_7b_overlap_hostile_rep1.json` | — |
| `toula/training/tmx_7b_overlap_hostile_rep2.json` | — |
| `toula/training/tmx_7b_overlap_hostile_rep3.json` | — |
| `toula/training/tmx_7b_sync_0ms_rep1.json` | — |
| `toula/training/tmx_7b_sync_0ms_rep2.json` | — |
| `toula/training/tmx_7b_sync_0ms_rep3.json` | — |
| `toula/training/tmx_7b_sync_80ms_rep1.json` | — |
| `toula/training/tmx_7b_sync_80ms_rep2.json` | — |
| `toula/training/tmx_7b_sync_80ms_rep3.json` | — |
| `toula/training/tmx_7b_sync_hostile_rep1.json` | — |
| `toula/training/tmx_7b_sync_hostile_rep2.json` | — |
| `toula/training/tmx_7b_sync_hostile_rep3.json` | — |
| `toula/training/tmx_moe_overlap_80ms_rep1.json` | — |
| `toula/training/tmx_moe_overlap_80ms_rep2.json` | — |
| `toula/training/tmx_moe_overlap_80ms_rep3.json` | — |
| `toula/training/tmx_moe_sync_0ms_rep1.json` | — |
| `toula/training/tmx_moe_sync_0ms_rep2.json` | — |
| `toula/training/tmx_moe_sync_0ms_rep3.json` | — |
| `toula/training/tmx_moe_sync_80ms_rep1.json` | — |
| `toula/training/tmx_moe_sync_80ms_rep2.json` | — |
| `toula/training/tmx_moe_sync_80ms_rep3.json` | — |
| `toula/training_status.json` | — |
| `toula/transport/transport_aggregate.json` | — |
| `toula/transport/transport_direct_0ms_rep1.json` | — |
| `toula/transport/transport_direct_0ms_rep2.json` | — |
| `toula/transport/transport_direct_0ms_rep3.json` | — |
| `toula/transport/transport_direct_80ms_rep1.json` | — |
| `toula/transport/transport_direct_80ms_rep2.json` | — |
| `toula/transport/transport_direct_80ms_rep3.json` | — |
| `toula/transport/transport_tunnel_0ms_rep1.json` | — |
| `toula/transport/transport_tunnel_0ms_rep2.json` | — |
| `toula/transport/transport_tunnel_0ms_rep3.json` | — |
| `toula/transport/transport_tunnel_80ms_rep1.json` | — |
| `toula/transport/transport_tunnel_80ms_rep2.json` | — |
| `toula/transport/transport_tunnel_80ms_rep3.json` | — |
| `toula/v2/12b/identity_80ms_rep1.json` | — |
| `toula/v2/12b/identity_80ms_rep2.json` | — |
| `toula/v2/12b/identity_80ms_rep3.json` | — |
| `toula/v2/12b/rtt_0ms_rep1.json` | — |
| `toula/v2/12b/rtt_0ms_rep2.json` | — |
| `toula/v2/12b/rtt_0ms_rep3.json` | — |
| `toula/v2/12b/rtt_20ms_rep1.json` | — |
| `toula/v2/12b/rtt_20ms_rep2.json` | — |
| `toula/v2/12b/rtt_20ms_rep3.json` | — |
| `toula/v2/12b/rtt_80ms_rep1.json` | — |
| `toula/v2/12b/rtt_80ms_rep2.json` | — |
| `toula/v2/12b/rtt_80ms_rep3.json` | — |
| `toula/v2/12b/rtt_hostilems_rep1.json` | — |
| `toula/v2/12b/rtt_hostilems_rep2.json` | — |
| `toula/v2/12b/rtt_hostilems_rep3.json` | — |
| `toula/v2/qwen36-35b-a3b/identity_80ms_rep1.json` | — |
| `toula/v2/qwen36-35b-a3b/identity_80ms_rep2.json` | — |
| `toula/v2/qwen36-35b-a3b/identity_80ms_rep3.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_0ms_rep1.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_0ms_rep2.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_0ms_rep3.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_20ms_rep1.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_20ms_rep2.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_20ms_rep3.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_80ms_rep1.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_80ms_rep2.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_80ms_rep3.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_hostilems_rep1.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_hostilems_rep2.json` | — |
| `toula/v2/qwen36-35b-a3b/rtt_hostilems_rep3.json` | — |
| `toula/v2/7b/identity_80ms.json` | — |
| `toula/v2/7b/identity_80ms_rep1.json` | — |
| `toula/v2/7b/identity_80ms_rep2.json` | — |
| `toula/v2/7b/identity_80ms_rep3.json` | — |
| `toula/v2/7b/rtt_0ms_rep1.json` | — |
| `toula/v2/7b/rtt_0ms_rep2.json` | — |
| `toula/v2/7b/rtt_0ms_rep3.json` | — |
| `toula/v2/7b/rtt_20ms_rep1.json` | — |
| `toula/v2/7b/rtt_20ms_rep2.json` | — |
| `toula/v2/7b/rtt_20ms_rep3.json` | — |
| `toula/v2/7b/rtt_80ms_rep1.json` | — |
| `toula/v2/7b/rtt_80ms_rep2.json` | — |
| `toula/v2/7b/rtt_80ms_rep3.json` | — |
| `toula/v2/7b/rtt_hostilems_rep1.json` | — |
| `toula/v2/7b/rtt_hostilems_rep2.json` | — |
| `toula/v2/7b/rtt_hostilems_rep3.json` | — |
| `toula/v2/realwan/rtt_realwan_rep1.json` | — |
| `toula/v2/realwan/rtt_realwan_rep2.json` | — |
| `toula/v2/realwan/rtt_realwan_rep3.json` | — |

## `results` — 72 artifacts

| artifact | producing commit |
|---|---|
| `e9/base_A.json` | — |
| `e9/gen_ids.json` | — |
| `e9/ids_A.json` | — |
| `e9/ids_B.json` | — |
| `e9/obf_A.json` | — |
| `e9/obf_B.json` | — |
| `e9/obf_C.json` | — |
| `e9/wire_attack.json` | — |
| `e9/wire_attack_base.json` | — |
| `e9/wire_attack_obf.json` | — |
| `training/tmx_7b_overlap_0ms_rep1.json` | — |
| `training/tmx_7b_overlap_0ms_rep2.json` | — |
| `training/tmx_7b_overlap_0ms_rep3.json` | — |
| `training/tmx_7b_overlap_80ms_rep1.json` | — |
| `training/tmx_7b_overlap_80ms_rep2.json` | — |
| `training/tmx_7b_overlap_80ms_rep3.json` | — |
| `training/tmx_7b_overlap_hostile_rep1.json` | — |
| `training/tmx_7b_overlap_hostile_rep2.json` | — |
| `training/tmx_7b_overlap_hostile_rep3.json` | — |
| `training/tmx_7b_sync_0ms_rep1.json` | — |
| `training/tmx_7b_sync_0ms_rep2.json` | — |
| `training/tmx_7b_sync_0ms_rep3.json` | — |
| `training/tmx_7b_sync_80ms_rep1.json` | — |
| `training/tmx_7b_sync_80ms_rep2.json` | — |
| `training/tmx_7b_sync_80ms_rep3.json` | — |
| `training/tmx_7b_sync_hostile_rep1.json` | — |
| `training/tmx_7b_sync_hostile_rep2.json` | — |
| `training/tmx_7b_sync_hostile_rep3.json` | — |
| `training/tmx_moe_overlap_80ms_rep1.json` | — |
| `training/tmx_moe_overlap_80ms_rep2.json` | — |
| `training/tmx_moe_overlap_80ms_rep3.json` | — |
| `training/tmx_moe_sync_0ms_rep1.json` | — |
| `training/tmx_moe_sync_0ms_rep2.json` | — |
| `training/tmx_moe_sync_0ms_rep3.json` | — |
| `training/tmx_moe_sync_80ms_rep1.json` | — |
| `training/tmx_moe_sync_80ms_rep2.json` | — |
| `training/tmx_moe_sync_80ms_rep3.json` | — |
| `training_status.json` | — |
| `v2/12b/identity_80ms_rep1.json` | — |
| `v2/12b/identity_80ms_rep2.json` | — |
| `v2/12b/identity_80ms_rep3.json` | — |
| `v2/12b/rtt_0ms_rep1.json` | — |
| `v2/12b/rtt_0ms_rep2.json` | — |
| `v2/12b/rtt_0ms_rep3.json` | — |
| `v2/12b/rtt_20ms_rep1.json` | — |
| `v2/12b/rtt_20ms_rep2.json` | — |
| `v2/12b/rtt_20ms_rep3.json` | — |
| `v2/12b/rtt_80ms_rep1.json` | — |
| `v2/12b/rtt_80ms_rep2.json` | — |
| `v2/12b/rtt_80ms_rep3.json` | — |
| `v2/12b/rtt_hostilems_rep1.json` | — |
| `v2/12b/rtt_hostilems_rep2.json` | — |
| `v2/12b/rtt_hostilems_rep3.json` | — |
| `v2/7b/identity_80ms.json` | — |
| `v2/7b/identity_80ms_rep1.json` | — |
| `v2/7b/identity_80ms_rep2.json` | — |
| `v2/7b/identity_80ms_rep3.json` | — |
| `v2/7b/rtt_0ms_rep1.json` | — |
| `v2/7b/rtt_0ms_rep2.json` | — |
| `v2/7b/rtt_0ms_rep3.json` | — |
| `v2/7b/rtt_20ms_rep1.json` | — |
| `v2/7b/rtt_20ms_rep2.json` | — |
| `v2/7b/rtt_20ms_rep3.json` | — |
| `v2/7b/rtt_80ms_rep1.json` | — |
| `v2/7b/rtt_80ms_rep2.json` | — |
| `v2/7b/rtt_80ms_rep3.json` | — |
| `v2/7b/rtt_hostilems_rep1.json` | — |
| `v2/7b/rtt_hostilems_rep2.json` | — |
| `v2/7b/rtt_hostilems_rep3.json` | — |
| `v2/realwan/rtt_realwan_rep1.json` | — |
| `v2/realwan/rtt_realwan_rep2.json` | — |
| `v2/realwan/rtt_realwan_rep3.json` | — |

## `results-h100-1` — 21 artifacts

| artifact | producing commit |
|---|---|
| `training/e7_private_ft.json` | c0b3a584 |
| `training/e7_private_ft_rep2.json` | c0b3a584 |
| `training/seq_inversion_06b_rep2.json` | c0b3a584 |
| `v2/privacy/e8_12b.json` | — |
| `v2/privacy/e8_8b.json` | — |
| `v2/privacy/e8_obfuscation_06b_fp32.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b_fp32_anchorcheck.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b_fp32_kvoid.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b_fp32_kvoid_seed123.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b_fp32_kvoid_seed99.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b_fp32_kvoid_sub100_seed42.json` | c0b3a584 |
| `v2/privacy/e8_robustness_06b_fp32_rep2.json` | c0b3a584 |
| `v2/privacy/e8kp_06b.json` | c0b3a584 |
| `v2/privacy/e8kp_7b.json` | c0b3a584 |
| `v2/privacy/output_inv_06b.json` | c0b3a584 |
| `v2/privacy/output_inv_7b.json` | c0b3a584 |
| `v2/privacy/sipit_06b.json` | — |
| `v2/privacy/ti_12b.json` | — |
| `v2/privacy/ti_7b.json` | — |
| `v2/privacy/ti_8b.json` | — |

## `results-h100-2` — 6 artifacts

| artifact | producing commit |
|---|---|
| `v2/privacy/dlgpp_06b.json` | 54ba1d59 |
| `v2/privacy/e8_7b.json` | — |
| `v2/privacy/e8_obfuscation_06b_fp32.json` | 54ba1d59 |
| `v2/privacy/e8_robustness_06b_fp32.json` | 54ba1d59 |
| `v2/privacy/ti_27b.json` | 94bc8421 |
| `v2/privacy/ti_27b_d8.json` | 54ba1d59 |

## `results-h100-3` — 2 artifacts

| artifact | producing commit |
|---|---|
| `v2/privacy/e8_obfuscation_06b_fp32_highest.json` | ac708a8c |
| `v2/privacy/e8_obfuscation_06b_fp32_tf32.json` | c0fed2f1 |

Also `v2/e8_obfuscation_06b_fp32_tf32_and_highest.log`, the run log for both.

**Known defects in these two artifacts.** They are the TF32-vs-true-fp32 pair
and are quoted in `COMPARISON_MATRIX.md` §5b.2. The measurements are sound —
the six deviations reproduce the published h100-1 values bit-identically — but
three things about the files are wrong and cannot be corrected in place,
because the node that produced them is gone and they are kept byte-frozen as
their run emitted them:

- **Their `interpretation` string sends a reader to the wrong arm.** It advises
  "rerun `--dtype fp32` to verify exactness". That flag alone leaves TF32
  matmuls on, which is the confound this pair exists to measure: the tf32 file
  records `dtype="fp32"` next to `matmul_precision_effective="high"` and
  `cuda_matmul_allow_tf32=true`. Reproduce with **both** `--dtype fp32` and
  `--matmul-precision highest`. `e8_obfuscation.py` now emits that wording.
- **Their `provenance.dtraining_commit` is unreachable from every ref.**
  `c0fed2f1` and `ac708a8c` are pre-rebase objects; `ac708a8` is the dropped
  "make attack 3's error bars cover key estimation" commit, superseded by
  `main`'s own `d3_oracle_repeatability`. Both resolve in a checkout that
  already fetched them and in no fresh clone. Treat them as naming the work,
  not as a reproducible tree.
- **They fail `probes/evidence_audit.py --strict`**, lacking
  `config.measurement_kind`, `evidence_status` and `known_limitations`. Both E8
  writers now emit all three, so a regenerated file will pass.

## `results-toula` — 31 artifacts

| artifact | producing commit |
|---|---|
| `v2/ea4/ea4_fedavg_client.json` | — |
| `v2/ea4/ea4_membership_property.json` | — |
| `v2/ea4/ea4_split_ft_train.json` | — |
| `v2/phase-d/tmx_7b_overlap_0ms_rep1.json` | — |
| `v2/phase-d/tmx_7b_overlap_0ms_rep2.json` | — |
| `v2/phase-d/tmx_7b_overlap_0ms_rep3.json` | — |
| `v2/phase-d/tmx_7b_overlap_80ms_rep1.json` | — |
| `v2/phase-d/tmx_7b_overlap_80ms_rep2.json` | — |
| `v2/phase-d/tmx_7b_overlap_80ms_rep3.json` | — |
| `v2/phase-d/tmx_7b_sync_0ms_rep1.json` | — |
| `v2/phase-d/tmx_7b_sync_0ms_rep2.json` | — |
| `v2/phase-d/tmx_7b_sync_0ms_rep3.json` | — |
| `v2/phase-d/tmx_7b_sync_80ms_rep1.json` | — |
| `v2/phase-d/tmx_7b_sync_80ms_rep2.json` | — |
| `v2/phase-d/tmx_7b_sync_80ms_rep3.json` | — |
| `v2/phase-d/tmx_7b_sync_hostile_rep1.json` | — |
| `v2/phase-d/tmx_moe_overlap_80ms_rep1.json` | — |
| `v2/phase-d/tmx_moe_overlap_80ms_rep2.json` | — |
| `v2/phase-d/tmx_moe_overlap_80ms_rep3.json` | — |
| `v2/phase-d/tmx_moe_sync_0ms_rep1.json` | — |
| `v2/phase-d/tmx_moe_sync_0ms_rep2.json` | — |
| `v2/phase-d/tmx_moe_sync_0ms_rep3.json` | — |
| `v2/phase-d/tmx_moe_sync_80ms_rep1.json` | — |
| `v2/phase-d/tmx_moe_sync_80ms_rep2.json` | — |
| `v2/phase-d/tmx_moe_sync_80ms_rep3.json` | — |
| `v2/qwen36-27b/ablation_qwen36-27b_0ms.json` | — |
| `v2/qwen36-27b/ablation_qwen36-27b_80ms.json` | — |
| `v2/qwen36-27b/perplexity_27b_80ms.json` | — |
| `v2/qwen36-27b/rtt_qwen36-27b_0ms.json` | — |
| `v2/qwen36-27b/rtt_qwen36-27b_20ms.json` | — |
| `v2/qwen36-27b/rtt_qwen36-27b_80ms.json` | — |

**Provenance coverage: 41/296.** Artifacts marked `—` carry no
resolvable `dtraining_commit`: either they predate the provenance stamping
added in `9fd2867`, or they predate the container git fix in `9c867d2` and
recorded `"unknown"`. They cannot be traced to a code version.
