# Five-node evidence recovery — 17 September 2026

**The external P0–P5 programme has been located and its small records recovered. Its historical results were explicitly retired by their owning project on 10 September.** Recovering matching bytes does not reinstate those claims. No experiments or uploads were launched by this audit.

## Located records and provenance

| Node | Located project records | Finding |
|---|---|---|
| toula | `/home/geo/r1-analysis-bc5242e` | Historical P1–P5 decisions, protocols, manifests, raw per-run loss curves, privacy qualification/analysis; no `.git`, so directory suffix is not a verified source commit |
| voula | `/home/geo/paper3-r1` | Git origin `geopolitis/paper-3-privacy-constrained-remote`, HEAD `525de9c3037abd6235af83a1add1480453107ae0`; current ledger and reset policy |
| toula, voula | `r1-bundle-sci-20260916/outputs/p3-r1-fresh-start/calibration--cal001` (toula); `/home/geo/r1-replicas/calibration--cal001` (voula) | Fresh R1 calibration, 76/76 complete; attack-spec proposal awaits freeze review |
| koula | `/home/geo/r1-bundle-sci-20260916/outputs/p3-r1-fresh-start/exploratory--exp001` | Fresh exploratory execution stopped with `KeyboardInterrupt`, nine completed runs, 16 September 18:45:48 UTC |
| sasa | `/home/geo/r1-replica/r1-bundle-sci-20260916` | Scientific bundle replica; no completed confirmation analysis located |
| soula | `/home/geo` and conventional mounted-data roots inspected | No matching P0–P5 project/result archive or SensiReplaced file found in this bounded search |

Read-only discovery covered `/home/geo`, `/mnt`, `/media`, `/data`, `/scratch`, `/srv`, pruning environments/caches and inspecting project-pattern candidates. No separate data filesystem appeared in `findmnt` for the inspected filesystem types. This is not an assertion about inaccessible paths, arbitrary differently named files, other user accounts, or unmounted media.

The local directory `results/external-p0p5-recovery-20260917/` contains 5,167 historical metadata/results files (184,223,900 bytes), plus twelve current R1 records and selected retired source files. Original bytes are preserved. `origin-index.json`, `current-r1-origin-index.json`, and `retired-source-origin-index.json` bind remote paths, Git identities where available, local paths, sizes and SHA-256 hashes. No tensors were downloaded. The historical release index records 4,492 manifest artifact references totalling 52,139,663,688 bytes. This audit verifies **3,720 recovered references with zero mismatches**; **772 references were not downloaded**, chiefly tensors/corpus data. The inspected toula analysis tree contains no `transcript_tensors.pt` files. Named compressed P2/P5 archive parts were not found in the five-node search.

The owning project's `evidence/R1_EVIDENCE_POLICY.json` was introduced at `3707df2d03ab9939f06107213403b88411a3bc5f` on 10 September and explicitly forbids historical outcomes, qualification and checkpoints as current scientific evidence. The preserved historical tag is `paper3-pre-reset-2026-09-10`, commit `a862baa559f5fbcef427997af2bf9ea170b5fa30`. All 768 old manifests cite scaffold commit `be32144570e6fdb6cbdbdf367c9e71ffa685d24a`, which lacks the executed implementation; the nine dirty-tree hashes do not recover the missing execution sources. The reset does not establish numerical falsehood. It does establish that the owner rejected these records for active claims.

## P2 raw paired losses and arithmetic

Reproduce with `.venv/bin/python experiments/audit_external_p2_20260917.py` (the script itself needs only Python's standard library). Results: `p2-paired-statistics-audit.json`; artifact checks: `manifest-artifact-verification.json`. The t-tail numerical implementation was independently checked against SciPy in the existing local venv, agreeing to better than 1e-15 absolute probability.

The script loads final held-out losses from the eight trained/frozen `metrics_summary.json` pairs, verifies equal pairing-block IDs, equal initial remote hashes, unchanged frozen remote hashes, and identical step-zero losses. It reproduces the archived final gains exactly. These checks establish consistency of saved records, not the missing historical execution provenance.

| Seed | Trained loss | Frozen-random loss | Gain |
|---:|---:|---:|---:|
| 11 | 1.676115531474 | 1.693942293525 | 0.017826762050 |
| 22 | 1.683799346909 | 1.702302342281 | 0.018502995372 |
| 33 | 1.673694428056 | 1.689275166020 | 0.015580737963 |
| 44 | 1.679714946076 | 1.694532737136 | 0.014817791060 |
| 55 | 1.673512235284 | 1.693657446653 | 0.020145211369 |
| 66 | 1.678916661069 | 1.697376705706 | 0.018460044637 |
| 77 | 1.680295620114 | 1.698659338057 | 0.018363717943 |
| 88 | 1.672721242532 | 1.682638792321 | 0.009917549789 |

Mean = **0.016701851273**, sample SD = **0.003230494044**, two-sided 95% t interval = **[0.014001090665, 0.019402611881]**.

| Null centring | t(7) | One-sided t p | Exact Wilcoxon W+ | Exact one-sided p |
|---|---:|---:|---:|---:|
| Zero | 14.623140774 | 8.354585995e-7 | 36 | 0.00390625 |
| Practical margin 0.01 | 5.867739630 | 0.000309678658 | 35 | 0.0078125 |

Therefore the requested `t=5.92, p≈2.9e-4` does **not** reproduce from the raw pairs; it is close to a calculation from rounded summaries. W+=36 is zero-centred only: seed 88 falls slightly below 0.01. The old decision prose's p≈3e-7 also differs from the generated zero-centred result. No manuscript has been altered by this audit.

## P0 and privacy qualification

No `PREREGISTRATION_P0.md` was found. P0's scope is documented in `RESEARCH_PLAN.md` section 3 and retired `tests/test_p0.py`; the P1 decision says thirty P0 tests passed before launch. These are engineering checks for parameter updates, pairing, FLOPs, schema, quantisation and seed separation. They are **not thirty sequence-property-v2/structure-semantic-v2 qualification experiments**. No independently anchored original pre-pilot thirty-test execution report was recovered. Later test source or logs cannot establish that timing.

The actual v2 property qualification is **P4b**, documented in `PREREGISTRATION_P4B.md` and `outputs/privacy_extended_p4b/privacy_extended_p4b.json`:

- `sequence_property_v2`: vowel fraction, punctuation fraction, newline count; ridge regression with lambda 1.0 from mean-pooled quantised latents. Advantage is **Pearson r minus shuffled-null r**, not R² difference. Only punctuation passes both the positive-control floor and null band: r-advantage 0.714848998, null-pipeline advantage 0.002865724. Vowel/newline null checks fail. Family-level qualification uses an at-least-one-target rule; it must not imply every target qualifies.
- `structure_semantic_v2`: next-byte classes newline/CR, space/tab, letter, digit, and punctuation/other; two-layer MLP, 200 epochs, learning rate .003. Accuracy advantage 0.293111165, null-pipeline advantage −0.013956706; qualified under floor .02 and null band .02.
- Canary membership and history accumulation v2 are unqualified. The P5 gradient qualification is separately preserved in `outputs/delegation/gradient_qualification.json`.

The later `outputs/revision_validation_20260910/p5_property_refit.json` reports punctuation-only checks separately and says qualification was reused without refitting. It remains retired, post-hoc verification. The historical source file recovered from tag a862baa is an inspectable later snapshot, **not** the lost execution source.

## Current R1 does not yet replace the old claims

The 76-cell calibration report qualifies only `input_byte` on inbound/outbound gradients under both defences; property and next-byte-class combinations are unqualified. Its status is `proposal_ready_for_freeze_review`. The exploratory run completed nine cells and stopped. No completed confirmation analysis was found. Current `CLAIM_EVIDENCE_LEDGER.md` admits no empirical scientific claims. Thus the later work does not supply a fresh confirmed substitute for the old P2 or P5 manuscript statements.

## SensiReplaced search

Only `koula:/home/geo/splitllm-ref/data/sfl/datasets/sensi/sensi.csv` was found. SHA-256: `5da54ac04f01937061a7aa5f5a55aec3d4f28b71375d00f6f56815667637183d`. All **23,665** rows have `sani_gpt4 == content`, confirming the documented raw CNN/DailyMail substitute. No original SensiReplaced version was identified across the five nodes. BiSR B1 remains NOT_ESTABLISHED.

## Candidate release scope

For **Paper A**, the recovered P0/P1 engineering records, P2 paired losses, P2b/P2c/P3 analyses, P4a/P4b qualification and P5 delegation records are the external evidence family requested by `docs/open-commitments-plan-2026-09-16.md`. They may be released as a clearly retired historical/audit supplement, with the reset policy and missing-source disclosure inseparable from their results. They cannot currently certify active scientific claims. For **Paper B**, this search adds no valid BiSR fidelity evidence; its relevant addition is the negative original-dataset search and existing B1 audit.

The archived release index points to the owning project's private GitHub evidence release `geopolitis/paper-3-privacy-constrained-remote`, tag `evidence-2026-09-10-v1`, with per-part hashes for future bounded retrieval. Remote availability of those release assets was not verified here. Keep any prospective HF supplement separate from the active upload snapshot. Resolve current manuscript claims against the owner's retirement policy before publishing the historical numbers as results.
