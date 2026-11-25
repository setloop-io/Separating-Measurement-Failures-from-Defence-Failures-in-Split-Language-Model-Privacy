# BiSR upstream repair and frozen inputs

The two preflight blockers are resolved under the prospective
[v2 amendment](bisr-conformance-amendment-v2-2026-09-17.md). This is **not** a
full fidelity pass: the three-seed matched real-data comparison remains pending.
Frozen v4 and the failed earlier diagnostic remain unchanged.

## Implementation result

Commit `bf6e50c` introduces the `splitllm-c376590` algorithm profile and makes the
separate prospective prediction runner use it. The legacy attack command retains
its previous equations, explicitly labelled `legacy-paper-variant` in its BRE
diagnostics. Do not use that command as the upstream-conformance path.

The repaired profile matches upstream GM initialisation, soft-label cross entropy,
squared-L2-plus-L1 objective, AdamW epsilon/decay, SM loss reduction and dot-product
decoding. It preserves SIP initialisation and GM-then-SM ordering. The comparison
executes pristine upstream methods from the pinned commit, not another copy of
the candidate equations.

The CPU FP32 one-step diagnostic passed all eight recorded checks: GM and SM
initial variables, gradients and updates, plus decoded tokens. All six tensor
maximum absolute differences were **0.0**. The earlier decoding counterexample
now agrees as well. This checks shared public models on a synthetic fixture;
it does not certify all remaining real-model, exact-grant or statistical gates.

The existing and new regression tests passed on koula: **51 passed, no skips**,
22.91 seconds. Package deprecation warnings did not affect the result.

## Frozen real inputs

The committed `experiments/bisr_frozen_inputs.v2.json` binds the source files,
exclusion inputs, tokenizer files, ordered validation identities and all selected
documents/windows. Its SHA-256 is
`727d79a0f9c474113b21e86afa3d751c1351f8886c4735c4f669d74de7e01dfd`.

| Input | Frozen count |
| --- | ---: |
| Source test articles | 62 |
| Eligible validation articles | 61 |
| Test articles excluded as shorter than 32 tokens | 1 |
| Selected auxiliary articles | 112 |
| Complete auxiliary windows, 32 tokens each | 10,918 |
| Auxiliary articles with zero complete windows | 13 |
| Selected auxiliary articles used historically | 112 |

All historical auxiliary reuse is disclosed, as required by the amendment. No
claim of fresh auxiliary text is made; the planned GRU initialisations and runs
are fresh. Validation text has no exact normalised-document overlap with the
selected auxiliary text. Historical B1 training sources, Sensi substitute and
auxiliary files, v3 IDs and pilot inputs were checked or conservatively excluded
as specified by the amendment. This does not certify absence from pretraining.

All output hashes, document counts, 32-token window lengths/masks and
validation/auxiliary separation were independently checked after copying the
files back from koula. `compare_frozen_scores` requires the frozen 61 identities;
it cannot silently substitute the number of successfully returned score rows.
The three seeds and ±0.01 equivalence margin are unchanged.

## Evidence locations and remaining work

Evidence is retained locally under `results/bisr-upstream-repair-20260917/`
and remotely under `koula:/tmp/bisr-upstream-repair-20260917/results/`.
`frozen-inputs/` holds the selected text, windows, scorer truth and manifest.
`mechanical.json`, `mechanical.pt` and `tests.log` record the mechanical checks.
The mechanical tensor bundle SHA-256 is
`ecd08ca647de8d5fdb97879648787c2d04da47963fc8d6b47e67af2900a55139`.
Raw generated evidence remains outside git; the frozen input manifest is tracked.
No evidence upload or GPU campaign was performed in this repair.

Next prepare and pin the fresh victim/SIP checkpoints and capture schedule,
complete real-model replay and the remaining conformance/resource checks, then
commit the complete launch manifest and run the three-seed matched comparison.
Input selection now succeeds, but its explicit status remains
`INPUTS_FROZEN_CHECKPOINTS_NOT_ADMITTED`. Original SensiReplaced and dual-cut
published-setting reproduction remain separate unresolved requirements.
