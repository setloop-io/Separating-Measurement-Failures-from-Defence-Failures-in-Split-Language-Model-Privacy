# Active claim–evidence ledger — R1 reset

Effective 2026-09-10. **No empirical scientific claim is currently admitted.**
This ledger replaces the retired manuscript's claim ledger. Its complete prior
version remains at tag `paper3-pre-reset-2026-09-10` / commit
`a862baa559f5fbcef427997af2bf9ea170b5fa30`.

## Admission policy

All pre-reset observations, derived analyses, and post-hoc verification outputs
are inadmissible for active claims. They remain preserved for audit. A successful
recalculation from an old capture does not recover its historical execution source.
The governing policy is [R1_EVIDENCE_POLICY.json](../evidence/R1_EVIDENCE_POLICY.json).

| Candidate claim | Current status | Required new evidence |
|---|---|---|
| Trained remote learning improves utility over freezing | Untested in R1 | Fresh matched seeds, exact initialization/control checks, verified runs, retained paired contrasts |
| Remote learning improves on a matched bypass | Untested in R1 | Same boundary width/precision/defense and trusted initialization, separate removal contrast |
| Precision, noise, capacity, or segment placement changes system behavior | Untested in R1 | Declared controlled factors, all executed cells and failures, retained continuous response estimates |
| At least one configuration meets utility, qualified privacy, and causal-gain requirements | Undetermined | The same cell must satisfy all requirements with complete required fresh qualification and scoring; no missing coverage treated as a pass |
| No candidate in the finite tested set meets the joint requirements | Undetermined | Every planned relevant cell accounted for, sufficient verified observations, and an established failing necessary condition for each rejected cell; incomplete cells leave the set unresolved |
| A universal trilemma holds | Not an empirical claim this design can establish | Explicit theoretical assumptions and a suitable lower bound beyond this finite sweep |

Engineering and performance-calibration outputs are non-scientific. They may
establish that files and control checks work and estimate compute/storage demand;
they cannot establish any row above. Exploration cannot be promoted into held-out
confirmation after its outcomes have been inspected.

An accepted result needs a stage/cell/seed/attempt identifier, the pre-execution
packet hash, source and environment identities, corpus/split hashes, per-run
manifest and raw artifact hashes, attack qualification dependency hashes, and
analysis code/input/output hashes. A ledger row is updated only when that complete
chain is available and the stated scope is justified.

Omitted channels and unqualified targets remain explicitly named in any future
privacy result, including captions and summaries. The current stage makes no
privacy result; its required coverage and remaining admission work are listed in
the [R1 protocol](PREREGISTRATION_R1.md). Real two-process CPU validation, draft
attack execution and semantic artifact checks remain engineering evidence and do
not change any row above.
