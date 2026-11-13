# Paper 4 — Defense Redesign Under Calibrated Privacy Instruments

Working title: **Can Structure-Aware Boundary Defenses Preserve Useful Remote Learning?**

Status: **Exploratory pilot complete (2026-09-22): no cell in the frozen tested
family is pilot-eligible.** The frozen 66-run matrix (`pilot-20260922-a`,
protocol [p4-defense-pilot-1.0](docs/PILOT_FREEZE.md)) executed on toula/koula
with qualified instruments on the pinned `gutenberg_p4` corpus; utility passed
d0/D1/d3-q3, the causal remote-gain gate passed every cell, and the privacy
gate failed or remained undetermined on every cell (punctuation advantage
0.79–0.89, next-byte structure 0.097–0.112 against the 0.02 acceptance
threshold). See the [pilot decision record](docs/PILOT_DECISION.md), the
[claim-evidence ledger](docs/CLAIM_EVIDENCE_LEDGER.md), and the
[freeze record](experiments/freeze/p4-defense-pilot-1.0.json). This is a finite
empirical statement about the completed tested cells — not a universal
impossibility claim and not a statement about untested mechanisms.

Start with the [source audit](outputs/deep_research_audit.md),
[implemented architecture](docs/ARCHITECTURE.md),
[P0 execution protocol](docs/P0_ENGINEERING_PROTOCOL.md), and
[environment instructions](docs/ENVIRONMENT.md).

The new [Bifrost source check](TEE-FTE.md) motivates a separate
[TEE–FHE deployment proposal](docs/TEE_FHE_TRACK.md), with its own trust model and
measurement plan. It is outside the current D1/D3 pilot.

Completed on 2026-09-10: **79 tests passed; all 10 P0 engineering runs completed**.
The [verified engineering report](outputs/runs/p0-20260910-a/report.md) links source,
manifest, and capture checks. These three-step diagnostics leave privacy undetermined.

The next implementation revision (`a8dff21`) passed **110 tests** and completed
[36 calibration scenarios](outputs/scenarios/controls-20260910-a/report.md) and
[six plaintext routing scenarios](outputs/scenarios/routing-20260910-a/report.md).
All 72 injected positive-dose checks exceeded their floors; 17 of 36 null checks
failed the proposed band, so scientific qualification remains pending. Routing
preserved the reference logits exactly in this fixture; it implements no TEE/FHE.

The shared-warmup revision (`ba8b23d`) passed **122 tests** and completed
[22 warmup/stability records](outputs/scenarios/warmup-20260910-a/report.md): two
shared setups and twenty branches. Larger panels reduced null-score variability,
but 171/288 diagnostic null checks still failed at 1,024 sequences. The warmed
basis tracked the later direction more closely than the initial basis, with
remaining drift. See the [engineering assessment and next steps](docs/WARMUP_STABILITY_DECISION.md).
These measurements leave scientific qualification and privacy undetermined.

The next revision (`e29fd63`) passed **137 tests** and completed
[six matched-null precision scenarios](outputs/scenarios/precision-20260910-a/report.md).
All positive-signal scores exceeded their floor and all quantizer-support checks
passed. At 1,024 sequences, averaging 64 matched nulls still left 85/192 no-signal
trials outside ±0.02. The [assessment](docs/NULL_PRECISION_DECISION.md) identifies
larger independent text panels as the next prerequisite; no instrument is qualified.

## Research question

Can specified boundary defenses reduce measured attack advantage while preserving utility
and causal remote-learning gain at the same operating point? A negative result concerns only
the tested mechanisms, configurations, training budgets, and qualified attacks. Paper 3 found
an empty empirical feasible set in its tested family; it did not establish universal
impossibility or that leaked properties are necessary for useful computation.

## Initial scope

- **D1:** property-targeted Gaussian noise along calibrated orthonormal directions, shared
  across positions within a sequence and applied before final forward quantization.
- **D3:** lower forward latent precision, with an otherwise matched reference configuration.
- **D1+D3:** optional controlled exploratory interaction, requiring both individual controls.
- **D2:** adversarial encoder training is a subsequent, separately specified extension.
- **D4:** dimension-selective quantization is deferred.
- **D5:** sequence permutation is an invariance control, not a privacy candidate: it preserves
  the mean-pooled representation used by the property probe.

Every candidate is anchored to span 4 at position 2, latent width 256, and the q=8 reference.
D3 varies q at that same anchor. Paper 4 uses fresh trained, frozen-random, removed-remote,
and all-local controls. It does not rerun Paper 3's historical experiment matrices.

## Evidence and gates

The governing prior is Paper 3's final [claim-evidence ledger](../Paper%203%20-%20Privacy-Constrained%20Remote/docs/CLAIM_EVIDENCE_LEDGER.md)
at commit `52191c430110bd8d9dee80538fde22bf1726e56a`; its manuscript clarifications supersede
broader historical decision prose. See [prior evidence](docs/PRIOR_EVIDENCE.md) for scope.
Punctuation is the primary sequence-property target; vowel/newline targets remain exploratory
until separately qualified. Next-byte structure describes byte classes, not unrestricted
semantic recovery. Earlier token-probe nondetection does not establish protection of tokens.

For a single eligible confirmatory cell, the proposed gates are mean replacement penalty
Δ_U ≤ 0.10 nats and its upper 95% confidence limit ≤ 0.12; mean remote gain G_R ≥ 0.01 nats
with a prespecified multiplicity-corrected paired test; and worst-seed advantage ≤ 0.02 for
every required qualified attack/target/channel combination. The property positive-control
floor of 0.10 qualifies an instrument; it is **not** the privacy acceptance threshold.

An activation-only audit cannot establish defense success. Missing or unqualified channels
remain undetermined. A positive defense claim requires the declared complete-transcript
evaluation and qualified active attacks. Stronger passive probes, auxiliary-data attacks,
and history analysis do not become active attacks merely by improving the attacker.

## Reviewer evidence

The repository remains private. Reviewers with access can obtain every locally
retained result, capture, checkpoint, panel, source archive, and the exact corpus
from the [versioned evidence release](https://github.com/geopolitis/paper-4-defense-redesign/releases/tag/evidence-2026-09-10-v1).
The [reviewer guide](docs/REVIEWER_EVIDENCE.md) gives authenticated download,
checksum verification, and reproduction commands. All 80 execution records remain
engineering diagnostics, with `scientific_evidence: false`.

## Reuse and reproducibility

The existing `pcr` implementation is reused from the pinned Paper 3 revision, with Paper 4
extensions isolated in this repository. Paper 3 source, protocols, and results remain
unchanged. See [inherited assets](docs/INHERITED_ASSETS.md) for integration checks and limits.
Do not install an unpinned editable sibling dependency and assume it reproduces the protocol.

Paper 4 uses per-token quantizer statistics along the latent dimension. Its quantization
does not aggregate future positions into a current token's min/max. This changes the
mechanism relative to Paper 3's per-tensor quantizer, so fresh Paper 4 D0 controls are
required; equal q does not make the two mechanisms identical or their results poolable.

The isolated environment is locked with `uv.lock` and contains unmodified vendored `pcr`
source and its license from the pinned revision. Preparation and diagnostic commands are:

```bash
uv sync --locked --extra dev
uv run python scripts/run_experiments.py --config experiments/configs/p0_diagnostics.yaml
```

Only the explicitly labeled engineering diagnostic configuration may run before the pilot
freeze. Use the runner's `--validate-only` option for draft pilot configurations. A P0
budget of three steps with batch size two and sequence length 32 checks execution; it
cannot estimate the planned pilot's utility, privacy, or convergence.

The Paper 4 runner reserves disjoint corpus regions: 80% training, 10% calibration,
5% probe fitting, and 5% evaluation. This differs from Paper 3's historical 90/10 split
and is another reason to use fresh controls. The validated draft pilot has 66 planned
runs; validation and engineering checks do not authorize its execution before freeze.

The manuscript and implementation status are recorded separately from scientific claims in
[the ledger](docs/CLAIM_EVIDENCE_LEDGER.md). `Deep-research.md` is retained as supplied
background; its theoretical claims and predicted defense outcomes are not evidence.

## Repository layout

- `docs/`: research plan, draft protocol, provenance, and claim ledger.
- `src/p4d/`: defense and evaluation extensions to the pinned `pcr` dependency.
- `experiments/configs/`: diagnostic and subsequently frozen study configurations.
- `experiments/manifests/`: execution records with cohort, source, and artifact hashes;
  a manifest is not an instrument qualification verdict.
- `scripts/`, `tests/`: execution, analysis, and implementation checks.
- `outputs/`, `analysis/`: generated artifacts and analysis, with explicit cohort labels.
- `paper/`: manuscript scaffold; prospective studies must not appear as completed results.

## Where the program stands (2026-09-22)

Executed and verified, in order: the 10 P0 engineering runs; 36 calibration and
6 routing scenario diagnostics; 22 warmup/stability records; 6 matched-null
precision scenarios; the [`gutenberg_p4` calibration text resource](docs/CALIBRATION_TEXT_PROTOCOL.md)
(75 public-domain documents, author-disjoint regions, zero sampled overlap with
the historical corpus); the [warmup-at-budget](docs/WARMUP_BUDGET_DECISION.md)
campaign (16/16, basis drift and null stability at the pilot point); the
[matched-null precision at pilot scale](docs/NULL_PRECISION_128_DECISION.md)
campaign (8/8, no-signal precision met at 16,384 windows); the
[structure precision](docs/STRUCTURE_PRECISION_DECISION.md) campaign (6/6,
exact nulls, injected-code sensitivity 0.254); the
[budget response and warmup policy](docs/BUDGET_SCAN_DECISION.md) checks
(12/12, plus no-warmup gain checks); the calibration-cohort
[instrument qualification](docs/QUALIFICATION_DECISION.md) (32/32, first
scientific cohort); and the frozen exploratory
[pilot](docs/PILOT_DECISION.md) (66/66, exploratory cohort, no eligible cell).

The 323-test suite passes on the Mac (CPU), toula/koula (CUDA), and
soula/voula/sasa (CPU). Every campaign's manifests re-verify; large captures
and checkpoints remain hash-bound on the executing hosts. Independent CPU-node
verification covers manifest schemas, freeze digests, seed audits, and small
artifact hashes.

## Follow-ups (each requires its own prospective protocol)

D2 adversarial encoder training; D1+D3 interaction cells; higher-rank or
multi-target D1 variants; active-attacker evaluation (required before any
positive defense claim); the complete declared-transcript evaluation;
replication beyond this architecture/corpus; a confirmatory freeze with fresh
seeds only if a future *eligible* cell exists (none does today). The frozen
pilot may not be re-executed without a new freeze record: post-freeze source
edits correctly refuse.
