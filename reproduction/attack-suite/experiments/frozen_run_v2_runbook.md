# Frozen validation run-book — protocol v2 (defended-capture arms)

Status: **NOT EXECUTED; measurement amendment required.** Review on 2026-09-09
found that the development pilot mixed pre-training transform diagnostics with
post-training held-out utility and compared replica cloud outputs against an
input-seam head. Corrected code retains actual trained replica state, measures
matched utility and captures the participant-visible input seam. Legacy defended
protocols are now blocked before held-out execution. Run a corrected development
pilot, review its results and budgets, then create a new protocol amendment with
`measurement_contract: matched-final-state-v2`. Do not modify frozen v2 in place.
The commands below are historical v2 instructions, not authorization to bypass
that gate. Synthetic smoke remains available and does not consume held-out data.

Matrix: 3,680 logical cells / 3,200 physical executions, of which 960 are the
v2 defended-capture arms (training-mode `forward_k{8,16,32}` and
`sipit_steps{32,64,128}` under `replica-median` and `secret-mixup`, at depths
{3,6}, all frozen victim/attack seeds and knowledge conditions, paired with
the undefended controls). Defended victim training reuses the splitbench
defense wiring (`make_defense_state` / `_apply_seam_defense` via
`splitbench.execute`); every defended cell records the defense provenance
sidecar path+hash, held-out utility before/after, the attacker-access spec of
its paired control, and runtime measurements, and a defended cell missing any
of these is FAILED in the report — never dropped — which blocks analysis
(the analyzer fails closed).

## Host and duration

Target host: the GB10 GPU box (the pilot host). Duration estimate, stated
with assumptions — these are planning figures, not bounds:

- Frozen budget review (`experiments/review_validation_budget.py`, recorded
  in the v2 amendment): **229.4 GB10 GPU-hours nominal online attack time**,
  286.8–344.2 GPU-hours with the prespecified 25–50% planning contingency.
  Excludes victim training, model loading, calibration, capture, scoring,
  I/O and analysis; assumes the runner's sequential per-depth execution and
  the measured pilot per-arm timings interpolated to K=16 / SIPIT=64.
- Defended-arm sanity check from the defended pilot
  (`results/pilot-defended-v1`): attacks 47–286 s/cell × 960 defended cells
  ≈ 12.5–76.3 GPU-h; defended victim training 5–15 s per run × 40 defended
  runs (2 defenses × 2 depths × 10 seeds, 128 steps each) ≈ negligible.
  Consistent with the budget review's +123.0 GPU-h defended delta.

## Pre-run checklist

1. **Frozen hash match.** `shasum -a 256 experiments/validation_protocol.v2.json`
   must equal `75365cae1a279a6323a9a5151b0fe7e07966f19401b7f4ef7c7d2a001f45966f`
   (the file frozen by the amendment; retain this digest externally). The
   pinned input must hash to the protocol's
   `frozen_inputs.tokenized_sha256`
   (`d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a`) —
   the runner refuses anything else.
2. **Clean journal.** `results/heldout_access.journal` must not exist (or be
   empty). Any entry means held-out outcomes were read before this run; stop.
3. **Fresh output dir.** The runner refuses a pre-existing `--output`.
4. **Provenance capture.** Record host, GPU, driver, `torch.__version__`,
   `torch.version.cuda`, git commit and dirty state alongside the run; the
   report embeds torch/CUDA/device runtime itself.
5. Smoke-verify the exact runner build on the host first:
   `.venv/bin/python -m pytest tests/test_validate_reconstruction_defended.py -q`.

## Run command

```bash
.venv/bin/python experiments/validate_reconstruction.py \
  --protocol experiments/validation_protocol.v2.json \
  --input results/validation-corpus-v1/validation-input.json \
  --input-sha256 d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a \
  --device cuda --output results/validation-heldout-v2
```

The first held-out outcome read is journaled to
`results/heldout_access.journal`; re-freezing the protocol is refused
afterwards by design.

## Two-host sharded execution (toula + koula)

The cell matrix can be partitioned across two GPU hosts **without amending
the protocol**: same frozen cells, partitioned execution, merged afterward.
`--shard-count 2 --shard-index i` assigns each physical execution root by
interleaving the sorted cell ids (sorted position j goes to shard j % 2), so
both hosts span all depths, seeds, knowledge conditions and arms — depth is
never confounded with host (the development pilot had that confound; do not
reintroduce it by splitting on depth). Reused inference cells are co-assigned
with their training-mode origin, so no shard references a measurement it did
not run. The partition is a pure function of the frozen protocol: both hosts
reproduce it with no communication. `--shard-count 1` (the default) is the
unchanged unsharded run.

Shard 0 on toula:

```bash
.venv/bin/python experiments/validate_reconstruction.py \
  --protocol experiments/validation_protocol.v2.json \
  --input results/validation-corpus-v1/validation-input.json \
  --input-sha256 d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a \
  --device cuda --shard-count 2 --shard-index 0 \
  --output results/validation-heldout-v2-shard0
```

Shard 1 on koula:

```bash
.venv/bin/python experiments/validate_reconstruction.py \
  --protocol experiments/validation_protocol.v2.json \
  --input results/validation-corpus-v1/validation-input.json \
  --input-sha256 d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a \
  --device cuda --shard-count 2 --shard-index 1 \
  --output results/validation-heldout-v2-shard1
```

Each host runs the identical pinned input, protocol hash and code build, and
the pre-run checklist above applies per host. Victim training, head export,
calibration and capture setup are duplicated on both hosts (small against the
229.4 GPU-h online attack budget, which is what sharding halves); the attack
cells themselves are disjoint. ETA: ~229 GPU-h total online attack time →
roughly **4.5–5 days wall-clock on two GB10s** (plus the prespecified 25–50%
planning contingency), versus ~9.5 days on one.

After both shards report COMPLETE, copy both shard directories to one host
and merge (pure stdlib, no GPU):

```bash
.venv/bin/python experiments/merge_validation_shards.py \
  results/validation-heldout-v2-shard0 results/validation-heldout-v2-shard1 \
  --output results/validation-heldout-v2
```

The merger fails closed unless: both shard evidence chains verify; both
reports are COMPLETE with complementary shard indices covering [0,2); the
protocol-source bytes, protocol/effective-protocol/input hashes and supplied
input binding are identical across shards; each shard's cells exactly equal
its deterministic assignment (so the union is disjoint and equals the full
frozen 3,680-cell matrix — no missing, no duplicates); and shared artifact
files hash identically (a cross-host `.pt` hash conflict aborts the merge —
investigate the builds, do not force it). The merged directory is a single
analyzer-ready report with its own fresh evidence chain recording both source
manifest hashes; analyze it with the unchanged post-run command below
(`--protocol-version 2`).

## No-retries policy (unchanged, documented not configured)

Any failed or incomplete cell makes the whole run unusable for comparison.
There are **no selective retries and no outcome-based re-execution**: the
only remedy is a complete new run of the identical frozen protocol in a
fresh output directory. Defended cells that cannot produce provenance,
utility, the access spec or runtime records are written to the report as
FAILED (never dropped), which sets the report status to FAILED and blocks
the analyzer — this is the fail-closed contract, not a retry hook. Under
sharded execution the policy is unchanged and applies to the shard set as a
whole: any FAILED or incomplete shard makes the merged run unusable, and the
only remedy is a complete new run of every shard in fresh output directories
(no re-execution of just the failed shard).

## Memory risk posture (v3 run, accepted 2026-09-10)

The runner has no preventive memory protection: replica-median's K-body
memory multiplier warns only (never downsizes — a silent config change would
contaminate the study), and defended-cell OOMs fail closed per the policy
above. For the v3 run the risk was assessed and accepted: GB10 hosts have
121 GB unified memory, pilot peak was ~2.5 GB per body (K=3 ≈ 7–10 GB
expected), and each host runs exactly one shard process (no contention).
Residual risk: fragmentation or an unanticipated defended-cell spike, costing
a full shard-set rerun. Future hardening (not applied to v3): NVML preflight
plus a per-cell memory watchdog that fails the cell cleanly.

## Post-run analysis

```bash
.venv/bin/python experiments/analyze_validation.py \
  --root results/validation-heldout-v2 \
  --output results/validation-heldout-v2/analysis \
  --protocol-version 2 \
  --expected-manifest-sha256 EXTERNALLY_RETAINED_MANIFEST_SHA256
```

`--protocol-version 2` pins the analysis to the v2 allow-list. The analyzer
verifies the evidence chain, protocol/input/artifact hashes, the complete
3,680-cell matrix, independently rescored predictions and matched prefix
work; any missing or FAILED cell aborts analysis. The analysis report must
state, alongside every contrast, the count of failures and nulls (a
COMPLETE report has zero by construction — say so explicitly), the defended
arms' per-run held-out utility (`defended_victims` entries and per-cell
`defense_utility`), and the physical vs logical execution counts. Defended
arms are a prespecified paired descriptive comparison against the
undefended controls; the v1 inferential contrast family (primary + 47
Bonferroni-corrected secondaries) is unchanged. Export and retain the final
evidence-manifest SHA256 outside the result directory.
