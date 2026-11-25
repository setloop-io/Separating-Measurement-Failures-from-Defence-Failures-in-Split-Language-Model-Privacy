# Prospective corrected P6 follow-up — 17 September 2026

Authorisation: the user's instruction to continue the outstanding Phase 2 work,
after disclosure of the historical driver deviations. The old P6 protocol,
driver and 192 pilot outcomes remain immutable. The new v2 protocol is a
prospective follow-up specification, not a retroactive claim that the pilot
satisfied preregistration. The original P5/Mode-1 claim remains outside scope.

## Fixed design

GPT-2 at revision 607a30d783dfa663caf39e06633721c8d4cfcd7e; 192 segment-trained
cells: spans 2/4/6, starts 0/1/2/3, D0/D1 and eight original victim seeds.
Use the hash-pinned v3 token file: 32 calibration, 128 training and 256
held-out evaluation documents. Train for 128 steps on a seed-shuffled pass of
training documents (32 tokens, batch 1, AdamW 5e-5). The 128-step choice follows
the original feasibility command template and is explicit here; the old
pilot used 100 steps. No pilot outcomes select cells or sample sizes.

Eight all_local runs train the full model without a wire defence. Their
before-training utility is the frozen pretrained control for that seed;
their after-training utility is the paired all_local comparator. These use
seam 1: without a wire defence or restricted trainable range it is the same
full-model computation. All three utility losses are measured on the same
256 held-out documents. Delta_U is segment-trained minus all_local loss.
The seed is the training replication unit; 256 documents are not 256 victims.

Privacy observations come from the final trained state on held-out documents,
with an eval-mode forward/backward pass and no optimiser update. Save forward
wire and post-defence backward wire, canonical evaluation tokens, document
identities and losses. This final-state observation differs from the old
training-trajectory pilot and is explicitly part of v2. Measurement preserves
model weights, training flags and RNG state. Save changed victim parameters
alongside the pinned base revision for replay.

The attacks are VMI with the public base head and the exact 32 public
calibration documents, plus gradient-norm label-scoring. Property tokens
286/357/764 are selected using only calibration prevalence: the three tokens
closest to half of the 32 documents, with token-ID tie-breaks. No evaluation
outcome selects properties. VMI uses a single fixed attack seed 17; per-run
gate output must not advertise three independent victim seeds. All_local
attacks supply linked undefended reference/positive-control evidence.

## Analysis fixed before new outcomes

Report utility, VMI accuracy/control and all three property AUCs separately.
For a named *normalised leakage index* use the maximum of zero, VMI recovery
minus its label-free control, and twice the absolute property-AUC deviation
from 0.5. This index is not a P5 privacy probability or the P5 0.02 gate.
It makes score reversal explicit and must not be substituted into Paper A's
three-gate claim.

For each defence, compare the span-only categorical fit with the full
span-by-start cell fit, on eight paired victim runs. Report partial R-squared,
10,000 seed-bootstrap draws, and a paired permutation test which shuffles
start labels within each seed/span (10,000 permutations). Require partial
R-squared >=0.10 and permutation p <0.05/4 across the two stochastic endpoints
and two defences. The bootstrap of a nonnegative R-squared is not, by itself,
a valid zero-effect significance test; this corrects an ambiguity in v1.
Analytic delegation span/12 has no start effect by construction and receives
no inferential test. Report the original 6-seed fit / 2-seed held-out ridge
comparison as a descriptive secondary result; no re-splitting or refitting.

Report crossed victim/document bootstrap intervals for paired utility and
VMI excess, and retain all per-document norm scores and property labels for
AUC re-analysis. The primary variance test conditions on the fixed corpus;
it does not estimate population-wide corpus or architecture variation.
No token-iid Wilson interval is treated as a document-clustered result.

## Execution and stop conditions

Freeze implementation hashes and protocol before real-model execution.
Run fresh infrastructure canaries, excluded from the 192-cell matrix. Prefer
a single GPU with a global shuffled matrix to avoid confounding span with
hardware. A CPU canary provides a fallback timing comparison. Launch the
full GPT-2 grid only after integrity checks and a projected GPU budget no
more than 20 hours; otherwise report the measured projection and retain the
prepared jobs. No new paid infrastructure is required or provisioned.
Qwen3 replication remains gated on GPT-2 completion and remaining budget.

Every cell seals evidence and pins all captured files. Resume checks every
completed artifact/hash; malformed journals fail closed. A failed cell is
retained and blocks comparison; repairs use fresh directories and an explicit
attempt reconciliation. Missing data or missing controls prevent analysis.
Source changes after a freeze require a new prospective record, never silent
reuse of the same frozen identifier. No new paper conclusions until the full
matrix, analysis, evidence checks and required tests have completed.
