# Latent defense campaign report: v7 -> v9 (as of 2026-08-16)

Scope: split training/inference over two DGX Spark GB10 nodes — toula
(trusted, private) and koula (fully compromised cloud). Attacker code frozen
throughout (`attacker/attacks/latent_probe.py`, nine arms, Bonferroni-adjusted
Wilson upper-95 excess over the label-free majority control; gate <= +1.0 pp).
All remote runs over TLS 1.3 (wss, pinned local CA, plaintext refused).
Detailed per-cell postmortems: `docs/experiments/LATENT_V7_DEFENSE_HARDENING.md`.
Raw evidence: `paper-data/collected/diagnostic/latent_v7_defense/` and
`paper-data/collected/diagnostic/latent_v9/`.

## Executive summary

- The v7 defense (D=16 bottleneck + fresh per-request gauges + chaff +
  boundary noise 0.35 + per-session isolation) drives the frozen nine-arm
  attacker to the statistical floor of the evaluation on every model tested
  (0.6B, 12B, 35B-A3B MoE hybrid). No arm exceeds its own majority control
  in the final configurations.
- Best campaign result: 35B-A3B at noise 0.40 + chaff-48: excess
  **+0.421 pp**, utility **-0.032** (protected path better than baseline),
  runtime **1.047x**.
- v8 equivariant-MoE cloud: mechanically correct (18/18 mechanism gates) but
  a documented null result — expert count and message-passing depth do not
  raise cloud contribution; the Gram-kernel function family saturates.
- v9 progressive program (production track) is in flight: E1 width frontier
  passed at D=32 (excess +0.426 pp, all gates); D=64 and E2 sequencing
  pending at this writing.

## Verified results ledger (all two-node, wss, frozen attacker)

| Version / cell | Model | Defenses | Excess (pp) | Loss delta | Eval ratio | Gates |
|---|---|---|---|---|---|---|
| v6 baseline (repro) | 0.6B | gauges, n0.35 | +0.652 | +0.097 | 1.006 | pass |
| v7 E2 (3 seeds) | 0.6B | + chaff-16 | +0.576/0.566/0.554 | +0.088..0.090 | 1.22-1.28 | pass |
| v7 L (3 seeds) | 0.6B | + v2 keys, wss | +0.541/0.642/0.522 | +0.084..0.094 | 1.50-1.58 | pass |
| v7 12B | 12B | same | +0.540 | -0.411 | 1.111 | pass |
| v7 35B (3 seeds n0.35) | 35B-MoE | same | +0.829/0.532/0.861 | -0.02..-0.04 | 1.03-1.05 | pass (thin margin) |
| v7.1 35B n0.40 | 35B-MoE | noise 0.40 | +0.618 | -0.029 | 1.052 | pass |
| v7.1 35B n0.40 chaff-48 | 35B-MoE | noise 0.40 + chaff-48 | **+0.421** | -0.032 | 1.047 | pass, all arms below majority |
| v8 0.6B MoE E=8 | 0.6B | + MoE cloud | +0.431 | +0.099 | 1.524 | pass* |
| v8 35B MoE E=32 | 35B-MoE | + MoE cloud, n0.40 | +0.578 | -0.035 | 1.039 | pass* |
| v8 depth L5/L8/L11 (E=32, n0.40, chaff-48) | 35B-MoE | + depth ladder | +0.424/+0.418/+0.402 | -0.033/-0.033/-0.030 | 1.05 | pass |
| v9.0 E1 D=32 chaff-48 | 0.6B | + width 32 | +0.426 | +0.082 | 1.576 | pass |
| v9.0 E1 D=64 chaff-48 | 0.6B | + width 64 | +0.411 | +0.061 | 1.812 | pass |
| v9.0 E2 seq-64 | 0.6B | + seq 64 | **+1.250** | +0.144 | 1.589 | **FAIL privacy** |
| v9.0 E2 seq-128 | 0.6B | + seq 128 | **+1.364** | +0.413 | 1.604 | **FAIL privacy+utility** |
| v9.1 E3 radial D=16 (scale gauge off) | 0.6B | + radial MoE E8 | +0.688 | +0.097 | 1.525 | pass, arms above majority |
| v9.1 E3 radial D=64 (scale gauge off) | 0.6B | + radial MoE E8 | +0.400 | +0.071 | 1.843 | pass, all arms below majority |
| v9.2 E4 K=2 D=16 | 0.6B | + 2 channels | +0.557 | +0.099 | 1.949 | pass |
| v9.2 E4 K=2 D=64 | 0.6B | + 2 channels | **+0.292** | +0.065 | 2.523 | pass, campaign-best excess |
| v9.2 E4 K=2 + radial D=64 (v9.1+K2) | 0.6B | full combo | +0.323 | +0.073 | 2.299 | pass |
| v9.3 E5 public-4k/private-1k (on v9.2) | 0.6B | + public pretrain | **+1.490** | -0.017 | 2.482 | **FAIL privacy** |
| v9.4 E6 K=4 across 2 endpoints (on v9.2) | 0.6B | + fleet channels | +0.646 | +0.073 | **3.975** | **FAIL time gate** |
| D=128 probe (v9.2 stack, K=2, radial, gauge off) | 0.6B | + width 128 | +0.287 | +0.061 | **3.720** | privacy PASS, **FAIL time gate** |

(*v8 sweep-6 JSONs carry three falsely-False gauge-presence gates from a
bookkeeping bug fixed the same day; mechanisms were verifiably applied.)

## Postmortems (attack-surface lessons)

1. **Silent chaff drop** — a 32-entry permutation over 48-row frames silently
   discarded the chaff; caught via attacker-artifact row counts; now
   fail-closed. Lesson: verify obfuscation at the wire artifact, never from
   code intent.
2. **Absmax int8 quantization strips the token-scale gauge** — per-row
   self-normalizing grids are scale-invariant by construction; it was the
   only cell where the graph arm beat majority. Fixed grid only.
3. **Majority-control dilution under chaff** — the runner's baseline was
   diluted by chaff rows in the denominator only; now matches the frozen
   attacker's definition exactly.
4. **Gauge-gate bookkeeping** — presence gates hard-coded cloud-kind
   membership; extended to new kinds (v8 MoE, radial).
5. **Statistical floor** — with 8,192-20,480 eval rows and 9 Bonferroni
   trials the excess metric has a ~+0.4-0.65 pp noise floor and ±0.2-0.25 pp
   single-seed variance; claims are made only on 3-seed runs or monotone
   dose-response.

## Production-readiness statement (unchanged verdict, stronger evidence)

v7.1/v9 remain `launchable: false`. Established: the passive wire view is at
the attacker's detection floor across three model generations, TLS transport,
v2 key stream, and utilization instrumentation. Open blockers: arbitrary
malicious-cloud integrity (bounded control only), surrogate capacity ceiling
(v8 null result), formal DP unusable, diagnostic scale (2,000 steps),
live/inference attacker drivers incomplete.

## v9 program status

- E1 (width frontier, chaff-48): **PASS at both D=32 (+0.426 pp, loss +0.082,
  1.58x) and D=64 (+0.411 pp, loss +0.061, 1.81x)** — the pre-chaff D=16
  choice was conservative; with chaff the boundary holds at 4x the width and
  utility improves monotonically with width. Note: E1 utilization logs
  predate the monitor fix (GB10 nvidia-smi exposes no memory.used; fixed via
  docker-stats RSS sampling — correct from E2 onward, plus exact
  torch-peak-VRAM in the runner JSON).
- E2 (seq 64/128 at D=16 reference): **FAILED — dropped from the
  combination.** Longer frames give the invariant arms richer per-row
  context: excess +1.250 pp (seq 64) and +1.364 pp (seq 128, utility +0.413
  also over gate). Frame length is a privacy parameter, not a free
  throughput knob; the combination stays at seq 32. First correct
  utilization capture: toula GPU mean 8-11% (peak 84-85%), koula GPU ~0%
  (by design), peak RSS 2.0-2.4 GiB toula / 1.2 GiB koula.
- E3 (radial+Gram hybrid experts, scale gauge OFF by construction):
  **PASS at D=64 (+0.400 pp, all arms below majority) — v9.1 = E1+E3.** The
  radial channel adds no measurable capacity (cloud non-bypass 1.479 vs
  1.488 with-gauge) at +0.03 runtime. The D=16 attribution cell is the
  payoff: with the scale gauge off at narrow width the invariant arm rises
  above majority (5.33 vs 5.09, excess +0.688) — the first direct
  measurement that the token-scale gauge is load-bearing at D=16 and
  dispensable at D=64 under chaff-48.
- E4 (K=2 parallel channels): running; includes the v9.2 combination cell
  (D=64 + radial + K=2).
- Transport robustness postmortem: the "transient TLS handshake timeout"
  seen on three first-cell-after-server-restart occasions was root-caused
  to cold CUDA init stalling the server event loop. Fixed both sides:
  server CUDA warmup before serving, client connect retry with backoff.
  v9.1 D=64 radial failed once with this signature before the fix.
- E5 (public-data cloud pretraining, 4000 public + 1000 private steps on the
  v9.2 winner): **FAILED privacy (+1.490 pp) — dropped; v9.3 stays at v9.2.**
  Postmortem: utility held (-0.017 at half the private steps) and cloud
  contribution actually rose (1.567, best of the campaign — public
  pretraining does add capacity), but the encoder co-adapted to the public
  distribution and 1,000 private steps of minimax pressure were not enough
  to suppress the private-distribution token signal. The mechanism works;
  the recipe needs private-phase adversarial training at full length (which
  defeats the compute-saving point at diagnostic scale). Noted as viable
  only if the public/private mix is rebalanced or the private adversarial
  phase is extended — revisit at convergence scale.
- E6 (K=4 channels across two endpoints, on the v9.2 winner): **FAILED the
  time gate (ratio 3.975 > 3) — dropped; v9.4 stays at v9.2.** Privacy held
  (+0.646 pp) but K sequential channel RTTs scale linearly in K; K=2 is the
  sweet spot. The multi-endpoint mechanics work (channels round-robin across
  two server endpoints with independent sessions/seeds); the
  multi-datacenter property is argued from per-channel gauge independence,
  not physically tested (single untrusted host).

## v9 final combination and verdict

**Successful combination (v9.2):** D=64 widened boundary (E1) + radial+Gram
MoE experts E=8 with the scale gauge off (E3) + K=2 parallel cloud channels
(E4), plus the invariant v7 stack: fresh v2-stream gauges (rotation +
permutation), chaff-48, boundary noise 0.35, per-session isolation, TLS 1.3.

Two finalist configurations, both all-gates-pass:

| Finalist | Excess | Loss delta | Eval ratio | Note |
|---|---|---|---|---|
| v9.2 radial (scale gauge off) | +0.323 pp | +0.073 | 2.30x | leaner runtime |
| v9.2 monomial K2 (scale gauge on) | **+0.292 pp** | +0.065 | 2.52x | keeps all three gauges |

Dropped stages with postmortems: E2 (longer frames leak: +1.25/+1.36 pp),
E5 (public pretraining shifts the encoder; +1.49 pp), E6 (K=4 breaks the
time budget: 3.98x).

Utilization profile of the final combination (v92_k2_radial_d64 cell):
toula GPU mean ~8% (peak 85%), CPU mean ~28-42%, peak RSS ~2.4 GiB;
koula GPU mean ~2%, CPU ~2%, RSS ~1.3 GiB total across all servers — the
untrusted side remains trivially cheap by design, and all v9 experiments
are CPU-feasible on the trusted side for small models (the GPU is a
throughput requirement, not a privacy one).

Runtime honesty note: the 2.3-2.5x eval ratio of the final combination is
driven by per-channel sequential RTTs; on a real WAN (80 ms profile) K=2
adds a full boundary RTT per step. forward_many coalescing already exists
for eval; training-time channel pipelining is the production TODO that
would bring K=2 back toward ~1.3x.

## Compromise-fraction battery (v9.2 winner, frozen gate re-scored + new sensitivity arms)

The frozen gate re-scored on regenerated bundles: **v9.2 on three seeds
(+0.318 / +0.647 / +0.427 pp, 0.6B) and transfers to the 35B MoE
(+0.390 pp, ratio 1.106)** — the final combination is now on three-seed
footing and holds on the hybrid architecture.

The new `latent-sensitivity` battery (a separate module — the frozen gate
is untouched) asks how much compromise cracks the defense. Convention note:
this battery's probe is a single invariant_only arm with two restarts and
scores over valid-label rows only, so its absolute levels sit BELOW the
gate's (every arm undershoots the majority control) — read TRENDS across
fractions, not levels.

| Arm (fraction grid) | 0.6B trend | 35B trend | Reading |
|---|---|---|---|
| capture 10/25/50/100% of frames | flat, -4.7 to -4.9 pp | flat, -4.6 to -5.1 pp | full wire capture yields nothing; no accumulation with volume |
| chaff-id 0/25/50/90% rows dropped | DROPS recovery: -4.9 -> -7.4 pp | -4.9 -> -5.6 pp | Gram-connectivity chaff identification is counterproductive for the attacker — chaff rows are real latents and are not separable; discarding "least-connected" rows removes structure the probe needed |
| gauge compromise ~10-20% canonical rows | -7.8 pp (canonical views), -4.7 pp (gauged eval) | -5.7 / -4.6 pp | **even canonical pre-gauge rows are not decodable above the control** — the adversarially-trained D=64 bottleneck + noise hides tokens on its own; the gauges are defense-in-depth, not the sole barrier |
| known-plaintext 1/5/20% eval labels | flat -3.8 to -4.4 pp | flat ~-4.0 pp | known labels do not unlock the rest |

Conclusion: at diagnostic scale the v9.2 defense's margin is not brittle
along any of these four compromise axes — there is no observed fraction,
up to full capture, at which recovery lifts off the floor. The residual
risk concentrates where this battery cannot see: live active-cloud
integrity, and capture-side drivers that do not exist yet.

- D=128 probe (post-program): privacy holds even at the old v5 failure width
  (+0.287 pp, all arms at/below majority) — but fails the time gate
  (3.72x; the pure-Python v2 QR/Box-Muller derivation at 128x128 dominates)
  and, tellingly, cloud contribution does NOT move (1.488 vs 1.488 at
  D=64). The capacity ceiling is the function family, not the width.
  D=64 remains the widest configuration that passes all gates; vectorizing
  the key-stream derivation is the fix if wider D is ever needed.

## Byzantine verification on CPU-only nodes (v9.5)

Top remaining production blocker was "arbitrary malicious cloud computation
has no independently verifiable integrity."  v9.5 adds and tests the
countermeasure: `--byzantine-verify` broadcasts each frame (same gauges) to
K identically-seeded replicas on independent nodes, Toula compares the
restored returns (relative deviation from the group mean, threshold 0.02),
uses the coordinate-wise median, and sends all replicas identical mean
gradients so honest replicas stay in lockstep.  Privacy is unchanged by
construction (every node receives the same already-gauged frame).

Setup: three CPU-only aarch64 nodes (192.168.1.120/121/122) running
`latent_cloud_server.py` in a plain venv (torch 2.13+cpu), TLS 1.3 with
per-node certs from the latent-link CA.  Config: v9.2 winner on 0.6B (D=64,
chaff-48, radial E=8, scale gauge off), K=3.

| Cell | Nodes | Verified frames | Flagged | Max rel. dev | Loss delta | Eval ratio |
|---|---|---|---|---|---|---|
| byz_honest | 3 honest CPU | 2,256 | **0 (0% FP)** | **0.000000** (bit-identical lockstep) | +0.071 | 9.30 |
| byz_malicious | 2 honest + 1 actively malicious (undeclared 0.5 sign perturbation; self-reported tampered = **0**, lying) | 2,256 | **2,256 (100% detection)** | 0.5947 | +0.088 | 9.13 |

Reading:

- **The malicious node lied (0 self-reported tampered frames) and was caught
  on 100% of frames anyway** — integrity now comes from cross-node
  replication, not from trusting the endpoint. The median path neutralized
  the corruption: utility under active attack +0.088 vs +0.071 honest.
- Honest CPU replicas are bit-identical (max deviation exactly 0.0), so the
  0.02 threshold has enormous headroom; mixed GPU/CPU fleets need a
  one-line calibration first (bf16 vs fp32 numerics), hence the flag.
- Cost honesty: 9.1-9.3x eval ratio is the weak-CPU per-step compute
  (0.595 s/step) — the verification itself adds ~zero Toula work beyond the
  median compare. On GPU-class replicas the same check runs at the channel
  RTT only; the time-gate failure here is the hardware, not the mechanism.
- Remaining gap: K=3 median tolerates one adversary; k-of-n generalization
  and WAN replica placement are the production steps.

## v10: delegation-share experiments

**v10.0 — two-segment delegation** (`bin/run_latent_native_v10_2seg.py`).
Layout on the 0.6B: prefix 0-15 | segment A 16-19 (Koula :5025) | private
island 20-21 | segment B 22-25 (Koula :5027) | tail 26-27 — **28.6% of the
architecture delegated**, each boundary with its own encoder/decoder, DP,
v2 gauges, and chaff pool; each segment an isolated TLS session.

| Metric | Value | Gate |
|---|---|---|
| Attacker excess (both boundaries, frozen probe) | +0.634 pp | PASS (<=1.0) |
| Utility delta | +0.550 | **FAIL (>0.35)** — chained surrogate error compounds |
| Eval time ratio | 8.91x | **FAIL (>3)** — unbatched 3-control eval path (3 sequential protected paths x 2 remote boundaries per block), engineering-fixable |
| Cloud contribution | A +0.023 / B +0.017 | small but real per segment |

Reading: **~30% architectural delegation is privacy-safe today**, but the
Gram-surrogate family does not approximate well enough to chain at
diagnostic scale, and the eval path needs batching.  The recipe to make it
practical: fewer+stronger segments, much longer cloud training, batched
eval.

**v10.1 — E5 done right.** The v9.3 failure mode (public pretraining +
halved private phase, +1.490 pp FAIL) was rerun with a full-length private
adversarial phase (4000 public + 2000 private on the v9.2 winner):
excess **+0.732 pp PASS**, utility **-0.007** (better than v9.2's +0.073),
cloud contribution **1.556** (campaign best — public pretraining does add
capacity).  The recipe works when the private minimax phase runs at full
length.  Koula carries the bulk of training steps on public data.

## v11.0: external matching-attack families vs the v9.2 winner

The 2026 literature search flagged that our floor was measured only against
trained probes. v11.0 implements the two published matching families as new
arms (`attacker/attacks/latent_matching.py`; the frozen gate is untouched):

- **vma_centroid_match** — vocabulary matching adapted from Hidden No More
  (ICML'25): contrastive cosine matching against per-class invariant
  centroids.
- **sorted_signature_matching_position_free** — permutation-reversal style
  matching (arXiv:2505.18332) with the observed-position channel excluded
  (an earlier draft that included it measured a tautological 80% "position
  recovery" and was corrected post-hoc; recorded here as a methodology
  lesson).

Results (regenerated v9.2 bundles; frozen gate re-scored on the same data):

| Model | Frozen gate excess | VMA match top-1 vs majority | Signature match top-1 vs majority |
|---|---|---|---|
| 0.6B (seed 42) | +0.541 pp | **0.03% vs 5.92%** | **1.56% vs 5.92%** |
| 35B-A3B (seed 42) | +0.285 pp | **0.02% vs 5.85%** | **1.54% vs 5.85%** |

Both matching families land far BELOW the no-attack majority control on
both architectures — the attacks that broke PermLLM/STIP/Centaur at >99%
extract nothing from the composite (rotation + permutation + scale +
chaff-48 + noise) boundary. The floor claim now holds against trained
probes AND the published matching families.

## v11.2: MI-budget estimation — honest negative result

`bin/measure_mi_budget.py` implements the variational/Fano-style lower
bound I(activations; current token) >= H(tokens) - CE_probe, with
temperature-calibrated probes, run on the undefended 0.6B backbone
(layers 2/8/14/21).

Result: **vacuous.** Prior entropy 5.771 nats; calibrated probe CE
5.775-6.013 nats — the lower bound is exactly 0 at every layer even though
the same probes recover 55-59% of tokens top-1.  Lesson recorded: CE-based
MI lower bounds are tail-calibration-limited and produce no usable bound at
LLM vocabularies; a real MI guarantee needs MINE/k-NN-class estimators or
the group-invariance proof route (gauges ARE provable).  The substantive
empirical statement stands: backbone probes recover 55-59% top-1 while the
defended boundary sits at the majority control (~5%) — the information
decay is real but currently only measurable, not boundable.

v11.1 (VFLAIR-LLM harness port): scoped — their defense API is a
tensor-transform hook while our defense is architectural (trained
encoder/decoder + per-request gauges), so a faithful port requires a
custom defense module and is queued, not executed.

## v12: the capacity ceiling is information-theoretic, not architectural

v12.0 tested the fifth independent capacity axis — a new cloud function
family (`invariant_mlp`: per-row MLP over gauge-invariant features gating
norm scale and Gram-message direction; mechanism test proves
rotation/permutation equivariance; the D-only audit fail-closed correctly
rejected its first draft).  Result on the 0.6B (D=64, chaff-48, K=1):

| Axis tried | Contribution |
|---|---|
| v9.2 reference (radial E8, D=64) | 1.488 |
| v8 experts 8→32 / depth 2→11 | 0.404 (35B) / flat |
| E1 width 16→128 | 1.488 (flat) |
| v10 segments 1→2 | A +0.023 / B +0.017 per segment |
| v10.1 public pretraining | **1.556** (the only mover) |
| **v12.0 invariant-MLP (new function family)** | **1.473 — flat** |

Five axes flat, one (training coverage) moving slightly.  Conclusion: the
usable task information crossing the D=64 boundary is the ceiling; the
cloud's function family does not matter.  Privacy on v12.0: +0.404 pp,
utility +0.077, ratio 1.83 — all gates pass (scale-gauge gate False by
design, as with radial).

v12.2 (convergence): 4,000 public + 10,000 private steps on the v9.2
winner — excess +0.417 pp PASS, but cloud contribution 1.427 (NOT higher
than the 1.488 diagnostic) and utility delta +0.123 (worse than the
2k-step +0.073: the 256-block train split overfits at 39 cycles — honest
caveat for any convergence claim at this corpus size).  With training
duration now also flat, the capacity conclusion closes from six axes: the
D=64 boundary's information content is the ceiling, full stop.

## v13: the reviewed-technique queue (MINE penalty, MI objective, fragmentation)

Three cells on the v9.2 winner (0.6B, D=64, chaff-48, radial E8, scale gauge
off, noise 0.35, wss), from the external-technique review:

| Cell | Technique | Excess | Loss delta | Ratio | Cloud | Extra |
|---|---|---|---|---|---|---|
| v13.0 (A9) | + MINE MI penalty β=0.1 (with minimax) | +0.576 pp | +0.073 | 1.844 | 1.476 | held-out MINE estimate ≈ -1.3e-6 nats |
| v13.1 (A2) | MINE penalty ONLY (probe minimax off) | +0.428 pp | +0.065 | 1.840 | 1.485 | MI ≈ -0.3e-3 nats |
| v13.2 (A1) | fragmentation: 2× D=32 halves across two nodes | +0.416 pp | +0.084 | 2.525 | 1.466 | no node holds a full row |

Findings:

1. **The MINE MI penalty works and reports ~0 nats on the defended
   boundary** — the first non-vacuous MI estimator reading we have (the
   v11.2 CE-based bound was vacuous).  Honest caveat: on an already-floored
   boundary, ~0 could mean "nothing to find" or estimator blindness; the
   estimator is validated to see information on undefended views (the v11.2
   backbone probes) — the paired reading is the evidence.
2. **A2 passes without the probe minimax** — the MI-penalty-only objective
   empirically replaces adversarial probes at diagnostic scale (+0.428 pp),
   simplifying the training story (one less moving part).
3. **Fragmentation holds**: two D=32 halves across two servers pass with
   essentially no cost (privacy +0.416, cloud 1.466).  Privacy from
   fragmentation composes with replication — the fleet story is complete:
   no single node ever holds a reconstruction-sufficient projection.

## Verdict re-evaluation under the v13 stack (0.6B, seed 42)

The failed verdicts were rerun under the v13 configuration (v9.2 winner +
MINE penalty β=0.1), frozen gate. Old = the published verdict; new =
re-evaluation. Caveats: single seed, and the seq-length cells are
corpus-dependent (PR-101 D4).

| Cell | Old verdict | New (v13 + MINE) | Delta reading |
|---|---|---|---|
| E2 seq-64 | **FAIL** +1.250 pp | **PASS** +0.758 pp | MINE suppresses the long-frame leak |
| E2 seq-128 | **FAIL** +1.364 pp | **PASS** +0.264 pp privacy, **FAIL** utility +0.380 | privacy rescued, utility now binding |
| E5 pub-4k/priv-1k | **FAIL** +1.490 pp | **FAIL** +1.553 pp | not an MI-leak problem — the E5 failure is distribution shift, as diagnosed in v10.1 |
| noise σ=0.30 tripwire | **FAIL** +1.039 pp (v6) | **PASS** +0.553 pp | chaff-48 moved the tripwire down; the composite buys noise budget |
| v9.2 + MINE (control) | PASS +0.541 | PASS +0.505 | consistent within seed variance |

Conclusions: (i) the MI penalty is the first mechanism since chaff that
moves a FAILED verdict to pass — it specifically targets the invariant
leak the long frames expose; (ii) σ=0.30 is now usable, revising the
operating point downward (more utility headroom); (iii) E5's failure is
robustly confirmed as non-MI (already fixed properly in v10.1 by the
full-length private phase).  The re-evaluation script is
`bin/latent_reval_stage.sh`; artifacts in
`paper-data/collected/diagnostic/latent_reval/`.

## Evidence
