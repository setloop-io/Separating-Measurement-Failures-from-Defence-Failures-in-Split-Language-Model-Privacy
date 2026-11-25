# Why the defense-aware surrogate arm recovered only ~5% while the naive baseline recovered 59–90%

Scope: bottleneck sweep v1 (`results/bottleneck-sweep-v1/`, 60 cells:
methods {pca-compress, tb-cotrain, tb-frozen} × dims {64, 128, 256} ×
splits {3, 6} × seeds {11, 22, 33}). All numbers below are read from the
recorded per-cell artifacts (`defense-aware-baseline.json`,
`forward-baseline.json`, `sipit-baseline.json`, `cell.json`); tensor
captures are not local and were not needed.

## The comparison, restated

Mean position-aligned token recovery (9 cells per split×dim, 3 methods ×
3 seeds):

| split | dim | DA fwd | DA SIPIT | naive fwd | naive SIPIT |
|-------|-----|--------|----------|-----------|-------------|
| 3 | 64  | 0.056 | 0.034 | 0.593 | 0.534 |
| 3 | 128 | 0.057 | 0.032 | 0.811 | 0.767 |
| 3 | 256 | 0.057 | 0.028 | 0.893 | 0.893 |
| 6 | 64  | 0.014 | 0.013 | 0.552 | 0.477 |
| 6 | 128 | 0.013 | 0.013 | 0.753 | 0.707 |
| 6 | 256 | 0.014 | 0.011 | 0.841 | 0.836 |

Undefended control (`none`): naive fwd 0.87–0.91.

## Mechanism (from the code)

Both arms attack the same captured wire and use the same public head,
calibration partition, and budgets. They differ in exactly one step — how
the wire is mapped back to hidden-state space
(`experiments/run_bottleneck_sweep.py`, `seam_baselines`):

- **Naive arm** uses the defense's *released public decoder* (granted by
  the matched-access spec; `public-pca-decoder.pt` /
  `public-bottleneck-decoder.pt` when the defense exposes one):
  `hidden = wire[:k] @ B_defenseᵀ + mean_defense`.
- **Defense-aware (DA) arm** may not use defense weights, so it fits its
  own PCA basis `B_sur` and mean on *public calibration* head-forward
  activations (`fit_pca_surrogate`,
  `attacker/reconstruction_baselines.py:230`) and decodes
  `hidden_hat = wire[:k] @ B_surᵀ + mean_sur`
  (`surrogate_decode`, `reconstruction_baselines.py:257`), then runs the
  identical forward/beam/SIPIT kernels on `hidden_hat`.

The wire coordinates are expressed in the defense's frame:
`wire[:k] = (h_victim − mean_def) @ B_def`. The DA decode therefore
produces

  hidden_hat = B_sur B_defᵀ (h − mean_def) + mean_sur,

i.e. the true hidden state **rotated by the cross-basis matrix
B_sur B_defᵀ and shifted by the wrong mean**. The forward/beam kernels
then verify candidates through the public head, which is extremely
direction-sensitive: a rotated/shifted vector is off the hidden-state
manifold, so forward scores are meaningless and beam verification
collapses. There is no double application of compression in the decode
path — the failure is the frame mismatch itself.

Two sources of frame mismatch, both structural:

1. **Basis mismatch.** The surrogate basis is the PCA of *pretrained
   public-head calibration activations*; the defense basis is fit (or
   learned, for the trained-bottleneck methods) on *victim* activations.
   Nothing in the DA threat model aligns them.
2. **Distribution/mean shift.** The wire is captured during victim
   training (64 steps; mean train loss 4.31 → 3.75 per `cell.json`), so
   the victim's activation distribution has drifted from the pretrained
   calibration distribution the surrogate's mean and basis describe.

## Hypotheses and evidence

**H1 — Explained-variance shortfall at k=64 (the surrogate is a bad fit
of its own distribution).** *Rejected.* The recorded
`surrogate_explained_variance_at_k` is already 0.9969 (split 3) / 0.9948
(split 6) at k=64, rising to 0.9995/0.9989 at k=256. If within-
distribution fit were the bottleneck, the DA–naive gap would shrink as k
grows; instead DA accuracy is flat (~0.056 at split 3, ~0.014 at
split 6) across k = 64→256 while naive accuracy climbs 0.59→0.89. Note
also that the defense's own PCA (pca-compress provenance) captures
*less* variance at k=64 (0.9967) than the surrogate captures of its own
distribution (0.9969) — yet the naive arm using the defense frame
recovers 59%.

**H2 — The DA decode path double-applies compression or otherwise
corrupts the wire.** *Rejected.* `surrogate_decode` is a single linear
map consuming `wire[:k]` once; the ridge decoder is refit
self-consistently on surrogate-round-tripped calibration (`hc_hat`), and
beam/SIPIT receive the same shape and budgets as the naive arm.

**H3 — The surrogate inverse optimizes a different objective than token
recovery (PCA minimizes reconstruction MSE, not head-verifiable
accuracy).** *Partially true but not the operative cause.* Within its
own frame the surrogate's residual is ≤0.5% of variance — tiny. The
objective mismatch would matter at the margin; it cannot explain a
0.06-vs-0.59 collapse, and H1's k-scaling controls for it.

**H4 — Cross-basis / cross-distribution mismatch (the surrogate decodes
defense-frame coordinates into the wrong frame).** *Supported; this is
the mechanism.* Evidence from the recorded numbers:

- DA accuracy is **identical across tb-cotrain and tb-frozen** for every
  (split, seed, dim) — e.g. split3/seed11: 0.029, 0.029, 0.026 (dims
  64/128/256) for both methods. The two defenses learn different
  encoders, so their wires differ; the invariance shows DA recovery is
  dominated by the surrogate frame, not by anything the wire contains.
- DA accuracy does not improve with k (flat 0.056/0.057/0.057 at
  split 3; 0.014/0.013/0.014 at split 6), exactly what a frame-rotation
  error predicts: the rotation B_sur B_defᵀ ≠ I is O(1) at every k.
- The naive arm's k-scaling (0.55→0.84 at split 6) shows the wire itself
  carries plenty of recoverable information; only the DA decode destroys
  it. For pca-compress the naive arm even uses the same *class* of
  linear decoder as the surrogate — differing only in whose basis —
  and recovers 0.58–0.90.

**H5 — SIPIT should fix what the forward pass misses (gradient
optimization toward the target).** *Rejected.* DA SIPIT ≤ DA forward in
37/54 cells and never exceeds it materially; at split 6 SIPIT
occasionally ties (~0.01). SIPIT minimizes head-reconstruction error
against an off-manifold target and its exact-match verification
(1e-5 rtol/atol against `hidden_hat`) can confirm a wrong answer — the
target itself is corrupted, so optimization cannot recover the frame.

**H6 — The DA arm's weakness is a budget artifact.** *Rejected.* Budgets
are matched across arms (candidates 32, beam 4, SIPIT 128 steps/token,
recorded in every baseline JSON), and the naive arm reaches 0.89 under
the same budgets.

## Conclusion: weak realization, not a fundamental limitation

The arm as realized is not a faithful defense-aware decoder, and the
recorded surrogate diagnostic (`explained_variance_at_k`) measures the
wrong thing: it certifies the surrogate *on its own calibration
distribution*, which is exactly the quantity that was never in doubt.
The number the arm needed — recovery when decoding defense-frame
coordinates — is the one it did not record. That said, nothing
fundamental forbids a strong class-and-calibration adversary: the wire
is demonstrably decodable (naive arm, same budgets). The flaw is the
realization's implicit assumption that the defense's encoder frame ≈ the
PCA frame of public calibration activations.

## Recommendation for the paper

Keep the arm, report it, and word it as: the PCA surrogate inverse is
*not* a faithful defense-aware attacker — it is a lower bound that fails
for an identified, non-fundamental reason (basis/distribution mismatch,
EV-certified only on its own fitting distribution). What would make a
faithful defense-aware attacker within the same threat model (defense
class + k + public calibration, no defense weights):

1. **Learn the inverse from wire observations, not from a PCA prior.**
   If the attacker can query the defended seam with public calibration
   text (standard for this threat model), it observes (input, wire)
   pairs and can regress wire → tokens/hidden directly, sidestepping
   frame alignment entirely. The current arm never uses the wire as a
   learning signal.
2. **Match the calibration distribution to the victim** (post-training
   activations) before fitting the surrogate mean/basis, reducing the
   shift component of the mismatch.
3. If the defense releases a public decoder, the honest defense-aware
   attacker uses it — which is precisely the naive arm here, and the
   reason the naive numbers are the correct headline.

## Diagnostic added (this note's accompanying change)

`run_bottleneck_sweep.py --da-self-consistency` (default off; completed
sweep cells are byte-identical without it) records, per defended cell,
the surrogate's self-consistency: end-to-end calibration-loop forward
accuracy (surrogate decode of surrogate-encoded calibration wires,
scored against known calibration tokens) and
`relative_reconstruction_mse` from the new
`surrogate_self_consistency()` in `attacker/reconstruction_baselines.py`.
Had this been recorded in sweep v1, it would have shown high
self-consistency alongside ~5% wire recovery, isolating H4 immediately.
