# Qwen3-0.6B production-v3 diagnostic

- Run ID: `poc06b_v3_20260814T2115`
- Code commits: `383a3bb` (v3 mechanisms), `aefae01` (live boundary DP),
  `c0b6632` (adaptive pooled-attack budget)
- Nodes: trusted Toula (local layers 0-15 and 26-27), compromised Koula
  model (middle layers 16-25, frozen)
- Model: Qwen3-0.6B; one optimizer step, 16 microbatches, sequence length 64
- Evidence status: **diagnostic**

## Result

The protected arm reduced pooled known-plaintext recovery to the label-free
floor at every tested adaptive budget: 3.12% at 8 rows/epoch (128 total),
3.12% at 16 rows/epoch (256 total), and 0% at 32 rows/epoch (512 total).
The plain arm recovered 42.97% with combined forward/backward observations at
32 solve rows. However, protected held-out loss was 16.508 versus 3.688 plain,
and protected step time was 60.74 s versus 2.66 s. This operating point therefore
passes the tested capture-resistance diagnostic but fails utility and cost.

**Accountant correction (2026-08-14):** the emitted boundary-DP epsilon 2.972
at delta 1e-6 counted each sequence as one release. The attacker labels token
rows, so this claim is superseded and must not be cited. The corrected runtime
clips and counts each final-dimension token row. Local weight DP-SGD separately
reported epsilon 10.150 at delta 1e-6; that local-update statement is unchanged.

## Limitations

- One step and one run; no uncertainty or convergence claim.
- The historical boundary-DP accountant used the wrong adjacency unit; its
  epsilon is a superseded metric even though the raw diagnostic is retained.
- The transform seed was pre-provisioned on Koula. Rotation is therefore a
  capture-resistance control, not protection from a Koula administrator.
- No nonlinear MPC or second non-colluding cloud peer was used.
- No learned bottleneck was used.
- The cloud logged three pending graphs at plain-client disconnect; this did
  not alter the emitted result but remains a lifecycle defect.
- Raw `.pt` and `.bin` captures, seeds, checkpoints, and private corpora were
  deleted and are not included here.
