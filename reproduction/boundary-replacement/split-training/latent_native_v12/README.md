# Latent-native v12 — attacking the cloud-capacity ceiling

Every previous capacity axis saturated the Gram-kernel family: experts
(8→32), depth (2→11), width (16→128) all left cloud contribution flat
(~1.4–1.5 on the 0.6B).  v12 changes the function class.

| Stage | Content | Status |
|---|---|---|
| v12.0 | `invariant_mlp` cloud kind: per-row MLP over gauge-invariant features (norms + Gram-row statistics, position channel excluded) gating norm scale and Gram-message direction on top of the monomial skeleton — a learned nonlinear family.  Scale gauge OFF (norms are read), as with the radial kind; mechanism test covers rotation+permutation equivariance | in progress |
| v12.1 | LM-head surrogate delegation (the largest single matmul in the step; output-side exposure → output-inversion arm required) | pending |
| v12.2 | 10k-step convergence cell (v9.2 winner + public pretraining) | pending |

Reference cell to beat (0.6B, D=64, chaff-48, K=1): cloud non-bypass
1.488, loss delta +0.061, attacker excess +0.411 pp, ratio 1.81.

`launchable: false`.
