# Latent-native v6 remote evidence

Safe accepted aggregate evidence for the Qwen3-0.6B Toula/Koula evaluation is
stored under `accepted/`. Raw `.pt` release bundles stayed on trusted Toula
only for the duration of `/attacker` scoring and were deleted afterward.
Koula's raw forward, return, and backward captures were counted for provenance
and then deleted; they are not part of this collection.

The accepted passive matrix uses isolated cloud state, seeds 42/43/44,
split-after 21, resume-before 26, D=16, noise multiplier 0.35, and the
monomial-equivariant cloud. The active control uses the same configuration and
adds a declared 0.5 signed perturbation to every cloud return.

The full interpretation and production-readiness decision are documented in
`docs/experiments/LATENT_NATIVE_V6_PRODUCTION_ASSESSMENT.md`.
