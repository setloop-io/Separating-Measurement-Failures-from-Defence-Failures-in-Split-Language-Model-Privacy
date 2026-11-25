# Production E2E privacy experiment — split inference

This directory isolates the production contract for split inference with trusted Toula and fully compromised Koula. It deliberately excludes encryption and TEE from the cloud-privacy claim.

The cross-mode implementation audit and paper rationale are in `docs/experiments/E2E_PRIVACY_ARCHITECTURE_AUDIT.md`.

## Security question

Can Toula delegate the middle of 7B inference to Koula while keeping user prompts and generated content below a predeclared attacker-recovery threshold when Koula can inspect all host/VRAM state and actively manipulate responses?

The cloud must not receive a reversible secret, an invertible representation plus its public reference, or a canonical activation. Current shared-key and ratchet modes protect a wire observer, not a compromised Koula that holds the key. They remain negative controls only.

## Required production boundary

- A private encoder and irreversible task bottleneck run only on Toula.
- Per-request activation clipping and calibrated local noise have a formal privacy accountant.
- The return direction is independently sanitized; protecting prompts but leaking generated-state activations is not sufficient.
- Request length, batching, cache behavior, retries, and timing are normalized or included as leakage surfaces.
- Repeated prompts/templates are randomized with padding and slot permutation as an auxiliary defense.
- Sessions have independent randomness and no cloud-visible reusable seed.
- A live malicious Koula driver performs active perturbation, replay, downgrade, and chosen-query tests.

The experiment measures utility/privacy tradeoffs. Exact identity with plain inference is not an acceptance requirement for irreversible arms. Five or more matched repetitions run across native, 80 ms RTT, and hostile-WAN profiles. Smoke, toy, and dry-run evidence is machinery-only.

## Acceptance rule

The protected arm passes only if every required attacker completes, the 95% confidence upper recovery bound is within 2 percentage points of its matched label-free control in both directions, no session exceeds the gate, held-out loss degradation is at most 0.35, and task/output quality passes. Missing artifacts or downgrade behavior invalidate the run.

Validate from the repository root:

```bash
python3 bin/validate_e2e_privacy_contract.py split-inference/e2e_privacy/contract.json
python3 bin/validate_e2e_privacy_contract.py --require-ready split-inference/e2e_privacy/contract.json
```

The readiness check must fail until every blocker is resolved.
