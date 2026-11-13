# Seed, stream, and cohort policy

## Paired trained/frozen blocks

Each trained/frozen comparison receives a unique `pairing_block_id`. Within a block, the two
runs share:

- backbone, encoder, decoder, and remote initial tensors;
- corpus order, minibatches, augmentation, and stopping budget;
- trusted-side random streams;
- hardware class and resolved configuration.

Only remote-parameter updating differs. The manifest records initial and final parameter
checksums. Frozen remote parameters must remain bit-identical.

## Stream separation

Derive domain-separated streams for model initialization, data order, boundary randomness,
attacker initialization, and control injection. Never reuse a trusted initialization stream as
a remote/adversarial stream.

## Cohort separation

- Calibration determines metric sensitivity and thresholds.
- Exploration estimates variance and selects candidate designs.
- Tuning selects configurations using a frozen rule.
- Confirmation estimates final effects on fresh seeds and captures.

These cohorts must have disjoint seed lists and artifacts. A result cannot change cohort status
after outcomes are inspected.

## Failure and retry rule

Before freeze, define operational failure classes and the maximum retry count. Retries keep the
same cell identity and record every attempt. Outcome-dependent failures or exclusions are not
permitted; ambiguous failures remain in the audit log and are handled by the frozen analysis.

