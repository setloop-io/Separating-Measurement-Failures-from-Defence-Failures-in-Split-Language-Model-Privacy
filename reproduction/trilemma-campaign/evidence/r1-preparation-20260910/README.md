# Prepared R1 remote workload — no experiment results

This package was prepared from clean commit
`9b45323c2068229cacfd07a49b9e08bba0fb13ef`. It contains 65
indexed source, protocol, configuration, environment-template, corpus and planned
workload files, plus the bundle manifest. The archive is small enough to retain
directly in this private repository. Reviewers need repository access.

- [workload.tar.gz](workload.tar.gz): exact source and raw input bytes for the
  prepared workload; no legacy results/checkpoints, installed CUDA environment,
  admitted scientific runs or fitted attacks.
- [r1_bundle.json](r1_bundle.json): every file hash, origin revision and clean
  source status.
- [plan-summary.json](plan-summary.json): 14 micro checks, four speed checks,
  76 proposed calibration runs and 232 proposed exploratory runs.
- [validation.json](validation.json): archive identity, fresh extraction check,
  software checks and explicit non-execution status.
- [SHA256SUMS](SHA256SUMS): checksums for the archive and machine-readable records.

The pre-reset reference remains `paper3-pre-reset-2026-09-10`. This new packet is
**preparation evidence**, not evidence for the trilemma or a claim of prospective
registration. The draft environment and budget fields intentionally remain unset.

## Retrieve and check

After cloning the private repository, verify from this directory:

```bash
shasum -a 256 -c SHA256SUMS
```

Extract `workload.tar.gz` into a new directory. Using an environment with this
project's Python dependencies available, invoke the **extracted** script:

```bash
python /absolute/new/directory/workload/scripts/run_r1.py verify-bundle
```

This verifies source, raw corpus and plan bytes without training or provisioning.
The archive was extracted to a disposable directory and this check passed.
The source Git metadata is in the bundle manifest; `.git` is intentionally absent.
Remote execution still needs a reviewed immutable GPU environment and budget.

Follow [the runbook](../../docs/R1_RUNBOOK.md) and
[cost estimate](../../docs/R1_REMOTE_COMPUTE.md) before the engineering stage.
The scientific panel loader, complete fresh attack qualification and scientific
checkpoint/capture requirements must be implemented and admitted before the
calibration or exploratory workload can execute. The existing R1 runner rejects
scientific launch. No run took place while preparing this package.
