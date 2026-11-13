# Prepared R1 remote workload — no experiment results

This package was prepared from clean commit
`5f9fcb0a509e975476b7d39bc86cfb79ae8d95cf`. The bundle manifest indexes 183
source, protocol, configuration, environment-template, corpus and planned
workload files; the archive holds 191 files in total, the extra eight being the
manifest, the plan and the two generated process packets with their set record.
The archive is small enough to retain directly in this private repository.
Reviewers need repository access.

This supersedes `evidence/r1-preparation-20260910/` in scope, which was prepared
from commit `9b45323` and indexed 65 files. That earlier package is preserved for
audit and is not withdrawn.

- [workload.tar.gz](workload.tar.gz): exact source and raw input bytes for the
  prepared workload, plus the two process packets; no legacy results/checkpoints,
  installed CUDA environment, admitted scientific runs or fitted attacks.
- [r1_bundle.json](r1_bundle.json): every file hash, origin revision and clean
  source status.
- [plan-summary.json](plan-summary.json): five planned stages totalling 340 runs,
  and the three outstanding blockers recorded in the plan.
- [validation.json](validation.json): archive identity, fresh extraction check and
  explicit non-execution status.
- [SHA256SUMS](SHA256SUMS): checksums for the archive and machine-readable records.

The pre-reset reference remains `paper3-pre-reset-2026-09-10`. This packet is
**preparation evidence**, not evidence for the trilemma and not a claim of
prospective registration. The draft environment and budget fields intentionally
remain unset. No run took place while preparing this package.

## What the process packets show

The prepared workload splits into two packets. The remote-service packet contains
no corpus text; the trusted-client packet contains
`data/raw/tinyshakespeare.txt`. The packet set digest
`0b3a5afd1fba491c414177af082c70c49b7640168171761fb03cfea2063d5a96` reproduced
identically across two independent preparations from the same source, so the
packet split is deterministic.

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
The archive was extracted to a disposable directory and this check passed. The
source Git metadata is in the bundle manifest; `.git` is intentionally absent.
Remote execution still needs a reviewed immutable GPU environment and budget.

Note that the bundle's own source inventory digest
`ee80b3aaa0c1c930f24fa28738f7acc664111e48d6207b529baf7805114ef307` differs from
the repository's `9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0`
because bundle preparation deliberately excludes non-R1 preregistration and
methodology documents. The 81 files shared between the two inventories agree
byte for byte, and no file is unique to the bundle.

Follow [the runbook](../../docs/R1_RUNBOOK.md) and
[cost estimate](../../docs/R1_REMOTE_COMPUTE.md) before the engineering stage.
The scientific protocol, thresholds, fresh attack qualification and held-out
admission must be accepted before the calibration or exploratory workload can
execute. The R1 runner rejects scientific launch.
