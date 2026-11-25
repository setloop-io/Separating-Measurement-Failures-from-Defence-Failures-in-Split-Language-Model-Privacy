# R1 preparation and gated engineering runbook

No scientific run or paid GPU launch is authorized. These preparation commands
read code and raw corpus bytes; they never load historical results or train.
Run from the Paper 3 repository with its existing software environment:

```bash
.venv/bin/python scripts/run_r1.py plan --stage engineering
.venv/bin/python scripts/run_r1.py plan --stage performance_calibration
.venv/bin/python scripts/run_r1.py plan --stage calibration
.venv/bin/python scripts/run_r1.py plan --stage exploratory
.venv/bin/python scripts/run_r1.py prepare --destination /absolute/new/r1-bundle
.venv/bin/python /absolute/new/r1-bundle/scripts/run_r1.py verify-bundle
```

## Local CPU systems gate

Before any GPU budget request, run the production transport and evidence path on
CPU with the exact corpus:

```bash
uv sync --frozen --all-groups
uv run pytest -q tests/test_wire.py tests/test_distributed_split.py \
  tests/test_r1_panels.py tests/test_r1_attacks.py tests/test_artifacts.py \
  tests/test_r1_packets.py tests/test_r1_cpu_e2e.py
uv run python scripts/run_r1.py local-validate \
  --output /absolute/new/r1-local-validation --execution-id local001
```

The remote subprocess is launched from a verified corpus-free packet. The run must
retain matching client/remote wire totals, durable step journals, coordinated
checkpoints, panel-bound attacks and semantic-validation records. Any uncertain
optimizer acknowledgement fail-stops the attempt; recovery starts only from a
coordinated checkpoint. These outputs remain `scientific_evidence: false`.

The destination must be new. Preparation copies bounded source files, the R1
protocol/matrix, environment templates, evidence policy and exact corpus bytes,
then records a complete hash inventory and all four planned workloads. Historical
outputs, checkpoints, manifests and non-R1 configurations are excluded. Source
scripts inherited from the reviewed repository remain available as code; their
presence does not admit legacy entry points for R1 use. The copied bundle's R1
runner is the sole supported launch path.

Archive the complete bundle, record the archive SHA-256, restore it to a second
directory, and execute that copy's `verify-bundle`. Store its index and private
retrieval location. This proves byte recovery and preparation, not scientific
reproduction or external preregistration.

The bundle omits `.git`. Its originating commit and clean/dirty status are retained
in `r1_bundle.json`; per-run archives retain the actual executed source bytes.
A generic Git collector inside the copy can therefore report `unknown-commit`
without losing source identity. Follow the launch's bundle binding to the origin
record instead of inventing a commit for the copied tree.

## After separate remote environment and cost acceptance

Provider selection and lifecycle control remain pending. Read
[R1_REMOTE_COMPUTE.md](R1_REMOTE_COMPUTE.md) and
[environment requirements](../environment/r1/README.md). From the **copied bundle**
in the reviewed remote CUDA environment, the future engineering commands are:

```bash
python scripts/run_r1.py inspect-environment
python scripts/run_r1.py freeze-engineering --stage engineering \
  --budget budget-review.json --environment-review environment-review.json \
  --admission engineering-admission.json
python scripts/run_r1.py execute-engineering \
  --admission engineering-admission.json --execution-id eng001
python scripts/analyze_r1.py \
  --execution-root outputs/p3-r1-fresh-start/engineering--eng001 \
  --output outputs/p3-r1-fresh-start/engineering--eng001/analysis.json
```

The two review JSONs must contain actual accepted values and references, not the
draft templates. No admission was created during preparation. Source, input,
environment and budget mismatches block launch; CUDA is mandatory. Execution
reserves a fresh directory and never overwrites an attempt. A failed run stops
the stage with preserved records. An admission is consumed by one launch. A new
attempt needs a new execution identifier and a newly reviewed remaining budget
and admission. Identifier changes are not permission to retry indefinitely.
Do not edit historical manifests to make a run appear complete. Split modes use
the same authenticated two-process transport validated on CPU; the trusted process
owns corpus/prefix/suffix/loss while the corpus-free process owns only the remote
surrogate and optimizer. Loopback on one GPU is an engineering deployment, not yet
a two-host latency measurement.

The separate four-run speed stage uses `--stage performance_calibration`, a new
budget/admission and its own execution identifier. The output namespace changes
accordingly. It has the same permanent engineering-only status.

The analyzer verifies the bundle, launch, acceptance, initial state, per-run
source/configuration and all required artifacts before descriptive contrasts.
Missing completions remain pending; corruption is an error. Privacy and joint
feasibility remain `UNDETERMINED`, regardless of engineering losses. Retain and
retrieve the entire bundle including outputs to replay verification after a move.

## Before scientific stages

Implement and review the four-region panel loader, target-model attack fitting,
all 12 proposed target/channel combinations and their fresh null/positive
controls, complete scientific capture/checkpoint requirements, and analysis.
Verify data separation, capture RNG invariance, raw-view fidelity and resource
accounting. Freeze the calibration packet before calibration; freeze the revised
exploratory protocol before its affected outcomes. The current runner cannot
execute those scientific stages. The confirmation design is a further task.
