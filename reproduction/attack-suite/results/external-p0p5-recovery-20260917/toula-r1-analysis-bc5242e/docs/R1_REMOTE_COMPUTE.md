# R1 remote GPU workload and preliminary cost estimate

Prepared 2026-09-10. **No paid execution or provisioning is authorized.** The
user requested a prepared remote GPU workload and estimate first. These are
advertised rates, not availability guarantees or an accepted budget. No legacy
runtime or scientific observation is used in the estimate.

> **Update 2026-09-16: executed on owned hardware; the metered estimates below
> are superseded drafts.** The owner authorized the owned nodes toula/koula
> (NVIDIA GB10, driver 580.159.03 / CUDA 13.0) on 2026-09-16; no metered cloud
> was used. Both engineering stages completed under bound admissions: Stage A
> micro engineering 14/14 runs on toula (~65 s whole-stage vs the 300 s cap) and
> Stage B performance calibration 4/4 runs on koula. Stage B measured the
> full-size reference (8 blocks, d256, batch 32, seq 128, 100 steps): split runs
> 12.6–15.3 s wall (~0.13–0.15 s/step including evaluations and captures),
> all-local 7.0 s, peak client ~686 MiB / remote ~174 MiB, ~1.43 GB actual wire
> per split run, 84–124 MiB outputs per run. Linear projection to the 308-run
> scientific draft at 2000 steps/run is roughly 26 GPU-hours (~13 h per node when
> split across two nodes), before attack fits and verification overhead; the
> stage budgets are reviewed against measured values, not the hypothetical
> scenarios below. Evidence: `evidence/r1-gpu-engineering-20260916/`,
> `docs/status/2026-09-16-gpu-engineering-status.md`.

| Stage | Runs | Steps/run | Total optimization steps | Status |
|---|---:|---:|---:|---|
| Micro engineering | 14 | 8 | 112 | Remote admission pending |
| Full-size performance calibration | 4 | 100 | 400 | Separate engineering admission required |
| Fresh attack calibration | 76 | 2000 | 152,000 | Scientific implementation and admission pending |
| Exploratory matrix | 232 | 2000 | 464,000 | Scientific implementation and admission pending |

The scientific draft totals **616,000 optimization steps**. Attack fitting,
qualification controls, evaluation, captures, checkpoint writing, provisioning,
installation and verification add work outside those counts. The 19-cell design
is in [R1_RESEARCH_PLAN.md](R1_RESEARCH_PLAN.md). Confirmation has reserved seeds
but no scheduled workload or cost commitment.

Use one GPU and serial runs for the first benchmark. Its full-size transformer
has eight blocks, hidden width 256, batch size 32 and sequence length 128. A
24 GiB GPU is a candidate to profile; no measured memory-fit claim exists yet.
Record actual device, software, peak allocated/reserved memory, step/evaluation
times, capture overhead, bytes written and total paid lifetime. The four-run
benchmark measures the reference configuration; higher-capacity cells need
additional profiling before the full matrix receives a budget.

## Advertised compute prices

| Service and offering | GPU-only USD/hour | Four GPU-hours |
|---|---:|---:|
| RunPod Secure Cloud RTX 4090, starting rate | 0.74 | 2.96 |
| Modal Functions L4 | 0.7992 | 3.1968 |
| Modal Functions A10 | 1.1016 | 4.4064 |

RunPod lists the RTX 4090 with 24 GB and Secure Cloud pricing from $0.74/hour.
Check the deployment quote before acceptance; availability and actual rate can
differ. [Official RTX 4090 page](https://www.runpod.io/gpu-models/rtx-4090).

Modal lists L4 at $0.000222/second and A10 at $0.000306/second. Functions also
charge CPU and memory: one physical core and 8 GiB add $0.111096/hour at the
listed rates. Those illustrative configurations total $0.910296/hour for L4
and $1.212696/hour for A10 before region or execution-mode multipliers. No
promotional credit is assumed. [Official Modal pricing](https://modal.com/pricing).

RunPod network volumes below 1 TB are listed at $0.07/GB/month: a continuously
retained 100 GB volume is $7/month. Container and pod-volume rates differ.
Budget retained storage after compute stops.
[Official Pod pricing](https://docs.runpod.io/pods/pricing).

## Sensitivity calculation, not a runtime prediction

`Training GPU-hours = 616000 × assumed seconds/step / 3600`.

| Assumed seconds/step | Training GPU-hours | At USD 0.74/hour |
|---:|---:|---:|
| 0.05 | 8.56 | 6.33 |
| 0.20 | 34.22 | 25.32 |
| 0.50 | 85.56 | 63.31 |

**These are hypothetical scenarios, not measured runtimes or total project
prices.** They exclude the additional work above and cannot authorize the
scientific sweep. Build the actual budget from measured per-cell/mode runtimes,
attack-fit costs and storage, with an explicit contingency and maximum spend.

The proposed micro limit is five minutes/256 MiB of outputs; the separate
performance-stage limit is 30 minutes/2 GiB. These are process limits, not
measured completion times. Thirty GPU minutes at the advertised RunPod rate
cost $0.37 for compute alone; setup, uploads and idle paid time are additional.
A process timeout does not stop a rented GPU. A cloud lifecycle controller must
enforce the accepted total paid lifetime and record termination independently.

Sparse training captures alone are estimated at roughly 13.3 GiB across the
scientific stages at width 256. This omits checkpoints, separate attack panels,
source/environment archives and varying widths. Actual storage admission follows
measurement; do not allocate an unreviewed large volume.

## Required remote admission packet

Before the first remote run, bind all of the following to the prepared bundle:

1. Provider, GPU, region, exact rate, maximum paid lifetime and total spend.
2. Immutable image digest, resolved package lock, CUDA/driver versions and CUDA
   availability; no silent CPU fallback.
3. Source/input hashes, exact command/stage, output limits and durable attempts.
   Engineering remains `scientific_evidence: false`.
4. A private reviewer retrieval destination, hash verification, preserved
   failures and independent cloud shutdown.

Current tooling prepares separate trusted-client and corpus-free remote packets,
uses authenticated two-process loopback execution, verifies actual frame bytes,
and gates engineering execution. The first GPU benchmark intentionally uses both
processes on one host; it is not a two-host latency result. Tooling does not
provision an account, create an immutable cloud environment, implement provider
billing shutdown or admit scientific runs. These are explicit remaining steps,
not assumptions hidden in the estimate.
