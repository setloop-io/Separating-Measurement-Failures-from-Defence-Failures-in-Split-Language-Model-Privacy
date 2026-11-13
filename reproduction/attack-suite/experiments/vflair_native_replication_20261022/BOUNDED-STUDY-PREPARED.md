# Prepared bounded native study — launch blocked

Six profiles are prepared under `profiles/bounded-{mid,none}-seed{11,22,33}.json`.
Every profile states `NOT_AUTHORIZED_TO_LAUNCH`; wall and aggregate allocation
fields are unset. No experiment, commit, model job or manuscript change is made
by this preparation. Scope and a fresh allocation remain pending.

The dedicated `bounded-study` phase uses source-book indices
`198,706,943,960,989,1158,1444,2372`, preserving native relative source order.
The shared hash-bound selection receipt pins the corrected proposal, complete
dataset shards/order, model/tokenizer files and original MID configuration.
Execution records text/domain hashes for the actual eight source rows. It keeps
the exact original 5,153-row test dataset object. Native preprocessing must yield
exactly 1,655 windows, 104 batches/epoch and 3,120 maximum updates before training.

The native configuration retains FP32, trainable head/tail, maximum30epochs,
early-stop1, original MID and matched NONE, VMI400/WBI300 and200 targets each.
The selected passive-party path supplies effective `std_shift=5`; the MID class
constructor's standalone default0.5 is overridden by that original native call.
No native scientific function or class body is edited.

Authorization is checked before bootstrap/model imports by the launcher,
direct runner and execution API. A persistent study marker prevents a phase
override from bypassing the block. Future authorization requires recorded scope
and allocation references, a finite positive aggregate allocation, and an
approved positive per-run wall guard. The parent ORX scheduler must enforce the
aggregate allocation across children; this code enforces each run's wall/step
guards. An earlier48-hour batch is not reused automatically.

Native final checkpoints remain after the original stopping/cap endpoint.
Timeout during training may leave no final model. Exact uninterrupted resume is
not implemented, and guarded partial runs are excluded from efficacy results.
Independent verification requires the frozen selection receipt and exact counts
in addition to the existing full utility/attack/checkpoint checks.

Validation: `python3 experiments/vflair_native_replication_20261022/test_bounded_study.py`
passes13 CPU/std-library fixtures. They cover all six blocked profiles, direct
runner/API/phase-override rejection, source/hash/order guards, exact selection,
complete test preservation, original native budgets, and old full/canary
configuration **and data-selection outputs** against the frozen timing source.
No model or CUDA execution occurs in these fixtures.

Before any launch, freeze a new child with the recorded user choice/allocation
and its authorized profile. The optional seed60-first choice requires its own
frozen profile. A categorical utility-preserving protection/PASS statement
requires a predeclared operational utility criterion; continuous native
privacy/utility reporting is already specified by the proposal.
