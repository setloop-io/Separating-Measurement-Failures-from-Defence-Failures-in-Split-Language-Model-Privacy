# Native VFLAIR-LLM reproduction harness

The fixed launch command is:

```bash
bash experiments/vflair_native_replication_20261022/entry.sh
```

The separate fixed-eight-book study is prepared with launch blocked; see
[BOUNDED-STUDY-PREPARED.md](BOUNDED-STUDY-PREPARED.md). Its six profiles require
an explicit user scope/allocation choice and a new frozen authorized child.

Freeze `run.json` in an ORX experiment snapshot before launching. `phase` is
`admission`, `canary`, or `full`; `condition` is `mid`, `none`, or `at`. Each run
uses a fresh directory under `output_root/ORX_RUN_ID`; an existing directory is
an error. There is no checkpoint resume or reuse of a previously trained model.

## Admission

Set `host.model_path` to the pinned GPT-2 snapshot, `host.model_revision` to
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, and `host.dataset_manifest` to the
input preparer's LAMBADA `manifest.json`. Its schema is:

```json
{"repository":"cimec/lambada","revision":"5953bd97664b64b95754f299b2309ecfbfbe81b9","splits":{"train":[{"path":"/absolute/train.parquet","sha256":"..."}],"test":[{"path":"/absolute/test.parquet","sha256":"..."}]}}
```

Set `bootstrap=true` to create a task-private environment. This environment
appends the existing host virtual environment's site-packages directory via a
recorded `.pth` file, and keeps private compatibility packages first. Both the
path and version of the inherited CUDA PyTorch must match before and after pip.
The base/shared environments are never modified. The requirements deliberately
retain host PyTorch instead of installing upstream's CUDA 11.8 build on GB10.

Set `prepare_inputs=true` when the snapshot includes `prepare_inputs.py` and
`inputs/lambada-{metadata,tree}.json` from the pinned input-preparation task.
Bootstrap and input preparation each have a setup timeout and streamed logs.
The parent task owns downloads, environment installation and all compute
launches through ORX.

Admission checks all 254 upstream file hashes; records host package versions,
CUDA device and free space; hashes model and dataset assets; validates required
`text` and `domain` fields; then **imports the actual native pipeline**. Import
failure produces a traceback and BLOCKED status. Admission never constructs a
neural model or reports privacy/utility results.

Optional `count_native_windows=true` tokenizes the complete admitted corpus on
CPU, using the literal retained LAMBADA start-offset/length/break rule. It emits
per-document counts, native batches per epoch and the maximum 30-epoch step
count, plus the untruncated native test-input length range. This is a workload
preflight, not training. The full source has 2,662 training **books**, rather
than 2,662 training examples after windowing.

## Native mechanisms and compatibility boundary

The source pin is `FLAIR-THU/VFLAIR-LLM` commit
`ff18565d89dce6367b6f13881a415c053b60cb0f`. A run-owned source copy is made only
after verifying the pinned manifest. `import_compat.py` removes unused vision,
Qwen/Paillier, alternative-model, unrelated-defence and remote-RPC imports,
narrows the loader registry to GPT-2, and retains the selected native attacks.
Every top-level native function and class AST is checked for exact preservation
before and after this import isolation. Every changed module has before/after
hashes and explicit edits in `compatibility.json`. Missing unrelated symbols
are unavailable; they are never replaced by dummy implementations.

The harness calls the original configuration loader, party constructors,
GPT-2 partition pipeline, `create_main_task`, native `train_vfl`, original
head/tail optimizers and schedulers, native MID/AT forward/backward paths, native
inference and native attacks. It supplies run-owned paths through the original
configuration fields and `party_utils.Trained_Models_Save_Path`. A data-I/O
adapter returns the exact verified Parquet splits at the native loader call;
native prompts, windows, tokenization, CLM training labels and next-token test
labels remain unchanged. Canary subsets are explicitly indexed in the receipt.

Observation wrappers log results **after the original method returns**; they do
not change losses, updates, token IDs, native RNG sampling or attack iterations.
The native algorithm/optimizer trajectory is retained, subject to the recorded
modern ARM/CUDA software environment and the upstream native CUDA settings
(`cudnn.deterministic=True` and `benchmark=True`). Numerical identity with the
authors' hardware is not asserted.

## Canary and complete profile

`canary` invokes the native `epochs==1` branch, which executes **one minibatch**,
then native final utility evaluation and the native VMI/WBI attacks on the
declared small test subset. Default length512 and batch16 are retained. Attack
iterations and sample count are reduced only in this engineering profile.
Canary values measure correctness/timing; they are not full benchmark results.

The explicit `canary.native_epochs=2` warm-timing profile exercises two ordinary
native epoch loops on the declared canary subset. With the first admitted book
(258 native windows) and batch16, this has a maximum34 updates, including two
native LR-decay/early-stop checks. `profiles/warm-{mid,none}.json` fix those
conditions, two utility rows, one target per attack and16 attack iterations.
The default remains one native epoch/one minibatch. Warm-timing outputs are
engineering evidence excluded from benchmark result tables.

`full` retains maximum30epochs, batch16, sequence/window512, main Adam LR0.001,
trainable head **and** tail, and the original early-stop threshold1. The native
early-stop code compares the final minibatch loss with the best earlier value;
its nonimprovement counter is cumulative and never reset. The harness preserves
that rule and reports the actual final epoch and executed optimizer steps.
Native LinearLR initialization/decay and parameter-group learning rates are
recorded as executed, rather than assuming that every update uses LR0.001.

The complete attack profile retains VMI400iterations/LR0.001 and WBI300iterations/
LR0.005/temperature0.05, each on200 native-randomly selected test passages,
with their original default `cross_entropy` objective. No MSE/100-step local
adaptation is substituted. Native VMI/WBI optimize through the original client
head; the harness preserves their native attacker access and does not relabel
them as adaptive known-defence attacks.

Full execution requires explicit frozen `maximum_training_windows`,
`maximum_test_rows` and `maximum_optimizer_steps` guards derived from admission,
plus a wall-time budget. Exceeding a guard or incomplete prediction coverage
terminates the run; partial evidence is retained and never marked complete.
The wall guard covers the child pipeline; setup has separate bounded guards.

The `mid` condition is the retained LAMBADA/GPT-2 MID config. The `none`
condition removes only its defence block and uses matching data, model,
training, seed and attacks. `at` transfers the original native CodeAlpaca AT
classes/hyperparameters to the GPT-2/LAMBADA training condition, with sequence
length512; this is a declared cross-dataset mechanism condition. A separate
CodeLlama/CodeAlpaca published-setting reproduction requires its own frozen
configuration and model/data assets. MID/no-defence are the first comparison.

BiSR remains excluded by default. Its native source first trains NaMoE experts,
gate and full model on auxiliary training windows, then runs reconstruction.
It has20 expert epochs,15 gate epochs,4 full epochs, auxiliary100 and200 attack
passages in the retained config, and caches the trained inverter under a name
that omits defence/seed. A fresh run-owned source prevents cross-condition
cache reuse. A separate explicit admission is required before enabling its
larger workload; it is never approximated with the local adapted BiSR route.

## Evidence and scoring

`steps.jsonl` contains every native training update, loss, train accuracy,
actual optimizer groups/LRs, timing and CUDA memory. Initial/final module state
hashes and trainable counts demonstrate which native components changed.
Native model/defence checkpoints, final optimizer states/RNG and checkpoint
hashes are retained. Native `training_time` is retained verbatim; the native
counter covers only the final executed epoch, so harness wall times are
reported separately.

`utility-predictions.json` retains predicted and target final BPE-token IDs plus
the native LAMBADA `best_pred` accuracy. This metric is not word-normalized
exact-match or perplexity. MID's native evaluation remains stochastic.

For each attack, the original transmitted wire and attention mask are exported
without precision conversion, together with the original input IDs, native
sample-selection receipts and reconstructed IDs. The tokenizer decode observer
delegates to the original decode function and reads its caller's already
computed IDs. Native token-membership precision/recall are independently
recomputed from these IDs and must match the native return within1e-10. An
additional positional/exact-document score excludes native pad positions when
the original and reconstructed lengths match. A mismatched-length positional
metric is explicitly undefined. All native outputs remain available separately.

Completion requires every expected prediction, successful independent scoring,
native checkpoint files and the terminal success marker. No paper or codebase
brand/runtime changes are part of this harness.
