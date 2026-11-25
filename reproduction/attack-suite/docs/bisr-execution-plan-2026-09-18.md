# Prospective C-A checkpoint and execution plan

This plan applies to the fresh v3 input freeze and upstream-compatible single-seam
algorithm. It is implementation conformance, not published-setting replication.
No original SensiReplaced or dual-cut fidelity claim is implied.

Use pretrained `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, FP32 eager attention, TF32 disabled,
deterministic PyTorch operations and `CUBLAS_WORKSPACE_CONFIG=:4096:8` on koula.
Depths are 3 and 6; grants are public and exact; seeds are 31415, 27182 and 16180.
Every candidate/reference pair receives identical checkpoint and observation bytes.

## Checkpoint preparation

Capture public auxiliary activations once per depth on all 15,604 frozen windows.
Train six GRU checkpoints, one per depth/seed, for 20 complete epochs, batch 32,
AdamW lr .001, betas (.9,.999), epsilon 1e-8, weight decay zero. Initialise and
shuffle with the paired seed. Include the final partial batch; no early stopping
or selection by validation recovery. Record epoch losses only as training diagnostics.

For C-A's shared victim, train one full GPT-2 checkpoint per seed for **128 steps**,
batch **8**, AdamW lr **5e-5**, betas (.9,.999), epsilon **1e-8**, weight decay **.01**.
Sample frozen auxiliary windows with replacement using that seed; enable the
pretrained model's dropout during training. This recipe makes the previously
unspecified conformance victim concrete. Its training text is shared with SIP
auxiliary training; heldout validation remains disjoint. This choice is not a
claim of independent auxiliary/victim training corpora or a reproduction of v3
victim training. Both depths use the same trained victim bytes for a given seed.

Disable dropout for capture/replay. Capture each of the 256 validation windows
individually, next-token mean cross entropy over 31 labels, batch scale one,
full masks, no victim update. Replay with the **actual complete victim model**,
not the attacker public mimic, and require exact saved h/g equality. Export exact
client heads at both depths. Public attacks use pretrained head/downstream;
exact attacks use exported victim head and pretrained downstream. Neither
attacker receives the private victim downstream checkpoint.

The preparer may read truth to capture and replay; prediction must be a separate
process whose manifest contains observations and granted checkpoints only.
Prepare immutable checkpoint hashes first, then freeze the complete prediction
manifest before generating any candidate validation predictions. No automatic
fidelity pass follows successful preparation.

## Resources and admission

The excluded real-model canary uses seed 81273, the first auxiliary window,
untrained GRU weights and a deterministic exact-head perturbation. It is never
included in the scored validation set. The training timing canary uses seed
81274 and 18 GRU steps on 32 auxiliary windows, discarding two warm-up timings;
its checkpoint is discarded. The six-run projection uses all 58,560 planned
GRU steps, including partial batches conservatively as full-size steps.

Full-budget online timing covers both candidate and upstream SIP/b/f/bf work at
both depths and grants, with GM32 and SM800 per forward arm. Multiply the measured
per-document total by 256 documents and three seeds, then by 1.5. Add the 1.5×
training projection, preparation overhead and canary cost. The whole new campaign
must fit the existing **30 GPU-hour ceiling**. A failed projection blocks full
launch; do not reduce documents, seeds or iterations to fit after seeing timings.
Record a source amendment and retain attempts for infrastructure repairs.

The real-model canary is a mechanical/resource test. Remaining work before a
fidelity claim includes trained-checkpoint mechanical checks, metric parity,
label-isolated predictions, complete pinned manifests, all eight equivalence
endpoints for each prespecified depth/grant stratum, and measured compute.
