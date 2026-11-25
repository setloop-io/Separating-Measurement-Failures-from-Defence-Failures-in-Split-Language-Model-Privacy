# Environment report card — `quick` tier

- **battery**: `benchmark.v1` (the versioned standard; constants in attacker/benchmark.py)
- **target**: model `gpt2`, split `6`, device `cpu`, mode `finetune`
- **gate**: `frozen-gate.v2` (excess <= +1.0 pp over `label_free_majority`, z=2.5392, seeds >= 3)
- **overall grade**: **OPEN**

| axis | surface | grade | cells (P/F/I/err) |
|---|---|---|---|
| S1 | seam activations, forward (h* crossing to the body) | **OPEN** | 2 (0/2/0/0) |
| S2 | seam gradients, backward (dL/dh* returned by the body) | **UNMEASURED** | 0 (0/0/0/0) |
| S5 | GPU memory residue (freed buffers, allocator state) | **UNMEASURED** | 1 (0/0/0/1) |
| S9 | labels / properties (property inference over seam signals) | **UNMEASURED** | 2 (0/0/2/0) |

## S1 — seam activations, forward (h* crossing to the body): **OPEN**

| defense | attack | headline | excess | verdict |
|---|---|---|---|---|
| none | seam-inversion (vmi) | `seq_accuracy` = 0.75 | +89.029 pp | FAIL |
| mid-lite | seam-inversion (vmi) | `seq_accuracy` = 0 | +16.770 pp | FAIL |
worst cell: `seam-inversion` vs `none` (excess +89.029 pp, FAIL).

## S2 — seam gradients, backward (dL/dh* returned by the body): **UNMEASURED**

_(no cells)_

## S5 — GPU memory residue (freed buffers, allocator state): **UNMEASURED**

| defense | attack | headline | excess | verdict |
|---|---|---|---|---|
| - | gpu-memory | - | - | ERROR: cell rc=1: [gpu_memory_residue] no hardware probe tier available on this host (need pynvml or torch+CUDA; see attacker/hwprobe.py). Use --toy for the pure-python simulation tier. |

## S9 — labels / properties (property inference over seam signals): **UNMEASURED**

| defense | attack | headline | excess | verdict |
|---|---|---|---|---|
| none | label-scoring (norm) | `n_info_cells` = 20 | - | INFO |
| mid-lite | label-scoring (norm) | `n_info_cells` = 20 | - | INFO |

## Utility (defense cost)

| defense | final_loss | mean_loss |
|---|---|---|
| mid-lite | 8.574 | 9.466 |
| none | 5.215 | 4.938 |

---

host `MacBookPro`, commit `ec08486849760f2233f7036d44f580412976088f`, 2026-09-16T00:55:10Z

> Corpus caveat: results apply only to the recorded model, corpus, attacker access and budget. Read the artifact's provenance and evidence tier; synthetic or incomplete runs do not establish deployment security.

