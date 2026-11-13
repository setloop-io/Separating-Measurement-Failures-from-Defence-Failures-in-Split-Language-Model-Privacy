# Long-horizon defense matrix — GPT-2 @6/12 at 100 steps (Track C)

Rebuttal answer to "20-step cells are machinery-grade, not convergence"
(paper/paper_b/main.tex:877): the paper's defense-matrix arms re-run at
the framework's full 100-step tier with cross-epoch probes and drift
proxies, on the pinned WikiText-2 pilot corpus
(`results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl`, flattened to
`inputs/pilot-corpus.txt`; hashes in the pilot corpus manifest).
CPU-only, torch 2.14.0, threads capped at 4 per cell subprocess.

## What ran

- **Matrix**: 6 arms × {20, 100} steps × 3 seeds {1234, 2026, 7} = 36
  splitbench finetune captures (`--save-canonical --evidence`), each
  attacked by seam-inversion VMI (`--samples` = full horizon, public
  gpt2 head, linearization submode — the battery's arm) and
  label-scoring (`contains-token:220`), all `--evidence`-sealed.
  Arms pinned to the paper's operating points: `none`; `gaussian
  --sigma 15.0`; `sparsify` (default 10%); `mid-lite --bottleneck-dim
  64`; `santext --santext-p 0.3`; `at-lite --at-eps 15.0`.
  benchmark.v1's battery could not be used for these arms: it pins only
  gaussian σ=15 and would run mid-lite k=8 / santext p=0.1 / at-lite
  ε=0.1 (splitbench defaults) — not the paper's cells.
- **Cross-epoch probes** (`attacks/accumulation`, real-capture path,
  `--phase fwd`, E ∈ {64, 256, 768}, 3 seeds × 3 solve-seeds): on the
  single-epoch 100-step `none` capture (3200 wire rows) and on two
  purpose-built epoch-wrap captures (`inputs/pilot-corpus-epochwrap.txt`
  = 800-token prefix of the same corpus → 25 windows/epoch → exactly 4
  epochs per 100 steps; defenses `none`, `mid-lite-k64`). Canonical
  replay files built by `experiments/canonical_replay.py` (public head,
  position-aligned rows).
- **Aligned-victim re-probe** (`experiments/accumulation_aligned_probe.py`)
  — see the framework note below.
- **Synthetic tier** (clearly labeled, secondary): `leak-accumulation
  --toy --epochs 50`, `subspace-joint --toy --epochs 50`.
- **Pilot/timing**: `--benchmark quick` (20-step, battery defaults) ran
  first for sizing (36 s wall for the whole tier) and was re-graded
  torch-free with `--benchmark-grade` (overall OPEN; S5 scrub-check
  errored fail-closed: no pynvml/CUDA on this host).
- Cells: 121 journaled in `matrix/run-index.jsonl`, **0 non-zero rc, no
  timeouts** (budgets sized from the pilot, not luck). 110/110 evidence
  manifests verify (`attacker.evidence.verify_evidence`). Framework
  scorecard over the 72 attack artifacts: 2232 rows, 2190 FAIL, 6 PASS
  (the six mid-lite-k64 pooled-summary rows, at the statistical floor —
  see the convention note below), 36 INFO, 0 not-shown
  (`matrix/scorecard.json`).

## Token recovery vs majority control (pooled over 3 seeds)

Excess = Bonferroni-Wilson upper-95 minus control; v2 = frozen-gate.v2
(z=2.5392, from the artifacts), v1z = recomputed at the paper's
frozen-gate.v1 z=2.62. Control = 0.00% everywhere (the attacker's
synthetic-calibration majority token never occurs in WikiText-2 prose),
so raw gap equals recovery. Paper column: 20-step reference from
paper_b "Defence matrix" (external: WikiText-2 first-steps corpus,
v1 gate, fleet, battery samples=8 budget) — not a paired run.

| arm | steps | recovery | excess v2 | excess v1z | paper 20-step |
|---|---|---|---|---|---|
| none | 20 | 58.18% | +63.02 | +63.17 | +86.0 |
| none | **100** | **56.68%** | **+58.89** | **+58.95** | |
| gaussian-s15 | 20 | 0.89% | +2.42 | +2.50 | +9.49 (loss 8.6) |
| gaussian-s15 | **100** | **2.84%** | **+3.69** | **+3.72** | |
| sparsify-10 | 20 | 19.27% | +23.52 | +23.67 | +55.32 |
| sparsify-10 | **100** | **23.08%** | **+25.03** | **+25.09** | |
| mid-lite-k64 | 20 | 0.00% | +1.00 | +1.06 | +2.08 (loss 7.6) |
| mid-lite-k64 | **100** | **0.00%** | **+0.20** | **+0.21** | |
| santext-30 | 20 | 41.15% | +46.15 | +46.31 | +66.08 |
| santext-30 | **100** | **40.45%** | **+42.67** | **+42.74** | |
| at-lite-e15 | 20 | 58.33% | +63.17 | +63.32 | +86.17 |
| at-lite-e15 | **100** | **56.79%** | **+59.00** | **+59.07** | |

Δ recovery (100 − 20 steps): none −1.50 pp, at-lite −1.54 pp, santext
−0.70 pp, gaussian +1.95 pp, sparsify +3.81 pp, mid-lite ±0.00 pp.

**Reading.** No arm changes its summary verdict class with the horizon:
the four leaking arms FAIL at 20 and at 100, and **mid-lite-k64 PASSes
the gate at the floor at both horizons** — 0.00% recovery, excess
+1.00 pp (20 steps, n=640) and +0.20 pp (100 steps, n=3200) ≤ +1.0 pp,
`at_floor=true`, on all 3 training seeds (the only 6 PASS rows in the
scorecard's 2232; the paper's 20-step fleet cell had it at +2.08 FAIL,
"the only cell approaching the gate" — under this corpus/budget it
crosses). Convention note: the battery's worst-cell enrichment (max
excess over per-step rows) still FAILs every arm including mid-lite —
at n=32/step the Wilson half-width alone exceeds the gate, so "PASS"
lives only at the pooled-summary level; both conventions are reported
by the framework and quoted here. The defense *ranking* is
horizon-invariant and matches the paper's ordering exactly
(bottleneck > noise > sparsify/sanitise ≫ PCA-subtraction ≡
undefended). Absolute excesses differ from the paper's fleet numbers
(corpus slice, gate z, attack budget n=640/3200 vs 256, CPU vs GPU) —
the paired same-corpus 20-step cells are the honest comparison, and
they move by ≤ ±3.8 pp at 5× horizon (seed sd 0.0–2.2 pp).

## Within-horizon drift (h100, recovery pooled over seeds)

Per-step cells windowed [0–7] / [8–19] / [20–49] / [50–99]:

- none: 58.9 → 57.7 → 56.2 → 56.4% (flat-to-declining as the body trains)
- at-lite-e15: 59.0 → 57.9 → 56.0 → 56.7% (≡ none at every window)
- santext-30: 41.4 → 41.0 → 39.4 → 40.8% (flat)
- sparsify-10: 17.5 → 20.5 → 20.9 → **25.9%** (creeps up with training)
- gaussian-s15: 0.3 → 1.3 → 1.8 → **4.2%** (noise erodes measurably but
  stays far below gate: window excess +5.0 pp at 50–99)
- mid-lite-k64: 0.00% in every window (excess +0.1–0.8 pp, at floor)

The critic's mechanism is real but small here: two defenses (sparsify,
gaussian) show upward leakage drift over prolonged fine-tuning; neither
changes verdict within 100 steps; mid-lite does not move at all.

## Cross-epoch accumulation probes (real captures, fwd phase)

Victim-token top-1 of the polar-lstsq seam-map solve vs label-free band
(0.0% everywhere here). Framework module and aligned re-probe agree.

| capture | epochs | E=64 | E=256 | E=768 | pooled≈within? |
|---|---|---|---|---|---|
| none h100 (full corpus) | 1 | 7.9% | 38.2% | 28.9% | yes (same-epoch partitions) |
| none epochwrap | 4 | 19.8% | 50.3% | 19.5% | yes (Δ ≤ ~3 pp) |
| mid-lite-k64 epochwrap | 4 | 0.0% | 0.0% | 0.5% | yes |

Descriptive reading: the labeled-pairs accumulation attacker recovers
substantial victim tokens on the undefended seam well above the band,
and cross-epoch pooling does **not** collapse the solve — expected on
splitbench, which applies no per-epoch rotation (the ER-ratchet world
the pooling control was designed for is the toy tier). mid-lite's
bottleneck holds at 0% at every E even with 4 epochs of pairs.
E=768 non-monotonicity (full-rank boundary; polar of a near-square
lstsq) is a solve-conditioning artifact, recorded not smoothed.
Caveat: recovery here uses the module's NN-mean surrogate decoder (its
documented real-run decoder injection point is not wired in the capture
path) — machinery-grade magnitudes, informative within/pooled
relationships.

**Framework note (found, not patched).** `attacks/accumulation.py`'s
real-capture path takes victim labels as `canon_tok[off-nv:off]` — the
LAST epoch's tail — while the victim wire rows are the FIRST epoch's
tail; for multi-epoch captures the labels are misaligned by
construction. It is numerically inert on these runs (epochs are exact
wraps, so epoch tails repeat the same windows and labels coincide —
verified: framework vs aligned outputs match within ~2 pp at every E),
but any capture with a partial final epoch would score garbage toward a
false PASS. `experiments/accumulation_aligned_probe.py` carries the
fixed-label variant (labels from epoch 0's canonical-token tail) using
the framework's own `solve_w`/`recovery_with_what_nn` primitives.

## Drift proxies (descriptive, never inferential)

splitbench records no held-out CE/PPL and no parameter-drift field;
those exist only in the `attacker.experiment` driver
(`client_parameter_change_l2`, held-out eval), which does not drive
benchmark cells, and splitbench saves no checkpoint to re-derive them
post-hoc. **Reported instead**: per-step loss journals (fail-closed,
crash-safe) and the epoch-wrap memorization proxy.

- 100-step loss trajectories (step0 → final, mean): none 4.87→4.41
  (4.67); gaussian 11.52→7.47 (8.23) — utility destroyed as the paper's
  loss-8.6 cell predicts; mid-lite 14.12→8.07 (8.30) — random-projection
  head retraining, final ≈ paper's 7.6; santext 5.86→5.36; sparsify
  6.41→5.58; at-lite 4.91→4.42 (≡ none).
- Memorization proxy (epoch-wrap, 4 passes over the same 25 windows):
  none loss step0 5.56 → final 0.54 (per-epoch means 3.95/2.55/1.73/
  1.14) — the body memorizes the repeated corpus fast; mid-lite-k64
  step0 14.43 → final 5.44 (epoch means 8.37/6.64/6.02/5.61). Yet the
  per-window recovery profile above does not grow across the horizon —
  memorization of the corpus by the body does not, within 100 steps,
  translate into rising seam inversion on these arms (descriptive).

## Wall-clock (this host; 4 threads/cell, shared with two other agents)

Total 2377 s for 121 cells. Captures 2096 s: h100 mean 33–36 s per cell
for five arms; **at-lite 368 s** (per-step EMA-PCA on CPU; h20 77 s).
Attacks 172 s total (VMI ≤ 4.5 s at samples=100; label-scoring ~1 s).
Replays 10 s; probes 78 s; synthetic tier 21 s; pilot quick tier 36 s.
Per-cell times are in `matrix/run-index.json{l}`.

## NOT run, and why

- **v13 composite** — belongs to an external project, out of scope for
  this framework rebuttal (as flagged in the track brief).
- **Multi-hundred-step horizons (>100)** — 100 is the framework's full
  tier (benchmark.v1 steps cap); longer horizons are a new tier, not
  this rebuttal's commitment. The h100 drift windows above are the
  offered evidence about horizon stability.
- **gradient-inversion TAG (S2 axis)** — battery's full-tier-only
  attack; skipped to fund 3 seeds × 6 arms × 2 horizons inside the
  ~80-min shared-host budget. The paper's defense matrix is a VMI table;
  S2 remains UNMEASURED at 100 steps (as the quick re-grade shows).
- **gpu-memory scrub-check (S5)** — host has no pynvml/CUDA; the pilot
  battery journaled the cell fail-closed (error + placeholder artifact),
  recorded not hidden.
- **custext, snd-lite arms (the battery's 7th/8th defenses)** — not in
  the paper's reported matrix; the brief prioritizes the paper's arms
  and ≥3 seeds over 1-seed breadth.
- **held-out PPL before/after, client_parameter_change_l2** — no
  splitbench machinery (see Drift proxies); adding it would mean code
  outside `experiments/` plus a checkpoint path, explicitly avoided.
- **`--benchmark full` as-is** — battery knobs ≠ the paper's operating
  points for 3 of 6 arms; the scripted loop reproduces the paper's
  matrix exactly (mirroring `benchmark.py:_capture_argv`).

## Artifacts

- `matrix/captures/<arm>_s<seed>_h<steps>/` — wire captures +
  canonical.pt (+ canonical-replay.pt on probe targets)
- `matrix/artifacts/` — 38 splitbench + 72 attack artifacts (all
  evidence-sealed), 3 framework + 3 aligned accumulation probes, 2
  synthetic-tier artifacts
- `matrix/scorecard.json` / `scorecard.txt` — framework scorecard
- `matrix/run-index.jsonl` / `run-index.json`, `matrix/run-manifest.json`
  — per-cell rc + wall-clock journal, run manifest
- `report.json` — this report's tables, machine-readable
- `pilot-quick/` — the `--benchmark quick` timing pilot + re-grade
- Drivers: `experiments/run_long_horizon.py`,
  `experiments/canonical_replay.py`,
  `experiments/accumulation_aligned_probe.py`,
  `experiments/analyze_long_horizon.py`
