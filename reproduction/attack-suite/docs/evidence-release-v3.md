# Evidence release and re-verification guide: validation-heldout-v3

15 September 2026 · Repository state `main` @ `e4d85a6` · §4 aligned to
the post-review manuscript revisions of 2026-09-15 (see §4.3)

This document is the release and provenance record for the completed
held-out validation campaign `validation-heldout-v3` (protocol v3, 16
shards). It states where every artefact lives, how a second party can
re-verify integrity and re-score predictions, how the published headline
numbers map to files, the full execution and restart history as far as
the records establish it, the known defence-provenance merge hazard, and
the access route. It deliberately keeps three operations apart:

1. **File-integrity verification** — checking recorded SHA-256 digests
   and hash-chained evidence journals against the bytes on disk. This
   establishes that files are unaltered, not that the science is correct.
2. **Prediction rescoring** — re-deriving document-level metrics from the
   stored prediction transcripts. This checks the reported metrics
   against the saved predictions without touching model weights.
3. **Prediction regeneration** — re-deriving the predictions themselves
   from the raw captured tensors, or re-executing the campaign. This is
   the only operation that tests the full pipeline, and it requires the
   binary capture tier. It has **not** been performed as part of any
   review to date.

## 1. Anchors

| Item | Value |
|---|---|
| Frozen protocol | `experiments/validation_protocol.v3.json` |
| Protocol SHA-256 | `a94bbd307455aba89ba01f192617147dac951b61ec61de1c4c144acf6d97b2c5` |
| Effective-protocol SHA-256 (canonicalised) | `b3d18d65e9bd51348259268e3d4389694273d6d985107514b628cb21a4e55565` |
| Frozen tokenized input SHA-256 | `d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a` |
| Input file | `results/validation-corpus-v1/validation-input.json` (copied into each shard directory as `input.json`) |
| Protocol status / version | `frozen`, `version: 3` (analyser requires `--protocol-version 3`) |
| Freeze commit | `064a64f` ("Freeze protocol v3 …"), checked out in the execution clones at 06:30 (koula) and 06:35 (toula) +0100 on 2026-09-10, before any campaign run |
| Evidence commits | `70e5686` (14 complete + 2 in-progress shards), `8663119` (all 16 finalised); merged via `f7efc50`; current HEAD `e4d85a6` |
| Cell matrix | 3,680 logical cells = 3,200 physical executions + 480 inference cells reused from identical training-mode executions |
| Document records | 942,080 = 3,680 cells × 256 held-out evaluation documents |
| Arms | 40 (28 undefended + 12 defended: 6 × `@replica-median`, 6 × `@secret-mixup`) |

## 2. Artefact locations

### 2.1 Two evidence tiers

**JSON tier (in this repository, tracked since commits `70e5686` /
`8663119`).** `results/validation-heldout-v3-r3-shardN/` for N = 0–15,
each containing:

- `validation.json` — the shard report (status `COMPLETE`; 200–280 cell
  records: 200 training-mode cells plus, on ten of the sixteen shards,
  40 or 80 inference-mode cells marked `measurement_reused`, which share
  their training-mode origin's prediction transcript);
- `validation.json.evidence.jsonl` — the hash-chained event journal
  (`run_start`, one `cell_complete` / `cell_reused` event per cell,
  `run_end`: 202 events on shards with no reused cells, up to 282 on
  shards 7 and 15);
- `validation.json.evidence.json` — the chain manifest (schema
  `llm_attacker.evidence.v2`);
- 200 distinct cell prediction transcripts (`<cell_id>.json`, 256
  documents each);
- `input.json`, `protocol-source.json`, `protocol-effective.json`
  (byte-identical across all 16 shards);
- per-defended-victim JSON records under `defended-<defense>-<seed>-<depth>/`
  (`run.json`, `run.json.jsonl`).

No `*.pt` files exist anywhere in the repository shard directories
(verified by recursive search). The repository therefore supports
verification (§3.1) and rescoring (§3.2), but not regeneration (§3.3).

**Binary tier (off-repository, two LAN GPU hosts).** The full shard
directories — including the `.pt` capture tier stripped from the
repository — are held on:

| Host | Address | Shards executed there | Path |
|---|---|---|---|
| `toula` | 192.168.1.107 | 0–7 | `~/llm-attacker-pilots/gpt2-wikitext2-v01/code/results/validation-heldout-v3-r3-shardN/` |
| `koula` | 192.168.1.108 | 8–15 | same |

The execution host of each shard is recorded in its evidence manifest
(`hostname` field) and matches this split. Following post-campaign
synchronisation, **both hosts now hold a full copy of all 16 final shard
directories**; the `.aborted` directories (§5) exist only on the host
where the aborted attempt ran. Each final shard directory occupies
≈19 GB on disk; the hash-referenced content is ≈13.0–13.6 GB per shard.

### 2.2 Recorded per-shard references

Each `validation.json` records SHA-256 digests for:

- `artifacts`: 60 top-level binary artefacts — `public-<seed>-<depth>.pt`,
  `exact-<seed>-<depth>.pt` and `observations-<seed>-<depth>.pt` for all
  10 victim seeds × 2 depths (the victim head exports and the
  evaluation-document observation captures). The same 60 names appear in
  every shard and hash **identically** across shards (verified across all
  16 reports; the merge script refuses a conflict);
- `cells[*].prediction_sha256`: the 200 prediction transcripts;
- `cells[*].defense_provenance` and `cells[*].defended_observations`:
  defence sidecars, where the cell is a defended arm (see §6);
- the protocol/input triple, bound by `protocol_sha256`,
  `effective_protocol_sha256`, `input_sha256` and
  `supplied_input_sha256` in the report header.

### 2.3 Report digests (primary anchors)

SHA-256 of each shard's `validation.json`. These were recomputed on the
execution hosts on 2026-09-15 and match the repository copies byte for
byte; a second party should anchor on these digests before verifying
anything else.

| Shard | `validation.json` SHA-256 | Host |
|---|---|---|
| 0 | `b0daba6d0c99c083822372aea93e555c6cccc065de14b4020193fb9af5ba3c96` | toula |
| 1 | `54a0204565bbc029f3d54de01b6ebbb625c8f842a867af3e90ec446dbeb1890e` | toula |
| 2 | `36f090b90854811a0391bc45ce93e6b272486feb4bd92b84f8fbf792a7338922` | toula |
| 3 | `718f6a02c751732e4203ef1ff79faccd53b56a0a72eff32920e5710e74f2a557` | toula |
| 4 | `19d261f998e563a73d03030dc76c2c637b2ab19604bb309e08048d3e9e24fc16` | toula |
| 5 | `4cd2648bbb78a7775a847135570db57b24f6d5e2d77f0a1575a499e687907baf` | toula |
| 6 | `2ce8c06acd1192481755daf6c2536ffd88876c4886475b100d3cb08693c67de9` | toula |
| 7 | `48779e7db6d5eee8a366be77cbd73bde86ec318fb5706f878d0a39e535435061` | toula |
| 8 | `121ab4981766654a309bd56ba79ee14b191e65d81d53fc3051e70b7445e18e1f` | koula |
| 9 | `cfcc883e3f6ec94a9725ec30fad2c19bee8531c14b360bb85360417651a4e14c` | koula |
| 10 | `52762a608035df5355b56067186f1242707490817fd1e55ef640f6f1a1630394` | koula |
| 11 | `3958c3e658a263bb083e9ad968653f288ceea45598bbd9b24ff3c375656db154` | koula |
| 12 | `c6ae73ee19b11517238f574bcc8ff2f3eb5873352401b2ee8e98af0c4d2055cc` | koula |
| 13 | `7954f9c8f1708e0965efb7b058b9e8b76115f49cda31ad08af9a65be062fa0d7` | koula |
| 14 | `a822df5d93b0d4ae2cfe6bd0a2e8b11a9547eb096ab730a35ad2de5f08d24dd0` | koula |
| 15 | `813356763aed1e184c4282ee565747dec249b5aef24eb03ce0c7e41d4092d140` | koula |

## 3. Re-verification recipes

All recipes assume a checkout of this repository at `e4d85a6` or later,
Python 3 with the repository root importable (the project virtual
environment `.venv` suffices; the evidence and metric modules are
stdlib-only apart from the analysis dependencies).

### 3.1 File-integrity verification (hashes and evidence chains)

**Evidence chains (JSON tier, works in this checkout).** Each shard's
journal is hash-chained: every event commits to its contents and the
previous event's hash, and the manifest re-commits to the terminal chain
hash, the artefact digest and the run metadata. Verify one shard with:

```bash
python3 -c "
from attacker.evidence import verify_evidence
ok, problems = verify_evidence('results/validation-heldout-v3-r3-shard0/validation.json.evidence.json')
print(ok, problems)"
```

`verify_evidence` returns `(True, [])` on success and a human-readable
problem list otherwise. Repeat for all 16 shards. All 16 chains verify
as of 2026-09-15.

**Recorded file hashes (binary tier, requires host access).** On each
host, every reference in the shard report can be re-hashed. The
reference implementation used on 2026-09-15 is committed at
`outputs/remote-evidence-check-2026-09-15/verify_remote.py`; it anchors
on the repository copies' report digests (§2.3), streams each shard's
recorded references on the source host over SSH, re-hashes them, and
re-verifies the chain, all read-only. Its result
(`outputs/remote-evidence-check-2026-09-15/verification.json`): **all
5,008 referenced files passed** — 960 raw artefact references
(60 × 16), 3,200 prediction files, 400 defence-provenance sidecars,
400 defended-observation files, and 48 protocol/input files (3 × 16) —
with all 16 evidence chains verifying and all 16 on-host reports
byte-identical to the reviewed repository copies.

A minimal manual spot check (used independently on 2026-09-15 for the
two longest-running shards) is:

```bash
ssh -o BatchMode=yes toula 'python3 - <<EOF
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0, "/home/geo/llm-attacker-pilots/gpt2-wikitext2-v01/code")
from attacker.evidence import verify_evidence
p = Path("/home/geo/llm-attacker-pilots/gpt2-wikitext2-v01/code/results/validation-heldout-v3-r3-shard7")
r = json.loads((p/"validation.json").read_text())
bad = [n for n, h in r["artifacts"].items()
       if hashlib.sha256((p/n).read_bytes()).hexdigest() != h]
print("mismatches:", bad)
print(verify_evidence(str(p/"validation.json.evidence.json")))
EOF'
```

Result for shard 7 (toula) and shard 15 (koula): 60/60 recorded
artefacts hash-verified, 0 mismatches, chain verifies — consistent with
the earlier spot check of the same two shards and with the full
5,008-file verification.

### 3.2 Prediction rescoring from stored cells

Rescoring re-derives every document-level metric from the stored
prediction transcripts and the frozen evaluation documents, and compares
against the reported cell metrics. It needs only the JSON tier.

- The reviewer-independent audit, `outputs/validation-v3-review-2026-09-15/audit_results.py`
  (run from the repository root: `.venv/bin/python outputs/validation-v3-review-2026-09-15/audit_results.py`),
  recomputed all **942,080** document records with the repository's
  metric definitions (`attacker.reconstruction_metrics.reconstruction_metrics`)
  and found an exact match; it also re-ran the crossed bootstrap with the
  prespecified seed and reproduced every published interval
  (see `outputs/validation-v3-review-2026-09-15/verdict.md`).
- The official analyser performs the same rescore as a gate:
  `experiments/analyze_validation.py:validate_report` re-derives each
  cell's metrics from its prediction file and refuses the analysis on
  any mismatch (`metric mismatch`). It additionally requires the binary
  tier (it hashes every artefact), an external manifest anchor
  (`--expected-manifest-sha256`), and `--protocol-version 3`, so it can
  only run against a complete merged package (§6).

### 3.3 Prediction regeneration from raw captures

Regeneration is **not** a packaged operation today and has not been
performed by any reviewer. Two distinct meanings should not be conflated:

- *Replay from stored tensors.* The binary tier retains per-setup head
  exports (`public-*.pt`, `exact-*.pt`), evaluation-document observation
  captures (`observations-*.pt`), and defended-capture wires
  (`defended-*/captures/wire_*.pt`). Re-deriving predictions from these
  without retraining would require new replay code; none exists in the
  repository at `e4d85a6`.
- *Full re-execution.* The exact command line for each shard is recorded
  in its evidence manifest (`argv`), e.g. shard 0:

  ```
  experiments/validate_reconstruction.py --protocol experiments/validation_protocol.v3.json \
    --input results/validation-corpus-v1/validation-input.json \
    --input-sha256 d79ef4fc3b27dbac21fd731f80873a5afc38481bc7a03acb0ee9fec1587bc21a \
    --device cuda --shard-count 16 --shard-index 0 --output results/validation-heldout-v3-r3-shard0
  ```

  Re-execution constitutes a new run (new `run_id`, fresh output
  directory, new journal entry) and is the only end-to-end regeneration
  route currently defined. Runtime per shard is ≈9–26 GPU-hours on the
  campaign hardware (torch 2.11.0+cu128, CUDA 12.8).

## 4. Table-to-file index

All headline numbers in both papers derive from the 16 shard reports and
their prediction transcripts, aggregated by the reviewer's committed
CSVs under `outputs/validation-v3-review-2026-09-15/`:

- `arm_summary.csv` — one row per (depth, knowledge, mode, arm): pooled
  means over that condition's cells;
- `cell_summary.csv` — one row per logical cell (3,680), with
  source-shard references;
- `contrasts.csv` — the 48 prespecified paired contrasts (1 primary +
  47 secondary) with crossed-bootstrap intervals.

This index matches the manuscripts **as revised on 2026-09-15 following
the adversarial review** (revisions in the working tree, pending commit
at the time of writing). Every value below was re-verified against the
CSVs on that date and matches the revised texts at displayed rounding;
§4.3 records what changed between the reviewed and revised manuscripts.

### 4.1 Paper A (`paper/paper_a/main.tex`)

The revised manuscript carries three tables; only the first two draw on
this campaign.

| Number in paper | Value | Source rows |
|---|---|---|
| Table 1 `tab:joint_reconstruction` (pooled token accuracy, L=3 / L=6) | backward 24.04 / 50.01; SIPIT-128 98.17 / 96.69; forward k32 97.60 / 94.51; joint-shuffled k32 97.46 / 94.21; joint-true k32 98.50 / 97.82 | `arm_summary.csv`: training-mode rows for the named arm, averaged over public/exact (caption's "10,240 document-level evaluations per pooled entry" = 10 victim seeds × 2 attack seeds × 2 knowledge conditions × 256 documents) |
| Table 2 `tab:primary_by_knowledge` (token accuracy; descriptive exact-sequence) | L3 public 97.39 / 98.29 (43.13 / 55.51); L3 exact 97.81 / 98.72 (51.41 / 66.13); L6 public 94.20 / 97.38 (13.71 / 37.77); L6 exact 94.81 / 98.27 (20.23 / 58.55) | `arm_summary.csv` rows `(depth, knowledge, training, forward_k32)` and `…, joint_true_k32`; `token_accuracy` and `sequence_exact` columns (5,120 evaluations per entry; the L3-public forward 43.13 % is exactly 43.125 % in the CSV, rounded half-up) |
| Table 3 `tab:phases` (appendix phase index, P0–P5) | — | no v3 content; provenance of the separate P0–P5 framework programme, outside this document's scope |
| Primary endpoint: +3.17 pp, 95 % CI [+2.72, +3.64], margin +2 | +3.1714 pp [+2.7246, +3.6438], `exceeds_practical_margin=True` | `contrasts.csv` row `(6, public, training, joint_true_k32, forward_k32)` |
| Perturbation paragraph (descriptive; L6, public, k32) | noise σ=0.1: 97.31; σ=1.0: 96.83; 8-bit: 97.33; 4-bit: 96.38; shuffled/random ≈ 93.9 | `arm_summary.csv` rows `(6, public, training, joint_noise_0.1_k32 / joint_noise_1.0_k32 / joint_bits_8_k32 / joint_bits_4_k32 / joint_shuffled_k32 / joint_random_k32)` |
| SIPIT comparison | 95.25 (L6 public) vs 97.38 joint; 98.89 (L3 exact) vs 98.72 joint | `arm_summary.csv` rows `sipit_steps128` |
| Defended-arms paragraph (descriptive): secret-mixup token recovery | range 83.01–90.67 % across the evaluated secret-mixup conditions | min/max of `token_accuracy` over all `arm_summary.csv` `*@secret-mixup` training rows |
| 〃 exact-sequence recovery | "at most 1.48 % (depth 3, forward k=8, both knowledge conditions)"; public forward k32: 0.31 % (L3) / 0.04 % (L6) | max of `sequence_exact` over the same rows is 1.4844 % at `(3, public, forward_k8@secret-mixup)` **and** `(3, exact, …)`; public k32 rows read 0.3125 % / 0.0391 % |
| 〃 annotated spans under secret-mixup | number-span recovery 93.94–96.50 %; identifier 90–100 % (two annotated spans) | `annotated_spans_number` min/max over the same rows; identifier counts from the frozen corpus annotations |
| 〃 utility | geometric-mean perplexity +31.8 % (L3) / +26.9 % (L6) | `descriptive_summary.json` `utility`: ratios 1.3181 / 1.2686, from the per-shard `defended_victims` records |
| 〃 replica-median | token recovery near-unchanged; exact-sequence moved by up to 7.7 points (public L3 SIPIT-128: 43.77 % → 36.11 %); perplexity 36.05 → 36.51–36.62 | `arm_summary.csv` `sipit_steps128` vs `sipit_steps128@replica-median` at (3, public); `descriptive_summary.json`: undefended mean PPL 36.05, replica-median 36.51 (L3) / 36.62 (L6); but see the precision note in §4.3 |
| "336.1 hours of summed, non-reused online attack wall-clock time" | 336.1 h | sum of `online_seconds` over non-reused rows of `cell_summary.csv` (all rows: 408.6 h) — **re-deriving from the shard JSON: use `work.wall_seconds`, not `cold_seconds`; see the note below** |

**Which field reproduces 336.1 hours.** `cell_summary.csv` is a derived
artefact and is not present in this checkout, so a re-deriver will normally
work from `results/validation-heldout-v3-r3-shard*/validation.json`. Each cell
there carries two different timings, and they do not agree. The published
figure is the sum of **`work.wall_seconds`** over the cells with
`measurement_reused == false` — the `online_seconds` column of
`cell_summary.csv` is that same quantity. Summing `cold_seconds` instead is
the easy mistake: it is a genuine field on the same cells, it is also a
wall-clock measurement, and it gives a slightly larger number, so a
re-deriver who picks it can wrongly conclude the papers are stale. Recomputed
over all 16 shards (3,680 cells; 3,200 non-reused):

| Field summed | Rows | Seconds | Hours | Matches a published figure? |
|---|---|---|---|---|
| `work.wall_seconds` | non-reused (3,200) | 1,209,821.487 | **336.0615 → 336.1 h** | **yes — this is the published figure** |
| `cold_seconds` | non-reused (3,200) | 1,216,191.676 | 337.8310 → 337.8 h | no — 1.77 h higher; not a published number |
| `work.wall_seconds` | all (3,680) | 1,471,037.843 | 408.6216 → 408.6 h | yes — the "all rows: 408.6 h" parenthetical above |
| `cold_seconds` | all (3,680) | 1,478,326.075 | 410.6461 → 410.6 h | no — not a published number |

`cold_seconds` measures the cell including its cold-start overhead, so it is
the larger of the two in every one of the 3,680 cells (smallest per-cell
difference 1.54 s; no cell has `cold_seconds < work.wall_seconds`). Neither
337.8 h nor 410.6 h appears in `paper/paper_a/main.tex` or
`paper/paper_b/main.tex`. Reproduction command:

```bash
python3 - <<'EOF'
import json, glob
w = c = 0.0
for f in sorted(glob.glob('results/validation-heldout-v3-r3-shard*/validation.json')):
    for cell in json.load(open(f))['cells']:
        if not cell['measurement_reused']:
            w += cell['work']['wall_seconds']
            c += cell['cold_seconds']
print(f"work.wall_seconds {w/3600:.4f} h   cold_seconds {c/3600:.4f} h")
EOF
# work.wall_seconds 336.0615 h   cold_seconds 337.8310 h
```

The revised subsection also states that this campaign is the single
canonical evidence source for the joint-observation result and that both
papers share that evidence base explicitly; Paper B now cites this
document (`docs/evidence-release-v3.md`) by name as the artefact index,
checksum and verification recipe.

**Figures.** All three Paper A figures are generated from the same
committed audit summaries by `experiments/make_v3_figures.py`
(regeneration command: `.venv/bin/python experiments/make_v3_figures.py`;
writes the PDFs into `paper/figures/`; the script performs no
recomputation from raw cells). Every plotted value was re-verified for
this index, including text extraction from the shipped PDFs:

| Figure | Plotted content | Source rows | Verification |
|---|---|---|---|
| Figure 1 `paper/figures/fig_arm_heatmap.pdf` | all 40 arms × 4 conditions (depth × knowledge), token accuracy printed at one decimal | every training-mode row of `arm_summary.csv` (160 cells; each entry 5,120 evaluations) | all 160 printed cell labels extracted from the PDF match the CSV rows at displayed rounding (multiset equality) |
| Figure 2 `paper/figures/fig_gradient_conditions.pdf` | gradient-condition axis at k32, depth 6, public: `joint_true / joint_noise_0.1 / joint_bits_8 / joint_noise_1.0 / joint_bits_4 / joint_shuffled / joint_random` token accuracy and exact-sequence bars, `forward_k32` reference lines | `arm_summary.csv` rows `(6, public, training, <arm>)` | all 14 bar labels and both reference values (94.20, 13.71) extracted from the PDF match the CSV rows at two decimals verbatim |
| Figure 3 `paper/figures/fig_privacy_utility.pdf` | defended-arm residual leakage vs utility cost: pooled token accuracy / exact-sequence over the 6 base arms × 2 knowledge conditions per defence × depth, against (geometric perplexity ratio − 1); undefended pooled reference diamonds | pooled from `arm_summary.csv` `forward_k{8,16,32}` and `sipit_steps{32,64,128}` rows with and without `@<defence>`; costs from `descriptive_summary.json` `utility` (x = 31.81 / 26.86 / 1.28 / 1.57 %) | points are unlabelled; all 8 scatter coordinates and 4 reference diamonds recomputed exactly from the same sources (e.g. secret-mixup L3: 89.2587 token, 0.5794 sequence at +31.81 %) |

### 4.2 Paper B (`paper/paper_b/main.tex`, §"Large-Scale Multi-Shard Validation (v3 Protocol)")

The revised subsection carries the corrected campaign descriptors (3,680
cells; 3,200 physical + 480 reused; **40 attack arms — 28 undefended and
12 defended**; ten victim seeds; two attack seeds; 256 held-out
documents; 16 shards on two hosts; **336.1 hours**), the primary
endpoint values in §4.1, the mechanism controls (true vs shuffled
+3.4705 pp [+2.7155, +4.2897]; true vs random +3.4882 pp
[+2.7363, +4.3138] — `contrasts.csv`), descriptive-only perturbation
language, and the corrected defended-arm summary (secret-mixup token
recovery 83.01–90.67 %, exact-sequence at most 1.48 %, perplexity
+31.8 % / +26.9 %; replica-median near-unchanged token recovery with the
7.7-point SIPIT-128 sequence movement) — all mapping to the same source
rows as in §4.1.

The revised subsection adds two artefacts:

- Figure `fig:v3pipeline` — a TikZ orchestration schematic (frozen
  protocol → cell matrix → sharding → toula/koula → merge → crossed
  bootstrap). It contains **no data**; there is nothing to recompute,
  and its caption again cites this document.
- Table `tab:v3primary` — forward-only versus joint-true-gradient token
  accuracy and descriptive exact-sequence recovery by depth × knowledge.
  It is a compact mirror of Paper A Table 2, and all sixteen printed
  values were verified identical to the same canonical `arm_summary.csv`
  rows (5,120 evaluations per entry).

### 4.3 Manuscript corrections recorded 2026-09-15

The reviewed manuscripts contained three statements inconsistent with
the records; the adversarial review (this document, earlier revision)
flagged them, and both manuscripts were corrected on 2026-09-15. A
fourth item arose as an imprecision in that first revision and was
absorbed in the current one. For provenance, old text versus corrected
text:

1. **Arm count.** Old (Paper B): "spanning 35 attack arms". Corrected
   (both papers): "40 attack arms (28 undefended and 12 defended)" —
   matching the 40 distinct arms in `arm_summary.csv`.
2. **Compute figure.** Old (Paper B): "≈342 GPU-hours". Corrected (both
   papers): "336.1 hours of summed, non-reused online attack wall-clock
   time" — exactly the non-reused `online_seconds` sum in
   `cell_summary.csv`, equivalently the sum of `work.wall_seconds` over
   the `measurement_reused == false` cells of
   `results/validation-heldout-v3-r3-shard*/validation.json` (336.0615 h).
   Do **not** sum `cold_seconds`, which gives 337.8310 h; see the field
   note in §4.1.
3. **Defended exact-sequence bound.** Old (Paper B): "secret-mixup
   suppresses exact-sequence recovery to at most 0.2 %". Corrected
   (both papers): token recovery 83.01–90.67 % and exact-sequence "at
   most 1.48 %" across the evaluated secret-mixup conditions — verified
   above against the CSV rows.
4. **Replica-median token-recovery bound.** First revision (both
   papers): "within half a percentage point of the undefended arms" —
   flagged here as mildly understating two depth-3 public SIPIT
   conditions (deviations 0.637 and 0.618 points across the 24
   `@replica-median` rows). Absorbed in the current revision: both
   papers now read "within two-thirds of a percentage point", which is
   consistent with the recorded maximum deviation of 0.637 points. No
   residual discrepancy remains on this point.

## 5. Execution history and restarts

### 5.1 What the records show

The campaign ran in three rounds against the same frozen protocol
(protocol hash identical in every surviving report). Times are UTC.

| Round | Geometry | Window | Outcome |
|---|---|---|---|
| r1 | 2 shards (`validation-heldout-v3-shard0/1`) | 2026-09-10 05:30–05:35 start; last chain events 2026-09-11 01:02–01:03 | Aborted mid-run. Directories retained on toula (352 of 1,760 cells checkpointed) and koula (192 of 1,920); reports frozen at status `RUNNING`; chains end without `run_end` and without error events. |
| r2 | 12 shards (`validation-heldout-v3-r2-shard0…11`) | started 2026-09-11 01:03:40 | Superseded within ≈50 minutes by r3. **No r2 outputs are retained on either host**; only the journal's start entries survive. |
| r3, first pass | 16 shards, launched staggered over ≈10–45 minutes per host, up to 8 concurrent per host | toula 01:54:10–02:37:31; koula 02:28:35–02:39:02 on 2026-09-11 | Shards 0–3 (toula) and 8 (koula) ran to completion. Shards 4–7 (toula) and 9–15 (koula) were aborted: their chains end 02:44:58–04:12:04 (6–102 minutes after start) with no error event and no `run_end`, reports frozen at `RUNNING` with **zero checkpointed cells** (verified for all 11). Directories retained as `…-shardN.aborted`. |
| r3, restarts | strictly sequential per host | toula: shard 4 from 2026-09-13 07:03:38, shard 7 completed 2026-09-15 20:30:48. koula: shard 9 from 2026-09-11 21:15:25, shard 15 completed 2026-09-15 19:19:44 | All 11 restarted shards completed with status `COMPLETE`. |

Quantitative observations from the chains, relevant to why the first
pass was restructured: before the concurrent launch, cells completed in
≈26–56 s (shard 0, 01:55–01:57 on 2026-09-11); once eight shards shared
the host, the same record shows cells taking 214–2,485 s in the
02:40–04:30 window, and the aborted shards' chains show individual cells
of 209–1,903 s. The surviving shards' wall-clock totals (≈33–53 h for
shards 0–3) are correspondingly inflated relative to the sequential
restarts (≈9–13 h; shards 7 and 15 ≈25–26 h).

### 5.2 The journal, and what it does not record

`results/heldout_access.journal` (tracked in git by explicit
`.gitignore` exception) is appended by
`experiments/validate_reconstruction.py:record_heldout_access`, which is
called **at run initialisation**, when the output directory is created —
before any held-out outcome exists (`validate_reconstruction.py:347`).
Its entries therefore record *run starts against an output path*, not
reads of results, notwithstanding the function docstring's "first read
of held-out evaluation outcomes" phrasing (the guard's purpose is to
make protocol re-freezing impossible once any run has begun).

Two further facts constrain its interpretation:

- The surviving journal is the **toula-side record**. It contains only
  toula-side starts: the r1 shard-0 entry, all twelve r2 entries, and
  the r3 entries for shards 0–7. No koula-side run appears anywhere in
  it — r1's koula shard-1 run (begun 05:30:25 on 2026-09-10) is absent,
  as are every first-pass and restart attempt for shards 8–15. The copy
  now on koula is byte-identical to toula's
  (SHA-256 `1760bb85…`, matching the repository copy) but carries an
  mtime of 2026-09-15 21:24 UTC — i.e. it was overwritten during
  post-campaign synchronisation. Whatever the koula-local journal
  previously contained is not recoverable from the synchronised copy;
  koula timings above are established from the evidence manifests and
  `.aborted` chains instead.
- The journal shows the r3 shards 4–7 output paths twice (2026-09-11
  02:33–02:37 and again 2026-09-13 → 2026-09-14), matching the retained
  `.aborted` directories and the second-pass manifests.

### 5.3 Git provenance of the final runs

Fifteen manifests record commit `064a64f` (the freeze commit) with a
clean tree; shard 7's manifest records `70e5686` with a dirty tree. The
host-side reflog resolves this: `70e5686` (the evidence commit
"14 completed + 2 in-progress shards") was created in toula's pilot
clone at 2026-09-15 00:26 +0100, while shard 7's final run
(2026-09-14 19:34 → 2026-09-15 21:30 +0100) was still executing; git
state is captured at finalisation, so only shard 7 saw the new HEAD,
and "dirty" reflects the untracked result directories in the working
tree. Crucially, all 16 manifests record the **same**
`source_sha256` (`d6538b78…`), a canonical hash over every
`attacker/**/*.py` file: the executed attacker package was byte-identical
across all shards, both hosts, and both recorded commits.

### 5.4 Assessment against the frozen no-retries policy

The frozen protocol's missing-data rule is: *"No complete-case deletion.
Any missing/failed cell blocks comparative analysis; retain failure and
rerun entire run in a fresh directory with same protocol."* Against that
rule, the records establish:

- every rerun was of an **entire run in a fresh directory** under the
  identical frozen protocol and input hashes — no selective cell retries
  appear in any journal, chain, or manifest;
- the aborted r3 attempts are retained on the execution hosts, as the
  policy requires ("retain failure"), and their reports checkpointed
  zero cells, so no completed metric from an aborted r3 attempt could
  have informed the decision to rerun;
- the final matrix is complete: 3,680/3,680 cells, no failures, so the
  comparative analysis was never unblocked by dropping data.

The records do **not** establish:

- the operator's reason for terminating any attempt (no error events
  exist; the severe slowdown under 8-way concurrency is documented
  above but is circumstantial evidence, not a recorded rationale);
- why r1's 2-shard geometry was abandoned for r2's 12 and then r3's 16,
  nor whether r1's partial checkpointed metrics (352 and 192 cells) or
  r2's outputs were inspected before relaunch; r2's outputs are not
  retained at all, which is a records gap relative to the r3 rounds;
- koula-side start times from the access journal (overwritten, §5.2).

Consequently, and consistent with the independent verdict
(`outputs/validation-v3-review-2026-09-15/verdict.md`, §6.3): the
restart pattern is fully consistent with an operational restart for
throughput reasons followed by the protocol-prescribed remedy, and the
structural facts (frozen contract pinned before any run, complete final
matrix, zero-cell aborted r3 reports) constrain what any retry could
have selected — but the no-outcome-driven-retry condition cannot be
certified from the final records alone. This section exists precisely so
that a reviewer does not have to take that on trust.

## 6. The 40 defence-provenance name conflicts and the required merge procedure

### 6.1 Mechanism (confirmed against the code and the reports)

`experiments/merge_validation_shards.py` merges the 16 shard directories
into one analyser-ready directory. For every defended cell it copies the
cell's `defense_provenance` / `defended_observations` sidecars to
**the same shard-relative path in the output**
(`target = output / ref['path']`, lines 139–147), after checking the
source against the cell's recorded hash. There is no collision check on
the destination.

Across the 16 reports there are exactly **40 distinct provenance paths**
(`defended-<defense>-<seed>-<depth>/captures/<defense>_provenance.json`;
20 per defence). Each path is referenced from **10 different shards**,
and each referencing shard records a **different SHA-256** for it — 400
references, 10 distinct byte strings per path, 40 paths. The cause is
benign and mechanical: each shard trains its own copy of every defended
victim its cells need, and the provenance JSON embeds the shard's own
output path in its `journal` field (a field-by-field diff of two shards'
copies of the same victim's provenance shows `journal` as the only
differing field). The defended *observation* tensors
(`defended/observations-*.pt`) are deterministic and carry a single
identical hash per path across all shards — no conflict there
(the reviewer's `audit.json` agrees: `raw_artifact_hash_conflicts: 0`,
`sidecar_hash_conflicts: 40`).

A naive merge therefore lets the **last writer win**: each copy is
checked against its own cell's hash at copy time, so no error is raised,
yet the merged tree can physically hold only one of the ten byte
strings. The merged `validation.json` still records all ten hashes, so
the merged package is left in a state where re-hashing the destination
against each referencing cell's recorded hash fails for nine of every
ten references. (The official merge was never run for the papers — the
analysis was produced per-shard by the audit tooling — and no merged
directory exists on either host. Had it been run, this is the state it
would have produced.)

### 6.2 Required procedure

1. **Preserve shard-specific paths.** Merge into per-shard subtrees
   (e.g. `<merged>/shardN/<shard-relative path>`) or otherwise rewrite
   sidecar destinations so that no two shards' files share a
   destination; record any path rewrite in the merged report and in a
   new evidence event. Do not rely on the current merger for the defence
   sidecars until it refuses or resolves destination collisions.
2. **Check the merge against the recorded hashes.** For every cell in
   the merged report, re-hash the file at the (possibly rewritten)
   destination of `defense_provenance` and `defended_observations` and
   require equality with the cell's recorded `sha256` — all 400
   provenance and 400 observation references must match exactly once.
   The same check applied to a flat (colliding) merge is precisely what
   detects the overwrite described above.
3. Only then run the official analyser:
   `python3 experiments/analyze_validation.py --root <merged> --output <dir> --protocol-version 3 --expected-manifest-sha256 <sha256 of merged validation.json.evidence.json>`.
   The analyser additionally re-hashes all 60 shared artefacts and every
   prediction file, and rescores every cell (§3.2), so it requires the
   binary tier present in the merged tree.

## 7. Access route and release plan

- **Current access (private).** SSH to the two LAN hosts
  (`ssh -o BatchMode=yes toula|koula`, 192.168.1.107 / 192.168.1.108),
  source root `~/llm-attacker-pilots/gpt2-wikitext2-v01/code/results/`.
  Verification to date has been strictly read-only on both nodes.
  Private SSH accessibility is not a public release.
- **Already public (JSON tier).** Everything under
  `results/validation-heldout-v3-r3-shardN/` except the `*.pt` tier is
  tracked in this repository; the complete JSON tier first appears in
  commit `8663119` (partial in `70e5686`); anchor on commit `e4d85a6`
  or later, the protocol hash, the input hash, and the 16 report
  digests in §2.3. `results/heldout_access.journal` is likewise tracked.
- **Planned (binary tier).** Release of the `.pt` capture tier to a
  Hugging Face dataset alongside the papers, preserving the shard
  directory layout so that the recorded hashes in §2.2 verify directly
  against the published files (including, per §6, the per-shard defence
  provenance). The release should also include the `.aborted`
  directories and the r1 leftovers (`validation-heldout-v3-shard0/1`),
  which exist only on the execution hosts, so that §5 remains checkable
  after the hosts are reclaimed.

## 8. Evidence-index status (`KEY_NUMBERS.md`, `evidence/`)

The reviewer correctly found both absent from this checkout; the state
of play is:

- `KEY_NUMBERS.md` is **not tracked in this repository** at `e4d85a6`
  and does not exist in this checkout. Copies exist on both GPU hosts at
  `~/llm-attacker/KEY_NUMBERS.md`, but: neither host copy is in a git
  repository; the two copies **differ** (toula 104 lines,
  SHA-256 `7672019f…`; koula 78 lines, SHA-256 `48d5fc6c…`); and neither
  contains any `validation-heldout-v3` number — they index the earlier
  campaigns only.
- No `evidence/` directory exists in this checkout, on either host's
  `~/llm-attacker/`, or in the pilot working copies.
- The canonical evidence index for v3 today is therefore the
  combination of: the 16 shard reports and their evidence chains
  (§2.3), the frozen protocol and input pins (§1), the review outputs in
  `outputs/validation-v3-review-2026-09-15/`, the remote verification
  outputs in `outputs/remote-evidence-check-2026-09-15/`, and this
  document.
- **To be published before or with the papers:** a v3 entry in a
  repository-tracked `KEY_NUMBERS.md` (or an equivalent index) carrying
  the §4 numbers with their file provenance; this release document; and
  the Hugging Face binary-tier release (§7). The manuscript corrections
  of 2026-09-15 — all four, including the replica-median bound — are
  recorded as resolved in §4.3.

## 9. Statement of verification

Verified directly in the preparation of this document (2026-09-15):

- protocol, effective-protocol and input hashes as stated in §1;
- the 16 repository shard directories contain the JSON tier only (no
  `.pt` files), and all 16 reports are `COMPLETE` with consistent shard
  records and identical protocol bindings;
- all 16 evidence chains verify with `attacker.evidence.verify_evidence`;
- all 60 shared artefact names hash identically across the 16 reports;
- all 16 on-host reports are byte-identical to the repository copies
  (digests in §2.3 recomputed over SSH);
- independent on-host spot check: shards 7 and 15, 60/60 recorded
  artefacts hash-verified each, chains verify;
- the 40 provenance conflicts exactly as characterised in §6 (40 paths,
  10 referencing shards, 10 distinct hashes each), and the absence of
  observation/artefact hash conflicts;
- every headline number in §4 recomputed from the committed CSVs —
  re-verified on 2026-09-15 against the revised manuscripts, including
  the corrected defended-arm ranges and the 1.48 % exact-sequence
  maximum;
- every plotted value in the three Paper A figures: all 160 heatmap
  cell labels and all 16 printed labels in the gradient-condition figure
  extracted from the shipped PDFs and matched to `arm_summary.csv` at
  displayed rounding, and the privacy–utility scatter coordinates
  recomputed exactly from `arm_summary.csv` and
  `descriptive_summary.json`; the Paper B mirror table verified
  identical to the canonical rows (the Paper B TikZ figure carries no
  data);
- the restart timeline in §5 from the journals, manifests, `.aborted`
  chains and host reflogs.

Relied upon from prior committed work (not re-executed here): the full
5,008-file on-host verification and the 942,080-record rescoring, both
of 2026-09-15, whose scripts and outputs are committed under `outputs/`.

Not performed by anyone to date: prediction regeneration from the binary
tier or any re-execution of the campaign (§3.3); an official merged
analysis run (§6.2). The binary tier remains on the two execution hosts
pending the Hugging Face release.

---

## Addendum 2026-09-16 — merged-package collision correction (complete)

The §6.1 caveat ("do not rely on the current merger … until it refuses or
resolves destination collisions") is resolved. `experiments/merge_validation_shards.py`
now uses deterministic per-shard sidecar destinations and refuses any
destination collision with divergent content hashes (fail closed); the
defended-victim records are kept per shard (no first-writer-wins). Verified
on copies of the canonical shards (`build/merge-fix-v1/`): all 16 source
chains plus the merged chain verify; all 3,680 prediction references, the
protocol/input triple, and all 17 hash-chained journals hash-verify from
the merged package; the cell identity set matches the frozen protocol
matrix exactly (3,680 cells, 480 reused-measurement rows). The 860
binary-tier references (60 shared `.pt` artifacts, 400 provenance files,
400 defended-observation files) exist only on `toula`/`koula` and are
carried as explicitly journaled deferred references; they verify against
the recorded hashes once the tier is present (per-shard verification of
2026-09-15 already covered them, 5,008/5,008). Regression tests:
`tests/test_validation_sharding.py` (15/15). Merged `validation.json`
SHA-256 `2ee462d9…`; verification record `build/merge-fix-v1/verification.json`.
