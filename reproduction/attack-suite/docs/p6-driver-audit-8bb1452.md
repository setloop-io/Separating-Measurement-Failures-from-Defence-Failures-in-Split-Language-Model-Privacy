# P6 driver audit: 8bb145235f12e4238ffa6037234e46c520c8efe8

Checked 17 September 2026 against the frozen protocol, collected run artifacts,
and current execution-host files. This commit adds only
`experiments/run_p6_grid.py` (332 lines). It is an ancestor of local HEAD.
The driver is unchanged between this commit, local HEAD, all three execution
hosts, and both staged HF code snapshots. Driver SHA-256:
`bbb43725f126bd949038db42b2810388783bb5dd85ef6ee6412d77b054fce913`.
The hosts currently report checkout `ae37fe7f7d1dd1c6815d76286f4f6fea65204827`;
that checkout ID alone does not prove when the uncommitted driver was copied.

## Findings

1. **High: missing utility endpoint and controls.** The frozen protocol requires
   held-out Delta_U relative to an all_local control and frozen-model evaluations.
   The driver runs only defended segment training and VMI, records final training
   minibatch loss (`capture_metrics`, lines 187–194), and never produces those
   controls or held-out evaluations. Training loss cannot substitute for Delta_U.
2. **High: input corpus differs from the frozen declaration.** Line 64 selects
   `results/long-horizon-v1/inputs/pilot-corpus.txt`; the frozen protocol declares
   the v3 calibration/train/evaluation partitions. Actual capture configs confirm
   the pilot path. Its SHA-256 is
   `026c4a6a930ec765c5d17b6b38783e42cb19be02b7e0f7fd4f6a34682aa9da43`, identical
   locally and on the three hosts. No driver assertion enforces the frozen input
   manifest. The pinned validation evaluation is not implemented by this runner.
3. **High: incomplete privacy instruments.** `attack_cmd` (lines 176–184) runs
   forward-latent VMI only. The protocol also calls for gradient-based
   `label-scoring`. A VMI-only outcome cannot be presented as the full declared
   privacy endpoint or worst outcome across the required instruments.
4. **Medium: execution order differs from the frozen rule.** The driver uses a
   deterministic Latin interleave; the protocol specifies randomisation with a
   prespecified seed. The freeze amendment documents the start=0 seam mapping,
   but does not amend the ordering rule. Record this as a deviation; do not
   retrofit a post-outcome amendment to portray these runs as conforming.
5. **Medium: resume validation is permissive.** `artifact_ok` accepts any nonempty
   results/summary container without checking evidence sealing, settings or
   capture files. A minimal `{"summary":[{}]}` passes. `load_journal` silently
   ignores malformed records. Failed cells are eligible to rerun in the same
   output directory, rather than the fresh whole-cell rerun required by the
   missing-data rule. These paths did not affect the collected runs: all three
   manifests say resume=false, journals contain no skipped phases, and all 192
   unique cells succeeded. All 384 evidence manifests independently verify.

## Release interpretation

The 192 cells are a completed **pilot-corpus, VMI-only Mode-2 grid**, with
verified artifact integrity. They are not a complete execution of the frozen
P6 protocol and cannot support its full confirmatory endpoints. Even a complete
Mode-2 run would not resolve the external Mode-1/P5 empty-set claim.

Preserve this driver and all evidence bytes as the historical execution record.
Any corrective implementation and further experiments need an explicit,
prospective protocol/analysis decision and fresh output directories. Do not
silently replace the historical driver or relabel training loss as utility.

For reproduction, the release includes the exact pilot text at
`code/results/long-horizon-v1/inputs/pilot-corpus.txt`. From the code directory,
after installing dependencies and caching the required GPT-2 model/tokenizer,
run `python experiments/run_p6_grid.py --span 2 --node reproduction --workdir
results/p6-reproduction-s2` (repeat with spans 4 and 6 in fresh directories).
The driver forces HF offline mode. This reproduces the historical driver
configuration, with the scientific limitations above; bitwise numerical
identity across environments has not been established.
