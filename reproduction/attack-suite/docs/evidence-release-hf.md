# Hugging Face evidence release

**20 September update:** the [complete adapted-comparison release and reviewed Paper B](v4-adaptation-completed-2026-09-20.md) are verified at immutable HF revisions. That record also links reproduction checks and distinguishes the remaining original-setting limitation. The dated release notes below are historical; they do not describe the current upload state.

Release work started 17 September 2026. Dataset repositories:

- https://huggingface.co/datasets/Setloop/llm-attacker-paper-a
- https://huggingface.co/datasets/Setloop/llm-attacker-paper-b

Both repositories retain their existing private visibility. The release is
in progress until both remote verification reports say `ok: true`.
A successful upload is not scientific validation or a fidelity-gate verdict.

## Maintainer commands

The staged trees are `build/hf-release-2026-09-17/paper-a` and `paper-b`.
Use cached HF credentials; never put a token in a command or a release file.

```sh
.venv/bin/python experiments/release_manifest.py build/hf-release-2026-09-17/paper-a
HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/release_upload.py paper-a
.venv/bin/python experiments/release_verify_remote.py paper-a
```

Repeat for `paper-b`. Uploads compare existing Git/LFS object digests and
commit changed files in batches of at most 500 files and approximately
256 MB, small files first. The manifest lands last. Rerunning resumes safely.
Verification pins the remote commit, checks every file's Git blob or LFS
SHA-256 digest, checks manifest coverage and hashes, and rejects missing,
unexpected or mismatched files (HF's `.gitattributes` is expected).

## Contents and scope

Each paper receives its manuscript, figures, a source snapshot with commit
anchor, frozen protocols, pinned inputs, v3 shard/merged evidence, numerical
analysis outputs, and the recorded binary references. Paper B also includes
domain-shift, long-horizon, agentic, clustered-gate and positive-control data.
Paper A receives the completed P6 Mode-2 grids. Experimental BiSR records
belong to Paper B and must be labelled as awaiting the fidelity decision.
Review letters, planning notes, literature caches and unrelated outputs are
excluded. Code is Apache-2.0; upstream assets retain their own licences.

The original uploader assumed every shard was complete on koula. On inspection,
400 provenance files were absent there from the attempted staged release;
original shards 0–7 are retrieved from toula and 8–15 from koula. The 60 shared
binary paths have no `shardN/` prefix and must be recovered separately from a
complete shard. All 860 expected references must be checked against the merged
report's recorded hashes, not merely hashed after copying.

Do not delete Git evidence until remote verification succeeds and paper links
reference an immutable HF revision. Retain reproduction code in Git. Removing
working-tree data does not erase Git history; no history rewrite is included.

## Reviewer use

Download a pinned revision with `huggingface_hub.snapshot_download` using
`repo_type="dataset"`, the repository ID above, and the release commit as
`revision`. Private repositories require authorised access. Run
`shasum -a 256 -c MANIFEST.sha256` from the downloaded root. Install the code
snapshot with `python -m pip install -e ./code` in a fresh environment; the
snapshot includes `requirements-research.lock` for the research dependencies.

`INDEX.md` maps claims to evidence and `PENDING.md` identifies gaps.
`v3-campaign/EVIDENCE-RELEASE.md` preserves the original verification,
rescoring and regeneration procedures and their limitations. Its historical
Git paths refer to the original execution layout; release locations are given
in INDEX. Saved evidence bytes must not be rewritten to change paths.

## Current worker

The detached worker is `experiments/release_finish.py`; its PID is stored
in `build/phase2/recovery-worker.pid`, and its log is
`build/phase2/recovery-worker.log`. It recovers and checks binary references,
waits for initial uploads, finalises the reviewed trees, uploads, and verifies.
`build/hf-release-2026-09-17/release-complete.json` is created only after both
remote verification reports pass. `RELEASE_FAILED` in the worker log indicates
an interrupted release requiring inspection; it must not be called complete.
The initial upload logs are `build/phase2/upload-paper-{a,b}-small.log`.

A reusable version of the completion worker is `experiments/release_finish.py`.
Run it only after reviewing the staged trees and checking that no existing
release worker is active. It supports `--wait-pid PID` for outstanding staging
or upload processes.

The Hub reported a 128-commits/hour limit during the initial small tiers.
The completion uploader now batches up to 500 files (still approximately
256 MB per commit) and retries rate limits over a two-hour window.

## Separate audit supplement

The five-node recovery and BiSR audits are staged separately in
`build/hf-audit-supplement-2026-09-17`. The baseline upload snapshot is frozen.
`experiments/release_audit_supplement.py` waits for the verified baseline,
uploads the eight supplementary files per paper, then verifies their object
digests at pinned remote commits. Its PID is in
`build/phase2/audit-supplement-worker.pid` and its log is
`build/phase2/audit-supplement-worker.log`. Completion requires
`build/hf-audit-supplement-2026-09-17/supplement-complete.json`.

At the user's request, Paper A's recovered P0–P5 archive and audit notice are
also being published ahead of the bulk baseline. Their independent pinned
verification is recorded in `paper-a-immediate-verification.json` in the
supplement staging directory. This does not mark the baseline complete.
The baseline verifier accepts these additional paths only when their remote
object hashes match the separately reviewed supplement; other extras fail.

The supplement's `RELEASE-UPDATE.md` explains that recovered historical P0–P5
evidence was retired by its owner, the original P6 grid was a pilot, and BiSR
fidelity remains unestablished. Archives retain source paths and include
reproduction scripts. `MANIFEST-AUDIT-20260917.sha256` covers the supplement;
the original `MANIFEST.sha256` continues to cover the baseline. The corrected
P6 run is not included in either snapshot and requires a later verified release.

## Progress monitoring

`experiments/monitor_hf_upload.py` reads local worker logs every 30 seconds and
writes `build/phase2/hf-upload-progress.md` and `.json`. It makes no Hub API
requests. Its PID and log are `build/phase2/hf-monitor.pid` and `hf-monitor.log`.
File-count progress is not byte progress or an ETA: large tensors upload last.
The monitor records pinned verification results separately and stops when both
releases/supplements verify or a worker exits before its completion marker.
It writes local status; it does not send notifications.

## Verification recovery

Paper A's first final verification found no missing files, digest mismatches
or manifest errors, but rejected two orphan rsync temporary fragments left on
HF by an early upload. Only those two unmanifested fragments were removed in
HF commit `a02fb09280dae643da4f5a11abf08ca8f62d8337`; canonical evidence was
unchanged. Details are in
`build/hf-release-2026-09-17/paper-a-stray-fragment-cleanup.json`.

`experiments/release_resume.py` reruns Paper A verification and resumes Paper B
upload/verification without modifying either frozen staging tree. The worker,
audit-supplement worker and progress monitor were restarted using their usual
PID/log files. Upload inventory now rejects rsync tensor fragments before
uploading, rather than allowing temporary bytes into a release.

## Paper A scientific correction and corrected P6 release

The revised manuscript and corrected P6 analysis/protocol/input/source package
are verified at `f7b2aef7649588e9021805f2edc25428b6f26c93`, under
`revision-20260917/manuscript/` (19 files). Its README supplies the corrected
claim map and reproduction instructions. It supersedes the baseline manuscript's
P0–P5 claims; the original baseline and audit manifests remain unchanged.

The approximately 33-GB full corrected P6 capture/checkpoint tier is a separate
active transfer/publication under `revision-20260917/p6-raw/`. Status is in
`build/paper-a-revision-20260917/raw-status.json`; it is complete only after
`raw-status-verification.json` records `ok: true` at a pinned remote revision.
See `docs/paper-a-closure-2026-09-17.md`. The baseline progress monitor does not
include this new revision tier. Both datasets retain private visibility.

## Paper B revision publication — verified 17 September 2026

The manuscript/analysis tier is remotely verified at immutable HF revision
`bd8e8d4582ea610d9d3f51d442b51df67bbf2b9b`, prefix
`revision-20260917/manuscript`: 23 files, 8,230,206 bytes, zero errors.
Source revision: `f9d3780` (full hash included in the package).
The package includes the PDF/LaTeX, reviewer response, pinned source and inputs,
metadata, summaries and reproduction instructions. Visibility remains private.

This pin does not certify the separate full white-box or Phase 2 raw tiers;
their publishers remain active. Sasa was still running
`convergence-mid-lite-33` when checked at 21:33 UTC; its incomplete extension
is excluded from the manuscript. BiSR provenance remains unestablished.

[Immutable manuscript package](https://huggingface.co/datasets/Setloop/llm-attacker-paper-b/tree/bd8e8d4582ea610d9d3f51d442b51df67bbf2b9b/revision-20260917/manuscript).
