# Hugging Face evidence release

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

The detached worker is `build/phase2/finish_hf_release.py`; its PID is stored
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
