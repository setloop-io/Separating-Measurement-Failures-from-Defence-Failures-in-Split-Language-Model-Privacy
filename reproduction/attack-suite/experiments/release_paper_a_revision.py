#!/usr/bin/env python3
"""Publish the separate Paper A correction; never modify a baseline manifest.

The manuscript tier must be reviewed and staged before invocation. The raw tier
waits for its copy, verifies all 200 frozen cells, then archives and publishes.
Remote verification pins each commit. No repository visibility changes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from release_upload import hashes, inventory, matches

REPO = 'Setloop/llm-attacker-paper-a'
PREFIX = 'revision-20260917'
PROTOCOL = '88e2286339109d8c6356b75b5263a23647261136eb0d039ecca15075c03d533b'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class PartWriter:
    """Bound archive objects to 4 GiB without making a second full archive."""
    def __init__(self, root, limit=4 * 1024**3):
        self.root, self.limit = root, limit
        self.index, self.size, self.stream = 0, 0, None

    def write(self, data):
        total = len(data)
        while data:
            if self.stream is None:
                self.stream = (self.root / f'evidence.tar.part{self.index:03d}').open('wb')
                self.index += 1
                self.size = 0
            n = min(len(data), self.limit - self.size)
            self.stream.write(data[:n])
            data = data[n:]
            self.size += n
            if self.size == self.limit:
                self.stream.close()
                self.stream = None
        return total

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def publish(stage, tier, state, publication_label="Paper A correction"):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    files = inventory(stage)
    manifest = stage / 'MANIFEST.sha256'
    entries = {r: hashes(p) for r, p in files.items() if r != 'MANIFEST.sha256'}
    manifest.write_text(''.join(f'{h[0]}  {r}\n' for r, h in sorted(entries.items())))
    files = inventory(stage)
    digests = {r: hashes(p) for r, p in files.items()}
    remote_prefix = f'{PREFIX}/{tier}'
    try:
        remote = {e.path: e for e in api.list_repo_tree(REPO, repo_type='dataset',
                  path_in_repo=remote_prefix, recursive=True) if hasattr(e, 'blob_id')}
    except Exception as exc:
        if getattr(getattr(exc, 'response', None), 'status_code', None) != 404:
            raise
        remote = {}
    changed = [r for r in files if f'{remote_prefix}/{r}' in remote
               and not matches(remote[f'{remote_prefix}/{r}'], digests[r])]
    if changed:
        raise RuntimeError('Frozen revision prefix already has different bytes: ' + str(changed))
    pending = [r for r in files if f'{remote_prefix}/{r}' not in remote]
    pending.sort(key=lambda r: (r == 'MANIFEST.sha256', files[r].stat().st_size, r))
    while pending:
        batch, size = [], 0
        while pending and len(batch) < 200:
            rel = pending[0]
            n = files[rel].stat().st_size
            if batch and size + n > 256_000_000:
                break
            batch.append(pending.pop(0)); size += n
        for attempt in range(120):
            try:
                result = api.create_commit(REPO, repo_type='dataset',
                    operations=[CommitOperationAdd(path_in_repo=f'{remote_prefix}/{r}',
                                path_or_fileobj=str(files[r])) for r in batch],
                    commit_message=f'{publication_label} {tier}: {len(batch)} files', num_threads=4)
                state('uploading', revision=result.oid, remaining=len(pending), tier=tier)
                break
            except Exception as exc:
                if attempt == 119:
                    raise
                print('UPLOAD_RETRY', type(exc).__name__, flush=True)
                time.sleep(60)
    info = api.repo_info(REPO, repo_type='dataset')
    remote = {e.path: e for e in api.list_repo_tree(REPO, repo_type='dataset',
              path_in_repo=remote_prefix, recursive=True, revision=info.sha) if hasattr(e, 'blob_id')}
    expected = {f'{remote_prefix}/{r}' for r in files}
    errors = [r for r in files if f'{remote_prefix}/{r}' not in remote or
              not matches(remote[f'{remote_prefix}/{r}'], digests[r])]
    extras = sorted(set(remote) - expected)
    if errors or extras:
        raise RuntimeError(f'Pinned remote verification failed: {errors}, extras={extras}')
    report = {'repo': REPO, 'revision': info.sha, 'private': info.private,
              'tier': tier, 'prefix': remote_prefix, 'files': len(files),
              'bytes': sum(p.stat().st_size for p in files.values()),
              'ok': True, 'errors': [], 'manifest_sha256': hashes(manifest)[0]}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tier', choices=['manuscript', 'p6-raw'])
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--raw', type=Path)
    ap.add_argument('--wait-pid', type=int)
    args = ap.parse_args()

    def state(status, **extra):
        record = {'status': status, 'utc': datetime.now(timezone.utc).isoformat(),
                  'tier': args.tier, **extra}
        save(args.status, record)
        print(json.dumps(record), flush=True)

    try:
        if args.tier == 'p6-raw':
            if not args.raw:
                raise ValueError('--raw is required')
            if args.wait_pid:
                state('waiting_for_copy', copy_pid=args.wait_pid, raw=str(args.raw))
                deadline = time.monotonic() + 48 * 3600
                while time.monotonic() < deadline:
                    try:
                        os.kill(args.wait_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(30)
                else:
                    raise TimeoutError('Copy process still alive after 48 hours')
            from experiments.run_p6_corrected import verify_cell, digest
            state('verifying_local_cells')
            marker = json.loads((args.raw / 'grid-complete.json').read_text())
            if marker.get('count') != 200 or marker.get('canary') or marker.get('protocol_sha256') != PROTOCOL:
                raise ValueError('Full frozen grid marker required')
            analysis = json.loads((ROOT / 'results/p6-corrected-followup-20260917/p6-corrected-gpt2-v2.analysis.json').read_text())
            anchors = {Path(k).parent.name: v for k, v in analysis['evidence_anchors'].items()}
            cells = sorted(args.raw.glob('*/cell.json'))
            if len(cells) != 200 or {p.parent.name for p in cells} != set(anchors):
                raise ValueError('Raw cells differ from analysed cells')
            for i, path in enumerate(cells):
                if digest(path) != anchors[path.parent.name]:
                    raise ValueError('Cell differs from analysis anchor')
                verify_cell(path.parent, PROTOCOL)
                if (i + 1) % 10 == 0:
                    state('verifying_local_cells', verified=i + 1, total=200)
            args.stage.mkdir(parents=True, exist_ok=True)
            # Only new, dedicated archive paths; never alter source evidence.
            old_parts = list(args.stage.glob('evidence.tar.part*'))
            if old_parts:
                raise RuntimeError('Archive parts already exist; inspect before restarting')
            state('archiving_verified_cells')
            writer = PartWriter(args.stage)
            try:
                with tarfile.open(fileobj=writer, mode='w|') as archive:
                    archive.add(args.raw, arcname='p6-corrected-gpt2-v2')
            finally:
                writer.close()
            (args.stage / 'README.md').write_text(
                '# Corrected GPT-2 P6 raw evidence\n\n'
                '200 verified final cells: 192 segment-trained and eight all-local controls.\n'
                'The original pilot, Qwen partial run and canaries are excluded.\n'
                'Verify MANIFEST.sha256, then concatenate parts in lexical order:\n\n'
                '```sh\ncat evidence.tar.part* | tar -xf -\n```\n\n'
                'Use the frozen source, protocol and reproduction instructions in the\n'
                'sibling manuscript tier. Original evidence paths/bytes are preserved.\n'
                'Cell manifests match all 200 anchors in the published analysis.\n')
        state('publishing', tier=args.tier)
        report = publish(args.stage, args.tier, state)
        save(args.status.with_name(args.status.stem + '-verification.json'), report)
        state('verified', **report)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        raise


if __name__ == '__main__':
    main()
