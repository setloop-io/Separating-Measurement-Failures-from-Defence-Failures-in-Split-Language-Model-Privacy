#!/usr/bin/env python3
"""Resume a staged HF evidence release, committing small files before tensors.

python experiments/release_upload.py paper-a --root build/hf-release-2026-09-17
Does not delete source files or change repository visibility.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            sha.update(chunk)
            git.update(chunk)
    return sha.hexdigest(), git.hexdigest()


def inventory(root):
    files = {p.relative_to(root).as_posix(): p for p in sorted(root.rglob('*'))
             if p.is_file() and '.cache' not in p.relative_to(root).parts}
    partials = [r for r, p in files.items()
                if re.match(r'^\..+\.pt\.[A-Za-z0-9]{6,}$', p.name)]
    if partials:
        raise ValueError('Staging contains unfinished rsync fragments: ' + str(partials))
    return files


def matches(entry, digest):
    return (entry.lfs.sha256 == digest[0] if entry.lfs else entry.blob_id == digest[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paper', choices=['paper-a', 'paper-b'])
    ap.add_argument('--root', type=Path, default=Path('build/hf-release-2026-09-17'))
    ap.add_argument('--small-only', action='store_true')
    args = ap.parse_args()
    root = args.root / args.paper
    repo = 'Setloop/llm-attacker-' + args.paper
    api = HfApi()
    api.repo_info(repo, repo_type='dataset')
    files = inventory(root)
    if args.small_only:
        files = {r:p for r,p in files.items() if p.stat().st_size < 1_000_000 and r != 'MANIFEST.sha256'}
    # Re-hash actual staged bytes, including MANIFEST, without trusting stale caches.
    local = {rel: hashes(p) for rel, p in files.items()}
    remote = {e.path:e for e in api.list_repo_tree(repo, repo_type='dataset', recursive=True)
              if hasattr(e, 'blob_id')}
    pending = [rel for rel in files if rel not in remote or not matches(remote[rel], local[rel])]
    if args.small_only:
        pending = [r for r in pending if files[r].stat().st_size < 1_000_000 and r != 'MANIFEST.sha256']
    pending.sort(key=lambda r:(r == 'MANIFEST.sha256', files[r].stat().st_size, r))
    print(f'{repo}: {len(files)} staged, {len(pending)} pending', flush=True)
    # Bound each commit to 500 files and approximately 256 MB.
    while pending:
        batch = []
        size = 0
        while pending and len(batch) < 500:
            rel = pending[0]
            n = files[rel].stat().st_size
            if batch and size + n > 256_000_000:break
            batch.append(pending.pop(0)); size += n
        for attempt in range(120):
            try:
                result = api.create_commit(repo, repo_type='dataset',
                    operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(files[r])) for r in batch],
                    commit_message=f'Evidence release: {len(batch)} files ({size} bytes)', num_threads=4)
                print(f'COMMIT {result.oid} files={len(batch)} bytes={size} remaining={len(pending)}',flush=True)
                break
            except Exception as exc:
                print(f'RETRY {attempt+1}: {type(exc).__name__}: {str(exc)[:200]}',flush=True)
                if attempt == 119: raise
                time.sleep(60 if getattr(getattr(exc, "response", None), "status_code", None) == 429 else min(60, 5 * 2**attempt))
    print('SMALL_TIER_DONE' if args.small_only else 'UPLOAD_DONE', flush=True)


if __name__ == '__main__':main()
