#!/usr/bin/env python3
"""Publish a frozen audit supplement only after baseline release verification.

The baseline snapshot is untouched. Supplement verification checks its own
manifest and every new Git/LFS object at an immutable remote revision.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from huggingface_hub import HfApi
from release_upload import hashes, inventory, matches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--baseline-pid', type=int, required=True)
    args = ap.parse_args()
    deadline = time.monotonic() + 24 * 3600
    marker = args.baseline / 'release-complete.json'
    print('WAITING_FOR_BASELINE_VERIFICATION', flush=True)
    while not marker.exists():
        os.kill(args.baseline_pid, 0)
        if time.monotonic() > deadline:
            raise TimeoutError('Baseline not verified within 24 hours')
        time.sleep(30)
    baseline = json.loads(marker.read_text())
    if len(baseline) != 2 or not all(r['ok'] for r in baseline):
        raise RuntimeError('Baseline verification failed')
    api = HfApi()
    reports = []
    for paper in ('paper-a', 'paper-b'):
        directory = args.root / paper
        files = inventory(directory)
        expected = {}
        for line in (directory / 'MANIFEST-AUDIT-20260917.sha256').read_text().splitlines():
            sha, path = line.split('  ', 1)
            expected[path] = sha
        if set(expected) != set(files) - {'MANIFEST-AUDIT-20260917.sha256'}:
            raise RuntimeError('Supplement manifest coverage mismatch')
        if any(hashes(files[p])[0] != sha for p, sha in expected.items()):
            raise RuntimeError('Supplement changed after staging')
        subprocess.run([sys.executable, 'experiments/release_upload.py', paper,
                        '--root', str(args.root)], check=True)
        repo = 'Setloop/llm-attacker-' + paper
        info = api.repo_info(repo, repo_type='dataset')
        remote = {e.path: e for e in api.list_repo_tree(repo, repo_type='dataset',
                  recursive=True, revision=info.sha) if hasattr(e, 'blob_id')}
        errors = [p for p, file in files.items()
                  if p not in remote or not matches(remote[p], hashes(file))]
        report = {'repo': repo, 'revision': info.sha, 'private': info.private,
                  'supplement_files': len(files), 'errors': errors, 'ok': not errors,
                  'baseline_revision': next(r['revision'] for r in baseline if r['repo'] == repo)}
        (args.root / (paper + '-verification.json')).write_text(json.dumps(report, indent=2) + '\n')
        if errors:
            raise RuntimeError('Remote supplement verification failed')
        reports.append(report)
    (args.root / 'supplement-complete.json').write_text(json.dumps(reports, indent=2) + '\n')
    print('SUPPLEMENT_COMPLETE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
