#!/usr/bin/env python3
"""Retrieve and verify the versioned reviewer evidence; never run experiments.

Python 3.11+, GitHub CLI (authenticated repository access), and zstd are required
for fetch. Verification alone uses the Python standard library.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
BUFFER = 4 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BUFFER), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in name:
        raise ValueError(f'Unsafe evidence path: {name!r}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Evidence path escapes destination: {name!r}')
    # Do not follow even an in-root symlink while restoring evidence.
    if any(parent.is_symlink() for parent in [target, *target.parents] if parent != root.parent):
        raise ValueError(f'Symlink in evidence destination: {name!r}')
    return target


def verify_record(root, record):
    path = safe_path(root, record['path'])
    if not path.is_file():
        return 'missing'
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        return 'hash_mismatch'
    return 'verified'


def verify_files(root, records):
    failures = []
    for record in records:
        status = verify_record(root, record)
        if status != 'verified':
            failures.append({'path': record['path'], 'status': status})
    return failures


def restore_tar(stream, root, records):
    """Accept only indexed regular files; never overwrite differing local evidence."""
    expected = {r['path']: r for r in records}
    seen = set()
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected archive member: {member.name!r}')
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError(f'Archive size mismatch: {member.name}')
            target = safe_path(root, member.name)
            if target.exists() and verify_record(root, record) != 'verified':
                raise ValueError(f'Refusing to overwrite different file: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            h = hashlib.sha256()
            # Temporary file is on the destination filesystem for atomic installation.
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.evidence-', delete=False) as out:
                temporary = Path(out.name)
                try:
                    for block in iter(lambda: source.read(BUFFER), b''):
                        h.update(block)
                        out.write(block)
                    out.flush()
                    if h.hexdigest() != record['sha256']:
                        raise ValueError(f'Extracted hash mismatch: {member.name}')
                    if not target.exists():
                        os.link(temporary, target)  # Fails if another writer created it.
                        target.chmod(record.get('mode', 0o644))
                finally:
                    temporary.unlink(missing_ok=True)
            seen.add(member.name)
    if seen != expected.keys():
        raise ValueError(f'Archive is missing {len(expected.keys() - seen)} indexed files')


def restore_parts(parts, root, records):
    errors = []
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(['zstd', '-q', '-d', '-c'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr)

        def feed():
            try:
                for path in parts:
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, process.stdin, BUFFER)
            except (OSError, ValueError) as exc:
                errors.append(exc)
            finally:
                process.stdin.close()

        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        try:
            restore_tar(process.stdout, root, records)
            # Drain padding so the decompressor completes and verifies its checksum.
            shutil.copyfileobj(process.stdout, _Discard(), BUFFER)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
            process.wait()
            thread.join()
        if process.returncode or errors:
            stderr.seek(0)
            raise RuntimeError(f'Decompression failed: {stderr.read().decode(errors="replace")} {errors}')


class _Discard:
    def write(self, data):
        return len(data)


def fetch(index, groups, root, cache):
    cache.mkdir(parents=True, exist_ok=True)
    for group in groups:
        records = [r for r in index['files'] if r['group'] == group]
        failures = verify_files(root, records)
        if not failures:
            print(f'{group}: {len(records)} files already hash-verified', flush=True)
            continue
        if any(r['status'] != 'missing' for r in failures):
            raise ValueError(f'{group}: local evidence differs; use a fresh checkout. {failures[:3]}')
        paths = []
        for asset in index['archives'][group]['parts']:
            target = safe_path(cache, asset['name'])
            if not target.exists():
                partial = target.with_name(target.name + '.partial')
                print(f'Downloading {asset["name"]} ({asset["bytes"]:,} bytes)', flush=True)
                subprocess.run(['gh', 'release', 'download', index['release_tag'],
                                '--repo', index['repository'], '--pattern', asset['name'],
                                '--output', str(partial), '--clobber'], check=True)
                if partial.stat().st_size != asset['bytes'] or digest(partial) != asset['sha256']:
                    raise ValueError(f'Download checksum mismatch: {asset["name"]}')
                partial.replace(target)
            if target.stat().st_size != asset['bytes'] or digest(target) != asset['sha256']:
                raise ValueError(f'Cached checksum mismatch: {target}')
            paths.append(target)
        print(f'Restoring {group}: {len(records)} indexed files', flush=True)
        restore_parts(paths, root, records)
        print(f'{group}: restored and verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'fetch', 'verify'])
    parser.add_argument('--index', type=Path, default=ROOT / 'evidence/release-20260910-v1.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache', type=Path, default=ROOT / '.evidence-downloads')
    parser.add_argument('--group', action='append', help='Repeat to select groups; default: all')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    if index['schema_version'] != 'reviewer-evidence-1':
        parser.error('Unsupported evidence index schema')
    groups = args.group or list(index['archives'])
    if set(groups) - index['archives'].keys():
        parser.error('Unknown group; use list to inspect the available groups')
    if args.action == 'list':
        for group in groups:
            records = [r for r in index['files'] if r['group'] == group]
            archive = index['archives'][group]
            print(f'{group}: {len(records)} files, {sum(r["bytes"] for r in records):,} extracted bytes; '
                  f'{sum(p["bytes"] for p in archive["parts"]):,} download bytes')
        return
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == 'fetch':
        fetch(index, groups, args.root, args.cache.resolve())
    records = [r for r in index['files'] if r['group'] in groups]
    failures = verify_files(args.root, records)
    result = {'release_tag': index['release_tag'], 'groups': groups,
              'files_checked': len(records), 'bytes_checked': sum(r['bytes'] for r in records),
              'failures': failures, 'status': 'verified' if not failures else 'failed',
              'scope': 'Byte availability and identity; no scientific recomputation.'}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
