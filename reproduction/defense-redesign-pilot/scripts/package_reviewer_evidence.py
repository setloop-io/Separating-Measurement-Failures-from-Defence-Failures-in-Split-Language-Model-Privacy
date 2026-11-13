#!/usr/bin/env python3
"""Package existing evidence without running or rewriting any experiment.

The snapshot includes all regular files in outputs/, data/raw/, and
experiments/manifests/. Source and analysis code are retained in Git; Paper 4's
historical source archives are also inside outputs/. Run before publication,
then commit the generated index and upload its listed parts to the release.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import stat
import subprocess
import tarfile

from reviewer_evidence import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--release-tag', default='evidence-2026-09-10-v1')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--part-bytes', type=int, default=1_500_000_000)
    args = parser.parse_args()
    root = args.root.resolve()
    staging = args.staging.resolve()
    if staging.is_relative_to(root) or not 0 < args.part_bytes < 2**31:
        parser.error('Use staging outside the repository and parts smaller than 2 GiB')
    staging.mkdir(parents=True, exist_ok=True)
    files = []
    for directory in ['outputs', 'data/raw', 'experiments/manifests']:
        for path in sorted((root / directory).rglob('*')):
            if path.name == '.DS_Store' or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink is not a retained regular artifact: {path}')
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if directory == 'outputs':
                parts = path.relative_to(root / directory).parts
                group = parts[0] if len(parts) > 1 else 'output_notes'
                if parts[0] in ['runs', 'scenarios']:
                    group = parts[1]
            else:
                group = 'corpora' if directory == 'data/raw' else 'manifests'
            files.append({'path': relative, 'group': group,
                          'bytes': path.stat().st_size, 'sha256': digest(path),
                          'mode': stat.S_IMODE(path.stat().st_mode) & 0o777})
    indexed = {r['path']: r for r in files}
    references = {}
    for path in sorted((root / 'experiments/manifests').glob('*.json')):
        manifest = json.loads(path.read_text())
        for artifact in manifest.get('artifacts', []):
            name = artifact['path']
            if name not in indexed or indexed[name]['sha256'] != artifact['sha256']:
                raise ValueError(f'Manifest artifact unavailable or changed: {name}')
            if name in references and references[name] != artifact['sha256']:
                raise ValueError(f'Conflicting historical artifact hashes: {name}')
            references[name] = artifact['sha256']
    index = {'schema_version': 'reviewer-evidence-1', 'repository': args.repository,
             'release_tag': args.release_tag, 'packaged_at': datetime.now(timezone.utc).isoformat(),
             'source_baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
             'scope': 'Recovered original files. Packaging date is not execution or preregistration date. '
                      'Availability and hashes do not establish scientific validity or recover missing historical source.',
             'manifest_reference_count': len(references),
             'manifest_referenced_bytes': sum(indexed[name]['bytes'] for name in references),
             'files': files, 'archives': {}}
    for group in sorted({r['group'] for r in files}):
        selected = [r for r in files if r['group'] == group]
        name = group + '.tar.zst'
        archive = staging / name
        print(f'Packaging {group}: {len(selected)} files, {sum(r["bytes"] for r in selected):,} bytes', flush=True)
        with archive.open('xb') as output:
            process = subprocess.Popen(['zstd', '-q', '-T4', '-3', '-c'], stdin=subprocess.PIPE, stdout=output)
            try:
                with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                    for record in selected:
                        path = root / record['path']
                        info = tar.gettarinfo(str(path), arcname=record['path'])
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        # Byte hashes bind evidence; archive timestamps describe packaging only.
                        info.mtime = 0
                        with path.open('rb') as source:
                            tar.addfile(info, source)
                process.stdin.close()
                if process.wait():
                    raise RuntimeError(f'zstd failed for {group}')
            except BaseException:
                process.kill()
                process.wait()
                raise
        subprocess.run(['zstd', '-q', '-t', str(archive)], check=True)
        entry = {'format': 'concatenated-tar.zst', 'sha256': digest(archive), 'parts': []}
        with archive.open('rb') as source:
            part_number = 1
            while True:
                block = source.read(min(4 * 1024 * 1024, args.part_bytes))
                if not block:
                    break
                part = staging / f'{name}.part{part_number:03d}'
                written = 0
                with part.open('xb') as output:
                    while block:
                        output.write(block)
                        written += len(block)
                        if written == args.part_bytes:
                            break
                        block = source.read(min(4 * 1024 * 1024, args.part_bytes - written))
                entry['parts'].append({'name': part.name, 'bytes': written, 'sha256': digest(part)})
                part_number += 1
        index['archives'][group] = entry
        print(f'{group}: {sum(p["bytes"] for p in entry["parts"]):,} compressed bytes, {len(entry["parts"])} parts', flush=True)
        (staging / 'progress.json').write_text(json.dumps(index, indent=2) + '\n')
    # A second pass detects mutations while packaging, without changing originals.
    for record in files:
        if digest(root / record['path']) != record['sha256']:
            raise ValueError(f'Evidence changed during packaging: {record["path"]}')
    target = root / 'evidence/release-20260910-v1.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, indent=2) + '\n')
    (staging / target.name).write_bytes(target.read_bytes())
    checksums = [f'{digest(target)}  {target.name}']
    for archive in index['archives'].values():
        checksums.extend(f'{part["sha256"]}  {part["name"]}' for part in archive['parts'])
    (staging / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    print(f'Packaged {len(files)} files; {len(references)} unique manifest artifacts verified.', flush=True)


if __name__ == '__main__':
    main()
