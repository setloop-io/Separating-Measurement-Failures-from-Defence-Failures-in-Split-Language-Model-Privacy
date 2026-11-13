"""Execute the frozen follow-up queue on existing nodes, preserving failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bundle(output, seed):
    import torch
    from attacker.defenses import ReferenceDefense
    from attacker.evidence import EvidenceLogger
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    meta = root/'generation.json'
    with EvidenceLogger(meta) as ev:
        for name in ('plain-split', 'composite'):
            path = root/(name+'.pt')
            torch.save(ReferenceDefense(name).generate(seed), path)
        meta.write_text(json.dumps({'seed': seed, 'scope': 'synthetic reference bundle; not external v13',
            'files': {p.name: sha(p) for p in root.glob('*.pt')}}, indent=2)+'\n')
        ev.event('generated', {'seed': seed})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', default='experiments/phase2_pending_20260917.json')
    ap.add_argument('--node')
    ap.add_argument('--bundle')
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()
    if args.bundle:
        return bundle(args.bundle, args.seed)
    os.chdir(ROOT)
    spec = json.loads(Path(args.protocol).read_text())
    for name, digest in spec['inputs'].items():
        if sha(name) != digest:
            raise ValueError('Pinned input mismatch: '+name)
    root = ROOT/'results/phase2-pending-v1'/args.node
    root.mkdir(parents=True, exist_ok=False)
    status = root/'status.json'
    def record(state, **extra):
        value = {'status': state, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'protocol_sha256': sha(args.protocol), **extra}
        status.write_text(json.dumps(value, indent=2)+'\n')
        print(json.dumps(value), flush=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    replacements = {'@PY': sys.executable, '@OUT': str(root), '@ROOT': str(ROOT),
                    '@GPT2': str(Path.home()/'.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'),
                    '@QWEN': str(Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')}
    def expand(s):
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s
    failed = []
    try:
        if args.node == 'koula':
            dependency = Path.home()/'llm-attacker-p6-qwen-20260917/results/qwen-queue-status.json'
            deadline = time.monotonic()+26*3600
            record('waiting_for_qwen_gpu_release', dependency=str(dependency))
            while True:
                state = json.loads(dependency.read_text()) if dependency.exists() else {}
                if state.get('status') in ('complete', 'failed', 'budget_blocked'):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('GPU predecessor has not terminated')
                time.sleep(30)
        for job in spec['queues'][args.node]:
            if any(d in failed for d in job.get('requires', [])):
                failed.append(job['id'])
                record('dependency_blocked', job=job['id'])
                continue
            command = [expand(x) for x in job['argv']]
            record('running', job=job['id'], command=command)
            start = time.monotonic()
            try:
                with (root/(job['id']+'.log')).open('x') as log:
                    rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=job['timeout_s']).returncode
                if rc:
                    raise RuntimeError('Exit code '+str(rc))
                from attacker.evidence import verify_evidence
                for name in job['verify']:
                    path = Path(expand(name))
                    if not path.is_file():
                        raise ValueError('Missing result '+str(path))
                    manifest = path.with_name(path.name+'.evidence.json')
                    ok, errors = verify_evidence(manifest)
                    if not ok:
                        raise ValueError(str(errors))
                entry = {'job': job['id'], 'status': 'complete', 'wall_s': time.monotonic()-start}
            except Exception as error:
                failed.append(job['id'])
                entry = {'job': job['id'], 'status': 'failed', 'error': str(error), 'wall_s': time.monotonic()-start}
            with (root/'journal.jsonl').open('a') as f:
                f.write(json.dumps(entry)+'\n')
            record(**{ 'state': entry.pop('status'), **entry})
        record('complete' if not failed else 'completed_with_failures', failed=failed)
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
