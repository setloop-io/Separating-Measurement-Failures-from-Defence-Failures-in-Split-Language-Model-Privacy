"""Publish the independently reproduced Sasa descriptive evidence supplement."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import release_paper_a_revision as release

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    reproduction = json.loads((args.stage / 'REPRODUCTION.json').read_text())
    if reproduction['status'] != 'PASS' or reproduction['hosts'] != ['sasa', 'soula']:
        raise ValueError('Independent CPU reproduction required')
    for name in ('analysis.json', 'trajectory.csv'):
        expected = reproduction['outputs_sha256'][name]
        for host in reproduction['hosts']:
            if digest(args.stage / ('analysis-' + host) / name) != expected:
                raise ValueError('Reproduction output changed: ' + host + '/' + name)
    report = json.loads((args.stage / 'analysis-sasa/analysis.json').read_text())
    matrix = {(x['defense'], x['seed']) for x in report['cells']}
    if (len(report['cells']) != 6
            or matrix != {(d, s) for d in ('none', 'mid-lite') for s in (11, 22, 33)}
            or any(x['n_steps'] != 500 or x['n_tokens'] != 16000 for x in report['cells'])
            or report['raw_files_verified'] != 12443
            or report['evidence_journals_verified'] != 13
            or report['grant_verdict'] != 'QUALIFIED_PUBLIC_SURROGATE_DESCRIPTIVE_ONLY'
            or report['privacy_gate_reissued'] or report['heldout_convergence_claim']):
        raise ValueError('Complete qualified descriptive audit required')
    if digest(args.stage / 'raw.tar.gz') != reproduction['raw_archive_sha256']:
        raise ValueError('Sealed raw archive changed')
    release.REPO = 'Setloop/llm-attacker-paper-b'
    release.PREFIX = 'sasa-500step-20260920'

    def status(state, **extra):
        value = {'status': state, **extra}
        release.save(args.status, value)
        print(json.dumps(value), flush=True)

    verified = release.publish(args.stage, 'evidence', status,
                               'Sasa 500-step descriptive extension')
    release.save(args.status.with_name('HF-VERIFICATION.json'), verified)
    status('verified', **verified)


if __name__ == '__main__':
    main()
