"""Verify and publish an additive amended agentic evidence tier at a pinned HF revision."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))


def publish_evidence(stage, output):
    from attacker.evidence import verify_evidence
    import release_paper_a_revision as release
    raw = stage / 'raw'
    report = json.loads((raw / 'FOLLOWUP.json').read_text())
    plan = json.loads((stage / 'agentic_amendment.v1.json').read_text())
    expected = {tuple(c) for c in plan['cells']}
    actual = {(r['attack'], r['arm'], r['world']) for r in report['results']}
    if len(report['results']) != 10 or actual != expected or any(
        r['status'] != 'MEASURED' or r['measurement']['trials'] != 2
        for r in report['results']
    ):
        raise ValueError('Complete matched ten-cell matrix required')
    if report['decode'] != {
        'thinking': False, 'greedy': True, 'max_new_tokens': 768, 'seed': 0,
    }:
        raise ValueError('Unexpected decoding condition')
    plan_hash = hashlib.sha256((stage / 'agentic_amendment.v1.json').read_bytes()).hexdigest()
    if plan_hash != report['plan_sha256']:
        raise ValueError('Protocol identity mismatch')
    ok, errors = verify_evidence(raw / 'FOLLOWUP.json.evidence.json')
    if not ok:
        raise ValueError(errors)
    traces_hash = hashlib.sha256((raw / 'planner-traces.jsonl').read_bytes()).hexdigest()
    if traces_hash != report['traces_sha256']:
        raise ValueError('Trace identity mismatch')
    rows = [json.loads(s) for s in (raw / 'planner-traces.jsonl').read_text().splitlines()]
    if {(r['attack'], r['arm'], r['world']) for r in rows} != expected:
        raise ValueError('Trace coverage incomplete')
    run = json.loads((stage / 'run-status.json').read_text())
    if run['returncode'] != 0 or run['wall_seconds'] + run['prior_canary_seconds'] > 9000:
        raise ValueError('Run failed or exceeds cumulative budget')
    release.REPO = 'Setloop/llm-attacker-paper-b'
    release.PREFIX = 'agentic-amendment-20260920'

    def status(state, **extra):
        value = {'status': state, **extra}
        release.save(output, value)
        print(json.dumps(value), flush=True)

    verified = release.publish(stage, 'evidence', status, 'Agentic non-thinking amendment')
    release.save(output.with_name('HF-VERIFICATION.json'), verified)
    status('verified', **verified)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    publish_evidence(args.stage, args.status)
