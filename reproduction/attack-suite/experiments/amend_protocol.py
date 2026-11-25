"""Protocol amendment mechanism: frozen vN -> amended v(N+1) draft -> explicit re-freeze.

Pure stdlib. An amendment record (schema llm_attacker.validation_amendment.v1)
carries per-axis-change values, rationale and a REQUIRED hash-pinned evidence
pointer to the pilot-feasibility results justifying each change. The amended
protocol keeps the llm_attacker.validation.v1 shape with "version" incremented
from whatever the frozen source protocol carries (v1 -> v2, v2 -> v3, ...) and
starts as draft_for_review; --freeze regenerates the input SHA-256 pins and
writes status frozen. Freeze is refused once any held-out outcome access has
been journaled by the frozen-study runner.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AMENDMENT_SCHEMA = 'llm_attacker.validation_amendment.v1'
BASE_SCHEMA = 'llm_attacker.validation.v1'
PROTECTED_KEYS = ('schema', 'status', 'version', 'frozen_inputs', 'budget_review', 'amendment')
DEFAULT_BASE = Path('experiments/validation_protocol.v2.json')
DEFAULT_INPUT = Path('results/validation-corpus-v1/validation-input.json')
DEFAULT_SUMMARY = Path('results/baseline-strengthening/analysis/summary.csv')
DEFAULT_JOURNAL = Path('results/heldout_access.journal')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_version(p):
    version = p.get('version', 1)
    if type(version) is not int or version < 1:
        raise ValueError('invalid protocol version')
    return version


def load_amendment(path):
    record = json.loads(Path(path).read_text())
    if record.get('schema') != AMENDMENT_SCHEMA:
        raise ValueError('invalid amendment schema')
    changes = record.get('changes')
    if not isinstance(changes, list) or not changes:
        raise ValueError('amendment requires at least one change')
    for change in changes:
        axis = change.get('axis')
        if not isinstance(axis, str) or not axis or axis in PROTECTED_KEYS:
            raise ValueError(f'invalid or protected amendment axis: {axis!r}')
        if 'value' not in change:
            raise ValueError(f'amendment change {axis!r} requires a new value')
        if not isinstance(change.get('rationale'), str) or not change['rationale'].strip():
            raise ValueError(f'amendment change {axis!r} requires a rationale')
        if 'new_axis' in change and change['new_axis'] is not True:
            raise ValueError(f'amendment change {axis!r}: new_axis must be exactly true when present')
        evidence = change.get('evidence')
        if not isinstance(evidence, dict) or not evidence.get('path') or not evidence.get('sha256'):
            raise ValueError(f'amendment change {axis!r} requires an evidence pointer (path + sha256)')
        evidence_path = Path(evidence['path'])
        if not evidence_path.is_file() or sha256(evidence_path) != evidence['sha256']:
            raise ValueError(f'amendment evidence hash mismatch for change {axis!r}')
    return record


def budget_summary(protocol, rows, documents):
    from experiments.review_validation_budget import estimate
    result = estimate(protocol, rows, documents)
    return {'logical_cells': result['cells'], 'physical_executions': result['physical_cells'],
            'estimated_online_gb10_gpu_hours': result['estimated_online_gpu_hours']}


def amend(base_path, amendment_path, output_path, *, freeze=False, input_path=DEFAULT_INPUT,
          summary_rows=None, journal_path=DEFAULT_JOURNAL):
    base_path, output_path = Path(base_path), Path(output_path)
    base = json.loads(base_path.read_text())
    if base.get('schema') != BASE_SCHEMA or base.get('status') != 'frozen':
        raise ValueError('amendments start from a frozen protocol')
    base_version = protocol_version(base)
    new_version = base_version + 1
    record = load_amendment(amendment_path)
    amended = copy.deepcopy(base)
    for change in record['changes']:
        if change['axis'] not in amended and change.get('new_axis') is not True:
            raise ValueError(f"amendment axis {change['axis']!r} is not a protocol key; "
                             "introducing a new axis requires an explicit 'new_axis': true")
        amended[change['axis']] = copy.deepcopy(change['value'])
    amended['version'] = new_version
    amended['status'] = 'draft_for_review'
    if summary_rows is None:
        with Path(DEFAULT_SUMMARY).open(newline='') as handle:
            summary_rows = list(csv.DictReader(handle))
    documents = base['frozen_inputs']['partition_counts']['evaluation']
    base_budget = budget_summary(base, summary_rows, documents)
    new_budget = budget_summary(amended, summary_rows, documents)
    record['budget'] = {f'v{base_version}': base_budget, f'v{new_version}': new_budget,
                        'delta': {key: new_budget[key] - base_budget[key] for key in base_budget}}
    amended['amendment'] = {'record': record, 'record_sha256': sha256(amendment_path),
                            'base_protocol_sha256': sha256(base_path),
                            'base_version': base_version}
    if freeze:
        journal_path = Path(journal_path)
        if journal_path.exists() and journal_path.read_text().strip():
            raise ValueError('held-out outcome access already journaled; protocol cannot be re-frozen')
        input_path = Path(input_path)
        if not input_path.is_file():
            raise ValueError('frozen input file required to regenerate pins')
        data = json.loads(input_path.read_text())
        counts = {key: len(data[key]) for key in ('calibration', 'train', 'evaluation')}
        if counts != amended['frozen_inputs']['partition_counts']:
            raise ValueError('frozen input partition counts differ from protocol')
        amended['frozen_inputs'] = dict(amended['frozen_inputs'],
                                        tokenized_sha256=sha256(input_path),
                                        name_annotations_sha256=sha256(amended['frozen_inputs']['name_annotations']),
                                        curation_manifest_sha256=sha256(amended['frozen_inputs']['manifest']))
        amended['budget_review'] = dict(amended['budget_review'], **new_budget)
        amended['status'] = 'frozen'
    if output_path.exists():
        raise ValueError('output must be fresh')
    output_path.write_text(json.dumps(amended, indent=2) + '\n')
    return amended


def default_output(base_path):
    base = json.loads(Path(base_path).read_text())
    return Path(f'experiments/validation_protocol.v{protocol_version(base) + 1}.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None,
                        help='default: validation_protocol.v<N+1>.json derived from the base version')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--journal', type=Path, default=DEFAULT_JOURNAL)
    args = parser.parse_args()
    output = args.output or default_output(args.base)
    with args.summary.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    amended = amend(args.base, args.amendment, output, freeze=args.freeze,
                    input_path=args.input, summary_rows=rows, journal_path=args.journal)
    budget = amended['amendment']['record']['budget']
    print(json.dumps({'output': str(output), 'version': amended['version'], 'status': amended['status'],
                      'budget': budget}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
