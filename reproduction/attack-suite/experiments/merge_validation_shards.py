"""Merge sharded frozen-validation reports into one analyzer-ready report.

Pure stdlib. Takes the per-shard output directories produced by
experiments/validate_reconstruction.py --shard-count N --shard-index i and
writes a single merged report directory that experiments/analyze_validation.py
accepts unchanged (--protocol-version 2). Fails closed with a descriptive
error on any mismatch:

- every shard's evidence chain must verify (attacker.evidence conventions)
- every shard report must be COMPLETE and carry a consistent shard record;
  the shard indices must be exactly {0..N-1}, one report each
- protocol source bytes, protocol/effective-protocol/input SHA256 and the
  supplied input binding must be identical across shards
- each shard's cells must be exactly its deterministic assignment, so their
  union is disjoint and equals the frozen protocol's full cell matrix
- artifact files shared across shards must hash identically

Destination collisions are resolved deterministically and never silently.
Every file the merge writes is registered as (destination, sha256): two
sources mapping to one destination with divergent content refuse the merge
(fail closed, never overwrite). Defence sidecars (defense_provenance /
defended_observations) are written under per-shard subtrees
(shard<index>/<shard-relative path>) because each shard trains its own copy
of every defended victim its cells need, so one shard-relative provenance
path carries different bytes per shard (the provenance journal embeds the
shard's own output path). The rewritten destinations are recorded in the
merged cells and defended_victims records, summarized in the report's
merge block, and committed to the shards_merged evidence event.

With --json-tier the sources carry only the repository (JSON) tier: the
binary capture tier (*.pt artifacts, defended observation tensors, the
captures/ provenance sidecars) is absent. References whose files are
missing from every shard are then deferred — carried with their recorded
hashes and listed under merge.deferred_reference_paths — instead of
refusing the merge. Every file that IS present is still hash-checked and
any mismatch refuses the merge. Prediction transcripts and the
protocol/input triple belong to the JSON tier and are always required.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger, file_sha256, verify_evidence
from experiments.validate_reconstruction import expected_cell_ids, shard_assignment

SIDECAR_FIELDS = ('defense_provenance', 'defended_observations')
SIDECAR_PATH_SCHEME = 'shard<shard_index>/<shard-relative path>'


def fail(message):
    raise ValueError(f'shard merge refused: {message}')


def sidecar_destination(shard_index, path):
    """Deterministic merged-relative destination for a defence sidecar."""
    return f'shard{shard_index}/{path}'


def safe_join(root, relative):
    """Join a recorded relative path onto root, refusing escapes."""
    relative = str(relative)
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts:
        fail(f'unsafe recorded path: {relative!r}')
    return Path(root) / candidate


class DestinationRegistry:
    """One writer per destination: refuse divergent content, never overwrite."""

    def __init__(self):
        self._planned = {}

    def plan(self, destination, sha256, kind, source):
        """Register (destination, sha256); True iff the file still needs copying."""
        destination = str(destination)
        previous = self._planned.get(destination)
        if previous is not None:
            if previous['sha256'] != sha256:
                fail(f'destination collision for {destination}: {source} offers '
                     f'{kind} sha256 {sha256}, but {previous["source"]} already planned '
                     f'{previous["sha256"]} — refusing to overwrite')
            return False
        self._planned[destination] = {'sha256': sha256, 'kind': kind,
                                      'source': str(source)}
        return True


def load_shard(directory):
    directory = Path(directory)
    report_path = directory / 'validation.json'
    if not report_path.is_file():
        fail(f'{directory}: validation.json missing')
    ok, problems = verify_evidence(str(directory / 'validation.json.evidence.json'))
    if not ok:
        fail(f'{directory}: evidence verification failed: {problems}')
    report = json.loads(report_path.read_text())
    if report.get('status') != 'COMPLETE':
        fail(f'{directory}: shard report status is {report.get("status")!r}, not COMPLETE')
    shard = report.get('shard')
    if not isinstance(shard, dict) or type(shard.get('shard_count')) is not int \
            or type(shard.get('shard_index')) is not int:
        fail(f'{directory}: shard record missing or invalid (re-run with --shard-count/--shard-index)')
    cell_ids = [cell['cell_id'] for cell in report['cells']]
    if sorted(cell_ids) != sorted(shard.get('cell_ids', [])):
        fail(f'{directory}: shard record cell_ids differ from the report cells')
    # Per-cell evidence: every reported cell must appear in the shard's
    # hash-chained event stream as completed or reused.
    logged = set()
    for line in (directory / 'validation.json.evidence.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') in ('cell_complete', 'cell_reused'):
            logged.add(event['payload'].get('cell_id'))
    missing = set(cell_ids) - logged
    if missing:
        fail(f'{directory}: cells missing from the evidence event stream: {sorted(missing)[:3]}...')
    return directory, report


def merge(shard_dirs, output, json_tier=False):
    output = Path(output)
    if output.exists():
        fail(f'output must be fresh: {output}')
    shards = [load_shard(d) for d in shard_dirs]
    count = shards[0][1]['shard']['shard_count']
    if count < 1 or any(r['shard']['shard_count'] != count for _, r in shards):
        fail('inconsistent shard_count across shard reports')
    indices = [r['shard']['shard_index'] for _, r in shards]
    if len(set(indices)) != len(indices):
        fail(f'duplicate shard index in {sorted(indices)}')
    if set(indices) != set(range(count)):
        fail(f'shard indices {sorted(indices)} do not cover [0,{count}) — missing shard(s)')
    shards.sort(key=lambda pair: pair[1]['shard']['shard_index'])

    base = shards[0][1]
    for field in ('protocol_sha256', 'effective_protocol_sha256', 'input_sha256',
                  'supplied_input_sha256', 'evidence_tier'):
        if any(r.get(field) != base.get(field) for _, r in shards):
            fail(f'{field} differs across shards')
    for name in ('input.json', 'protocol-source.json', 'protocol-effective.json'):
        digests = {file_sha256(d / name) for d, _ in shards}
        if len(digests) != 1:
            fail(f'{name} bytes differ across shards')
    if any(r['protocol'] != base['protocol'] for _, r in shards):
        fail('embedded protocol differs across shards')
    p = base['protocol']

    # Coverage: each shard must hold exactly its deterministic assignment;
    # this makes the union disjoint and equal to the full cell matrix.
    assignment = shard_assignment(p, count)
    expected = set(expected_cell_ids(p))
    union = set()
    for directory, report in shards:
        index = report['shard']['shard_index']
        want = {cid for cid, shard in assignment.items() if shard == index}
        have = {cell['cell_id'] for cell in report['cells']}
        if have != want:
            fail(f'{directory}: shard {index} holds {len(have)} cells, assignment requires '
                 f'{len(want)} (missing {sorted(want - have)[:3]}, unexpected {sorted(have - want)[:3]})')
        union |= have
    if union != expected:
        fail(f'cell union differs from the protocol matrix: '
             f'missing {sorted(expected - union)[:3]}, unexpected {sorted(union - expected)[:3]}')

    registry = DestinationRegistry()
    deferred = set()

    # Artifacts: shared files must be byte-identical wherever both shards
    # produced them; the union must cover every recorded hash. Under
    # --json-tier an artifact absent from every shard is deferred (carried
    # with its recorded hash), not fatal; any present copy must match.
    artifacts = {}
    artifact_sources = {}
    for directory, report in shards:
        for name, digest in report['artifacts'].items():
            if name in artifacts and artifacts[name] != digest:
                fail(f'artifact hash conflict for {name} across shards')
            artifacts[name] = digest
            source = safe_join(directory, name)
            if source.is_file():
                if file_sha256(source) != digest:
                    fail(f'{directory}: recorded hash mismatch for {name}')
                artifact_sources.setdefault(name, source)
            elif not json_tier:
                fail(f'{directory}: recorded artifact missing: {name}')

    output.mkdir(parents=True)
    for name in ('input.json', 'protocol-source.json', 'protocol-effective.json'):
        source = shards[0][0] / name
        if not source.is_file():
            fail(f'{shards[0][0]}: {name} missing')
        registry.plan(name, file_sha256(source), 'protocol_input', source)
        shutil.copy2(source, output / name)
    for name, digest in sorted(artifacts.items()):
        source = artifact_sources.get(name)
        if source is None:
            deferred.add(name)
            continue
        if registry.plan(name, digest, 'artifact', source):
            shutil.copy2(source, output / name)

    cells_by_id = {}
    for directory, report in shards:
        for cell in report['cells']:
            cells_by_id[cell['cell_id']] = (report['shard']['shard_index'], directory, cell)
    merged_cells = []
    rewritten_refs = 0
    for cid in expected_cell_ids(p):
        index, directory, cell = cells_by_id[cid]
        prediction = safe_join(directory, cell['prediction_file'])
        if not prediction.is_file():
            fail(f'{directory}: prediction file missing for {cid}')
        if file_sha256(prediction) != cell['prediction_sha256']:
            fail(f'{directory}: prediction hash mismatch for {cid}')
        if registry.plan(cell['prediction_file'], cell['prediction_sha256'],
                         'prediction', prediction):
            shutil.copy2(prediction, safe_join(output, cell['prediction_file']))
        merged_cell = copy.deepcopy(cell)
        # Defended cells reference provenance/observation sidecars; copy them
        # so the merged directory carries the full per-cell evidence. The
        # destination is shard-prefixed: one shard-relative path holds
        # different bytes per shard, so a flat destination would collide.
        for field in SIDECAR_FIELDS:
            ref = cell.get(field)
            if ref and ref.get('path'):
                if not isinstance(ref.get('sha256'), str):
                    fail(f'{directory}: {field} sidecar reference without sha256 for {cid}')
                destination = sidecar_destination(index, ref['path'])
                source = safe_join(directory, ref['path'])
                if source.is_file():
                    if file_sha256(source) != ref['sha256']:
                        fail(f'{directory}: {field} sidecar hash mismatch for {cid}')
                    if registry.plan(destination, ref['sha256'], field, source):
                        target = safe_join(output, destination)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                elif json_tier:
                    registry.plan(destination, ref['sha256'], field,
                                  f'{directory}:{ref["path"]}')
                    deferred.add(destination)
                else:
                    fail(f'{directory}: {field} sidecar missing for {cid}: {ref["path"]}')
                merged_cell[field] = dict(ref, path=destination)
                rewritten_refs += 1
        merged_cells.append(merged_cell)

    merged = dict(base)
    merged['cells'] = merged_cells
    seen_setup = {}
    for _, report in shards:
        for setup in report['setup']:
            seen_setup.setdefault(setup['id'], setup)
    merged['setup'] = list(seen_setup.values())
    seen_victims = {}
    for _, report in shards:
        for victim in report['victims']:
            seen_victims.setdefault(victim['seed'], victim)
    merged['victims'] = list(seen_victims.values())
    # Each shard trains its own defended victims, so the per-victim records
    # differ across shards (their provenance hashes do). Keep one record per
    # (shard, seed, depth, defense) with the same shard-prefixed path
    # rewrite as the cell sidecars; identical duplicates collapse.
    seen_defended = {}
    for directory, report in shards:
        index = report['shard']['shard_index']
        for victim in report.get('defended_victims', []):
            record = copy.deepcopy(victim)
            record['capture_dir'] = sidecar_destination(index, victim['capture_dir'])
            provenance = victim.get('provenance')
            if provenance and provenance.get('path'):
                record['provenance'] = dict(provenance, path=sidecar_destination(
                    index, provenance['path']))
            key = (index, victim['seed'], victim['depth'], victim['defense'])
            if key in seen_defended and seen_defended[key] != record:
                fail(f'{directory}: conflicting defended_victims record for '
                     f'{key[1]} d{key[2]} {key[3]}')
            seen_defended.setdefault(key, record)
    merged['defended_victims'] = list(seen_defended.values())
    merged['artifacts'] = artifacts
    merged['physical_execution_count'] = sum(not c['measurement_reused'] for c in merged_cells)
    merged['logical_cell_count'] = len(merged_cells)
    merged['status'] = 'COMPLETE'
    merged['shard'] = {'shard_count': count, 'merged_from': indices,
                       'cell_ids': sorted(expected)}
    merged['shard_runtimes'] = [r.get('runtime') for _, r in shards]
    merged['merge'] = {'tool': 'experiments/merge_validation_shards.py',
                       'sidecar_path_scheme': SIDECAR_PATH_SCHEME,
                       'rewritten_sidecar_references': rewritten_refs,
                       'deferred_reference_paths': sorted(deferred),
                       'json_tier': json_tier}

    artifact = output / 'validation.json'
    with EvidenceLogger(artifact) as ev:
        ev.event('shards_merged', {'shard_count': count, 'shard_indices': indices,
                                   'source_manifests': [file_sha256(d / 'validation.json.evidence.json')
                                                        for d, _ in shards],
                                   'sidecar_path_scheme': SIDECAR_PATH_SCHEME,
                                   'rewritten_sidecar_references': rewritten_refs,
                                   'deferred_reference_count': len(deferred),
                                   'json_tier': json_tier})
        artifact.write_text(json.dumps(merged, indent=2, allow_nan=False))
    ok, problems = verify_evidence(str(output / 'validation.json.evidence.json'))
    if not ok:
        fail(f'merged evidence chain does not verify: {problems}')
    return {'output': str(output), 'shard_count': count, 'shard_indices': indices,
            'logical_cells': merged['logical_cell_count'],
            'physical_executions': merged['physical_execution_count'],
            'protocol_sha256': merged['protocol_sha256'],
            'rewritten_sidecar_references': rewritten_refs,
            'deferred_references': len(deferred),
            'manifest_sha256': file_sha256(output / 'validation.json.evidence.json')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('shards', nargs='+', type=Path, help='per-shard output directories')
    ap.add_argument('--output', type=Path, required=True, help='fresh merged output directory')
    ap.add_argument('--json-tier', action='store_true',
                    help='sources carry only the repository (JSON) tier: defer references '
                         'whose files are absent (binary capture tier) instead of refusing; '
                         'every present file is still hash-checked')
    args = ap.parse_args()
    print(json.dumps(merge(args.shards, args.output, json_tier=args.json_tier), indent=2))
