"""Execution sharding for the frozen-study runner — smoke tier only.

All runner invocations here use --smoke (synthetic partitions, tiny random
GPT-2, engineering_smoke tier). Smoke mode never reads held-out outcomes and
never writes results/heldout_access.journal; that invariant is asserted.
"""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from attacker.evidence import file_sha256
from experiments import merge_validation_shards as mvs
from experiments import validate_reconstruction as vr
from experiments.analyze_validation import validate_report
from experiments.merge_validation_shards import merge

V2 = Path('experiments/validation_protocol.v2.json')
JOURNAL = Path('results/heldout_access.journal')


def journal_snapshot():
    return JOURNAL.read_bytes() if JOURNAL.exists() else None


def smoke_run(output, shard_count=1, shard_index=0, shard_flags=True):
    args = SimpleNamespace(protocol=V2, input=None, input_sha256=None,
                           output=Path(output), device='cpu', smoke=True)
    if shard_flags:
        args.shard_count, args.shard_index = shard_count, shard_index
    assert vr.run(args) == 0
    return json.loads((Path(output) / 'validation.json').read_text())


@pytest.fixture(scope='module')
def smoke_matrix(tmp_path_factory):
    """Unsharded run plus a two-shard pair of the same smoke protocol."""
    before = journal_snapshot()
    root = tmp_path_factory.mktemp('shardsmoke')
    full = smoke_run(root / 'full', shard_flags=False)
    one = smoke_run(root / 'one', shard_count=1, shard_index=0)
    s0 = smoke_run(root / 's0', shard_count=2, shard_index=0)
    s1 = smoke_run(root / 's1', shard_count=2, shard_index=1)
    assert journal_snapshot() == before, 'smoke shards must not touch the held-out access journal'
    return root, full, one, s0, s1


def smoke_protocol():
    """The effective (smoke-reduced) protocol the matrix is computed from."""
    p = json.loads(V2.read_text())
    p.update(depths=[1], victim_seeds=[101, 202], attack_seeds=[17, 29], sequence_length=4,
             minimum_documents=3, train_steps=2, candidate_counts=[4], sipit_steps=[2])
    return p


def test_partition_deterministic_covering_disjoint():
    p = smoke_protocol()
    expected = set(vr.expected_cell_ids(p))
    for count in (1, 2, 3):
        first = vr.shard_assignment(p, count)
        assert vr.shard_assignment(p, count) == first, 'partition must be reproducible'
        shards = [set() for _ in range(count)]
        for cid, shard in first.items():
            assert 0 <= shard < count
            shards[shard].add(cid)
        assert set().union(*shards) == expected, 'shards must cover the full matrix'
        for i in range(count):
            for j in range(i + 1, count):
                assert not shards[i] & shards[j], 'shards must be disjoint'


def test_partition_interleaves_depths_and_reuse_pairs():
    p = json.loads(V2.read_text())  # frozen matrix: depths {3,6}, 10 seeds
    for count in (2, 3):
        assignment = vr.shard_assignment(p, count)
        for shard in range(count):
            depths = {int(cid.split('-')[1]) for cid, s in assignment.items() if s == shard}
            assert depths == set(p['depths']), 'depth must not be confounded with shard/host'
        for cid, shard in assignment.items():
            if '-inference-' in cid:
                origin = cid.replace('-inference-', '-training-')
                assert assignment[origin] == shard, 'reused inference cell must share its origin\'s shard'


def test_shard_validation_rejects_bad_indices(tmp_path):
    for count, index in ((0, 0), (2, 2), (2, -1)):
        args = SimpleNamespace(protocol=V2, input=None, input_sha256=None,
                               output=tmp_path / f'bad-{count}-{index}', device='cpu', smoke=True,
                               shard_count=count, shard_index=index)
        with pytest.raises(ValueError, match='shard'):
            vr.run(args)


def test_shard_zero_of_one_equals_unsharded(smoke_matrix):
    _, full, one, _, _ = smoke_matrix
    full_ids = [c['cell_id'] for c in full['cells']]
    one_ids = [c['cell_id'] for c in one['cells']]
    assert one_ids == full_ids
    assert one['shard'] == {'shard_count': 1, 'shard_index': 0, 'cell_ids': sorted(full_ids)}
    assert one['status'] == full['status'] == 'COMPLETE'
    assert one['physical_execution_count'] == full['physical_execution_count']
    assert one['logical_cell_count'] == full['logical_cell_count']


def test_shards_partition_the_run(smoke_matrix):
    _, full, _, s0, s1 = smoke_matrix
    ids0 = {c['cell_id'] for c in s0['cells']}
    ids1 = {c['cell_id'] for c in s1['cells']}
    assert ids0.isdisjoint(ids1)
    assert ids0 | ids1 == {c['cell_id'] for c in full['cells']}
    assert s0['shard']['shard_index'] == 0 and s1['shard']['shard_index'] == 1
    # Both shards must span defended and undefended cells.
    for ids in (ids0, ids1):
        assert any('@' in cid for cid in ids) and any('@' not in cid for cid in ids)


@pytest.fixture(scope='module')
def merged(smoke_matrix, tmp_path_factory):
    root, _, _, _, _ = smoke_matrix
    out = tmp_path_factory.mktemp('merged') / 'run'
    summary = merge([root / 's0', root / 's1'], out)
    return out, summary


def test_merge_accepted_by_analyzer(merged):
    out, summary = merged
    assert summary['logical_cells'] == 128 and summary['shard_count'] == 2
    report, cells = validate_report(out, allowed_protocol_versions=(2,))
    assert len(cells) == summary['logical_cells']
    assert report['status'] == 'COMPLETE'


def test_merge_rejects_missing_shard(smoke_matrix, tmp_path):
    root, _, _, _, _ = smoke_matrix
    with pytest.raises(ValueError, match='cover'):
        merge([root / 's0'], tmp_path / 'out')


def test_merge_rejects_duplicate_shard(smoke_matrix, tmp_path):
    root, _, _, _, _ = smoke_matrix
    with pytest.raises(ValueError, match='duplicate shard index'):
        merge([root / 's0', root / 's0'], tmp_path / 'out')


def test_merge_rejects_protocol_mismatch(smoke_matrix, tmp_path):
    root, _, _, _, _ = smoke_matrix
    tampered = tmp_path / 's1'
    shutil.copytree(root / 's1', tampered)
    (tampered / 'protocol-source.json').write_text('{}')
    with pytest.raises(ValueError, match='protocol-source.json'):
        merge([root / 's0', tampered], tmp_path / 'out')


def test_merge_rejects_tampered_evidence(smoke_matrix, tmp_path):
    root, _, _, _, _ = smoke_matrix
    tampered = tmp_path / 's1'
    shutil.copytree(root / 's1', tampered)
    events = (tampered / 'validation.json.evidence.jsonl').read_text().splitlines()
    event = json.loads(events[1])
    event['payload'] = {'tampered': True}
    events[1] = json.dumps(event)
    (tampered / 'validation.json.evidence.jsonl').write_text('\n'.join(events) + '\n')
    with pytest.raises(ValueError, match='evidence verification failed'):
        merge([root / 's0', tampered], tmp_path / 'out')


def test_destination_registry_refuses_divergent_content():
    registry = mvs.DestinationRegistry()
    assert registry.plan('dest', 'hash-a', 'kind', 'source-a') is True
    assert registry.plan('dest', 'hash-a', 'kind', 'source-b') is False
    with pytest.raises(ValueError, match='collision'):
        registry.plan('dest', 'hash-b', 'kind', 'source-b')


def sidecar_refs(report):
    return [(cell['cell_id'], field, cell[field]) for cell in report['cells']
            for field in ('defense_provenance', 'defended_observations') if cell.get(field)]


def test_merge_rewrites_sidecar_destinations(merged):
    out, _ = merged
    report = json.loads((out / 'validation.json').read_text())
    refs = sidecar_refs(report)
    assert refs, 'smoke protocol must produce defended sidecar references'
    for cid, field, ref in refs:
        assert ref['path'].startswith(('shard0/', 'shard1/')), \
            f'{cid}: {field} destination not shard-prefixed: {ref["path"]}'
        target = out / ref['path']
        assert target.is_file() and file_sha256(target) == ref['sha256'], \
            f'{cid}: {field} does not hash-verify at its rewritten destination'
    for victim in report['defended_victims']:
        provenance = victim['provenance']
        assert provenance['path'].startswith(('shard0/', 'shard1/'))
        target = out / provenance['path']
        assert target.is_file() and file_sha256(target) == provenance['sha256']
    assert report['merge']['sidecar_path_scheme'] == mvs.SIDECAR_PATH_SCHEME
    assert report['merge']['rewritten_sidecar_references'] == len(refs)
    assert report['merge']['deferred_reference_paths'] == []


@pytest.fixture(scope='module')
def smoke_trio(tmp_path_factory):
    """Three-shard smoke run: unlike the two-shard matrix (where each defense
    lands entirely on one shard), three shards make every defended victim
    shared across shards — the v3 campaign's 40-conflict pattern in
    miniature (per-shard divergent provenance, identical observations)."""
    before = journal_snapshot()
    root = tmp_path_factory.mktemp('shardsmoke3')
    shards = [smoke_run(root / f's{i}', shard_count=3, shard_index=i) for i in range(3)]
    assert journal_snapshot() == before, 'smoke shards must not touch the held-out access journal'
    return root, shards


def test_merge_preserves_divergent_sidecars_per_shard(smoke_trio, tmp_path):
    """The v3 campaign pattern: one shard-relative provenance path carries
    different bytes per shard; every byte string must survive the merge and
    each must verify against its own recorded hash."""
    root, shards = smoke_trio
    paths = {}
    for index, report in enumerate(shards):
        for _, _, ref in sidecar_refs(report):
            paths.setdefault(ref['path'], {})[index] = ref['sha256']
    divergent = {path: hashes for path, hashes in paths.items()
                 if len(set(hashes.values())) > 1}
    benign = {path: hashes for path, hashes in paths.items()
              if len(hashes) > 1 and len(set(hashes.values())) == 1}
    assert divergent, 'three smoke shards must reproduce the per-shard provenance divergence'
    assert benign, 'defended observation tensors must stay identical across shards'
    out = tmp_path / 'merged'
    merge([root / f's{i}' for i in range(3)], out)
    for path, hashes in paths.items():
        for index, digest in hashes.items():
            target = out / f'shard{index}' / path
            assert target.is_file() and file_sha256(target) == digest


def test_merge_refuses_flat_divergent_destination(smoke_trio, tmp_path, monkeypatch):
    """Fail closed on the pre-fix defect: with the old flat destination
    scheme, divergent per-shard sidecars map to one destination and the
    merge must refuse rather than silently overwrite."""
    root, _ = smoke_trio
    monkeypatch.setattr(mvs, 'sidecar_destination', lambda index, path: path)
    with pytest.raises(ValueError, match='collision'):
        merge([root / f's{i}' for i in range(3)], tmp_path / 'out')


def test_merge_json_tier_defers_missing_references(smoke_matrix, tmp_path):
    root, _, _, _, _ = smoke_matrix
    copies = []
    for name in ('s0', 's1'):
        target = tmp_path / name
        shutil.copytree(root / name, target)
        copies.append(target)
    removed = 0
    for directory in copies:
        for fp in list(directory.rglob('*')):
            if fp.is_file() and (fp.suffix == '.pt' or 'captures' in fp.parts):
                fp.unlink()
                removed += 1
    assert removed, 'strip must remove the binary tier from the copies'
    with pytest.raises(ValueError, match='missing'):
        merge(copies, tmp_path / 'strict')
    out = tmp_path / 'deferred'
    summary = merge(copies, out, json_tier=True)
    assert summary['deferred_references'] > 0
    report = json.loads((out / 'validation.json').read_text())
    deferred = set(report['merge']['deferred_reference_paths'])
    assert len(deferred) == summary['deferred_references']
    for _, _, ref in sidecar_refs(report):
        assert ref['path'] in deferred and not (out / ref['path']).exists()
    for cell in report['cells']:
        assert file_sha256(out / cell['prediction_file']) == cell['prediction_sha256']
