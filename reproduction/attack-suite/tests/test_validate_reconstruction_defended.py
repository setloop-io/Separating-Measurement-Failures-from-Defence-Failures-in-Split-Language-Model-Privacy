"""Defended-capture execution in the frozen-study runner — smoke tier only.

Every test here runs experiments/validate_reconstruction.py in --smoke mode
(synthetic 3-document partitions, tiny random GPT-2, engineering_smoke
evidence tier). Smoke mode never reads held-out outcomes and never writes
results/heldout_access.journal; that invariant is asserted explicitly.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from attacker.evidence import file_sha256
from experiments.analyze_validation import validate_report
from experiments import validate_reconstruction as vr

V2 = Path('experiments/validation_protocol.v2.json')
JOURNAL = Path('results/heldout_access.journal')


def journal_snapshot():
    return JOURNAL.read_bytes() if JOURNAL.exists() else None


def smoke_run(output, protocol=V2):
    args = SimpleNamespace(protocol=Path(protocol), input=None, input_sha256=None,
                           output=Path(output), device='cpu', smoke=True)
    assert vr.run(args) == 0
    return json.loads((Path(output) / 'validation.json').read_text())


@pytest.fixture(scope='module')
def defended_smoke(tmp_path_factory):
    before = journal_snapshot()
    root = tmp_path_factory.mktemp('v2smoke')
    report = smoke_run(root / 'run')
    assert journal_snapshot() == before, 'smoke mode must not touch the held-out access journal'
    return root / 'run', report


def defended_cells(report):
    return [c for c in report['cells'] if '@' in c['arm']]


def test_smoke_defended_run_completes_with_full_defended_matrix(defended_smoke):
    root, report = defended_smoke
    p = report['protocol']
    assert report['status'] == 'COMPLETE'
    assert report['evidence_tier'] == 'engineering_smoke'
    cells = defended_cells(report)
    expected = (len(p['capture_defenses']['defended']) * len(p['victim_seeds']) * len(p['depths'])
                * len(p['knowledge']) * len(p['attack_seeds'])
                * (len(p['candidate_counts']) + len(p['sipit_steps'])))
    assert len(cells) == expected > 0
    assert all(c['status'] == 'COMPLETE' for c in cells)
    assert len(report['defended_victims']) == (len(p['capture_defenses']['defended'])
                                               * len(p['victim_seeds']) * len(p['depths']))
    # The undefended matrices stay exact (the analyzer enforces this too).
    assert len(report['victims']) == len(p['victim_seeds'])
    assert len(report['setup']) == len(p['victim_seeds']) * len(p['depths']) * len(p['knowledge'])
    assert not (root / 'observations-replica-median-101-1.pt').exists()  # defended .pt stay out of the root artifact glob


def test_defended_cell_records_provenance_utility_access_and_runtime(defended_smoke):
    root, report = defended_smoke
    by_id = {c['cell_id']: c for c in report['cells']}
    for cell in defended_cells(report):
        # provenance sidecar: path + hash of a real file inside the run dir
        prov = cell['defense_provenance']
        assert prov['schema'] == f"llm_attacker.{cell['defense'].replace('-', '_')}_provenance.v1"
        prov_path = root / prov['path']
        assert prov_path.parent != root and file_sha256(prov_path) == prov['sha256']
        # mandatory held-out utility before/after
        for side in ('utility_before', 'utility_after'):
            assert cell['defense_utility'][side]['cross_entropy'] > 0
        # attacker-access spec identical to the paired undefended control
        control = by_id[cell['paired_control']]
        assert '@' not in control['arm']
        for field in ('victim_seed', 'depth', 'knowledge', 'attack_seed', 'mode'):
            assert control[field] == cell[field]
        assert control['arm'] == cell['access']['budget_arm'] == cell['arm'].split('@')[0]
        assert cell['access']['calibration_documents'] > 0
        # runtime measurements
        assert cell['work']['wall_seconds'] >= 0 and cell['cold_seconds'] >= cell['work']['wall_seconds']
        assert cell['amortized_seconds_per_document'] >= 0
        # defended capture artifact recorded with hash
        obs = root / cell['defended_observations']['path']
        assert file_sha256(obs) == cell['defended_observations']['sha256']
        # defended cells are always original physical measurements
        assert cell['measurement_reused'] is False and cell['execution_id'] == cell['cell_id']
    for entry in report['defended_victims']:
        assert entry['training_seconds'] > 0 and entry['evaluation_nll'] > 0
        assert (root / entry['provenance']['path']).is_file()


def test_analyzer_validates_smoke_report_with_defended_cells(defended_smoke):
    root, report = defended_smoke
    _, cells = validate_report(root, allowed_protocol_versions=(2,))
    assert len(cells) == report['logical_cell_count'] == len(report['cells'])
    assert sum('@' in key[5] for key in cells) == len(defended_cells(report))


def test_missing_provenance_fails_closed_and_cells_are_never_dropped(tmp_path, monkeypatch):
    before = journal_snapshot()
    monkeypatch.setitem(vr.DEFENSE_PROVENANCE, 'secret-mixup', 'nonexistent_sidecar.json')
    protocol = json.loads(V2.read_text())
    protocol['capture_defenses'] = {'defended': ['secret-mixup']}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    report = smoke_run(tmp_path / 'run', path)
    assert journal_snapshot() == before
    assert report['status'] == 'FAILED' and 'defended-capture' in report['error']
    cells = defended_cells(report)
    assert cells, 'defended cells must be retained in the report'
    assert all(c['status'] == 'FAILED' for c in cells)
    assert all('provenance sidecar missing' in c['error'] for c in cells)
    # the fail-closed contract: the analyzer refuses a run with FAILED status
    with pytest.raises(ValueError, match='incomplete or failed run'):
        validate_report(tmp_path / 'run', allowed_protocol_versions=(2,))


def test_unsupported_defense_refuses_to_start(tmp_path):
    before = journal_snapshot()
    protocol = json.loads(V2.read_text())
    protocol['capture_defenses'] = {'defended': ['trained-bottleneck']}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='implemented only for'):
        smoke_run(tmp_path / 'run', path)
    assert not (tmp_path / 'run').exists()
    assert journal_snapshot() == before
