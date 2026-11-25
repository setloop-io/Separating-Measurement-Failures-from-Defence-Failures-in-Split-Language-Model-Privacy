"""Amendment/versioning mechanism checks; synthetic evidence fixtures only."""
import copy
import json
from pathlib import Path

import pytest

from attacker.evidence import EvidenceLogger, file_sha256
from attacker.reconstruction_metrics import reconstruction_metrics
from experiments.amend_protocol import amend
from experiments.analyze_validation import validate_report
from experiments.validate_reconstruction import arms, record_heldout_access

PROTOCOL_PATH = Path('experiments/validation_protocol.json')
FROZEN_INPUT = Path('results/validation-corpus-v1/validation-input.json')


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def data():
    return {'calibration': [{'id': 'cal', 'source_document_id': 'source-cal', 'tokens': [1, 2, 3]}],
            'train': [{'id': 'train', 'source_document_id': 'source-train', 'tokens': [4, 5, 6]}],
            'evaluation': [{'id': 'eval0', 'source_document_id': 'source-eval0', 'tokens': [7, 8, 9]},
                           {'id': 'eval1', 'source_document_id': 'source-eval1', 'tokens': [10, 11, 12]}],
            'provenance': {'untouched_evaluation': True}}


def budget_rows():
    rows = []
    for depth in (3, 6):
        for mode in ('training', 'inference'):
            for arm, value in [('forward_beam_k8', 1), ('forward_beam_k32', 4), ('joint_union_k8', 1),
                               ('joint_union_k32', 4), ('sipit_steps32', 5), ('sipit_steps128', 20),
                               ('direction_backward', .01)]:
                rows.append({'depth': str(depth), 'mode': mode, 'arm': arm,
                             'attack_seconds_per_batch': str(value)})
    return rows


@pytest.fixture
def amendment(tmp_path):
    evidence = tmp_path / 'pilot-feasibility.json'
    evidence.write_text(json.dumps({'pilot': 'synthetic feasibility fixture'}))
    record = {'schema': 'llm_attacker.validation_amendment.v1',
              'amendment_id': 'synthetic-v2-mechanism-fixture',
              'changes': [{'axis': 'victim_seeds', 'value': protocol()['victim_seeds'] + [1111, 1212],
                           'rationale': 'pilot feasibility supports two more victim seeds',
                           'evidence': {'path': str(evidence), 'sha256': file_sha256(evidence)}}]}
    path = tmp_path / 'amendment.json'
    path.write_text(json.dumps(record))
    return path


def test_v2_draft_production(amendment, tmp_path):
    out = tmp_path / 'v2.json'
    v2 = amend(PROTOCOL_PATH, amendment, out, summary_rows=budget_rows())
    assert v2['version'] == 2 and v2['status'] == 'draft_for_review'
    assert v2['schema'] == 'llm_attacker.validation.v1'
    assert len(v2['victim_seeds']) == 12
    # The frozen v1 stays immutable.
    assert json.loads(PROTOCOL_PATH.read_text()).get('version', 1) == 1
    assert json.loads(PROTOCOL_PATH.read_text())['status'] == 'frozen'


def test_freeze_regenerates_pins(amendment, tmp_path):
    out = tmp_path / 'v2.json'
    v1 = protocol()
    v2 = amend(PROTOCOL_PATH, amendment, out, freeze=True, input_path=FROZEN_INPUT,
               summary_rows=budget_rows(), journal_path=tmp_path / 'no-journal.journal')
    assert v2['status'] == 'frozen'
    # Inputs unchanged by this amendment: regenerated pins match the v1 pins.
    for field in ('tokenized_sha256', 'name_annotations_sha256', 'curation_manifest_sha256'):
        assert v2['frozen_inputs'][field] == v1['frozen_inputs'][field]
    assert v2['frozen_inputs']['tokenized_sha256'] == file_sha256(FROZEN_INPUT)
    assert v2['budget_review']['logical_cells'] == 3264


def test_budget_delta_recorded(amendment, tmp_path):
    v2 = amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json', summary_rows=budget_rows())
    budget = v2['amendment']['record']['budget']
    assert budget['v1']['logical_cells'] == 2720
    assert budget['v2']['logical_cells'] == 3264
    assert budget['delta']['logical_cells'] == 544
    assert budget['delta']['estimated_online_gb10_gpu_hours'] > 0


def test_new_axis_requires_explicit_flag(amendment, tmp_path):
    record = json.loads(amendment.read_text())
    spec = {'defended': ['replica-median', 'secret-mixup']}
    record['changes'] = [{'axis': 'capture_defenses', 'value': spec,
                          'rationale': 'pilot feasibility supports defended-capture arms',
                          'evidence': record['changes'][0]['evidence']}]
    amendment.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='new_axis'):
        amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json', summary_rows=budget_rows())
    record['changes'][0]['new_axis'] = True
    amendment.write_text(json.dumps(record))
    v2 = amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json', summary_rows=budget_rows())
    assert v2['capture_defenses'] == spec
    assert v2['amendment']['record']['budget']['delta']['logical_cells'] == 960
    record['changes'][0]['new_axis'] = False
    amendment.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='new_axis must be exactly true'):
        amend(PROTOCOL_PATH, amendment, tmp_path / 'v2b.json', summary_rows=budget_rows())


def test_amendment_without_evidence_pointer_rejected(amendment, tmp_path):
    record = json.loads(amendment.read_text())
    del record['changes'][0]['evidence']
    amendment.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='evidence pointer'):
        amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json', summary_rows=budget_rows())
    record = json.loads(amendment.read_text())
    record['changes'][0]['evidence'] = {'path': str(amendment), 'sha256': '0' * 64}
    amendment.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='evidence hash mismatch'):
        amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json', summary_rows=budget_rows())


def test_freeze_refused_after_heldout_access(amendment, tmp_path):
    journal = tmp_path / 'heldout_access.journal'
    record_heldout_access(tmp_path / 'run-output', journal=journal)
    with pytest.raises(ValueError, match='held-out outcome access'):
        amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json', freeze=True, input_path=FROZEN_INPUT,
              summary_rows=budget_rows(), journal_path=journal)
    # Draft production is still allowed; only re-freeze is blocked.
    v2 = amend(PROTOCOL_PATH, amendment, tmp_path / 'v2.json',
               summary_rows=budget_rows(), journal_path=journal)
    assert v2['status'] == 'draft_for_review'


@pytest.fixture
def synthetic_report(tmp_path):
    """Tiny sealed fixture: dummy checkpoints are hash inputs, never loaded."""
    p = protocol()
    p['execution'] = {'reuse_identical_activation_modes': False}
    p.update(victim_seeds=[101, 202], depths=[1], knowledge=['public'], attack_seeds=[17],
             candidate_counts=[4], sipit_steps=[2], gradient_conditions=['true', 'shuffled'])
    p['primary'].update(depth=1, left='joint_true_k4', right='forward_k4')
    for name, value in [('protocol-source.json', p), ('protocol-effective.json', p), ('input.json', data())]:
        (tmp_path / name).write_text(json.dumps(value))
    report = {'protocol': p, 'status': 'COMPLETE', 'evidence_tier': 'engineering_smoke',
              'protocol_sha256': file_sha256(tmp_path / 'protocol-source.json'),
              'effective_protocol_sha256': file_sha256(tmp_path / 'protocol-effective.json'),
              'input_sha256': file_sha256(tmp_path / 'input.json'), 'artifacts': {},
              'setup': [], 'victims': [], 'cells': []}
    documents = data()['evaluation']
    pred = {'document_ids': [r['id'] for r in documents], 'predictions': [r['tokens'] for r in documents]}
    (tmp_path / 'prediction.json').write_text(json.dumps(pred))
    metrics = [{'document_id': r['id'], **reconstruction_metrics(r['tokens'], r['tokens'])} for r in documents]
    for seed in p['victim_seeds']:
        report['victims'].append({'seed': seed})
        report['setup'].append({'id': f'{seed}-1-public'})
        for kind in ('public', 'exact', 'observations'):
            name = f'{kind}-{seed}-1.pt'
            (tmp_path / name).write_bytes(b'synthetic fixture, not a real checkpoint')
            report['artifacts'][name] = file_sha256(tmp_path / name)
        for mode in ('training', 'inference'):
            for arm in arms(p, mode):
                report['cells'].append(dict(victim_seed=seed, depth=1, knowledge='public',
                    attack_seed=17, mode=mode, arm=arm, prediction_file='prediction.json',
                    prediction_sha256=file_sha256(tmp_path / 'prediction.json'), metrics=copy.deepcopy(metrics),
                    work=dict(head_forward_calls=3, head_forward_sequences=8, head_forward_token_positions=24)))
    return tmp_path, report


def seal_report(root, report):
    path = root / 'validation.json'
    with EvidenceLogger(path):
        path.write_text(json.dumps(report))


def test_analyzer_accepts_frozen_v1(synthetic_report):
    root, report = synthetic_report
    seal_report(root, report)
    validated, cells = validate_report(root)
    assert validated['protocol'].get('version', 1) == 1


def test_analyzer_accepts_frozen_v2(synthetic_report):
    root, report = synthetic_report
    report['protocol']['version'] = 2
    for name in ('protocol-source.json', 'protocol-effective.json'):
        (root / name).write_text(json.dumps(report['protocol']))
    report['protocol_sha256'] = file_sha256(root / 'protocol-source.json')
    report['effective_protocol_sha256'] = file_sha256(root / 'protocol-effective.json')
    seal_report(root, report)
    validated, cells = validate_report(root)
    assert validated['protocol']['version'] == 2


def test_analyzer_rejects_draft_v2(synthetic_report):
    root, report = synthetic_report
    report['protocol'].update(version=2, status='draft_for_review')
    report['evidence_tier'] = 'held_out_controlled_probe'
    (root / 'protocol-source.json').write_text(json.dumps(report['protocol']))
    (root / 'protocol-effective.json').write_text(json.dumps(report['protocol']))
    report['protocol_sha256'] = file_sha256(root / 'protocol-source.json')
    report['effective_protocol_sha256'] = file_sha256(root / 'protocol-effective.json')
    seal_report(root, report)
    digest = file_sha256(root / 'validation.json.evidence.json')
    with pytest.raises(ValueError, match='unchanged frozen protocol'):
        validate_report(root, digest)


def test_analyzer_rejects_disallowed_version(synthetic_report):
    root, report = synthetic_report
    report['protocol']['version'] = 2
    (root / 'protocol-source.json').write_text(json.dumps(report['protocol']))
    (root / 'protocol-effective.json').write_text(json.dumps(report['protocol']))
    report['protocol_sha256'] = file_sha256(root / 'protocol-source.json')
    report['effective_protocol_sha256'] = file_sha256(root / 'protocol-effective.json')
    seal_report(root, report)
    with pytest.raises(ValueError, match='protocol version not permitted'):
        validate_report(root, allowed_protocol_versions=(1,))


def test_analyzer_rejects_hash_mismatch_v2(synthetic_report):
    root, report = synthetic_report
    report['protocol']['version'] = 2
    (root / 'protocol-source.json').write_text(json.dumps(report['protocol']))
    (root / 'protocol-effective.json').write_text(json.dumps(report['protocol']))
    report['protocol_sha256'] = file_sha256(root / 'protocol-source.json')
    report['effective_protocol_sha256'] = file_sha256(root / 'protocol-effective.json')
    seal_report(root, report)
    (root / 'input.json').write_text(json.dumps({'tampered': True}))
    with pytest.raises(ValueError, match='input hash mismatch'):
        validate_report(root)


PROTOCOL_V2_PATH = Path('experiments/validation_protocol.v2.json')


def test_v3_draft_from_frozen_v2(amendment, tmp_path):
    """The amendment mechanism generalizes: it amends the CURRENT frozen
    protocol (v2), incrementing whatever version the source carries."""
    out = tmp_path / 'v3.json'
    v3 = amend(PROTOCOL_V2_PATH, amendment, out, summary_rows=budget_rows())
    assert v3['version'] == 3 and v3['status'] == 'draft_for_review'
    assert v3['schema'] == 'llm_attacker.validation.v1'
    v2_seeds = json.loads(PROTOCOL_V2_PATH.read_text())['victim_seeds']
    assert len(v3['victim_seeds']) == len(v2_seeds) + 2
    budget = v3['amendment']['record']['budget']
    assert set(budget) == {'v2', 'v3', 'delta'}
    assert budget['delta']['logical_cells'] > 0
    # The frozen v2 source stays immutable.
    v2 = json.loads(PROTOCOL_V2_PATH.read_text())
    assert v2['version'] == 2 and v2['status'] == 'frozen'


def test_v3_freeze_regenerates_pins(amendment, tmp_path):
    out = tmp_path / 'v3.json'
    v3 = amend(PROTOCOL_V2_PATH, amendment, out, freeze=True,
               input_path=FROZEN_INPUT, summary_rows=budget_rows(),
               journal_path=tmp_path / 'no-journal.journal')
    assert v3['version'] == 3 and v3['status'] == 'frozen'
    v2 = json.loads(PROTOCOL_V2_PATH.read_text())
    for field in ('tokenized_sha256', 'name_annotations_sha256',
                  'curation_manifest_sha256'):
        assert v3['frozen_inputs'][field] == v2['frozen_inputs'][field]
    assert v3['frozen_inputs']['tokenized_sha256'] == \
        file_sha256(FROZEN_INPUT)


def test_v3_freeze_refused_after_heldout_access(amendment, tmp_path):
    journal = tmp_path / 'heldout_access.journal'
    record_heldout_access(tmp_path / 'run-output', journal=journal)
    with pytest.raises(ValueError, match='held-out outcome access'):
        amend(PROTOCOL_V2_PATH, amendment, tmp_path / 'v3.json',
              freeze=True, input_path=FROZEN_INPUT,
              summary_rows=budget_rows(), journal_path=journal)


def test_unfrozen_base_rejected(amendment, tmp_path):
    draft = tmp_path / 'draft.json'
    p = json.loads(PROTOCOL_V2_PATH.read_text())
    p['status'] = 'draft_for_review'
    draft.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='frozen'):
        amend(draft, amendment, tmp_path / 'v3.json',
              summary_rows=budget_rows())
