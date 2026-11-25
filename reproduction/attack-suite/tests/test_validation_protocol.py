"""Adversarial checks on the new protocol; exclusively synthetic inputs."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from attacker.evidence import EvidenceLogger, file_sha256
from attacker.reconstruction_metrics import reconstruction_metrics
from experiments.analyze_validation import crossed_interval, contrasts, validate_report
from experiments.validate_reconstruction import arms, validate_data


def protocol():
    return json.loads(Path('experiments/validation_protocol.json').read_text())


def test_primary_not_repeated_and_secondary_family_prespecified():
    p = protocol()
    main, secondary = contrasts(p)
    assert main not in secondary
    assert len(secondary) == len(set(secondary)) == 47
    assert all(k[2] == 'training' for k in secondary)
    assert all('joint_' not in arm for arm in arms(p, 'inference'))


def test_crossed_bootstrap_constant_and_deterministic():
    kwargs = dict(alpha=.05, replicates=2000, seed=912)
    result = crossed_interval(np.full((4, 8), .25), **kwargs)
    assert result['lower'] == result['upper'] == result['difference'] == .25
    varying = np.arange(32).reshape(4, 8) / 32
    assert crossed_interval(varying, **kwargs) == crossed_interval(varying, **kwargs)


def test_repeated_documents_do_not_create_independent_victim_runs():
    kwargs = dict(alpha=.05, replicates=4000, seed=97)
    # Perfect within-run correlation retains between-run uncertainty even with
    # hundreds of document columns. An IID token/document bootstrap would shrink.
    few = crossed_interval([[0, 0], [1, 1]], **kwargs)
    many = crossed_interval([[0] * 500, [1] * 500], **kwargs)
    assert few['lower'] == many['lower'] == 0
    assert few['upper'] == many['upper'] == 1
    transposed = crossed_interval([[0, 1]] * 500, **kwargs)
    assert transposed['lower'] == 0 and transposed['upper'] == 1


@pytest.mark.parametrize('matrix', [[[1, 2]], [[1], [2]], [[1, 2], [3, np.nan]], [1, 2]])
def test_crossed_bootstrap_rejects_incomplete_or_unreplicated_matrix(matrix):
    with pytest.raises(ValueError):
        crossed_interval(matrix, alpha=.05, replicates=100, seed=1)


@pytest.mark.parametrize('alpha,replicates', [(0, 100), (1, 100), (.05, 0), (.05, True), (np.nan, 100)])
def test_crossed_bootstrap_rejects_invalid_configuration(alpha, replicates):
    with pytest.raises(ValueError):
        crossed_interval([[0, 1], [1, 0]], alpha=alpha, replicates=replicates, seed=1)


def data():
    return {'calibration': [{'id': 'cal', 'source_document_id': 'source-cal', 'tokens': [1, 2, 3]}],
            'train': [{'id': 'train', 'source_document_id': 'source-train', 'tokens': [4, 5, 6]}],
            'evaluation': [{'id': 'eval0', 'source_document_id': 'source-eval0', 'tokens': [7, 8, 9]}, {'id': 'eval1', 'source_document_id': 'source-eval1', 'tokens': [10, 11, 12]}],
            'provenance': {'untouched_evaluation': True}}


def test_validate_data_rejects_leakage_and_out_of_vocab_tokens():
    p = dict(sequence_length=3, minimum_documents=2)
    validate_data(data(), p, 64)
    for mutation in ('duplicate_id', 'duplicate_prefix', 'bad_token', 'bad_span', 'insufficient', 'provenance'):
        d = data()
        if mutation == 'duplicate_id': d['train'][0]['id'] = 'cal'
        if mutation == 'duplicate_prefix': d['train'][0]['tokens'] = [1, 2, 3]
        if mutation == 'bad_token': d['evaluation'][0]['tokens'][0] = 64
        if mutation == 'bad_span': d['evaluation'][0]['annotations'] = [dict(kind='name', start=2, end=4)]
        if mutation == 'insufficient': d['evaluation'].pop()
        if mutation == 'provenance': d['provenance'] = {}
        with pytest.raises(ValueError): validate_data(d, p, 64)


@pytest.mark.parametrize('mutation', ['empty_id', 'false_string'])
def test_validate_data_rejects_ambiguous_identity_or_provenance(mutation):
    d = data()
    if mutation == 'empty_id': d['train'][0]['id'] = ''
    else: d['provenance']['untouched_evaluation'] = 'false'
    with pytest.raises(ValueError):
        validate_data(d, dict(sequence_length=3, minimum_documents=2), 64)


def test_gradient_controls_preserve_shape_and_shuffle_without_fixed_points():
    torch = pytest.importorskip('torch')
    from attacker.validation_controls import transform_gradient
    g = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4)
    shuffled, details = transform_gradient(g, 'shuffled', 19)
    donors = details['donor_indices']
    assert sorted(donors) == list(range(4))
    assert all(i != donor for i, donor in enumerate(donors))
    assert torch.equal(shuffled, g[donors])
    assert torch.equal(shuffled, transform_gradient(g, 'shuffled', 19)[0])
    unchanged, _ = transform_gradient(g, 'true', 19)
    assert torch.equal(unchanged, g) and unchanged.data_ptr() != g.data_ptr()
    for condition in ('noise_0.1', 'noise_1.0', 'bits_4', 'bits_8'):
        out, _ = transform_gradient(g, condition, 19)
        assert out.shape == g.shape and torch.isfinite(out).all()
    zeros = torch.zeros_like(g)
    assert torch.equal(transform_gradient(zeros, 'bits_4', 1)[0], zeros)
    assert torch.equal(transform_gradient(g, 'noise_0', 1)[0], g)


@pytest.mark.parametrize('condition', ['noise_nan', 'noise_inf', 'noise_-1', 'bits_1', 'bits_17', 'unknown'])
def test_gradient_control_rejects_invalid_condition(condition):
    torch = pytest.importorskip('torch')
    from attacker.validation_controls import transform_gradient
    with pytest.raises(ValueError): transform_gradient(torch.ones(3, 2, 4), condition, 1)


def test_random_candidate_control_matches_forward_first_position_and_distinct_budget():
    torch = pytest.importorskip('torch')
    from attacker.validation_controls import random_candidates
    from attacker.reconstruction_baselines import candidate_ids
    scores = torch.arange(16, dtype=torch.float32).expand(3, 4, -1)
    ids = random_candidates(scores, 8, 91)
    forward = candidate_ids(scores, None, 8, 'forward')
    assert torch.equal(ids[:, 0], forward[:, 0])
    assert torch.equal(ids[:, 1:, :4], forward[:, 1:, :4])
    assert all(len(set(row)) == 8 for row in ids.reshape(-1, 8).tolist())
    assert torch.equal(ids, random_candidates(scores, 8, 91))
    with pytest.raises(ValueError): random_candidates(scores, 3, 91)


@pytest.fixture
def synthetic_report(tmp_path):
    """Tiny sealed fixture: dummy checkpoints are hash inputs, never loaded."""
    p = protocol()
    p['execution']={'reuse_identical_activation_modes':False}
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


def test_complete_synthetic_report_rescores_successfully(synthetic_report):
    root, report = synthetic_report
    seal_report(root, report)
    _, cells = validate_report(root)
    assert len(cells) == len(report['cells'])


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'missing cells'), ('duplicate', 'duplicate or unexpected'),
    ('score', 'metric mismatch'), ('work', 'matched prefix work'),
    ('setup', 'setup matrix'), ('victim', 'victim matrix'),
    ('artifact', 'artifact matrix'), ('failed', 'incomplete or failed'),
    ('protocol', 'embedded protocol mismatch'), ('path', 'prediction path/hash'),
])
def test_report_fails_closed_even_when_mutated_report_is_sealed(synthetic_report, mutation, reason):
    root, report = synthetic_report
    if mutation == 'missing': report['cells'].pop()
    if mutation == 'duplicate': report['cells'].append(copy.deepcopy(report['cells'][0]))
    if mutation == 'score': report['cells'][0]['metrics'][0]['token_accuracy'] = 0
    if mutation == 'work':
        next(c for c in report['cells'] if c['arm'] == 'joint_true_k4')['work']['head_forward_calls'] += 1
    if mutation == 'setup': report['setup'].pop()
    if mutation == 'victim': report['victims'].pop()
    if mutation == 'artifact': report['artifacts'].pop(next(iter(report['artifacts'])))
    if mutation == 'failed': report['status'] = 'FAILED'
    if mutation == 'protocol': report['protocol']['candidate_counts'] = [8]
    if mutation == 'path': report['cells'][0]['prediction_file'] = '../prediction.json'
    seal_report(root, report)
    with pytest.raises(ValueError, match=reason): validate_report(root)


def test_changed_prediction_bytes_fail_digest_verification(synthetic_report):
    root, report = synthetic_report
    seal_report(root, report)
    (root / 'prediction.json').write_text('{}')
    with pytest.raises(ValueError, match='prediction path/hash'): validate_report(root)


def test_attack_boundary_is_truth_free_and_multibatch_controls_match_prefix_work():
    import inspect
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker.splitbench import build_split, SplitPlan
    from attacker.reconstruction_baselines import MeteredHead
    from experiments.validate_reconstruction import attack_observations
    assert not {'truth', 'tokens', 'labels', 'evaluation', 'data'} & set(inspect.signature(attack_observations).parameters)
    torch.set_num_threads(1)
    torch.manual_seed(192)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=8, n_layer=2, n_head=2,
        n_positions=16, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval().requires_grad_(False)
    head = MeteredHead(build_split(model, SplitPlan.parse('1', n_layers=2)))
    ids = torch.tensor([[1, 3, 5], [2, 4, 6], [7, 8, 9]])
    hidden = head(ids=ids).detach()
    gradient = torch.randn_like(hidden)
    embedding = head.split.wte.weight.detach()
    decoder = torch.eye(8)
    p = {'attack_batch_size': 2, 'beam_width': 2}
    work = []
    for arm in ('forward_k4', 'joint_true_k4', 'joint_shuffled_k4', 'joint_random_k4',
                'joint_noise_0.1_k4', 'joint_bits_4_k4'):
        with head.measure() as measured:
            prediction, diagnostics = attack_observations(head, hidden, gradient, embedding,
                                                          decoder, 1, arm, p, 42)
        assert len(prediction) == 3 and all(len(row) == 3 for row in prediction)
        assert len(diagnostics['batches']) == 2
        work.append((measured.head_forward_calls, measured.head_forward_sequences,
                     measured.head_forward_token_positions))
        if arm == 'joint_shuffled_k4':
            assert all(i != j for i, j in enumerate(diagnostics['donor_indices']))
    assert len(set(work)) == 1


def test_source_identity_and_near_duplicate_prefixes_rejected():
    d = data()
    for i, part in enumerate(('calibration', 'train', 'evaluation')):
        for j, row in enumerate(d[part]):
            row['tokens'] = list(range(i * 32 + j * 8, i * 32 + j * 8 + 8))
    p = dict(sequence_length=8, minimum_documents=2)
    validate_data(d, p, 128)
    duplicate = copy.deepcopy(d)
    duplicate['train'][0]['source_document_id'] = duplicate['calibration'][0]['source_document_id']
    with pytest.raises(ValueError, match='source_document_id'): validate_data(duplicate, p, 128)
    # Same set of five-grams with a repeated prefix, but nonidentical token tuple.
    p['sequence_length'] = 12
    for part in d.values():
        if isinstance(part, list):
            for row in part: row['tokens'] = row['tokens'] + row['tokens'][:4]
    d['calibration'][0]['tokens'] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]
    d['train'][0]['tokens'] = [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
    with pytest.raises(ValueError, match='near-duplicate'): validate_data(d, p, 128)


def test_real_analysis_requires_external_anchor(synthetic_report):
    root, report = synthetic_report
    report['evidence_tier'] = 'held_out_controlled_probe'
    seal_report(root, report)
    with pytest.raises(ValueError, match='external manifest anchor'):
        validate_report(root)


def test_external_anchor_rejects_resealed_replacement(synthetic_report):
    root, report = synthetic_report
    seal_report(root, report)
    with pytest.raises(ValueError, match='external manifest anchor mismatch'):
        validate_report(root, '0' * 64)


def test_real_analysis_rejects_draft_protocol_even_with_anchor(synthetic_report):
    root, report = synthetic_report
    report['protocol']['status']='draft_for_review'
    report['evidence_tier'] = 'held_out_controlled_probe'
    seal_report(root, report)
    digest = file_sha256(root / 'validation.json.evidence.json')
    with pytest.raises(ValueError, match='unchanged frozen protocol'):
        validate_report(root, digest)


def test_defended_capture_arms_enumerate_and_validate():
    from experiments.validate_reconstruction import arms, validate_protocol
    p = protocol()
    assert not any('@' in a for a in arms(p, 'training'))
    p['capture_defenses'] = {'defended': ['replica-median', 'secret-mixup']}
    validate_protocol(p)
    defended = [a for a in arms(p, 'training') if '@' in a]
    assert len(defended) == 2 * (len(p['candidate_counts']) + len(p['sipit_steps']))
    assert 'forward_k8@replica-median' in defended and 'sipit_steps128@secret-mixup' in defended
    assert not any('@' in a for a in arms(p, 'inference'))
    for bad in ({'defended': []}, {'defended': ['none']}, {'defended': ['a', 'a']},
                {'defended': ['x@y']}, {'defended': 'x'}):
        broken = protocol()
        broken['capture_defenses'] = bad
        with pytest.raises(ValueError, match='capture_defenses'):
            validate_protocol(broken)


def test_analyzer_accepts_frozen_v2_with_defended_arms(synthetic_report):
    root, report = synthetic_report
    report['protocol']['version'] = 2
    report['protocol']['capture_defenses'] = {'defended': ['replica-median']}
    extra = []
    for cell in report['cells']:
        if cell['mode'] == 'training' and cell['arm'] in ('forward_k4', 'sipit_steps2'):
            extra.append(dict(cell, arm=f"{cell['arm']}@replica-median"))
    report['cells'] += extra
    for name in ('protocol-source.json', 'protocol-effective.json'):
        (root / name).write_text(json.dumps(report['protocol']))
    report['protocol_sha256'] = file_sha256(root / 'protocol-source.json')
    report['effective_protocol_sha256'] = file_sha256(root / 'protocol-effective.json')
    seal_report(root, report)
    validated, cells = validate_report(root)
    assert validated['protocol']['version'] == 2
    assert sum('@' in key[5] for key in cells) == len(extra)


def test_default_protocol_has_required_axes_and_primary():
    from experiments.validate_reconstruction import validate_protocol
    validate_protocol(protocol())
    broken = protocol()
    broken['victim_seeds'] = [101] * 10
    with pytest.raises(ValueError, match='duplicate protocol axis'):
        validate_protocol(broken)


def test_identical_activation_reuse_is_explicit_and_does_not_repeat_measurement():
    from experiments.validate_reconstruction import reuse_activation_cell
    original={'cell_id':'101-3-public-17-training-forward_k32','execution_id':'measurement-1',
              'mode':'training','arm':'forward_k32','measurement_reused':False,
              'work':{'wall_seconds':3.0},'metrics':[{'token_accuracy':.5}]}
    reused=reuse_activation_cell(original)
    assert reused['mode']=='inference' and reused['measurement_reused'] is True
    assert reused['execution_id']==original['execution_id']
    assert reused['reused_from']==original['cell_id']
    assert original['mode']=='training' and original['measurement_reused'] is False
    with pytest.raises(ValueError): reuse_activation_cell(dict(original,arm='joint_true_k32'))


def test_annotation_binding_requires_complete_identity_and_matching_text():
    import hashlib
    from experiments.prepare_validation import attach_annotations
    raw={'evaluation':[{'id':'a','source_document_id':'source-a','text':'Ada arrived.'}]}
    report={'annotation_source':'AI-assisted test fixture','reviewed_document_count':1,
            'documents':[{'id':'a','source_document_id':'source-a','prefix_character_count':3,
                          'prefix_text_sha256':hashlib.sha256(b'Ada').hexdigest(),
                          'char_annotations':[{'kind':'name','start':0,'end':3}]}]}
    attach_annotations(raw,report)
    assert raw['evaluation'][0]['char_annotations']==report['documents'][0]['char_annotations']
    bad=copy.deepcopy(report); bad['documents']=[]
    with pytest.raises(ValueError,match='coverage/identity'): attach_annotations(raw,bad)
    bad=copy.deepcopy(report); bad['documents'][0]['prefix_text_sha256']='0'*64
    with pytest.raises(ValueError,match='text hash'): attach_annotations(raw,bad)
    bad=copy.deepcopy(report); bad['documents'][0]['source_document_id']='other'
    with pytest.raises(ValueError,match='source identity'): attach_annotations(raw,bad)


def test_budget_counts_distinguish_logical_rows_from_physical_execution():
    from experiments.review_validation_budget import estimate
    p=protocol()
    rows=[]
    for depth in (3,6):
        for mode in ('training','inference'):
            for arm,value in [('forward_beam_k8',1),('forward_beam_k32',4),('joint_union_k8',1),
                              ('joint_union_k32',4),('sipit_steps32',5),('sipit_steps128',20),('direction_backward',.01)]:
                rows.append({'depth':str(depth),'mode':mode,'arm':arm,'attack_seconds_per_batch':str(value)})
    reused=estimate(p,rows,256)
    assert reused['cells']==2720 and reused['physical_cells']==2240
    assert reused['physical_document_reconstructions']==573440
    assert reused['estimated_online_gpu_hours']<reused['nominal_all_executed_gpu_hours']
    full=estimate(dict(p,execution={'reuse_identical_activation_modes':False}),rows,256)
    assert full['physical_cells']==2720 and full['duplicate_execution_savings_gpu_hours']==0
    with pytest.raises(ValueError): estimate(p,rows,255)
