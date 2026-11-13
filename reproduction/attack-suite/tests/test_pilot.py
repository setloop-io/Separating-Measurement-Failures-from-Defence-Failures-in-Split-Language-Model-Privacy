"""Pilot data separation and fail-closed analysis contracts."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'experiments'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_article_grouping_never_splits_subheadings_into_independent_documents():
    prepare=load_script('prepare_pilot')
    rows=['ignored preamble',' = Article A = ','first paragraph',' == Subsection == ','second paragraph',
          ' = Article B = ','third paragraph']
    assert list(prepare.articles(rows))==[('Article A','first paragraph\n\nsecond paragraph'),('Article B','third paragraph')]


def test_protocol_has_complete_matrix_and_holds_out_official_test_data():
    p=json.loads((ROOT/'experiments/pilot_gpt2_wikitext2.json').read_text())
    count=len(p['seeds'])*len(p['split_depths'])*len(p['defenses'])*sum(map(len,p['attack_arms'].values()))
    assert count==p['expected_cells']==100
    assert p['source_split']=='train'
    assert set(p['reserved_splits'])=={'validation','test'}
    assert len(set(p['seeds']))==5
    assert p['statistics']['family_size']==60


def test_analysis_refuses_unverified_or_partial_bundle(tmp_path,monkeypatch):
    analysis=load_script('analyze_pilot')
    monkeypatch.setattr(analysis,'verify_run',lambda root,anchor=None:(False,['experiment incomplete']))
    with pytest.raises(ValueError,match='verification failed'):
        analysis.analyze(tmp_path,ROOT/'experiments/pilot_gpt2_wikitext2.json')
    assert not (tmp_path/'analysis').exists()


def test_analysis_rejects_missing_matrix_cell_before_writing_report(tmp_path,monkeypatch):
    analysis=load_script('analyze_pilot')
    p=json.loads((ROOT/'experiments/pilot_gpt2_wikitext2.json').read_text())
    config={k:p[k] for k in ['seeds','seq_len','train_steps','decode_steps','attack_steps','calibration_seqs',
                              'joint_beam','joint_candidates','data_seed','adaptive','defenses']}
    config.update(sigma=p['gaussian_sigma'],lr=p['training_lr'],split='3')
    folder=tmp_path/'split3';folder.mkdir()
    (folder/'experiment.json').write_text(json.dumps({'evidence_tier':'local_model_experiment',
        'corpus_sha256':p['corpus_sha256'],'config':config,'results':[]}))
    monkeypatch.setattr(analysis,'verify_run',lambda root,anchor=None:(True,[]))
    with pytest.raises(ValueError,match='missing, duplicate or unexpected cells'):
        analysis.analyze(tmp_path,ROOT/'experiments/pilot_gpt2_wikitext2.json')
    assert not (tmp_path/'analysis').exists()
