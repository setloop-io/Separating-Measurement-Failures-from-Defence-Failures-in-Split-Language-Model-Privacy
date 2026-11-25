"""Analysis must refuse incomplete or unanchored research evidence."""
import json
from pathlib import Path

import pytest

from experiments import analyze_baselines as analysis


@pytest.fixture
def fixture_bundle(tmp_path, monkeypatch):
    protocol = json.loads((Path(__file__).parents[1] / 'experiments/baseline_protocol.json').read_text())
    folder = tmp_path / 'split3'
    folder.mkdir()
    (tmp_path / 'origin-anchor-split3.json').write_text(json.dumps({'manifest_sha256': 'external-anchor'}))
    manifest = {'git': {'commit': '6292d4aafa88c8f7fbe92853531c242cf2cf8c82', 'dirty': False}}
    (folder / 'comparison.json.evidence.json').write_text(json.dumps(manifest))
    raw = {'depth': 3, 'status': 'COMPLETE', 'protocol': protocol, 'setup': [], 'results': []}
    (folder / 'comparison.json').write_text(json.dumps(raw))
    monkeypatch.setattr(analysis, 'verify_evidence', lambda *a: (True, []))
    monkeypatch.setattr(analysis, 'verify_run', lambda *a: pytest.fail('incomplete bundle reached source analysis'))
    return folder, protocol, raw, manifest


def test_invalid_seal_is_rejected(fixture_bundle, monkeypatch):
    folder, protocol, _, _ = fixture_bundle
    monkeypatch.setattr(analysis, 'verify_evidence', lambda *a: (False, ['hash mismatch']))
    with pytest.raises(ValueError, match='evidence failed'):
        analysis.verify_comparison(folder, protocol, folder.parent)


def test_dirty_execution_is_rejected(fixture_bundle):
    folder, protocol, _, manifest = fixture_bundle
    manifest['git']['dirty'] = True
    (folder / 'comparison.json.evidence.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='clean frozen'):
        analysis.verify_comparison(folder, protocol, folder.parent)


def test_depth_substitution_is_rejected(fixture_bundle):
    folder, protocol, raw, _ = fixture_bundle
    raw['depth'] = 6
    (folder / 'comparison.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='depth/folder'):
        analysis.verify_comparison(folder, protocol, folder.parent)


def test_missing_setup_and_attack_cells_are_rejected(fixture_bundle):
    folder, protocol, raw, _ = fixture_bundle
    with pytest.raises(ValueError, match='setup'):
        analysis.verify_comparison(folder, protocol, folder.parent)
    raw['setup'] = [{'id': f'{seed}-{defense}'} for seed in protocol['seeds'] for defense in protocol['defenses']]
    (folder / 'comparison.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='missing, duplicate'):
        analysis.verify_comparison(folder, protocol, folder.parent)
