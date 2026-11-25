#!/usr/bin/env python3
"""CPU-only, post-hoc review analysis of completed corrected P6 evidence.

Reads immutable summaries and source snapshots. Never trains a model, loads
weights, performs inference, contacts a service, or changes original evidence.
New tests and alternative prediction models are exploratory, not preregistered.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
SPANS, STARTS = [2, 4, 6], [0, 1, 2, 3]
GPT_PACKAGE = ROOT / 'output/hf-v1-reference-fix/gpt2-partial-offloading/reproduction-package'
QWEN = ROOT / 'build/qwen-fp64-closure-20260921'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Input SHA256 mismatch: {path}')
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': actual,
            'bytes': Path(path).stat().st_size}


def partial_r2(a):
    """Accept trailing seed x span x start dimensions, including batched draws."""
    a = np.asarray(a, dtype=float)
    reduced = a.mean(axis=(-3, -1), keepdims=True)
    full = a.mean(axis=-3, keepdims=True)
    sse0 = ((a - reduced) ** 2).sum(axis=(-3, -2, -1))
    sse1 = ((a - full) ** 2).sum(axis=(-3, -2, -1))
    return np.divide(np.maximum(0, sse0 - sse1), sse0,
                     out=np.zeros_like(sse0), where=sse0 > 1e-24)


def component_test(a, draws, random_seed):
    """Descriptive seed bootstrap and start-label permutation, paired by seed/span."""
    rng = np.random.default_rng(random_seed)
    actual = float(partial_r2(a))
    # These draws are newly specified for the review, not the historical test RNG.
    sampled_seeds = rng.integers(8, size=(draws, 8))
    boot = partial_r2(a[sampled_seeds])
    permutation = np.argsort(rng.random((draws, 8, 3, 4)), axis=-1)
    null = partial_r2(np.take_along_axis(np.broadcast_to(a, permutation.shape),
                                       permutation, axis=-1))
    p = float((1 + (null >= actual - 1e-12).sum()) / (draws + 1))
    return {'partial_r2': actual, 'seed_bootstrap_95_descriptive':
            np.quantile(boot, [.025, .975]).tolist(), 'permutation_p_unadjusted': p,
            'bonferroni_family_size': 16, 'permutation_p_bonferroni_16': min(1., 16*p),
            'scope': 'Post-hoc association with joint observation/training placement; '
                     'fixed corpus, eight training seeds; no causal isolation.'}


def design(kind, full):
    span = np.repeat(np.asarray(SPANS, float), 4)
    start = np.tile(np.asarray(STARTS, float), 3)
    if kind == 'original_numeric_ridge':
        return np.column_stack([np.ones(12), span] + ([start] if full else []))
    columns = [np.ones(12), span == 4, span == 6]
    if full:
        columns.extend(start == b for b in STARTS[1:])
        columns.extend((span == s) & (start == b)
                       for s in SPANS[1:] for b in STARTS[1:])
    return np.column_stack(columns).astype(float)


def prediction(a, kind):
    out = {'feature_kind': kind, 'post_hoc': kind != 'original_numeric_ridge',
           'fit_seeds': SEEDS[:6], 'heldout_seeds': SEEDS[6:],
           'lambda': 0. if kind == 'matched_categorical_ols' else .001,
           'intercept_penalized': False, 'feature_scaling': 'none'}
    for full, label in [(False, 'span_only'), (True, 'span_plus_start')]:
        x = design(kind, full)
        xx = np.tile(x, (6, 1))
        penalty = np.eye(x.shape[1]) * out['lambda']
        penalty[0, 0] = 0.
        coef = np.linalg.solve(xx.T @ xx + penalty, xx.T @ a[:6].reshape(-1))
        if kind == 'matched_categorical_ols':
            expected = a[:6].mean(axis=0) if full else np.broadcast_to(
                a[:6].mean(axis=(0, 2))[:, None], (3, 4))
            if not np.allclose(x @ coef, expected.reshape(-1), atol=1e-12, rtol=0):
                raise ValueError('Categorical OLS does not match training-cell means')
        residual = a[6:].reshape(2, 12) - x @ coef
        out[label + '_rmse'] = float(np.sqrt(np.mean(residual ** 2)))
        out[label + '_per_seed_rmse'] = {
            str(seed): float(np.sqrt(np.mean(row ** 2)))
            for seed, row in zip(SEEDS[6:], residual)}
        out[label + '_features'] = int(x.shape[1])
    out['relative_rmse_reduction'] = 1 - out['span_plus_start_rmse']/out['span_only_rmse']
    return out


def archive_sources(path, protocol, wanted):
    """Hash and parse the source that actually accompanied the experiment."""
    sources = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if member.name in wanted:
                content = archive.extractfile(member).read()
                digest = hashlib.sha256(content).hexdigest()
                if digest != protocol['implementation_sha256'][member.name]:
                    raise ValueError('Frozen source member mismatch: ' + member.name)
                text = content.decode()
                tree = ast.parse(text)
                function = next(node for node in tree.body
                                if isinstance(node, ast.FunctionDef)
                                and node.name == 'prediction_comparison')
                sources[member.name] = {'sha256': digest,
                    'prediction_function': ast.get_source_segment(text, function)}
    if set(sources) != set(wanted):
        raise ValueError('Frozen analyzer source missing')
    return sources


def prepare():
    manifest = {}
    for line in (GPT_PACKAGE/'MANIFEST.sha256').read_text().splitlines():
        digest, rel = line.split(None, 1)
        manifest[rel] = digest
    gpt_analysis = ROOT/'results/p6-corrected-followup-20260917/p6-corrected-gpt2-v2.analysis.json'
    gpt_protocol = ROOT/'experiments/crossed_span_protocol.p6.v2.json'
    gpt_source = GPT_PACKAGE/'source/p6-source-d5ec03e.tar.gz'
    closure = read(ROOT/'docs/qwen-fp64-closure-2026-09-21.json')
    qwen_analysis = QWEN/'collected/grid.analysis.json'
    qwen_protocol = QWEN/'metadata/crossed_span_protocol.p6.qwen3.fp64.v2.json'
    qwen_source = QWEN/'metadata/execution-source.tar.gz'
    provenance = [checked(gpt_analysis, manifest['p6/p6-corrected-gpt2-v2.analysis.json']),
                  checked(gpt_protocol, manifest['p6/crossed_span_protocol.p6.v2.json']),
                  checked(gpt_source, manifest['source/p6-source-d5ec03e.tar.gz']),
                  checked(qwen_analysis, closure['analysis_sha256']),
                  checked(qwen_protocol, closure['protocol_sha256']),
                  checked(qwen_source, closure['metadata_sha256']['execution-source.tar.gz'])]
    inputs = {
        'GPT-2': (read(gpt_analysis), read(gpt_protocol), gpt_source,
                  ROOT/'build/paper-a-revision-20260917/p6/raw'),
        'Qwen3': (read(qwen_analysis), read(qwen_protocol), qwen_source,
                  QWEN/'collected/grid')}
    for model, (analysis, protocol, source, raw) in inputs.items():
        if analysis['protocol_sha256'] != sha(gpt_protocol if model == 'GPT-2' else qwen_protocol):
            raise ValueError('Summary protocol mismatch')
        if len(analysis['evidence_anchors']) != 200:
            raise ValueError('Complete final-grid anchors required')
    return inputs, provenance


def raw_availability(raw, analysis, tokens):
    anchors = {Path(k).parent.name: v for k, v in analysis['evidence_anchors'].items()}
    if len(anchors) != len(analysis['evidence_anchors']):
        raise ValueError('Duplicate anchor directory names')
    expected_trained = {(r['span'], r['start'], r['defense'], r['seed'])
                        for r in analysis['cells']}
    checked_cells = []
    seen_keys = set()
    for cell_path in sorted(raw.glob('*/cell.json')):
        cell = read(cell_path)
        if cell.get('status') != 'complete' or cell['protocol_sha256'] != analysis['protocol_sha256']:
            raise ValueError('Unqualified raw cell')
        if sha(cell_path) != anchors.get(cell_path.parent.name):
            raise ValueError('Raw cell anchor mismatch')
        if cell['kind'] not in ['trained', 'all_local']:
            raise ValueError('Unexpected raw cell kind')
        identity = (cell['span'], cell['start'], cell['defense'], cell['seed'])
        key = (cell['kind'], cell['seed']) if cell['kind'] == 'all_local' else (cell['kind'], *identity)
        if key in seen_keys:
            raise ValueError('Duplicate retained raw cell identity')
        seen_keys.add(key)
        if cell['kind'] == 'trained' and identity not in expected_trained:
            raise ValueError('Retained raw cell outside complete summary grid')
        small_files = []
        for name in ['measurements.json', 'vmi.json']:
            path = cell_path.parent/name
            if path.exists():
                small_files.append(checked(path, cell['files'][name]))
        reconstructed = None
        if cell['kind'] == 'trained' and len(small_files) == 2:
            documents = read(cell_path.parent/'measurements.json')['documents']
            records = sorted(read(cell_path.parent/'vmi.json')['results'], key=lambda r:r['step'])
            if len(documents) != 256 or len(records) != 256:
                raise ValueError('Incomplete retained document records')
            excess, norms, labels = [], [], []
            for i, (document, record) in enumerate(zip(documents, records)):
                if document['step'] != i or record['step'] != i:
                    raise ValueError('Retained document order changed')
                sequence = record['per_sequence'][0]
                correct = sum(p == t for p,t in zip(sequence['prediction'], sequence['truth']))
                if correct != sequence['correct']:
                    raise ValueError('Retained VMI prediction count mismatch')
                excess.append((correct-sequence['control_correct'])/sequence['n_tokens'])
                norms.append(document['gradient_norm'])
                labels.append([token in sequence['truth'] for token in tokens])
            # Direct pairwise AUC is independent of the archived rank implementation.
            labels = np.asarray(labels, bool)
            norms = np.asarray(norms)
            aucs = []
            for k in range(3):
                present = norms[labels[:, k]][:, None]
                absent = norms[~labels[:, k]][None, :]
                aucs.append(float(((present > absent)+.5*(present == absent)).mean()))
            original = next(r for r in analysis['cells'] if
                (r['span'],r['start'],r['defense'],r['seed']) ==
                (cell['span'],cell['start'],cell['defense'],cell['seed']))
            if not np.isclose(np.mean(excess), original['vmi_excess'], atol=1e-12, rtol=0) or not np.allclose(
                    aucs, original['property_auc'], atol=1e-12, rtol=0):
                raise ValueError('Retained document metrics differ from complete summary')
            reconstructed = {'vmi_excess':float(np.mean(excess)), 'property_auc':aucs,
                             'documents':256, 'summary_matches_within':1e-12}
        checked_cells.append({'cell': cell_path.parent.name, 'kind': cell['kind'],
                              'span': cell['span'], 'start': cell['start'],
                              'defense': cell['defense'], 'seed': cell['seed'],
                              'verified_files': small_files,
                              'document_metrics_reconstructed':reconstructed})
    trained = [x for x in checked_cells if x['kind'] == 'trained']
    reconstructed = [x for x in trained if x['document_metrics_reconstructed'] is not None]
    reconstructed_keys = {(x['span'], x['start'], x['defense'], x['seed']) for x in reconstructed}
    return {'verified_cells': len(checked_cells), 'verified_trained_cells': len(trained),
            'verified_document_metric_cells': len(reconstructed),
            'expected_trained_cells': len(expected_trained),
            'missing_document_metric_cell_keys': [list(key) for key in sorted(expected_trained-reconstructed_keys)],
            'complete_document_grid': reconstructed_keys == expected_trained,
            'label_permutation_null': (
                'COMPLETE_DOCUMENT_INPUTS_AVAILABLE: see the separate '
                'paper_a_review_null.py calibration; no selected-subset inference.'
                if reconstructed_keys == expected_trained else
                'INCOMPLETE_DOCUMENT_INPUTS: no selected-subset null is substituted '
                'for a full-grid result.'),
            'cells': checked_cells}


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'artifacts/paper-a-review-20261020/offloading')
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--gpt2-scoring', type=Path,
                        help='Explicit complete retained GPT-2 small-scoring-file root')
    parser.add_argument('--qwen-scoring', type=Path,
                        help='Explicit complete retained FP64 Qwen small-scoring-file root')
    parser.add_argument('--null-calibration', type=Path,
                        help='Completed separate document-correspondence analysis.json')
    args = parser.parse_args()
    if args.draws < 1000:
        raise ValueError('At least 1000 review resamples required')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'analysis.json').exists():
        raise ValueError('Review analysis output already exists; use a fresh directory')
    inputs, provenance = prepare()
    report = {'schema': 'paper-a-review.offloading.v1', 'status': 'COMPLETE_SUMMARY_REANALYSIS',
              'scope': 'CPU analysis of completed corrected GPT-2 and completed FP64 Qwen3 '
                       'summaries and retained scoring records. No pilot or failed FP32 outcomes. All added analyses '
                       'are post-hoc; causal observation/training confound remains.',
              'draws': args.draws, 'random_seed': args.seed, 'inputs': provenance,
              'models': {}, 'limitations': [
                  'No new training, inference, GPU work or network access.',
                  'Analysis and source hashes validated; original model binaries and '
                  'all historical evidence journals were not reverified in this review.',
                  'Permutation calibration of the composite maximum is a separate '
                  'post-hoc reference analysis, not part of the original frozen tests.',
                  'Eight training seeds and a fixed corpus; two held-out seeds have '
                  'already been examined. Alternative predictors are post-hoc diagnostics.',
                  'Start remains coupled to observation depth and frozen prefix.']}
    if args.null_calibration:
        null_path = args.null_calibration.resolve()
        null_report = read(null_path)
        if null_report['status'] != 'COMPLETE_BOTH_GRIDS':
            raise ValueError('Complete both-grid null calibration required')
        report['document_null_calibration'] = checked(null_path, sha(null_path))
        report['document_null_calibration']['scope'] = (
            'Post-hoc fixed-output document-correspondence reference under '
            'document exchangeability; nonsignificance does not establish noise.')
    placements, seed_rows, tests, prediction_rows = [], [], [], []
    for model, (analysis, protocol, archive, raw) in inputs.items():
        requested_raw = args.gpt2_scoring if model == 'GPT-2' else args.qwen_scoring
        if requested_raw is not None:
            raw = requested_raw.resolve()
        wanted = ['experiments/analyze_p6_corrected.py' if model == 'GPT-2'
                  else 'experiments/qwen_p6/analyze.py']
        source = archive_sources(archive, protocol, wanted)
        cells = analysis['cells']
        indexed = {(r['defense'], r['seed'], r['span'], r['start']): r for r in cells}
        expected = {(d, v, s, b) for d in ['D0', 'D1'] for v in SEEDS
                    for s in SPANS for b in STARTS}
        if set(indexed) != expected or len(cells) != len(expected):
            raise ValueError('Complete unique summary grid required')
        model_report = {'property_tokens': protocol['property_tokens'],
                        'executed_analyzer': source,
                        'raw_availability': raw_availability(raw, analysis, protocol['property_tokens']),
                        'defenses': {}}
        for defense in ['D0', 'D1']:
            arrays = {name: np.empty((8, 3, 4)) for name in
                      ['delta_utility', 'vmi_excess', 'normalised_leakage_index',
                       'property_max'] + [f'property_auc_{t}' for t in protocol['property_tokens']]}
            for i, victim in enumerate(SEEDS):
                for j, span in enumerate(SPANS):
                    for k, start in enumerate(STARTS):
                        cell = indexed[(defense, victim, span, start)]
                        aucs = np.asarray(cell['property_auc'], float)
                        if len(aucs) != 3 or not np.isfinite(aucs).all() or (aucs < 0).any() or (aucs > 1).any():
                            raise ValueError('Invalid property AUCs')
                        value = max(0., cell['vmi_excess'], float(np.max(2*np.abs(aucs-.5))))
                        if not np.isclose(value, cell['normalised_leakage_index'], atol=1e-12, rtol=0):
                            raise ValueError('Composite reconstruction mismatch')
                        for name in ['delta_utility', 'vmi_excess', 'normalised_leakage_index']:
                            arrays[name][i,j,k] = cell[name]
                        arrays['property_max'][i,j,k] = np.max(2*np.abs(aucs-.5))
                        for token, auc in zip(protocol['property_tokens'], aucs):
                            arrays[f'property_auc_{token}'][i,j,k] = auc
                        seed_rows.append({'model': model, 'defense': defense, 'seed': victim,
                            'span': span, 'start': start,
                            **{name: float(a[i,j,k]) for name,a in arrays.items()}})
            d_report = {'component_tests': {}, 'prediction': {}, 'frozen_prefix_checks': []}
            for component in ['vmi_excess'] + [f'property_auc_{t}' for t in protocol['property_tokens']]:
                test = component_test(arrays[component], args.draws, args.seed)
                d_report['component_tests'][component] = test
                tests.append({'model': model, 'defense': defense, 'component': component,
                              **{k:v for k,v in test.items() if not isinstance(v, (list, dict))}})
            for endpoint in ['delta_utility', 'normalised_leakage_index']:
                a = arrays[endpoint]
                r2 = float(partial_r2(a))
                previous = analysis['defenses'][defense][endpoint]
                if not np.isclose(r2, previous['variance']['partial_r2'], atol=1e-12, rtol=0):
                    raise ValueError('Historical partial R2 reconstruction mismatch')
                predictions = {kind: prediction(a, kind) for kind in
                    ['original_numeric_ridge', 'matched_categorical_ridge', 'matched_categorical_ols']}
                for name in ['span_only_rmse', 'span_plus_start_rmse']:
                    if not np.isclose(predictions['original_numeric_ridge'][name],
                                      previous['heldout_prediction'][name], atol=1e-12, rtol=0):
                        raise ValueError('Historical ridge reconstruction mismatch')
                d_report['prediction'][endpoint] = {
                    'historical_partial_r2_reproduced': r2, 'models': predictions}
                for kind, result in predictions.items():
                    prediction_rows.append({'model':model, 'defense':defense,
                        'endpoint':endpoint, **{k:v for k,v in result.items()
                                               if not isinstance(v, (list, dict))}})
            for start in STARTS[1:]:
                k = STARTS.index(start)
                a = arrays['vmi_excess'][:,:,k]
                d_report['frozen_prefix_checks'].append({'start':start,
                    'vmi_range_over_all_seeds_and_lengths':float(np.ptp(a)),
                    'vmi_max_length_range_within_seed':float(np.max(np.ptp(a, axis=1)))})
            for j, span in enumerate(SPANS):
                for k, start in enumerate(STARTS):
                    vmi = arrays['vmi_excess'][:,j,k]
                    prop = arrays['property_max'][:,j,k]
                    row = {'model':model, 'defense':defense, 'span':span, 'start':start}
                    for name,a in arrays.items():
                        row[name+'_mean'] = float(a[:,j,k].mean())
                        row[name+'_sd_across_seeds'] = float(a[:,j,k].std(ddof=1))
                    row['vmi_dominant_seed_count'] = int(((vmi > prop) & (vmi > 0)).sum())
                    row['property_dominant_seed_count'] = int((prop > np.maximum(vmi,0)).sum())
                    placements.append(row)
            model_report['defenses'][defense] = d_report
        report['models'][model] = model_report
        print(f'{model}: complete 192-cell summary reconstructed; '
              f'{model_report["raw_availability"]["verified_document_metric_cells"]}/192 '
              'document-level cells available.', flush=True)
    # Different architectures have different token IDs; use a union CSV schema.
    fields = list(dict.fromkeys(k for row in seed_rows for k in row))
    save_csv(args.output/'per-seed-components.csv', [{k:r.get(k,'') for k in fields} for r in seed_rows])
    fields = list(dict.fromkeys(k for row in placements for k in row))
    save_csv(args.output/'placement-components.csv', [{k:r.get(k,'') for k in fields} for r in placements])
    save_csv(args.output/'component-placement-tests.csv', tests)
    save_csv(args.output/'heldout-prediction.csv', prediction_rows)
    (args.output/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'], 'output':str(args.output),
                      'component_tests':len(tests), 'prediction_comparisons':len(prediction_rows),
                      'analysis_sha256':sha(args.output/'analysis.json')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
