"""Complete-set, fail-closed analysis of the separately frozen v4 adaptation.

No model dependencies; all predictions and controls are verified before any
candidate recovery is computed. Frozen crossed bootstrap is reused unchanged.
"""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attacker.bisr_fidelity import sha256
from attacker.reconstruction_metrics import reconstruction_metrics
from experiments.bisr_campaign_score import verified_rows
from experiments.analyze_validation import crossed_interval

VICTIMS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
DEPTHS = (3, 6)
GRANTS = ('public', 'exact')
ATTACKS = (17, 29)
CONTROLS = ('forward_k32', 'sipit_steps128', 'joint_true_k32')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cellkey(v, d, a, g):
    return f'v{v}-d{d}-a{a}-{g}'


def metrics(documents, predictions):
    require(len(predictions) == len(documents), 'Prediction document count')
    for doc, pred in zip(documents, predictions):
        require(len(pred) == len(doc['tokens']) and
                all(type(t) is int and 0 <= t < 50257 for t in pred),
                'Invalid prediction token shape or vocabulary')
    return [{'document_id': d['id'], **reconstruction_metrics(
        d['tokens'], p, d.get('annotations', []))} for d, p in zip(documents, predictions)]


def analyze(predictions, output):
    import numpy as np
    require(not output.exists(), 'Output must be fresh')
    admission = ROOT / 'experiments/v4-final-admission'
    inputs = ROOT / 'experiments/v4-analysis-inputs'
    pins = read(inputs / 'pins.json')
    for name in ('controls.json', 'v3-input.json'):
        digest = pins[name]
        require(sha256(inputs / name) == digest, 'Input pin mismatch: ' + name)
    protocol = read(ROOT / 'experiments/validation_protocol.v3.json')
    require(pins['v3-input.json'] == protocol['frozen_inputs']['tokenized_sha256'], 'Wrong v3 inputs')
    docs = read(inputs / 'v3-input.json')['evaluation']
    ids = [d['id'] for d in docs]
    require(len(ids) == len(set(ids)) == 256, 'Expected 256 distinct documents')
    launch = read(admission / 'launch.json')
    launch_sha = sha256(admission / 'launch.json')
    expected = {cellkey(v, d, a, g) for v, d, a, g in itertools.product(VICTIMS, DEPTHS, ATTACKS, GRANTS)}
    require(set(launch['cells']) == expected, 'Launch matrix differs from frozen 80 cells')
    for name, digest in launch['evidence_files'].items():
        require(sha256(admission / name) == digest, 'Admission evidence changed: ' + name)
    complete = {}
    integrity = {'status': 'COMPLETE_SET_VERIFIED_BEFORE_COMPARATIVE_SCORING',
                 'launch_sha256': launch_sha, 'input_pins': pins,
                 'hosts': {}, 'cells': {}, 'controls': {}, 'documents': 256,
                 'candidate_joint_cells': 80, 'candidate_document_executions': 20480,
                 'control_cells': 240, 'controls_source_sha256': None}
    for host in ('toula', 'koula'):
        hp = predictions / host
        result = read(hp / 'result.json')
        resource = read(hp / 'resource.json')
        keys = {k for k, c in launch['cells'].items() if c['host'] == host}
        require(len(keys) == 40 and set(result['cells']) == keys, 'Host matrix mismatch')
        require(result['status'] == 'HOST_PREDICTIONS_COMPLETE' and result['host'] == host
                and result['documents'] == 10240 and result['launch_sha256'] == launch_sha,
                'Host result incomplete or wrong launch')
        require(resource['exit_code'] == 0 and resource['host'] == host and
                resource['elapsed_seconds'] <= launch['per_host_limit_seconds'], 'Host resource failure')
        require({p.name for p in hp.iterdir() if p.is_dir()} == keys, 'Unexpected cell directory')
        require(not (hp / 'failed.json').exists(), 'Failed host attempt present')
        integrity['hosts'][host] = {'result_sha256': sha256(hp / 'result.json'),
                                    'resource_sha256': sha256(hp / 'resource.json'), 'resource': resource}
        for key in sorted(keys):
            manifest = admission / 'records' / (key + '.json')
            digest = launch['cells'][key]['manifest_sha256']
            require(sha256(manifest) == digest, 'Changed cell manifest')
            cp = hp / key
            require({p.name for p in cp.iterdir()} ==
                    {f'{i:05d}.json' for i in range(256)} | {'result.json', 'journal.jsonl'},
                    'Unexpected or missing cell files: ' + key)
            require(sha256(cp / 'result.json') == result['cells'][key]['result_sha256'], 'Cell result hash mismatch')
            finish = read(cp / 'result.json')
            require(finish['journal_tip'] == result['cells'][key]['journal_tip'] and
                    finish['online_seconds'] == result['cells'][key]['online_seconds'], 'Cell summary mismatch')
            rows = verified_rows(cp, ids, digest)
            declared = read(manifest)
            for row in rows:
                require(set(row['arms']) == {'sip', 'b', 'bf'} and row['engine'] == 'candidate'
                        and row['budgets'] == {'gm': 32, 'sm_bf': 800}, 'Changed attack arms/budgets')
                require(row['grant']['boundary'] == declared['boundary'] and
                        row['grant']['knowledge'] == declared['knowledge'] and
                        row['grant']['downstream'] == 'public' and
                        row['grant']['head_sha256'] == (declared['head']['sha256'] if declared['head'] else None),
                        'Prediction grant differs from frozen manifest')
                require(set(row['work']) == {'sip', 'gm', 'sm_bf'} and row['bf_reuses_b_stage'] is True,
                        'Stage accounting differs from frozen algorithm')
                for arm in row['arms'].values():
                    require(len(arm) == 1 and len(arm[0]) == 32 and
                            all(type(t) is int and 0 <= t < 50257 for t in arm[0]), 'Invalid candidate predictions')
                for work in row['work'].values():
                    require(math.isfinite(work['wall_seconds']) and work['wall_seconds'] >= 0, 'Invalid stage time')
            stage_time = sum(w['wall_seconds'] for r in rows for w in r['work'].values())
            require(math.isclose(stage_time, finish['online_seconds'], rel_tol=1e-10, abs_tol=1e-8), 'Online accounting mismatch')
            complete[key] = rows
            integrity['cells'][key] = {'manifest_sha256': digest,
                                       'result_sha256': sha256(cp / 'result.json'),
                                       'journal_sha256': sha256(cp / 'journal.jsonl'),
                                       'journal_tip': finish['journal_tip'],
                                       'online_seconds': finish['online_seconds'],
                                       'cold_load_seconds': finish['cold_load_seconds']}
        print('Verified candidate host ' + host, flush=True)
    controls = read(inputs / 'controls.json')
    integrity['controls_source_sha256'] = controls['source_sha256']
    require(controls['input_sha256'] == pins['v3-input.json'], 'Controls input mismatch')
    seen = {}
    expected_controls = set(itertools.product(VICTIMS, DEPTHS, GRANTS, ATTACKS, CONTROLS))
    expected_files = set()
    for cell in controls['cells']:
        key = tuple(cell[f] for f in ('victim_seed', 'depth', 'knowledge', 'attack_seed', 'arm'))
        require(key in expected_controls and key not in seen and cell['mode'] == 'training', 'Control matrix or duplicate')
        require(cell['measurement_reused'] is False and cell['execution_id'] == cell['cell_id'], 'Control execution identity')
        name = cell['cell_id'] + '.json'
        expected_files.add(name)
        path = inputs / 'control-predictions' / name
        require(sha256(path) == cell['prediction_sha256'], 'Frozen control prediction changed')
        pred = read(path)
        require(pred['document_ids'] == ids, 'Control document order mismatch')
        scored = metrics(docs, pred['predictions'])
        require(scored == cell['metrics'], 'Frozen control metrics fail rescoring')
        seen[key] = scored
        integrity['controls'][cell['cell_id']] = cell['prediction_sha256']
    require(set(seen) == expected_controls and len(seen) == 240, 'Missing control cells')
    require({p.name for p in (inputs / 'control-predictions').iterdir()} == expected_files, 'Extra controls')
    output.mkdir(parents=True)
    sip_equal_pairs = sum(
        all(left['arms']['sip'] == right['arms']['sip'] for left, right in zip(
            complete[cellkey(v, d, a, 'public')], complete[cellkey(v, d, a, 'exact')]))
        for v, d, a in itertools.product(VICTIMS, DEPTHS, ATTACKS))
    integrity['sip_grant_pair_identity'] = {
        'pairs_checked': 40, 'pairs_with_all_256_predictions_identical': sip_equal_pairs,
        'independent_attack_replications_added_by_identical_grant_pairs': 0}
    write(output / 'integrity.json', integrity)
    print('INTEGRITY PASS: 80 joint cells, 240 rescored frozen controls; comparative scoring starts', flush=True)
    scores = dict(seen)
    stage_totals = defaultdict(float)
    layer_work = defaultdict(lambda: defaultdict(int))
    with (output / 'document-metrics.jsonl').open('w') as handle:
        for v, d, a, g in itertools.product(VICTIMS, DEPTHS, ATTACKS, GRANTS):
            rows = complete[cellkey(v, d, a, g)]
            for short, arm in (('sip', 'bisr_sip'), ('bf', 'bisr_bre_bf')):
                scores[v, d, g, a, arm] = metrics(docs, [r['arms'][short][0] for r in rows])
            for row in rows:
                for stage, work in row['work'].items():
                    stage_totals[stage] += work['wall_seconds']
                    for name, value in work['layer_work'].items():
                        layer_work[stage][name] += value
        for (v, d, g, a, arm), rows in sorted(scores.items()):
            for row in rows:
                handle.write(json.dumps({'victim_seed': v, 'depth': d, 'knowledge': g,
                                         'attack_seed': a, 'arm': arm, **row}) + '\n')
    main = (6, 'public', 'bisr_bre_bf', 'joint_true_k32')
    secondary = [(d, g, left, right) for left, right in
                 (('bisr_bre_bf', 'forward_k32'), ('bisr_bre_bf', 'sipit_steps128'),
                  ('bisr_sip', 'forward_k32')) for d, g in itertools.product(DEPTHS, GRANTS)]
    secondary += [(d, g, 'bisr_bre_bf', 'joint_true_k32') for d, g in itertools.product(DEPTHS, GRANTS)
                  if (d, g) != (6, 'public')]
    require(len(secondary) == len(set(secondary)) == 15, 'Wrong secondary family')
    contrasts = []
    for index, (d, g, left, right) in enumerate([main] + secondary):
        matrix = np.array([[np.mean([scores[v, d, g, a, left][j]['token_accuracy'] -
                                     scores[v, d, g, a, right][j]['token_accuracy'] for a in ATTACKS])
                            for j in range(256)] for v in VICTIMS])
        ci = crossed_interval(matrix, alpha=.05 if index == 0 else .05 / 15,
                              replicates=100000, seed=93281)
        result = {'depth': d, 'knowledge': g, 'left': left, 'right': right,
                  'primary': index == 0, **ci,
                  'victim_seed_differences': dict(zip(map(str, VICTIMS), matrix.mean(axis=1).tolist())),
                  'above_zero': ci['lower'] > 0, 'below_zero': ci['upper'] < 0,
                  'above_practical_margin': ci['lower'] > .02 if index == 0 else None,
                  'below_negative_practical_margin': ci['upper'] < -.02 if index == 0 else None}
        contrasts.append(result)
        print('CONTRAST ' + json.dumps(result), flush=True)
    arms = []
    for d, g, arm in itertools.product(DEPTHS, GRANTS, CONTROLS + ('bisr_sip', 'bisr_bre_bf')):
        per_run = {str(v): float(np.mean([q['token_accuracy'] for a in ATTACKS
                                        for q in scores[v, d, g, a, arm]])) for v in VICTIMS}
        arms.append({'depth': d, 'knowledge': g, 'arm': arm,
                     'token_accuracy': float(np.mean(list(per_run.values()))), 'victim_means': per_run})
    write(output / 'contrasts.json', contrasts)
    write(output / 'arm-summaries.json', arms)
    host_seconds = sum(h['resource']['elapsed_seconds'] for h in integrity['hosts'].values())
    summary = {'status': 'COMPLETE', 'scope': launch['scope'], 'published_setting_reproduced': False,
               'primary': contrasts[0], 'secondary_family_size': 15,
               'bootstrap': {'replicates': 100000, 'seed': 93281, 'unit': 'independently resampled victim and document axes; attack seeds averaged first',
                             'secondary_alpha': .05/15, 'secondary_expected_draws_per_tail': 100000*.05/30},
               'arm_summaries': arms, 'sip_grant_pair_identity': integrity['sip_grant_pair_identity'], 'compute': {'stage_seconds': dict(stage_totals),
               'layer_work': dict(layer_work), 'cold_load_seconds': sum(c['cold_load_seconds'] for c in integrity['cells'].values()),
               'prediction_host_elapsed_gpu_hours': host_seconds/3600,
               'prior_and_preparation_reserved_gpu_hours': launch['prior_checks_reserved_gpu_hours'] + launch['completed_preparation_reserved_gpu_hours'],
               'approved_gpu_hours': launch['approved_gpu_hours'], 'host_resources': integrity['hosts']},
               'limitations': ['Single-seam adaptation; original dual-cut setting not reproduced.',
               'Distinct attack budgets; no equal-compute claim. Layer work is not FLOPs.',
               'Crossed bootstrap conditional on fixed training corpus and four shared trained GRUs (two attack seeds by two depths); no uncertainty over auxiliary training or attacker retraining; approximate with ten victim runs.',
               'Only token-accuracy contrasts are inferential; other reconstruction metrics descriptive.',
               'An interval crossing zero is not evidence of equivalence.',
               'SIP versus forward compares initialisation methods; it does not isolate the BRE stage.',
               'SIP grant-pair token identities are measured in sip_grant_pair_identity; identical predictions do not add independent attacker replications.'],
               'provenance': {'python': platform.python_version(), 'numpy': np.__version__, 'analyzer_sha256': sha256(Path(__file__)),
                              'launch_sha256': launch_sha, 'controls_sha256': pins['controls.json']},
               'outputs': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write(output / 'result.json', summary)
    print(json.dumps({'status': 'COMPLETE', 'primary': summary['primary'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.predictions, args.output)
