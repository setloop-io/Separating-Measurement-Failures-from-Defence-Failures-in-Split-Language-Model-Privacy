#!/usr/bin/env python3
"""Post-hoc CPU permutation references for completed P6 scoring records.

No training, inference, model loading, binary captures or network access.
Requires all 200 sealed scoring cells for each architecture. Document identity
is permuted jointly across property labels and VMI truths; fixed predictions
and gradient norms remain unchanged. This is an exchangeability reference,
not a universal information-theoretic or deployment privacy guarantee.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper_a_review_offloading import (  # noqa: E402
    SEEDS, SPANS, STARTS, QWEN, archive_sources, checked, prepare, read, sha,
)


def average_ranks(values):
    """Average ranks with exact ties, without scipy or model dependencies."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    unique_ranks = np.cumsum(counts) - (counts - 1)/2
    return unique_ranks[inverse]


def auc_from_ranks(ranks, labels):
    labels = np.asarray(labels, float)
    n = labels.sum(axis=0)
    if (n == 0).any() or (n == len(labels)).any():
        raise ValueError('A property lacks one evaluation class')
    return (ranks @ labels - n*(n+1)/2)/(n*(len(labels)-n))


def validate_reference_algebra():
    """Known four-document example; exhaustively verify all 24 permutations."""
    norms = np.asarray([4.,3.,2.,1.])
    labels = np.asarray([[1,1,1],[1,0,1],[0,1,0],[0,0,0]],bool)
    ranks = average_ranks(norms)
    if not np.allclose(auc_from_ranks(ranks,labels),[1.,.75,1.],atol=1e-15,rtol=0):
        raise ValueError('Known-case AUC validation failed')
    if not np.allclose(auc_from_ranks(average_ranks(np.ones(4)),labels),.5,atol=1e-15,rtol=0):
        raise ValueError('Tie handling validation failed')
    truth = np.asarray([[1,2],[1,3],[4,2],[4,3]])
    predictions = np.asarray([[1,2],[1,4],[4,2],[3,3]])
    matches = (predictions[:,None,:] == truth[None,:,:]).sum(axis=-1)
    aucs = []
    for permutation in itertools.permutations(range(4)):
        perm = np.asarray(permutation)
        permuted_labels = labels[perm]
        actual = auc_from_ranks(ranks,permuted_labels)
        direct = []
        for j in range(3):
            positive = norms[permuted_labels[:,j]][:,None]
            negative = norms[~permuted_labels[:,j]][None,:]
            direct.append(((positive>negative)+.5*(positive==negative)).mean())
        if not np.allclose(actual,direct,atol=1e-15,rtol=0):
            raise ValueError('Permutation AUC differs from direct pairwise scoring')
        if matches[np.arange(4),perm].sum()/8 != np.mean(predictions==truth[perm]):
            raise ValueError('Permutation token matching differs from direct scoring')
        aucs.append(actual)
    if not np.allclose(np.mean(aucs,axis=0),.5,atol=1e-15,rtol=0):
        raise ValueError('Exhaustive null AUC mean differs from one half')
    return {'known_cases':'AUC 1,.75,1; all-tied AUC .5; all 24 four-document permutations',
            'pairwise_auc_and_token_matching':'PASS','exhaustive_null_mean_auc':.5,
            'full_grid_identity_reconstruction':'Checked before permutation calculation for each architecture'}


def load_complete_scoring(model, root, analysis, protocol, input_path):
    """Verify only sealed small scoring files, then independently rescore them."""
    input_record = checked(input_path, protocol['input_sha256'])
    data = read(input_path)
    canonical = data['evaluation']
    if len(canonical) != 256 or any(len(r['tokens']) != 32 for r in canonical):
        raise ValueError('Unexpected canonical evaluation size')
    truth = np.asarray([r['tokens'] for r in canonical], dtype=np.int32)
    labels = np.column_stack([(truth == t).any(axis=1) for t in protocol['property_tokens']])
    anchors = {Path(k).parent.name:v for k,v in analysis['evidence_anchors'].items()}
    if len(anchors) != 200:
        raise ValueError('Complete unique final-grid anchors required')
    paths = {p.parent.name:p for p in root.glob('*/cell.json')}
    if set(paths) != set(anchors):
        missing = sorted(set(anchors)-set(paths))
        extra = sorted(set(paths)-set(anchors))
        raise ValueError(f'{model}: incomplete/unexpected cell set; missing={missing[:5]}, extra={extra[:5]}')
    expected = {(d, v, s, b) for d in ['D0','D1'] for v in SEEDS for s in SPANS for b in STARTS}
    originals = {(r['defense'],r['seed'],r['span'],r['start']):r for r in analysis['cells']}
    if set(originals) != expected:
        raise ValueError('Summary grid is incomplete')
    cal_counts = collections.Counter(t for r in data['calibration'] for t in r['tokens'])
    maximum = max(cal_counts.values())
    allowed_control = {t for t,count in cal_counts.items() if count == maximum}
    controls, cells, files = {}, {}, []
    seen = set()
    for name, path in sorted(paths.items()):
        files.append(checked(path, anchors[name]))
        cell = read(path)
        if cell.get('status') != 'complete' or cell['protocol_sha256'] != analysis['protocol_sha256']:
            raise ValueError('Unqualified cell status/protocol: '+name)
        if cell['input_sha256'] != protocol['input_sha256'] or cell['model_revision'] != protocol['model_revision']:
            raise ValueError('Cell input/model pin mismatch: '+name)
        if cell['kind'] not in ['trained','all_local']:
            raise ValueError('Unexpected cell kind')
        identity = (cell['defense'],cell['seed'],cell['span'],cell['start'])
        key = ('all_local',cell['seed']) if cell['kind']=='all_local' else ('trained',*identity)
        if key in seen:
            raise ValueError('Duplicate cell identity')
        seen.add(key)
        measurements_path, vmi_path = path.parent/'measurements.json', path.parent/'vmi.json'
        for file in [measurements_path,vmi_path]:
            files.append(checked(file,cell['files'][file.name]))
        measurements = read(measurements_path)['documents']
        records = sorted(read(vmi_path)['results'], key=lambda r:r['step'])
        if len(measurements) != 256 or len(records) != 256:
            raise ValueError('Incomplete document scoring: '+name)
        predictions, norms, losses, control_scores = [], [], [], []
        for i,(document,record) in enumerate(zip(measurements,records)):
            if (document['step'] != i or record['step'] != i or
                document['document_id'] != canonical[i]['id'] or
                document['source_document_id'] != canonical[i]['source_document_id'] or
                len(record['per_sequence']) != 1):
                raise ValueError('Document identity/order mismatch: '+name)
            sequence = record['per_sequence'][0]
            if sequence['truth'] != canonical[i]['tokens'] or sequence['n_tokens'] != 32:
                raise ValueError('VMI truth differs from pinned input')
            prediction = np.asarray(sequence['prediction'], dtype=np.int32)
            if prediction.shape != (32,):
                raise ValueError('VMI prediction shape mismatch')
            correct = int((prediction == truth[i]).sum())
            control_token = record['control_token']
            control_correct = int((truth[i] == control_token).sum())
            if (control_token not in allowed_control or correct != sequence['correct'] or
                control_correct != sequence['control_correct']):
                raise ValueError('VMI or majority-control rescore mismatch')
            predictions.append(prediction)
            norms.append(document['gradient_norm'])
            losses.append(document['loss'])
            control_scores.append(control_correct/32)
        norms, losses = np.asarray(norms,float), np.asarray(losses,float)
        if not np.isfinite(norms).all() or not np.isfinite(losses).all():
            raise ValueError('Non-finite scoring input')
        predictions = np.asarray(predictions,dtype=np.int32)
        ranks = average_ranks(norms)
        aucs = auc_from_ranks(ranks,labels)
        control_mean = float(np.mean(control_scores))
        vmi_excess = float(np.mean(predictions == truth)-control_mean)
        index = max(0.,vmi_excess,float(np.max(2*np.abs(aucs-.5))))
        if cell['kind']=='all_local':
            controls[cell['seed']] = losses
        else:
            if identity not in expected:
                raise ValueError('Unexpected trained-cell identity')
            previous = originals[identity]
            for actual,wanted in [(vmi_excess,previous['vmi_excess']),
                                  (index,previous['normalised_leakage_index'])]:
                if not np.isclose(actual,wanted,atol=1e-12,rtol=0):
                    raise ValueError('Complete-grid metric reconstruction mismatch')
            if not np.allclose(aucs,previous['property_auc'],atol=1e-12,rtol=0):
                raise ValueError('Complete-grid property AUC reconstruction mismatch')
            cells[identity] = {'predictions':predictions,'norms':norms,'ranks':ranks,
                              'losses':losses,'control_mean':control_mean,
                              'vmi_excess':vmi_excess,'property_index':float(np.max(2*np.abs(aucs-.5))),
                              'index':index}
    if set(cells) != expected or set(controls) != set(SEEDS):
        raise ValueError('Complete trained/control grids required')
    for (d,v,s,b),cell in cells.items():
        utility = float(np.mean(cell['losses']-controls[v]))
        if not np.isclose(utility,originals[(d,v,s,b)]['delta_utility'],atol=1e-12,rtol=0):
            raise ValueError('Complete paired utility reconstruction mismatch')
    order = [(d,v,s,b) for d in ['D0','D1'] for v in SEEDS for s in SPANS for b in STARTS]
    return {'cells':cells,'order':order,'truth':truth,'labels':labels,
            'source_document_ids':[r['source_document_id'] for r in canonical],
            'verification':{'input':input_record,'scoring_files':files,
                            'verified_cells':200,'rescored_trained_cells':192,
                            'rescored_documents_per_cell':256,
                            'metric_comparison_absolute_tolerance':1e-12,
                            'scope':'Hashes/anchors and small scoring JSON only; '
                                    'binary captures and all historical journals not reverified.'}}


def describe_null(values, observed):
    return {'observed':float(observed), 'null_mean':float(values.mean()),
            'null_q025':float(np.quantile(values,.025)),
            'null_q950':float(np.quantile(values,.95)),
            'null_q975':float(np.quantile(values,.975)),
            'permutation_p_unadjusted':float((1+(values >= observed-1e-12).sum())/(len(values)+1))}


def references(model, scored, permutations, batch):
    order, cells = scored['order'],scored['cells']
    truth, labels = scored['truth'],scored['labels']
    n, length = truth.shape
    draws = len(permutations)
    ranks = np.stack([cells[k]['ranks'] for k in order])
    control = np.array([cells[k]['control_mean'] for k in order])
    vmi = np.array([cells[k]['vmi_excess'] for k in order])
    obs_property = np.array([cells[k]['property_index'] for k in order])
    obs_index = np.array([cells[k]['index'] for k in order])
    # Each matrix entry counts position-wise token matches between a fixed
    # prediction for document i and the full truth sequence for document j.
    match_counts = np.stack([np.sum(cells[k]['predictions'][:,None,:] == truth[None,:,:],
                                   axis=-1,dtype=np.uint8) for k in order])
    identity_vmi = match_counts[:,np.arange(n),np.arange(n)].sum(axis=-1,dtype=float)/(n*length)-control
    identity_property = np.max(2*np.abs(auc_from_ranks(ranks,labels)-.5),axis=-1)
    identity_index = np.maximum(identity_property,np.maximum(identity_vmi,0))
    if (not np.allclose(identity_vmi,vmi,atol=1e-12,rtol=0) or
        not np.allclose(identity_property,obs_property,atol=1e-12,rtol=0) or
        not np.allclose(identity_index,obs_index,atol=1e-12,rtol=0)):
        raise ValueError('Full-grid identity-permutation reconstruction mismatch')
    positives = labels.sum(axis=0)
    denominator = positives*(n-positives)
    offset = positives*(positives+1)/2
    null = {name:np.empty((len(order),draws),float) for name in
            ['property_only','conditional_fixed_vmi','joint_document_correspondence','vmi_document_correspondence']}
    next_progress = 25
    for start in range(0,draws,batch):
        stop = min(draws,start+batch)
        perm = permutations[start:stop]
        permuted_labels = labels[perm].transpose(1,0,2).reshape(n,-1).astype(float)
        auc = ((ranks @ permuted_labels).reshape(len(order),stop-start,3)-offset)/denominator
        property_null = np.max(2*np.abs(auc-.5),axis=-1)
        recovery_null = (match_counts[:,np.arange(n)[None,:],perm].sum(axis=-1,dtype=float)/(n*length)
                         -control[:,None])
        null['property_only'][:,start:stop] = property_null
        null['vmi_document_correspondence'][:,start:stop] = recovery_null
        null['conditional_fixed_vmi'][:,start:stop] = np.maximum(property_null,np.maximum(vmi[:,None],0))
        null['joint_document_correspondence'][:,start:stop] = np.maximum(property_null,np.maximum(recovery_null,0))
        if stop/draws*100 >= next_progress:
            print(f'{model}: permutation references {int(stop/draws*100)}% complete',flush=True)
            next_progress += 25
    cell_rows = []
    for i,(d,v,s,b) in enumerate(order):
        row = {'model':model,'defense':d,'seed':v,'span':s,'start':b,
               'observed_vmi_excess':float(vmi[i]),'observed_property_index':float(obs_property[i]),
               'observed_composite_index':float(obs_index[i])}
        for name,a in null.items():
            observed = obs_property[i] if name=='property_only' else vmi[i] if name=='vmi_document_correspondence' else obs_index[i]
            row.update({name+'_'+key:value for key,value in describe_null(a[i],observed).items() if key!='observed'})
        cell_rows.append(row)
    placement_rows, placement_draws = [],{}
    for d in ['D0','D1']:
        for s in SPANS:
            for b in STARTS:
                indices = [i for i,key in enumerate(order) if key[0]==d and key[2:]==(s,b)]
                if len(indices)!=8:
                    raise ValueError('Unpaired placement seeds')
                row = {'model':model,'defense':d,'span':s,'start':b,'victim_seeds':8,
                       'observed_vmi_excess':float(vmi[indices].mean()),
                       'observed_property_index':float(obs_property[indices].mean()),
                       'observed_composite_index':float(obs_index[indices].mean())}
                for name,a in null.items():
                    values = a[indices].mean(axis=0)
                    observed = (obs_property[indices].mean() if name=='property_only' else
                                vmi[indices].mean() if name=='vmi_document_correspondence' else obs_index[indices].mean())
                    summary = describe_null(values,observed)
                    row.update({name+'_'+key:value for key,value in summary.items() if key!='observed'})
                    placement_draws[f'{model}_{d}_s{s}_b{b}_{name}'] = values
                p = row['joint_document_correspondence_permutation_p_unadjusted']
                row['joint_document_correspondence_p_bonferroni_48'] = min(1.,48*p)
                row['joint_document_correspondence_family_size'] = 48
                placement_rows.append(row)
    return cell_rows,placement_rows,placement_draws


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpt2-root',type=Path,required=True)
    parser.add_argument('--qwen-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,
                        default=ROOT/'artifacts/paper-a-review-20261020/offloading/null-calibration')
    parser.add_argument('--draws',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--batch',type=int,default=250)
    args = parser.parse_args()
    if args.draws < 1000 or args.batch < 1:
        raise ValueError('At least 1000 draws and a positive batch size required')
    if args.output.exists():
        raise ValueError('Null-calibration output must be fresh')
    algebra_validation = validate_reference_algebra()
    inputs,provenance = prepare()
    roots = {'GPT-2':args.gpt2_root.resolve(),'Qwen3':args.qwen_root.resolve()}
    input_paths = {'GPT-2':ROOT/'results/validation-corpus-v1/validation-input.json',
                   'Qwen3':QWEN/'metadata/input.json'}
    scored,source_provenance = {},{}
    # Both complete grids must verify before any permutation inference runs.
    for model,(analysis,protocol,source,_) in inputs.items():
        wanted = ['experiments/analyze_p6_corrected.py' if model=='GPT-2'
                  else 'experiments/qwen_p6/analyze.py']
        source_provenance[model] = archive_sources(source,protocol,wanted)
        scored[model] = load_complete_scoring(model,roots[model],analysis,protocol,input_paths[model])
        print(f'{model}: all 200 sealed small-file cells verified and 192 trained-cell metrics rescored.',flush=True)
    if scored['GPT-2']['source_document_ids'] != scored['Qwen3']['source_document_ids']:
        raise ValueError('Cross-architecture source-document alignment differs')
    rng = np.random.default_rng(args.seed)
    permutations = np.argsort(rng.random((args.draws,256)),axis=1).astype(np.int16)
    cell_rows,placement_rows,all_draws = [],[],{}
    for model in inputs:
        cells,placements,draws = references(model,scored[model],permutations,args.batch)
        cell_rows.extend(cells);placement_rows.extend(placements);all_draws.update(draws)
    group_summary = []
    for model in inputs:
        for defense in ['D0','D1']:
            group = [r for r in placement_rows if r['model']==model and r['defense']==defense]
            group_summary.append({'model':model,'defense':defense,'placements':12,
                'joint_reference_bonferroni_48_below_005':sum(
                    r['joint_document_correspondence_p_bonferroni_48'] < .05 for r in group),
                'observed_composite_range':[min(r['observed_composite_index'] for r in group),
                                            max(r['observed_composite_index'] for r in group)],
                'joint_null_mean_range':[min(r['joint_document_correspondence_null_mean'] for r in group),
                                         max(r['joint_document_correspondence_null_mean'] for r in group)],
                'joint_null_q950_range':[min(r['joint_document_correspondence_null_q950'] for r in group),
                                         max(r['joint_document_correspondence_null_q950'] for r in group)]})
    args.output.mkdir(parents=True)
    report = {'schema':'paper-a-review.null-references.v1','status':'COMPLETE_BOTH_GRIDS',
              'draws':args.draws,'seed':args.seed,'batch':args.batch,
              'document_permutations_sha256':hashlib.sha256(permutations.tobytes()).hexdigest(),
              'quantile_interpretation':'q025/q975 delimit a central 95% permutation-reference '
                  'envelope; q950 is a one-sided 95th percentile. These are not confidence '
                  'intervals for population leakage or effect sizes.',
              'computation_bounds':{'cells_per_architecture':192,'documents':256,
                  'tokens_per_document':32,'maximum_pair_match_array_bytes':192*256*256,
                  'null_arrays_per_architecture_bytes':4*192*args.draws*8,
                  'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
                  'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')},
              'scope':'Post-hoc CPU rescoring and permutation reference; no inference or training.',
              'source_inputs':provenance,'executed_analyzer_sources':source_provenance,
              'algebra_validation':algebra_validation,
              'scoring_verification':{model:item['verification'] for model,item in scored.items()},
              'null_definitions':{
                  'property_only':'Jointly permute complete document-label rows across the three '
                      'properties; same document permutation across all seeds/placements and '
                      'both architectures. Recompute max_j 2|AUC_j-.5|.',
                  'conditional_fixed_vmi':'Use property-label permutations but keep observed VMI '
                      'excess fixed in max(0,VMI,property-index). This is conditional and is '
                      'not a joint no-correspondence null.',
                  'joint_document_correspondence':'Using the same document permutation, rematch '
                      'every fixed VMI prediction sequence to a permuted complete truth sequence, '
                      'and assign the corresponding three property labels to fixed gradient norms. '
                      'Recompute VMI excess, property AUCs and the composite maximum. '
                      'The aggregate calibration-majority control is unchanged by truth permutation.',
                  'vmi_document_correspondence':'VMI excess under the same full-truth document '
                      'permutation, without the composite maximum.'},
              'multiplicity':'Joint document-correspondence placement-mean tests form a separate '
                  '48-test post-hoc Bonferroni family: 2 architectures x 2 defences x 3 lengths '
                  'x 4 starts. Per-seed p-values and other reference p-values are descriptive '
                  'unadjusted diagnostics, not additional confirmatory tests.',
              'limitations':[
                  'Document exchangeability is the permutation assumption; this is a fixed-output '
                  'correspondence reference, not a universal leakage definition.',
                  'Conditioned on the fixed corpus, calibration attack, checkpoint predictions '
                  'and norm scores. No generalisation to new corpora, models or attacker families.',
                  'Source documents are permuted as whole 32-token sequences. Positional token '
                  'marginals and all three property prevalences/correlations are preserved.',
                  'Shared document permutations preserve dependence across victim seeds, placements '
                  'and aligned source documents in both architectures.',
                  'The observation-depth/trained-segment confound remains; frozen-prefix repeats '
                  'are not treated as independently permuted documents.',
                  'Complete scoring JSON verified against sealed cell anchors; raw binary captures '
                  'and every historical journal were not freshly reverified.'],
              'group_summary':group_summary,'placement_results':placement_rows}
    receipts = [ROOT/'build/paper-a-review-20261020/gpt2-scoring-recovery.json']
    report['local_recovery_receipts'] = [checked(path,sha(path)) for path in receipts if path.exists()]
    write_csv(args.output/'per-seed-null-reference.csv',cell_rows)
    write_csv(args.output/'placement-null-reference.csv',placement_rows)
    (args.output/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    paths = [Path(__file__),ROOT/'experiments/paper_a_review_offloading.py'] + sorted(args.output.iterdir())
    with (args.output/'MANIFEST.sha256').open('w') as stream:
        for path in paths:
            stream.write(sha(path)+'  '+str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)+'\n')
    print(json.dumps({'status':report['status'],'placements':len(placement_rows),
                      'per_seed_cells':len(cell_rows),'analysis_sha256':sha(args.output/'analysis.json'),
                      'output':str(args.output)},indent=2),flush=True)


if __name__=='__main__':
    main()
