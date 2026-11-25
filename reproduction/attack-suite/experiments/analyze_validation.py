"""Fail-closed paired crossed-cluster analysis of the validation matrix."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger, file_sha256, verify_evidence
from experiments.validate_reconstruction import arms


def crossed_interval(matrix, *, alpha, replicates, seed):
    """Resample independent victim runs and documents, preserving paired arms.

    Attack seeds must be averaged first. Documents recur across victim runs;
    drawing both axes independently preserves this crossed sampling design.
    """
    import numpy as np
    if not 0 < alpha < 1 or type(replicates) is not int or replicates < 2:
        raise ValueError('alpha in (0,1) and at least two replicates required')
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or min(a.shape)<2 or not np.isfinite(a).all():
        raise ValueError('finite complete run-by-document matrix required')
    rng=np.random.default_rng(seed)
    draws=np.empty(replicates)
    for i in range(replicates):
        rows=rng.integers(a.shape[0],size=a.shape[0])
        cols=rng.integers(a.shape[1],size=a.shape[1])
        draws[i]=a[rows[:,None],cols].mean()
    lo,hi=np.quantile(draws,[alpha/2,1-alpha/2])
    return {'difference':float(a.mean()),'lower':float(lo),'upper':float(hi),'alpha':alpha,
            'expected_draws_per_tail':replicates*alpha/2}


def contrasts(p):
    primary=p['primary']
    main=(primary['depth'],primary['knowledge'],primary['mode'],primary['left'],primary['right'])
    secondary=[]
    for d,k,n in itertools.product(p['depths'],p['knowledge'],p['candidate_counts']):
        for right in (f'forward_k{n}',f'joint_shuffled_k{n}',f'joint_random_k{n}','backward'):
            key=(d,k,'training',f'joint_true_k{n}',right)
            if key!=main: secondary.append(key)
    return main,secondary


def validate_report(root, expected_manifest_sha256=None, allowed_protocol_versions=(1, 2)):
    from attacker.reconstruction_metrics import reconstruction_metrics
    ok,problems=verify_evidence(str(root/'validation.json.evidence.json'))
    if not ok: raise ValueError(f'evidence verification failed: {problems}')
    report=json.loads((root/'validation.json').read_text()); p=report['protocol']
    version=p.get('version',1)
    if type(version) is not int or version not in allowed_protocol_versions:
        raise ValueError('protocol version not permitted for analysis')
    tier=report.get('evidence_tier')
    if tier not in ('engineering_smoke','held_out_controlled_probe'): raise ValueError('unknown evidence tier')
    if expected_manifest_sha256 is not None and file_sha256(root/'validation.json.evidence.json')!=expected_manifest_sha256:
        raise ValueError('external manifest anchor mismatch')
    if tier=='held_out_controlled_probe':
        from experiments.validate_reconstruction import validate_protocol
        if expected_manifest_sha256 is None: raise ValueError('real analysis requires external manifest anchor')
        if p['status']!='frozen' or json.loads((root/'protocol-source.json').read_text())!=p:
            raise ValueError('real evaluation requires unchanged frozen protocol')
        validate_protocol(p)
        if report['input_sha256']!=p['frozen_inputs']['tokenized_sha256'] or report.get('supplied_input_sha256')!=p['frozen_inputs']['tokenized_sha256']:
            raise ValueError('input binding differs from frozen protocol')

    if file_sha256(root/'protocol-source.json')!=report['protocol_sha256'] or file_sha256(root/'protocol-effective.json')!=report['effective_protocol_sha256']:
        raise ValueError('protocol hash mismatch')
    if json.loads((root/'protocol-effective.json').read_text())!=p:
        raise ValueError('embedded protocol mismatch')
    for name,digest in report['artifacts'].items():
        path=root/name
        if path.parent!=root or file_sha256(path)!=digest: raise ValueError('artifact hash mismatch')
    expected_artifacts={f'{kind}-{s}-{d}.pt' for kind,s,d in itertools.product(('public','exact','observations'),p['victim_seeds'],p['depths'])}
    if set(report['artifacts'])!=expected_artifacts: raise ValueError('artifact matrix mismatch')
    expected_setup={f'{s}-{d}-{k}' for s,d,k in itertools.product(p['victim_seeds'],p['depths'],p['knowledge'])}
    if len(report['setup'])!=len(expected_setup) or {s['id'] for s in report['setup']}!=expected_setup: raise ValueError('setup matrix mismatch')
    if len(report['victims'])!=len(p['victim_seeds']) or {v['seed'] for v in report['victims']}!=set(p['victim_seeds']): raise ValueError('victim matrix mismatch')
    if report['status']!='COMPLETE': raise ValueError('incomplete or failed run')
    if file_sha256(root/'input.json')!=report['input_sha256']: raise ValueError('input hash mismatch')
    data=json.loads((root/'input.json').read_text()); docs=data['evaluation']
    expected={(s,d,k,a,m,arm) for s,d,k,a,m in itertools.product(p['victim_seeds'],p['depths'],p['knowledge'],p['attack_seeds'],('training','inference')) for arm in arms(p,m)}
    seen={}
    for cell in report['cells']:
        key=tuple(cell[f] for f in ('victim_seed','depth','knowledge','attack_seed','mode','arm'))
        if key in seen or key not in expected: raise ValueError('duplicate or unexpected cell')
        path=root/cell['prediction_file']
        if path.parent!=root or file_sha256(path)!=cell['prediction_sha256']: raise ValueError('prediction path/hash mismatch')
        pred=json.loads(path.read_text())
        if pred['document_ids']!=[r['id'] for r in docs] or len(pred['predictions'])!=len(docs): raise ValueError('document mismatch')
        rescored=[{'document_id':r['id'],**reconstruction_metrics(r['tokens'],x,r.get('annotations',[]))} for r,x in zip(docs,pred['predictions'])]
        if rescored!=cell['metrics']: raise ValueError('metric mismatch')
        seen[key]=cell
    if set(seen)!=expected: raise ValueError('missing cells: analysis cannot drop failures')
    if p.get('execution',{}).get('reuse_identical_activation_modes'):
        from experiments.validate_reconstruction import reuse_activation_cell
        for key,cell in seen.items():
            if key[4]=='inference':
                origin=seen[key[:4]+('training',key[5])]
                if cell!=reuse_activation_cell(origin): raise ValueError('invalid shared execution reference')
            elif cell.get('measurement_reused') is not False or cell.get('execution_id')!=cell.get('cell_id'):
                raise ValueError('training cell must contain original measurement')
        if report.get('physical_execution_count')!=sum(not c['measurement_reused'] for c in seen.values()) or report.get('logical_cell_count')!=len(seen):
            raise ValueError('execution count mismatch')
    # Mechanism controls must spend exactly the same prefix evaluation work.
    for s,d,k,a in itertools.product(p['victim_seeds'],p['depths'],p['knowledge'],p['attack_seeds']):
        for count in p['candidate_counts']:
            reference=seen[s,d,k,a,'training',f'forward_k{count}']['work']
            for condition in p['gradient_conditions']+['random']:
                actual=seen[s,d,k,a,'training',f'joint_{condition}_k{count}']['work']
                for field in ('head_forward_calls','head_forward_sequences','head_forward_token_positions'):
                    if actual[field]!=reference[field]: raise ValueError('matched prefix work violated')
    return report,seen


def plot_compute(cells,p,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for mode in ('training','inference'):
        fig,axes=plt.subplots(len(p['depths']),len(p['knowledge']),squeeze=False,figsize=(10,4*len(p['depths'])))
        for i,d in enumerate(p['depths']):
            for j,k in enumerate(p['knowledge']):
                ax=axes[i,j]
                families=['forward','sipit']+(['joint_true','joint_shuffled','joint_random'] if mode=='training' else [])
                for family in families:
                    methods=([f'sipit_steps{n}' for n in p['sipit_steps']] if family=='sipit' else [f'{family}_k{n}' for n in p['candidate_counts']])
                    x,y=[],[]
                    for method in methods:
                        selected=[v for key,v in cells.items() if key[1]==d and key[2]==k and key[4]==mode and key[5]==method]
                        x.append(np.mean([v['work']['wall_seconds']/len(v['metrics']) for v in selected]))
                        y.append(100*np.mean([m['token_accuracy'] for v in selected for m in v['metrics']]))
                    ax.plot(x,y,'o-',label=family)
                ax.set(title=f'{mode}, depth {d}, {k} head',xlabel='Online seconds per document (batch amortized)',ylabel='Token recovery (%)')
                ax.set_xscale('log'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('Descriptive budget curves — '+('synthetic engineering smoke' if len(p['victim_seeds'])<10 else 'controlled validation'))
        fig.tight_layout()
        for ext in ('png','pdf','svg'): fig.savefig(output/f'compute-{mode}.{ext}',dpi=160)
        plt.close(fig)


def analyze(root,output,expected_manifest_sha256=None,allowed_protocol_versions=(1,2)):
    import numpy as np
    report,cells=validate_report(root,expected_manifest_sha256,allowed_protocol_versions); p=report['protocol']; main,secondary=contrasts(p)
    if output.exists(): raise ValueError('analysis output must be fresh')
    output.mkdir(parents=True)
    results=[]
    for key in [main]+secondary:
        d,k,m,left,right=key
        matrix=[]
        for s in p['victim_seeds']:
            delta=[]
            for a in p['attack_seeds']:
                l=cells[s,d,k,a,m,left]['metrics']; r=cells[s,d,k,a,m,right]['metrics']
                delta.append([x['token_accuracy']-y['token_accuracy'] for x,y in zip(l,r)])
            matrix.append(np.mean(delta,axis=0))
        primary=key==main
        interval=crossed_interval(matrix,alpha=p['statistics']['alpha']/(1 if primary else len(secondary)),
                                  replicates=p['statistics']['bootstrap_replicates'],seed=p['statistics']['bootstrap_seed'])
        results.append({'depth':d,'knowledge':k,'mode':m,'left':left,'right':right,'primary':primary,**interval,
                        'exceeds_practical_margin':interval['lower']>p['primary']['practical_margin'] if primary else None})
    # Long-form per-document metrics and compute are suitable for budget curves.
    rows=[]
    for key,cell in sorted(cells.items()):
        metrics=cell['metrics']
        for row in metrics:
            flat={f:cell[f] for f in ('victim_seed','depth','knowledge','attack_seed','mode','arm')}
            flat.update({f:row[f] for f in ('document_id','token_accuracy','sequence_exact','longest_exact_run')})
            for family in ('contiguous_spans','annotated_spans'):
                for category,score in row[family].items():
                    for field,value in score.items(): flat[f'{family}_{category}_{field}']=value
            flat.update({'execution_id':cell.get('execution_id'),'measurement_reused':cell.get('measurement_reused',False)})
            flat.update({'online_batch_seconds':cell['work']['wall_seconds'],'cold_seconds':cell['cold_seconds'],
                         'amortized_seconds_per_document':cell['amortized_seconds_per_document']})
            rows.append(flat)
    with (output/'metrics_compute.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output/'contrasts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    plot_compute(cells,p,output)
    result={'evidence_tier':report['evidence_tier'],'primary':results[0],'secondary_family_size':len(secondary),
            'source_manifest_sha256':file_sha256(root/'validation.json.evidence.json'),
            'logical_cells':len(cells),'physical_executions':report.get('physical_execution_count',len(cells)),
            'limitations':['Percentile crossed bootstrap is approximate, especially with few independent runs.',
              'Only token-accuracy contrasts are inferential. Span, sequence, noise, precision, knowledge and cost curves are descriptive.',
              'A lower bound above zero supports improvement; above 0.02 supports the prespecified practical margin. No equivalence conclusion from crossing zero.',
              'Fixed sample is not a power guarantee; bootstrap Monte Carlo tail resolution is reported.',
              'Controlled post-training loss-gradient probes do not reproduce an evolving training trajectory.']}
    artifact=output/'analysis.json'
    with EvidenceLogger(artifact):
        result['outputs']={path.name:file_sha256(path) for path in output.iterdir() if path.suffix in ('.csv','.png','.pdf','.svg')}
        artifact.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expected-manifest-sha256')
    ap.add_argument('--protocol-version',type=int,action='append',default=None,
                    help='allowed protocol version (repeatable); default accepts frozen v1 and v2')
    args=ap.parse_args()
    versions=tuple(args.protocol_version) if args.protocol_version else (1,2)
    print(json.dumps(analyze(args.root,args.output,args.expected_manifest_sha256,versions),indent=2))
