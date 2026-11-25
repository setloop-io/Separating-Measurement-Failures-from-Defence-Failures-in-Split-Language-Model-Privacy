#!/usr/bin/env python3
"""Complete-matrix P6 follow-up analysis; seed is the replication unit.

Consumes only verified cells bound to one frozen protocol. Missing data,
missing controls, duplicate attempts and canary files block inference.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.run_p6_corrected import digest, verify_cell, save


def partial_r2(values):
    """Seed x span x start. Full categorical cells vs span-only fit."""
    a=np.asarray(values,dtype=float)
    if a.ndim!=3 or not np.isfinite(a).all():raise ValueError('finite seed x span x start required')
    reduced=a.mean(axis=(0,2),keepdims=True)
    full=a.mean(axis=0,keepdims=True)
    sse0=float(((a-reduced)**2).sum());sse1=float(((a-full)**2).sum())
    return max(0.,(sse0-sse1)/sse0) if sse0>1e-24 else 0.


def variance_test(values, replicates, seed, alpha):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    actual=partial_r2(a);boot=np.empty(replicates);null=np.empty(replicates)
    for i in range(replicates):
        boot[i]=partial_r2(a[rng.integers(len(a),size=len(a))])
        # Randomise start labels within each paired seed/span; preserve spans.
        perm=np.argsort(rng.random(a.shape),axis=2)
        null[i]=partial_r2(np.take_along_axis(a,perm,axis=2))
    lo,hi=np.quantile(boot,[alpha/2,1-alpha/2])
    p=(1+int((null>=actual-1e-12).sum()))/(replicates+1)
    return {'partial_r2':actual,'seed_bootstrap_interval':[float(lo),float(hi)],
            'permutation_p':p,'alpha_adjusted':alpha,
            'start_matters':bool(actual>=.10 and p<alpha),
            'interpretation':'Start plus span-by-start interaction, conditional on span; not a P5 gate.'}


def prediction_comparison(values, seeds):
    a=np.asarray(values,dtype=float)
    fit=[i for i,s in enumerate(seeds) if s in [11,22,33,44,55,66]]
    test=[i for i,s in enumerate(seeds) if s in [77,88]]
    if len(fit)!=6 or len(test)!=2:raise ValueError('Frozen 6/2 split is incomplete')
    span=np.repeat([2.,4.,6.],4);start=np.tile([0.,1.,2.,3.],3)
    errors=[]
    for include_start in [False,True]:
        x=np.column_stack([np.ones(12),span]+([start] if include_start else []))
        train_x=np.tile(x,(6,1));y=a[fit].reshape(-1)
        penalty=np.eye(x.shape[1])*1e-3;penalty[0,0]=0.
        coef=np.linalg.solve(train_x.T@train_x+penalty,train_x.T@y)
        errors.append(float(np.sqrt(np.mean((a[test].reshape(2,12)-x@coef)**2))))
    return {'span_only_rmse':errors[0],'span_plus_start_rmse':errors[1],
            'ten_percent_improvement':bool(errors[1]<=.9*errors[0] and errors[0]>0),
            'interpretation':'Prespecified held-out-seed descriptive comparison, no refitting or re-splitting.'}


def auc(norms, labels):
    from attacker.attacks.lira_membership import rank_auc
    present=[float(x) for x,l in zip(norms,labels) if l]
    absent=[float(x) for x,l in zip(norms,labels) if not l]
    if not present or not absent:raise ValueError('Property lacks both classes')
    return rank_auc(present,absent)


def metrics(root, tokens):
    measurement=json.loads((root/'measurements.json').read_text())
    rows=measurement['documents'];vmi=json.loads((root/'vmi.json').read_text())
    records=sorted(vmi['results'],key=lambda r:r['step'])
    if len(rows)!=256 or len(records)!=256:raise ValueError('Incomplete heldout documents')
    recovery=[];norms=[];labels=[]
    for i,(doc,record) in enumerate(zip(rows,records)):
        if doc['step']!=i or record['step']!=i or len(record['per_sequence'])!=1:raise ValueError('Document order drift')
        seq=record['per_sequence'][0]
        correct=sum(p==t for p,t in zip(seq['prediction'],seq['truth']))
        if correct!=seq['correct']:raise ValueError('Prediction rescore mismatch')
        recovery.append((correct-seq['control_correct'])/seq['n_tokens'])
        norms.append(doc['gradient_norm']);labels.append([t in seq['truth'] for t in tokens])
    recovery=np.asarray(recovery);norms=np.asarray(norms);labels=np.asarray(labels)
    au=[auc(norms,labels[:,i]) for i in range(len(tokens))]
    # A named normalised leakage index, never the external P5 privacy gate.
    worst=max(0.,float(recovery.mean()),*[2*abs(x-.5) for x in au])
    return {'losses':np.asarray([r['loss'] for r in rows]),'vmi_excess':recovery,
            'gradient_norms':norms,'property_labels':labels,'auc':au,'worst':worst,
            'document_ids':[r['document_id'] for r in rows]}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,action='append',required=True)
    ap.add_argument('--protocol',type=Path,default=ROOT/'experiments/crossed_span_protocol.p6.v2.json')
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Analysis output must be fresh')
    p=json.loads(args.protocol.read_text());pin=digest(args.protocol)
    reports={};controls={};anchors={}
    for directory in args.root:
        marker=json.loads((directory/'grid-complete.json').read_text())
        if marker.get('canary') or marker.get('protocol_sha256')!=pin:raise ValueError('Canary/protocol mismatch')
        for path in directory.glob('*/cell.json'):
            r=verify_cell(path.parent,pin);key=(r['span'],r['start'],r['defense'],r['seed'])
            target=controls if r['kind']=='all_local' else reports
            key=r['seed'] if r['kind']=='all_local' else key
            if key in target:raise ValueError('Duplicate cell/attempt')
            target[key]=(path.parent,r);anchors[str(path)]=digest(path)
    expected={(s,b,d,v) for s in p['spans'] for b in p['starts'] for d in p['defenses'] for v in p['victim_seeds']}
    if set(reports)!=expected or set(controls)!=set(p['victim_seeds']):raise ValueError('Complete grid plus all controls required')
    control_metrics={v:metrics(root,p['property_tokens']) for v,(root,r) in controls.items()}
    output={'protocol_sha256':pin,'evidence_anchors':anchors,'defenses':{},'scope':p['scope']}
    details=[];seeds=p['victim_seeds'];rep=p['statistics']['bootstrap_replicates'];alpha=.05/4
    for defense in p['defenses']:
        utility=np.empty((8,3,4));privacy=np.empty_like(utility)
        per_cell={}
        for i,v in enumerate(seeds):
            for j,span in enumerate(p['spans']):
                for k,start in enumerate(p['starts']):
                    root,r=reports[(span,start,defense,v)];m=metrics(root,p['property_tokens']);c=control_metrics[v]
                    if m['document_ids']!=c['document_ids']:raise ValueError('Unpaired utility documents')
                    utility[i,j,k]=float((m['losses']-c['losses']).mean());privacy[i,j,k]=m['worst']
                    per_cell[(i,j,k)]=m
                    details.append({'span':span,'start':start,'defense':defense,'seed':v,
                                    'delta_utility':utility[i,j,k],'normalised_leakage_index':privacy[i,j,k],
                                    'vmi_excess':float(m['vmi_excess'].mean()),'property_auc':m['auc']})
        summary={}
        for name,a in [('delta_utility',utility),('normalised_leakage_index',privacy)]:
            summary[name]={'variance':variance_test(a,rep,p['statistics']['bootstrap_seed'],alpha),
                           'heldout_prediction':prediction_comparison(a,seeds)}
        # Document-clustered paired utility/VMI intervals, crossed with victim seeds.
        rng=np.random.default_rng(p['statistics']['cluster_seed']);cluster=[]
        for j,span in enumerate(p['spans']):
            for k,start in enumerate(p['starts']):
                losses=np.array([per_cell[i,j,k]['losses']-control_metrics[v]['losses'] for i,v in enumerate(seeds)])
                vmi=np.array([per_cell[i,j,k]['vmi_excess'] for i in range(8)])
                draws=np.empty((rep,2))
                for b in range(rep):
                    vs=rng.integers(8,size=8);ds=rng.integers(256,size=256)
                    draws[b]=[losses[vs[:,None],ds].mean(),vmi[vs[:,None],ds].mean()]
                ci=np.quantile(draws,[.025,.975],axis=0)
                cluster.append({'span':span,'start':start,'utility_interval':ci[:,0].tolist(),
                                'vmi_excess_interval':ci[:,1].tolist(),
                                'interpretation':'Descriptive crossed victim/document intervals; no token-iid inference or P5 PASS claim.'})
        summary['clustered_checks']=cluster;output['defenses'][defense]=summary
    output['analytic_delegation']={'span2':2/p['num_hidden_layers'],'span4':4/p['num_hidden_layers'],'span6':6/p['num_hidden_layers'],'start_effect':0,
                                   'inference':'none: start-independent by construction'}
    output['cells']=details
    save(args.output,output);print(args.output,flush=True)


if __name__=='__main__':main()
