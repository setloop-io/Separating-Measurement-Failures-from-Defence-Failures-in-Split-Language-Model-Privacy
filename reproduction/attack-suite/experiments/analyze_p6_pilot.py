#!/usr/bin/env python3
"""Post-hoc descriptive analysis of the immutable 8bb1452 pilot grid."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.analyze_p6_corrected import variance_test,prediction_comparison
from attacker.evidence import verify_evidence


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT/'results')
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Fresh analysis output required')
    seeds=[11,22,33,44,55,66,77,88];spans=[2,4,6];out={'scope':'Post-hoc descriptive pilot analysis. Not frozen P6 confirmation; utility controls, heldout partitions and property attack absent.',
        'driver_commit':'8bb145235f12e4238ffa6037234e46c520c8efe8','defenses':{},'cell_summaries':[],'source_hashes':{}}
    bydef={d:np.empty((8,3,4)) for d in ['D0','D1']};groups={}
    for span in spans:
        root=args.root/f'p6-grid-s{span}-v1'
        index=json.loads((root/'run-index.json').read_text());cells=[c for c in index['cells'] if c['kind']=='cell']
        if len(cells)!=64 or len({c['tag'] for c in cells})!=64 or any(c['rc']!=0 for c in cells):raise ValueError('Complete original grid required')
        for c in cells:
            path=root/'artifacts'/f"seam-inversion_vmi_{c['tag']}.json"
            ok,problems=verify_evidence(str(path)+'.evidence.json')
            if not ok:raise ValueError(problems)
            out['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            artifact=json.loads(path.read_text());clusters={}
            for record in artifact['results']:
                for seq in record['per_sequence']:
                    if sum(a==b for a,b in zip(seq['prediction'],seq['truth']))!=seq['correct']:raise ValueError('Prediction rescore mismatch')
                    key=hashlib.sha256(json.dumps(seq['truth']).encode()).hexdigest()
                    r=clusters.setdefault(key,[0,0]);r[0]+=seq['correct']-seq['control_correct'];r[1]+=seq['n_tokens']
            vals=np.asarray(list(clusters.values()),dtype=float);excess=float(vals[:,0].sum()/vals[:,1].sum())
            bydef[c['defense']][seeds.index(c['seed']),spans.index(span),c['start']]=excess
            groups.setdefault((span,c['start'],c['defense']),[]).append(vals)
    for defense,a in bydef.items():
        out['defenses'][defense]={'vmi_excess_mean':float(a.mean()),
          'conditional_start_analysis':variance_test(a,10000,41777,.05/2),
          'heldout_seed_comparison':prediction_comparison(a,seeds)}
    rng=np.random.default_rng(20260917)
    for (span,start,defense),seed_clusters in sorted(groups.items()):
        draws=[]
        for _ in range(2000):
            numerator=denominator=0.
            for seed in rng.integers(8,size=8):
                a=seed_clusters[seed];sample=a[rng.integers(len(a),size=len(a))]
                numerator+=sample[:,0].sum();denominator+=sample[:,1].sum()
            draws.append(numerator/denominator)
        a=bydef[defense][:,spans.index(span),start]
        out['cell_summaries'].append({'span':span,'start':start,'defense':defense,'mean_vmi_excess':float(a.mean()),
          'hierarchical_seed_token_window_interval':np.quantile(draws,[.025,.975]).tolist(),
          'limitation':'Truth-token-identical windows grouped; source article IDs unavailable, so this is not an article-clustered CI.'})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['defenses'],indent=2))


if __name__=='__main__':main()
