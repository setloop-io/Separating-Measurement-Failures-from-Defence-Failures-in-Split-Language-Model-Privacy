#!/usr/bin/env python3
"""Recompute Paper B descriptive tables from collected, evidence-sealed reports.

No canary, running node, old missing-head result or historical co-trained
observation is admitted. Binary regeneration is a separate operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from attacker.evidence import verify_evidence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise(root, whitebox):
    anchors={}; nodes={}; domains=[]; agents={}; synthetic=[]
    for node in ('soula','voula','koula'):
        d=root/node
        status=json.loads((d/'status.json').read_text())
        if status['status']!='complete' or status['failed']:
            raise ValueError('Incomplete node: '+node)
        journals=list(d.rglob('*.evidence.json'))
        for p in journals:
            ok, errors=verify_evidence(str(p))
            if not ok: raise ValueError(str(errors))
        nodes[node]={'status':'complete','verified_journals':len(journals)}
        for p in sorted(d.glob('*/experiment.json')):
            if p.parent.name=='canary':continue
            x=json.loads(p.read_text());anchors[str(p.relative_to(root))]=sha(p)
            if x['status']!='COMPLETE':raise ValueError('Incomplete experiment')
            for row in x['summary']:
                # Independently recompute each displayed mean from per-seed records.
                records=[q for q in x['results'] if all(q[k]==row[k] for k in ('mode','defense','arm'))]
                if len(records)!=3 or {q['seed'] for q in records}!={11,22,33}:
                    raise ValueError('Expected three-seed matrix')
                accuracy=statistics.mean(q['accuracy'] for q in records)
                if abs(accuracy-row['accuracy_mean'])>1e-12:raise ValueError('Mean mismatch')
                domains.append({'experiment':p.parent.name,'node':node,**{k:row[k] for k in ('mode','defense','arm')},
                                'accuracy_percent':100*accuracy,'raw_excess_pp':row['attack_minus_control_pp']['mean'],
                                'scope':'Descriptive, fixed corpus; not a gate re-issuance'})
        if node=='koula':
            for name in ('agent-goal-hijack','cross-agent-deputy'):
                p=d/(name+'.json');x=json.loads(p.read_text());anchors[str(p.relative_to(root))]=sha(p)
                agents[name]={'config':x['config'],'cells':[{'arm':q['arm'],'world':q['world'],
                    'status':'UNMEASURED' if q.get('error') else 'MEASURED',
                    'trials':q.get('trials'),'total_violations':q.get('total_violations'),
                    'violation_categories':q.get('violations'),'false_blocks':q.get('false_blocks')}
                    for q in x['results']]}
        if node=='soula':
            for p in sorted(d.glob('bundle-*-latent-matching.json')):
                x=json.loads(p.read_text());anchors[str(p.relative_to(root))]=sha(p)
                synthetic.append({'file':p.name,**x['summary'][0]})
    w=json.loads((whitebox/'full-summary.json').read_text())
    m=json.loads((whitebox/'full-manifest.json').read_text())
    ready=json.loads((whitebox/'PACKAGE-READY.json').read_text())
    if sha(whitebox/'full-manifest.json')!=ready['manifest_sha256']:raise ValueError('Package manifest changed')
    if sha(whitebox/'full-summary.json')!=m['files']['summary.json']['sha256']:raise ValueError('Summary changed')
    if len(w['stable_cells'])!=198 or len(w['cotrain_final_inference'])!=99:raise ValueError('Incomplete white-box grid')
    table=[]
    for depth in range(1,12):
        row={'depth':depth}
        for method in ('pca-compress','tb-frozen'):
            cells=[q for q in w['stable_cells'] if q['depth']==depth and q['method']==method]
            if {(q['seed'],q['dim']) for q in cells}!={(seed,dim) for seed in (11,22,33) for dim in (64,128,256)}:raise ValueError('Cell matrix mismatch')
            row[method]=100*statistics.mean(q['accuracy'] for q in cells)
        cells=[q for q in w['cotrain_final_inference'] if q['depth']==depth]
        if len(cells)!=9:raise ValueError('Co-trained matrix incomplete')
        row['cotrain_final_inference']=100*statistics.mean(q['accuracy'] for q in cells)
        table.append(row)
    return {'scope':'Paper B descriptive follow-ups; no historical external claims or universal guarantees',
        'nodes':nodes,'report_sha256':anchors,'whitebox_summary_sha256':sha(whitebox/'full-summary.json'),
        'whitebox_depth_means_percent':table,'domain_and_adaptive_rows':domains,
        'agentic':agents,'synthetic_order':synthetic,'excluded':['canaries','sasa incomplete 500-step extension',
        '18 historical co-trained observations','old missing-head results','external v13/PR-101 and August campaign']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase2',type=Path,required=True);ap.add_argument('--whitebox',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.write_text(json.dumps(summarise(a.phase2,a.whitebox),indent=2)+'\n')

if __name__=='__main__':main()
