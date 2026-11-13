"""Bounded clipping replay of previously inspected, externally anchored pilot captures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger,file_sha256,verify_evidence
from attacker.experiment import verify_run
from experiments.compare_baselines import load_observations


def run(bundle,output,depth):
    import torch
    from attacker.attacks import gradient_hardening as gh
    from attacker.attacks.seam_inversion import majority_control_token
    if output.exists():raise ValueError('fresh output required')
    p=json.loads(Path('experiments/baseline_protocol.json').read_text())
    anchor=p['source_manifest_sha256'][str(depth)]
    ok,problems=verify_run(bundle,anchor)
    if not ok:raise ValueError(str(problems))
    source=json.loads((bundle/'experiment.json').read_text())
    if int(source['config']['split'])!=depth:raise ValueError('source depth mismatch')
    ids=p['selected_document_ids']['training']
    h,g,indices,truth_path=load_observations(bundle,11,'none','training',ids,source['partitions'])
    output.mkdir(parents=True)
    torch.save({'g':g,'h':h},output/'observations.pt')
    # Evaluator preparation is separate from attack API; truth is never supplied
    # to harden_gradient or decode and is scored only after prediction writes.
    truth=torch.load(truth_path,map_location='cpu',weights_only=True)[indices]
    torch.save(truth,output/'evaluation.pt')
    head=torch.load(bundle/'public-head-11.pt',map_location='cpu',weights_only=True)
    torch.save(head['state_dict']['transformer.wte.weight'],output/'embedding.pt')
    calibration=torch.load(bundle/'calibration.pt',map_location='cpu',weights_only=True)
    first=majority_control_token(calibration.tolist())
    args=gh.build_parser().parse_args(['--observations-pt',str(output/'observations.pt'),
       '--embedding-pt',str(output/'embedding.pt'),'--canonical-pt',str(output/'evaluation.pt'),
       '--first-token',str(first),'--seeds','17','--clip','1','.1','.001','.00001','--output',str(output/'gradient.json')])
    args.mode='training'
    artifact=output/'replay.json'
    with EvidenceLogger(artifact):
        gh.run(args)
        ok,problems=verify_evidence(str(output/'gradient.json.evidence.json'))
        if not ok:raise ValueError(str(problems))
        report=json.loads((output/'gradient.json').read_text())
        summary=[]
        for row in report['results']:
            metrics=row.get('metrics',{}).get('backward',[])
            summary.append({'condition':row['condition'],'status':row['sanitizer']['status'],
                'clipped_units':row['sanitizer'].get('clipped_units'),
                'token_accuracy':sum(m['correct_tokens'] for m in metrics)/sum(m['token_count'] for m in metrics) if metrics else None})
        result={'status':'COMPLETE','source_manifest_sha256':anchor,'depth':depth,'victim_seed':11,'document_ids':ids,
                'canonical_indices':indices,'summary':summary,'gradient_manifest_sha256':file_sha256(output/'gradient.json.evidence.json'),
                'limits':['Eight previously inspected GPT2/WikiText2 training documents, one victim run; exploratory replay.',
                          'Clipping granularity is per token; exact deployed sanitizer not reproduced.',
                          'No newly frozen evaluation data accessed.']}
        artifact.write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--depth',type=int,choices=[3,6],required=True)
    args=ap.parse_args();run(args.bundle,args.output,args.depth)
