"""Post-hoc diagnostic: oracle ceiling imposed by the frozen joint candidate set.

This reads sealed captures and scoring truth; it does not tune or rerun attacks.
Position zero is generously assumed correct, giving an upper bound independent
of the forward candidate list or beam ranking.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from attacker import captures
    from attacker.attacks.gradient_inversion import _tag_candidates_torch
    from attacker.evidence import file_sha256
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    raw=json.loads((args.bundle/'experiment.json').read_text())
    records=[]
    for cell in raw['results']:
        if cell['mode']!='training' or cell['arm']!='joint':continue
        seed,defense=cell['seed'],cell['defense']
        head=torch.load(args.bundle/f'public-head-{seed}.pt',map_location='cpu',weights_only=True)
        embedding=head['state_dict']['transformer.wte.weight'].float().cuda()
        cap=args.bundle/f'seed-{seed}'/defense/'training'
        truth=torch.load(cap/'canonical.pt',map_location='cpu',weights_only=True)
        records_bwd=[(m,p) for m,p in captures.scan_captures(str(cap),'training') if m['phase']=='bwd']
        covered=0;positions=0;cursor=0;sequences=[]
        with torch.no_grad():
            for meta,path in records_bwd:
                grad=captures.load_tensor(path).float().cuda()
                if grad.ndim==2:grad=grad.unsqueeze(0)
                labels=truth[cursor:cursor+grad.shape[0]].cuda()
                scores=_tag_candidates_torch(grad,embedding)
                k=raw['config']['joint_candidates']
                candidates=scores[:,:-1].topk(k,dim=-1).indices
                hit=(candidates==labels[:,1:,None]).any(-1)
                count=int(hit.sum());total=hit.numel()
                covered+=count;positions+=total
                sequences.append({'step':meta['step'],'included_labels_after_first_position':count,'positions_after_first':total})
                cursor+=grad.shape[0]
        ceiling=(covered+cursor)/(positions+cursor)
        if cell['accuracy']>ceiling+1e-10:raise AssertionError('observed joint result exceeds its candidate oracle ceiling')
        records.append({'seed':seed,'defense':defense,'depth':int(raw['config']['split']),
                        'backward_topk':k,'candidate_recall_after_first':covered/positions,
                        'oracle_sequence_accuracy_ceiling':ceiling,'actual_joint_accuracy':cell['accuracy'],
                        'forward_accuracy':cell['attack_summary'][0]['fwd_linearization_seq_accuracy_mean'],
                        'n_sequences':cursor,'n_tokens':positions+cursor,'sequences':sequences})
    result={'schema':'llm_attacker.joint_candidate_diagnostic.v1','analysis_type':'post_hoc_descriptive',
            'source_experiment_sha256':file_sha256(args.bundle/'experiment.json'),
            'script_sha256':file_sha256(__file__),'results':records,
            'interpretation':'Ceiling from correct-token inclusion in backward-only top-k candidate sets at positions 1..L-1, granting perfect position zero. This bounds this implementation, not what joint observations can reveal.'}
    args.output.write_text(json.dumps(result,indent=2));print(args.output)


if __name__=='__main__':main()
