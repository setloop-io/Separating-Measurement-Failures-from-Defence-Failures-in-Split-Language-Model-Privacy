"""Label-isolated paired prediction worker and trained-checkpoint mechanical gate."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import (sha256,gru_class,granted_split,validate_observations,
                                    numerical_check,write_predictions)
from attacker.bisr_upstream import adapter,reference_gru


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--engine',choices=['gate','candidate','reference'],required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args();start=time.monotonic()
    manifest=json.loads(args.manifest.read_text())
    keys={'schema','public_model','inverter','observations','head','boundary','knowledge','sources','source_sha256','documents'}
    if set(manifest)!=keys or manifest['schema']!='bisr.campaign_prediction.v1':
        raise ValueError('Unknown manifest fields; truth is not accepted')
    root=Path(__file__).resolve().parents[1]
    for name,digest in manifest['source_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Implementation changed')
    def pinned(entry):
        if set(entry)!={'path','sha256'}:raise ValueError('Unpinned input')
        path=args.manifest.parent/entry['path']
        if sha256(path)!=entry['sha256']:raise ValueError('Input hash mismatch')
        return path
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    from attacker.attacks.bisr_baseline import _bre_gradient_match,_bre_smashed_match
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def sync():
        if args.device=='cuda':torch.cuda.synchronize()
    blob=torch.load(pinned(manifest['public_model']),map_location='cpu',weights_only=True)
    config=GPT2Config(**blob['config']);config._attn_implementation='eager'
    public=GPT2LMHeadModel(config);public.load_state_dict(blob['state_dict'],strict=True)
    public=public.float().to(args.device).eval().requires_grad_(False)
    head=manifest['head']
    split=granted_split(public,manifest['boundary'],knowledge=manifest['knowledge'],
        head_path=pinned(head) if head else None,head_sha256=head['sha256'] if head else None)
    checkpoint=torch.load(pinned(manifest['inverter']),map_location='cpu',weights_only=True)
    if checkpoint['format']!='llm_attacker.bisr_gru.v1':raise ValueError('GRU required')
    arch=checkpoint['arch']
    if arch['hidden_size']!=256 or arch['dropout']!=.1:raise ValueError('Wrong GRU architecture')
    inv=gru_class()(**arch).to(args.device).eval().requires_grad_(False)
    inv.load_state_dict(checkpoint['state_dict'],strict=True)
    sources={k:pinned(v).read_text() for k,v in manifest['sources'].items()}
    if set(sources)!={'dlg','sma','model','sip'}:raise ValueError('Incomplete reference sources')
    ref_inv=reference_gru(sources,arch,checkpoint['state_dict'],args.device)
    gm,sm=adapter(split,sources)
    observations=torch.load(pinned(manifest['observations']),map_location='cpu',weights_only=True)
    validate_observations(observations)
    if len(observations['sample_ids'])!=manifest['documents']:raise ValueError('Incomplete observations')
    args.output.mkdir(parents=True,exist_ok=False)
    sync();cold_seconds=time.monotonic()-start
    def candidate_gm(h,g,scores,iters=32):
        ids,info=_bre_gradient_match(split,h,g,scores,iters=iters,lr=.09,beta=.85,tau=1.2,opt_batch=1,upstream=True)
        if not math.isfinite(info['gm_final_loss_mean']):raise ValueError('Nonfinite GM')
        return ids
    def candidate_sm(h,initial,iters=800):
        ids,info=_bre_smashed_match(split,split.wte.weight,h,initial,iters=iters,lr=.01,wd=.01,opt_batch=1,upstream=True)
        if not math.isfinite(info['sm_final_loss_mean']):raise ValueError('Nonfinite SM')
        return ids
    def decoded(scores):
        if not torch.isfinite(scores).all():raise ValueError('Nonfinite attack logits')
        return scores.argmax(-1)
    if args.engine=='gate':
        # One-step checks on fixed first/last observations, before any full-budget
        # recovery is generated or scored. Canonical tokens are never loaded.
        checks=[]
        for idx in sorted({0,len(observations['sample_ids'])-1}):
            h=observations['h'][idx:idx+1].to(args.device)
            g=observations['g'][idx:idx+1].to(args.device)*float(observations['gradient_batch_scale'][idx])
            with torch.no_grad():scores=inv(h);rs=ref_inv(h);numerical_check(scores,rs)
            old_opt=torch.optim.AdamW;old_backward=torch.Tensor.backward;traces=[];losses=[]
            class Trace(old_opt):
                def step(self,closure=None):
                    p=self.param_groups[0]['params'][0]
                    row={'before':p.detach().cpu().clone(),'gradient':p.grad.detach().cpu().clone(),
                         'loss':losses[-1]}
                    out=super().step(closure);row['after']=p.detach().cpu().clone();traces.append(row);return out
            def backward(t,*a,**kw):
                losses.append(t.detach().cpu().clone());return old_backward(t,*a,**kw)
            torch.optim.AdamW=Trace;torch.Tensor.backward=backward
            try:
                gm(h,g,rs,iters=1);candidate_gm(h,g,scores,iters=1)
                sm(h,rs,iters=1);candidate_sm(h,scores.argmax(-1),iters=1)
            finally:torch.optim.AdamW=old_opt;torch.Tensor.backward=old_backward
            for a,b in ((traces[0],traces[1]),(traces[2],traces[3])):
                for field in ('before','gradient','after','loss'):numerical_check(a[field],b[field])
            path=args.output/f'traces-{idx}.pt'
            torch.save({'h':h.cpu(),'g':g.cpu(),'sip':scores.cpu(),'upstream_sip':rs.cpu(),'traces':traces},path)
            checks.append({'index':idx,'sha256':sha256(path),'passed':True})
        write_predictions(args.output/'result.json',{'status':'TRAINED_MECHANICAL_PASS','checks':checks,
            'manifest_sha256':sha256(args.manifest),'heldout_performance_scored':False})
        return
    # Observed module-level counts include higher-order reverse visits. They are
    # not FLOPs or equal-compute claims; stage budgets are also recorded explicitly.
    counters={};handles=[]
    def hook(name):
        def forward(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            tokens=int(value.shape[0]*value.shape[1])
            counters[name+'_forward_token_positions']=counters.get(name+'_forward_token_positions',0)+tokens
            if value.requires_grad:
                def backward(grad):
                    key=name+'_output_reverse_token_positions';counters[key]=counters.get(key,0)+tokens
                value.register_hook(backward)
        return forward
    for i,block in enumerate(split.head.model.transformer.h[:manifest['boundary']]):
        handles.append(block.register_forward_hook(hook('client_block_'+str(i))))
    for i,block in enumerate(split.public.model.transformer.h[manifest['boundary']:],manifest['boundary']):
        handles.append(block.register_forward_hook(hook('downstream_block_'+str(i))))
    previous='0'*64;total=0
    import hashlib
    for idx,sample_id in enumerate(observations['sample_ids']):
        h=observations['h'][idx:idx+1].to(args.device)
        g=observations['g'][idx:idx+1].to(args.device)*float(observations['gradient_batch_scale'][idx])
        stages={}
        def measure(name,fn):
            counters.clear()
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            sync();t=time.monotonic();value=fn();sync()
            stages[name]={'wall_seconds':time.monotonic()-t,'layer_work':dict(counters),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None}
            return value
        with torch.no_grad():scores=measure('sip',lambda:(inv if args.engine=='candidate' else ref_inv)(h))
        sip=decoded(scores)
        if args.engine=='candidate':
            b=measure('gm',lambda:candidate_gm(h,g,scores))
            bf=measure('sm_bf',lambda:candidate_sm(h,b))
        else:
            b_scores=measure('gm',lambda:gm(h,g,scores));b=decoded(b_scores)
            bf=measure('sm_bf',lambda:decoded(sm(h,b_scores.detach())))
        record={'schema':'bisr.predictions.v1','sample_ids':[sample_id],
            'arms':{k:v.detach().cpu().tolist() for k,v in {'sip':sip,'b':b,'bf':bf}.items()},
            'engine':args.engine,'algorithm_profile':'splitllm-c376590','grant':split.grant,
            'input_manifest_sha256':sha256(args.manifest),'work':stages,
            'budgets':{'gm':32,'sm_bf':800},'bf_reuses_b_stage':True}
        digest=write_predictions(args.output/f'{idx:05d}.json',record)
        event={'index':idx,'sample_id':sample_id,'prediction_sha256':digest,'previous':previous}
        previous=hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest();event['hash']=previous
        with (args.output/'journal.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        total+=sum(x['wall_seconds'] for x in stages.values())
        print(json.dumps({'index':idx,'documents':manifest['documents'],'engine':args.engine,'online_seconds':total}),flush=True)
    write_predictions(args.output/'result.json',{'status':'PREDICTIONS_COMPLETE','documents':manifest['documents'],
        'manifest_sha256':sha256(args.manifest),'journal_tip':previous,'cold_load_seconds':cold_seconds,
        'online_seconds':total,'scoring_performed':False})


if __name__=='__main__':main()
