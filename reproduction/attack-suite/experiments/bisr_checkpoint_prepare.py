"""Prepare shared C-A checkpoints and label-free captures; never attack heldout data.

The resource-canary mode trains only an excluded timing fixture. Preparation
uses the frozen auxiliary windows; heldout tokens are read only for capture/replay.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import sha256,gru_class,replay_wire
from experiments.bisr_upstream_preflight import save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['canary','prepare'])
    for n in ('model','inputs','output'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--deadline-seconds',type=float,default=108000)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import GPT2LMHeadModel
    from attacker.splitbench import build_split,SplitPlan
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda';started=time.monotonic()
    def check_time():
        if time.monotonic()-started>args.deadline_seconds:raise TimeoutError('Preparation budget exhausted')
    def event(kind,**kw):
        record={'event':kind,'elapsed_seconds':time.monotonic()-started,**kw}
        with (args.output/'journal.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    if manifest['schema']!='bisr.frozen_inputs.v3':raise ValueError('Fresh input freeze required')
    for name,digest in manifest['outputs'].items():
        if sha256(args.inputs/name)!=digest:raise ValueError('Input changed: '+name)
    aux=torch.tensor([r['token_ids'] for r in json.loads((args.inputs/'auxiliary-windows.json').read_text())])
    model=GPT2LMHeadModel.from_pretrained(args.model,local_files_only=True,attn_implementation='eager').float().cuda().eval()
    model.requires_grad_(False)
    training={'epochs':20,'batch_size':32,'lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0}
    arch={'vocab':model.config.vocab_size,'seam_dim':model.config.n_embd,'seq_len':32,'hidden_size':256,'dropout':.1}
    result={'schema':'bisr.checkpoint_preparation.v1','mode':args.mode,
            'input_manifest_sha256':sha256(args.inputs/'manifest.json'),
            'model_files':{p.name:sha256(p) for p in args.model.iterdir() if p.is_file()},
            'training':training,'auxiliary_windows':len(aux),'files':{},'launch_admitted':False}
    def persist(name,value):
        path=args.output/name
        if path.exists():raise FileExistsError(path)
        torch.save(value,path);result['files'][name]=sha256(path)
        return path
    try:
        if args.mode=='canary':
            torch.manual_seed(81274)
            split=build_split(model,SplitPlan.parse('6',n_layers=12))
            ids=aux[:32].cuda()
            with torch.no_grad():rows=split.head_forward(ids)
            inv=gru_class()(**arch).cuda().train()
            opt=torch.optim.AdamW(inv.parameters(),lr=.001,weight_decay=0,eps=1e-8)
            timings=[]
            for i in range(18):
                torch.cuda.synchronize();t=time.monotonic()
                loss=torch.nn.functional.cross_entropy(inv(rows).flatten(0,1),ids.flatten())
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                torch.cuda.synchronize()
                if i>=2:timings.append(time.monotonic()-t)
            seconds=sum(timings)/len(timings)
            result['seconds_per_gru_step']=seconds
            result['six_gru_projection_hours']=seconds*((len(aux)+31)//32)*20*6/3600
            result['six_gru_projection_1_5_hours']=result['six_gru_projection_hours']*1.5
            result['status']='TRAINING_TIMING_COMPLETE'
        else:
            persist('public.pt',{'config':model.config.to_dict(),
                                 'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}})
            aux_rows={}
            for depth in (3,6):
                split=build_split(model,SplitPlan.parse(str(depth),n_layers=12))
                rows=[];t=time.monotonic()
                with torch.no_grad():
                    for i in range(0,len(aux),32):
                        check_time();rows.append(split.head_forward(aux[i:i+32].cuda()).cpu())
                aux_rows[depth]=torch.cat(rows)
                persist(f'aux-rows-d{depth}.pt',aux_rows[depth])
                event('auxiliary_captured',depth=depth,seconds=time.monotonic()-t)
            for seed in (31415,27182,16180):
                for depth in (3,6):
                    torch.manual_seed(seed)
                    inv=gru_class()(**arch).cuda().train()
                    opt=torch.optim.AdamW(inv.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
                    generator=torch.Generator().manual_seed(seed)
                    for epoch in range(20):
                        order=torch.randperm(len(aux),generator=generator)
                        for start in range(0,len(aux),32):
                            check_time();indices=order[start:start+32]
                            scores=inv(aux_rows[depth][indices].cuda())
                            loss=torch.nn.functional.cross_entropy(scores.flatten(0,1),aux[indices].cuda().flatten())
                            if not torch.isfinite(loss):raise ValueError('Nonfinite SIP training loss')
                            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                        event('sip_epoch',seed=seed,depth=depth,epoch=epoch+1,loss=float(loss.detach()))
                    persist(f'sip-s{seed}-d{depth}.pt',{'format':'llm_attacker.bisr_gru.v1',
                        'arch':arch,'state_dict':{k:v.cpu() for k,v in inv.state_dict().items()},
                        'seed':seed,'training':training,'auxiliary_manifest_sha256':result['input_manifest_sha256']})
                    del inv,opt,scores;torch.cuda.empty_cache()
                # Explicit C-A victim recipe: shared across depths, fresh SGD order/dropout seed.
                torch.manual_seed(seed)
                victim=deepcopy(model).train().requires_grad_(True)
                opt=torch.optim.AdamW(victim.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
                generator=torch.Generator().manual_seed(seed)
                for step in range(128):
                    check_time();indices=torch.randint(len(aux),(8,),generator=generator)
                    ids=aux[indices].cuda()
                    logits=victim(input_ids=ids,use_cache=False).logits
                    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,arch['vocab']),ids[:,1:].reshape(-1))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite victim loss')
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    if (step+1)%16==0:event('victim_step',seed=seed,step=step+1)
                victim.eval().requires_grad_(False);del opt
                persist(f'victim-s{seed}.pt',{'config':victim.config.to_dict(),
                    'state_dict':{k:v.cpu() for k,v in victim.state_dict().items()},'seed':seed})
                truth=json.loads((args.inputs/'truth.json').read_text())
                for depth in (3,6):
                    split=build_split(victim,SplitPlan.parse(str(depth),n_layers=12))
                    head={k:v.cpu() for k,v in victim.state_dict().items() if k.startswith(
                        ('transformer.wte.','transformer.wpe.')+tuple(f'transformer.h.{i}.' for i in range(depth)))}
                    persist(f'head-s{seed}-d{depth}.pt',{'format':'llm_attacker.seam_head.v1',
                        'arch':'gpt2','boundary':depth,'config':victim.config.to_dict(),'state_dict':head})
                    hs=[];gs=[]
                    for token_ids in truth['token_ids']:
                        check_time();ids=torch.tensor([token_ids],device=device)
                        h=split.head_forward(ids).detach().requires_grad_(True)
                        logits=split.rest_forward(h)
                        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,arch['vocab']),ids[:,1:].reshape(-1))
                        g,=torch.autograd.grad(loss,h)
                        replay_wire(split,ids,h.detach(),g.detach())
                        hs.append(h.detach().cpu());gs.append(g.detach().cpu())
                    persist(f'observations-s{seed}-d{depth}.pt',{'schema':'bisr.observations.v1',
                        'sample_ids':truth['sample_ids'],'h':torch.cat(hs),'g':torch.cat(gs),
                        'mask':torch.ones(256,32,dtype=torch.long),'gradient_batch_scale':torch.ones(256)})
                    event('capture_replay_pass',seed=seed,depth=depth,documents=256)
                del victim,split;torch.cuda.empty_cache()
            result['status']='CHECKPOINTS_PREPARED_REPLAY_VERIFIED'
        result['wall_seconds']=time.monotonic()-started
        save(args.output/'result.json',result)
        event('complete',status=result['status'])
    except Exception as exc:
        result['status']='PREPARATION_FAILED';result['error']=repr(exc)
        save(args.output/'result.json',result);event('failed',error=repr(exc));raise


if __name__=='__main__':main()
