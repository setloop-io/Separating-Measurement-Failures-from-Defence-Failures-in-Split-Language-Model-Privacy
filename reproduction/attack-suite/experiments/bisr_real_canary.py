"""Real GPT-2 mechanical and full-budget resource canary; no recovery scores."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.bisr_upstream_preflight import source, save
from attacker.bisr_fidelity import sha256, granted_split, gru_class, numerical_check, replay_wire
from attacker.bisr_upstream import adapter, reference_gru


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','inputs','upstream','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import GPT2LMHeadModel
    from attacker.attacks.bisr_baseline import _bre_gradient_match,_bre_smashed_match
    torch.set_num_threads(1);torch.manual_seed(81273)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Resource canary requires GPU')
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    if sha256(args.inputs/'auxiliary-windows.json')!=manifest['outputs']['auxiliary-windows.json']:
        raise ValueError('Auxiliary input hash mismatch')
    ids=torch.tensor(json.loads((args.inputs/'auxiliary-windows.json').read_text())[0]['token_ids'],device='cuda')[None]
    sources={k:source(args.upstream,p) for k,p in {
        'dlg':'sfl/model/attacker/dlg_attacker.py','sma':'sfl/model/attacker/sma_attacker.py',
        'model':'sfl/utils/model.py','sip':'sfl/model/attacker/sip/inversion_models.py'}.items()}
    model=GPT2LMHeadModel.from_pretrained(args.model,local_files_only=True,attn_implementation='eager').float().cuda().eval()
    model.requires_grad_(False)
    result={'scope':'single-seam C-A real-model engineering/resource canary; not empirical fidelity',
        'input_manifest_sha256':sha256(args.inputs/'manifest.json'), 'seed':81273,
        'sources':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
        'model_files':{p.name:sha256(p) for p in args.model.iterdir() if p.is_file()},
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),'cells':[]}
    save(args.output/'inputs.json',result)
    try:
        for depth in (3,6):
            for knowledge in ('public','exact'):
                prefix=f'd{depth}-{knowledge}'
                head_path=None
                if knowledge=='exact':
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                        if k.startswith(('transformer.wte.','transformer.wpe.')+
                                        tuple(f'transformer.h.{i}.' for i in range(depth)))}
                    # Excluded canary perturbation makes exact grant observably distinct.
                    state['transformer.h.0.ln_1.weight']+=.001
                    head_path=args.output/(prefix+'-head.pt')
                    torch.save({'format':'llm_attacker.seam_head.v1','arch':'gpt2',
                        'boundary':depth,'config':model.config.to_dict(),'state_dict':state},head_path)
                split=granted_split(model,depth,knowledge=knowledge,head_path=head_path,
                                    head_sha256=sha256(head_path) if head_path else None)
                arch={'vocab':model.config.vocab_size,'seam_dim':model.config.n_embd,
                      'seq_len':32,'hidden_size':256,'dropout':.1}
                inv=gru_class()(**arch).cuda().eval().requires_grad_(False)
                ref_inv=reference_gru(sources,arch,inv.state_dict(),'cuda')
                h=split.head_forward(ids).detach().requires_grad_(True)
                logits=split.rest_forward(h)
                loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,arch['vocab']),ids[:,1:].reshape(-1))
                g,=torch.autograd.grad(loss,h)
                h=h.detach();g=g.detach()
                replay_wire(split,ids,h,g)
                with torch.no_grad():
                    scores=inv(h);ref_scores=ref_inv(h);numerical_check(scores,ref_scores)
                gm,sm=adapter(split,sources)
                traces=[];opt_cls=torch.optim.AdamW
                class Trace(opt_cls):
                    def step(self,closure=None):
                        p=self.param_groups[0]['params'][0]
                        rec={'before':p.detach().cpu().clone(),'gradient':p.grad.detach().cpu().clone()}
                        value=super().step(closure);rec['after']=p.detach().cpu().clone();traces.append(rec)
                        return value
                torch.optim.AdamW=Trace
                try:
                    gm(h,g,ref_scores,iters=1)
                    _bre_gradient_match(split,h,g,scores,iters=1,lr=.09,beta=.85,tau=1.2,opt_batch=1,upstream=True)
                    sm(h,ref_scores,iters=1)
                    _bre_smashed_match(split,split.wte.weight,h,scores.argmax(-1),iters=1,lr=.01,wd=.01,opt_batch=1,upstream=True)
                finally:torch.optim.AdamW=opt_cls
                for a,b in ((traces[0],traces[1]),(traces[2],traces[3])):
                    for name in ('before','gradient','after'):numerical_check(a[name],b[name])
                torch.save({'h':h.cpu(),'g':g.cpu(),'sip':scores.cpu(),'reference_sip':ref_scores.cpu(),
                            'traces':traces},args.output/(prefix+'-mechanical.pt'))
                cell={'depth':depth,'knowledge':knowledge,'mechanical_pass':True,'arms':{}}
                for engine in ('candidate','reference'):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize();start=time.monotonic()
                    if engine=='candidate':
                        b,_=_bre_gradient_match(split,h,g,scores,iters=32,lr=.09,beta=.85,tau=1.2,opt_batch=1,upstream=True)
                        f,_=_bre_smashed_match(split,split.wte.weight,h,scores.argmax(-1),iters=800,lr=.01,wd=.01,opt_batch=1,upstream=True)
                        bf,_=_bre_smashed_match(split,split.wte.weight,h,b,iters=800,lr=.01,wd=.01,opt_batch=1,upstream=True)
                    else:
                        b_scores=gm(h,g,ref_scores,iters=32)
                        f=sm(h,ref_scores,iters=800).argmax(-1)
                        bf=sm(h,b_scores.detach(),iters=800).argmax(-1);b=b_scores.argmax(-1)
                    torch.cuda.synchronize()
                    cell['arms'][engine]={'wall_seconds':time.monotonic()-start,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'gm_steps':32,'sm_steps_per_arm':800,'samples':1}
                    torch.save({'b':b.cpu(),'f':f.cpu(),'bf':bf.cpu()},args.output/(prefix+'-'+engine+'.pt'))
                result['cells'].append(cell)
                save(args.output/(prefix+'.json'),cell)
                print(json.dumps(cell),flush=True)
                del split,inv,ref_inv,gm,sm,traces,scores,ref_scores,h,g,logits
                torch.cuda.empty_cache()
        seconds=sum(v['wall_seconds'] for c in result['cells'] for v in c['arms'].values())
        result['online_projection_hours']=seconds*256*3/3600
        result['online_projection_with_1_5_margin_hours']=result['online_projection_hours']*1.5
        result['resource_ceiling_hours']=30
        result['status']='CANARY_PASS' if result['online_projection_with_1_5_margin_hours']<30 else 'BLOCKED_RESOURCE_PROJECTION'
        result['full_launch_admitted']=False # training/resources, remaining gates and checkpoint manifest required
    except Exception as exc:
        result['status']='CANARY_FAILED';result['error']=repr(exc)
        save(args.output/'result.json',result)
        raise
    save(args.output/'result.json',result)


if __name__=='__main__':main()
