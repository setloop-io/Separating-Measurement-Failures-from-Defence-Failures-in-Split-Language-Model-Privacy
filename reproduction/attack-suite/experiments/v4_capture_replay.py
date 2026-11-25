"""Mechanical original-head replay; truth access restricted to this preparation."""
import json,os,sys,socket,time
from pathlib import Path
from v4_adaptation_preflight import digest

def main():
    py='/home/geo/llm-attacker/.venv/bin/python'
    if not os.environ.get('V4_REPLAY_REEXEC'):
        os.environ['V4_REPLAY_REEXEC']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.execv(py,[py,__file__])
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from attacker.bisr_fidelity import granted_split,numerical_check,validate_observations
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    host=socket.gethostname().split('.')[0].lower()
    plan=json.loads((root/'experiments/v4_adaptation.v1.json').read_text())
    pins=json.loads((root/'experiments/v4_adaptation_expected.json').read_text())
    base=Path('/home/geo/llm-attacker-pilots/gpt2-wikitext2-v01/code/results')/('validation-heldout-v3-r3-shard0' if host=='toula' else 'validation-heldout-v3-r3-shard8')
    inp=base/'input.json';protocol=json.loads((root/'experiments/validation_protocol.v3.json').read_text())
    if digest(inp)!=protocol['frozen_inputs']['tokenized_sha256']:raise ValueError('Input mismatch')
    data=json.loads(inp.read_text())['evaluation'];tokens=torch.tensor([x['tokens'] for x in data],device='cuda')
    out=Path.cwd()/'replay';out.mkdir(exist_ok=False);records=[];start=time.monotonic()
    for seed in plan['fleet'][host]['victim_seeds']:
        for depth in (3,6):
            hn=f'exact-{seed}-{depth}.pt';on=f'observations-{seed}-{depth}.pt'
            for n in (hn,on):
                if digest(base/n)!=pins[n]:raise ValueError('Binary hash mismatch')
            blob=torch.load(base/hn,map_location='cpu',weights_only=True)
            config=GPT2Config(**blob['config']);config._attn_implementation='eager'
            # Synthetic downstream is never executed; this checks only the exact
            # head loader and capture identity, not full gradient replay.
            public=GPT2LMHeadModel(config).float().cuda().eval().requires_grad_(False)
            split=granted_split(public,depth,knowledge='exact',head_path=base/hn,head_sha256=pins[hn])
            # Replay original capture with its SDPA forward kernel. The qualified
            # eager attack remains unchanged and consumes the original saved wire.
            split.head.model.set_attn_implementation('sdpa')
            obs=torch.load(base/on,map_location='cpu',weights_only=True)
            if set(obs)!={'h','g'} or obs['h'].shape!=obs['g'].shape or obs['h'].shape[:2]!=(256,32):raise ValueError('Capture schema mismatch')
            maxdiff=0.
            with torch.no_grad():
                for i in range(0,256,8):
                    h=split.head.head_forward(tokens[i:i+8]);ref=obs['h'][i:i+8].cuda()
                    if not torch.equal(h,ref):
                        raise ValueError(f'Original SDPA capture differs at seed={seed} depth={depth} batch={i}; max_abs={float((h-ref).abs().max())}')
                    maxdiff=max(maxdiff,float((h-ref).abs().max()))
            converted={'schema':'bisr.observations.v1','sample_ids':[x['id'] for x in data],'h':obs['h'],'g':obs['g'],'mask':torch.ones((256,32),dtype=torch.bool),'gradient_batch_scale':torch.full((256,),256,dtype=torch.float32)}
            validate_observations(converted)
            target=out/on;torch.save(converted,target)
            record={'seed':seed,'depth':depth,'head_sha256':pins[hn],'original_observations_sha256':pins[on],'converted_sha256':digest(target),'input_sha256':digest(inp),'head_replay_pass':True,'capture_attention':'sdpa','attack_attention':'eager','bitwise_equal':True,'max_activation_abs_difference':maxdiff,'gradient_replay_performed':False,'gradient_multiplier':256,'gradient_multiplier_basis':'original mean CE denominator 256*(32-1)'}
            records.append(record);(out/'progress.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
            del public,split,blob,obs,converted;torch.cuda.empty_cache()
    report={'status':'ORIGINAL_HEAD_REPLAY_PASS','rows_checked':len(records)*256,'captures':records,'elapsed_seconds':time.monotonic()-start,'gradient_replay_performed':False,'predictions_generated':False,'capture_attention':'sdpa','attack_attention':'eager','bitwise_equal':True,'full_launch_admitted':False}
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
