"""Fresh adapted SIP training and label-isolated mechanical gates; no scoring."""
import json,os,sys,time,subprocess,signal
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[1]
 plan=json.loads((root/'experiments/v4_sip_preparation_plan.json').read_text())
 out=Path.cwd()/'sip-preparation'
 if '--worker' not in sys.argv:
  out.mkdir(exist_ok=False)
  env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
  start=time.monotonic();proc=subprocess.Popen(['/home/geo/llm-attacker/.venv/bin/python',__file__,'--worker'],env=env,start_new_session=True)
  try:code=proc.wait(timeout=plan['this_stage_ceiling_seconds'])
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
  (out/'resource.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-start,'stage_ceiling_seconds':plan['this_stage_ceiling_seconds'],'total_approved_gpu_hours':50,'exit_code':code,'full_comparison_launched':False},indent=2))
  raise SystemExit(code)
 sys.path.insert(0,str(root))
 import torch,transformers
 from attacker.bisr_fidelity import sha256,gru_class,train_gru,validate_observations
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 started=time.monotonic()
 def event(kind,**kw):
  row={'event':kind,'elapsed_seconds':time.monotonic()-started,**kw}
  with (out/'journal.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 for name,sha in plan['source_sha256'].items():
  if sha256(root/name)!=sha:raise ValueError('Qualified code changed: '+name)
 inputs=Path('/home/geo/llm-attacker-bisr-fresh-inputs-20260918/results/frozen-inputs')
 checkpoints=Path('/home/geo/llm-attacker-bisr-fidelity-20260918/run-v1/checkpoints')
 frozen=json.loads((root/'experiments/v4-preparation-pins/inputs.json').read_text())
 if sha256(inputs/'manifest.json')!=plan['auxiliary_manifest_sha256']:raise ValueError('Input manifest changed')
 for name in ('auxiliary-windows.json','auxiliary-documents.json'):
  if sha256(inputs/name)!=frozen['outputs'][name]:raise ValueError('Auxiliary changed')
 prepared=json.loads((root/'experiments/v4-preparation-pins/ca-preparation.json').read_text())
 for name in ('public.pt','aux-rows-d3.pt','aux-rows-d6.pt'):
  if sha256(checkpoints/name)!=prepared['files'][name]:raise ValueError('Cache changed: '+name)
 aux=torch.tensor([r['token_ids'] for r in json.loads((inputs/'auxiliary-windows.json').read_text())])
 if aux.shape!=(15604,32):raise ValueError('Unexpected auxiliary shape')
 env={'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'tf32':False,'deterministic':True,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}
 (out/'environment.json').write_text(json.dumps(env,indent=2));event('INPUTS_VERIFIED',auxiliary_windows=len(aux))
 arch={'vocab':50257,'seam_dim':768,'seq_len':32,'hidden_size':256,'dropout':.1};trained={}
 for depth in plan['depths']:
  rows=torch.load(checkpoints/f'aux-rows-d{depth}.pt',map_location='cpu',weights_only=True)
  if rows.shape!=(len(aux),32,768) or not torch.isfinite(rows).all():raise ValueError('Bad auxiliary rows')
  for seed in plan['seeds']:
   torch.manual_seed(seed);inv=gru_class()(**arch).cuda();calls=0
   def fetch(indices):
    nonlocal calls
    calls+=1
    if calls%488==0:event('EPOCH_PROGRESS',seed=seed,depth=depth,epoch=calls//488)
    return aux[indices].cuda(),rows[indices].cuda()
   training=train_gru(inv,fetch,len(aux),epochs=20,batch_size=32,seed=seed,lr=.001)
   name=f'sip-s{seed}-d{depth}.pt';target=out/name
   torch.save({'format':'llm_attacker.bisr_gru.v1','arch':arch,'state_dict':{k:v.detach().cpu() for k,v in inv.state_dict().items()},'seed':seed,'training':training,'auxiliary_manifest_sha256':plan['auxiliary_manifest_sha256']},target)
   trained[name]=sha256(target);event('CHECKPOINT_TRAINED',name=name,sha256=trained[name]);del inv;torch.cuda.empty_cache()
  del rows
 # Canonical v3 tokens are never opened here. Sample identifiers come from the
 # hash-pinned original corpus manifest; observations retain original h/g bytes.
 corpus=root/'experiments/validation_corpus_manifest.json'
 proto=json.loads((root/'experiments/validation_protocol.v3.json').read_text())
 if sha256(corpus)!=proto['frozen_inputs']['curation_manifest_sha256']:raise ValueError('Corpus identity changed')
 ids=json.loads(corpus.read_text())['selected_ids']['evaluation']
 base=Path('/home/geo/llm-attacker-pilots/gpt2-wikitext2-v01/code/results/validation-heldout-v3-r3-shard8')
 binary=json.loads((root/'experiments/v4_adaptation_expected.json').read_text())
 template=json.loads((root/'experiments/v4-preparation-pins/ca-manifest.json').read_text())
 records=out/'records';records.mkdir();gates=out/'gates';gates.mkdir();cells={}
 for victim in (101,202,303,404,505,606,707,808,909,1010):
  for depth in (3,6):
   name=f'observations-{victim}-{depth}.pt';original=base/name
   if sha256(original)!=binary[name]:raise ValueError('Original observation changed')
   obs=torch.load(original,map_location='cpu',weights_only=True)
   bundle={'schema':'bisr.observations.v1','sample_ids':ids,'h':obs['h'],'g':obs['g'],'mask':torch.ones((256,32),dtype=torch.bool),'gradient_batch_scale':torch.full((256,),256,dtype=torch.float32)}
   validate_observations(bundle);converted=out/name;torch.save(bundle,converted)
   for seed in (17,29):
    for grant in ('public','exact'):
     key=f'v{victim}-d{depth}-a{seed}-{grant}';head=base/f'exact-{victim}-{depth}.pt'
     if sha256(head)!=binary[head.name]:raise ValueError('Exact head changed')
     m={'schema':'bisr.campaign_prediction.v1','public_model':{'path':str(checkpoints/'public.pt'),'sha256':prepared['files']['public.pt']},'inverter':{'path':str(out/f'sip-s{seed}-d{depth}.pt'),'sha256':trained[f'sip-s{seed}-d{depth}.pt']},'observations':{'path':str(converted),'sha256':sha256(converted)},'head':{'path':str(head),'sha256':binary[head.name]} if grant=='exact' else None,'boundary':depth,'knowledge':grant,'sources':template['sources'],'source_sha256':plan['source_sha256'],'documents':256}
     manifest=records/(key+'.json');manifest.write_text(json.dumps(m,sort_keys=True)+'\n')
     with (gates/(key+'.log')).open('w') as log:
      subprocess.run([sys.executable,str(root/'experiments/bisr_campaign_predict.py'),'--manifest',str(manifest),'--engine','gate','--output',str(gates/key)],check=True,stdout=log,stderr=subprocess.STDOUT)
     result=json.loads((gates/key/'result.json').read_text())
     if result['status']!='TRAINED_MECHANICAL_PASS':raise ValueError('Mechanical gate failed')
     cells[key]={'manifest_sha256':sha256(manifest),'gate_sha256':sha256(gates/key/'result.json')};event('MECHANICAL_GATE_PASS',cell=key,completed=len(cells),total=80)
 summary={'status':'ADAPTED_SIP_PREPARED_GATES_PASSED','checkpoints':trained,'cells':cells,'gate_count':80,'auxiliary_manifest_sha256':plan['auxiliary_manifest_sha256'],'environment_sha256':sha256(out/'environment.json'),'wall_seconds':time.monotonic()-started,'capture_attention':'sdpa','attack_attention':'eager','gradient_loss_normalisation_multiplier':256,'prediction_scoring_performed':False,'full_comparison_admitted':False}
 (out/'result.json').write_text(json.dumps(summary,indent=2));event('PREPARATION_COMPLETE',gates=80)
if __name__=='__main__':main()
