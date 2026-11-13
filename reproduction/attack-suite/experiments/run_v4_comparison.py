"""Bounded two-host adapted comparison. No retries or partial-set scoring."""
import json,os,sys,time,signal,socket,subprocess,fcntl
from pathlib import Path
from v4_adaptation_preflight import digest

def main():
 root=Path(__file__).resolve().parents[1];admission=root/'experiments/v4-final-admission';plan=json.loads((admission/'launch.json').read_text())
 host=socket.gethostname().split('.')[0].lower()
 if host not in ('toula','koula'):raise ValueError('Unassigned GPU host')
 out=Path.cwd()/'comparison'
 if '--worker' not in sys.argv:
  lock=Path('/home/geo/.cache/llm-attacker-v4-comparison.lock');lock.parent.mkdir(exist_ok=True)
  with lock.open('a') as handle:
   fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   out.mkdir(exist_ok=False);started=time.monotonic()
   env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
   proc=subprocess.Popen(['/home/geo/llm-attacker/.venv/bin/python',__file__,'--worker'],env=env,start_new_session=True)
   try:code=proc.wait(timeout=plan['per_host_limit_seconds']-10)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    code=124
   (out/'resource.json').write_text(json.dumps({'host':host,'elapsed_seconds':time.monotonic()-started,'limit_seconds':plan['per_host_limit_seconds'],'total_ceiling_gpu_hours':50,'exit_code':code,'no_automatic_retry':True},indent=2))
   if code:
    (out/'failed.json').write_text(json.dumps({'exit_code':code,'partial_results_retained':True,'comparative_scoring_blocked':True}))
   raise SystemExit(code)
 import torch,transformers
 if torch.__version__!=plan['environment']['torch'] or transformers.__version__!=plan['environment']['transformers'] or torch.version.cuda!=plan['environment']['cuda']:raise ValueError('Environment differs from trained gates')
 if torch.cuda.get_device_name(0)!=plan['environment']['device']:raise ValueError('Different GPU')
 for name,sha in plan['evidence_files'].items():
  if digest(admission/name)!=sha:raise ValueError('Admission evidence changed: '+name)
 for name,item in plan['assets'].items():
  if digest(Path('/home/geo/v4-adaptation-assets-20260919')/name)!=item['sha256']:raise ValueError('Asset mismatch: '+name)
 keys=sorted(k for k,v in plan['cells'].items() if v['host']==host)
 if len(keys)!=40 or len(plan['cells'])!=80:raise ValueError('Incomplete matrix')
 status={'host':host,'status':'HOST_ASSETS_VERIFIED','launch_sha256':digest(admission/'launch.json'),'cells_total':40,'cells_completed':0,'documents_per_cell':256,'scoring_performed':False,'capture_attention':'sdpa','attack_attention':'eager'}
 def report():
  temp=out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'status.json');print(json.dumps(status),flush=True)
 report();results={}
 for key in keys:
  manifest=admission/'records'/(key+'.json')
  if digest(manifest)!=plan['cells'][key]['manifest_sha256']:raise ValueError('Manifest changed')
  status.update(status='PREDICTING',cell=key,current_cell_documents=0);report()
  cell=out/key
  with (out/(key+'.log')).open('w') as log:
   proc=subprocess.Popen([sys.executable,str(root/'experiments/v4_adaptation_predict.py'),'--manifest',str(manifest),'--engine','candidate','--output',str(cell)],stdout=log,stderr=subprocess.STDOUT)
   while True:
    try:code=proc.wait(timeout=60);break
    except subprocess.TimeoutExpired:
     status['current_cell_documents']=len(list(cell.glob('[0-9][0-9][0-9][0-9][0-9].json')));report()
   if code:raise RuntimeError('Prediction failed: '+key+'; inspect retained log')
  result=json.loads((cell/'result.json').read_text())
  if result['status']!='PREDICTIONS_COMPLETE' or result['documents']!=256:raise ValueError('Incomplete predictions')
  results[key]={'result_sha256':digest(cell/'result.json'),'journal_tip':result['journal_tip'],'online_seconds':result['online_seconds']}
  status['cells_completed']=len(results);status['current_cell_documents']=256;report()
 status.update(status='HOST_PREDICTIONS_COMPLETE',scoring_performed=False);report()
 (out/'result.json').write_text(json.dumps({'status':'HOST_PREDICTIONS_COMPLETE','host':host,'cells':results,'documents':len(results)*256,'launch_sha256':status['launch_sha256'],'scoring_requires_both_hosts_complete':True},indent=2))
if __name__=='__main__':main()
