"""Sequential all-depth canaries, stable grid and repaired co-trained tier."""
import json,os,random,signal,subprocess,sys,time
from pathlib import Path
from experiments.whitebox35.rerun import check_cell,sha
ROOT=Path(__file__).resolve().parents[2]

def main():
 os.chdir(ROOT);out=ROOT/'results/whitebox35-followup-v1';out.mkdir(parents=True,exist_ok=False)
 protocol=ROOT/'experiments/whitebox35/protocol.json';p=json.loads(protocol.read_text())
 def state(status,**extra):
  value={'status':status,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'protocol_sha256':sha(protocol),**extra}
  tmp=out/'status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,out/'status.json');print(json.dumps(value),flush=True)
 def run(cmd,name):
  started=time.monotonic();state('running',job=name)
  with (out/(name+'.log')).open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=p['max_seconds_per_job'])
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=124
  rec={'job':name,'rc':rc,'wall_s':time.monotonic()-started,'command':cmd}
  with (out/'journal.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  if rc:raise RuntimeError(f'{name} failed rc={rc}; retained, no automatic retry')
  return rec
 try:
  state('waiting_for_original36')
  marker=ROOT/'results/whitebox35-original36-v1/verified-complete.json'
  while not marker.exists():
   pid=int((ROOT/'original36.pid').read_text());os.kill(pid,0);time.sleep(15)
  assert json.loads(marker.read_text())['corrected_cells']==36
  canaries=[]
  for depth in p['internal_depths']:
   cmd=[sys.executable,'-m','experiments.whitebox35.grid','--root',str(out/'canaries'/f'depth{depth}'),'--depth',str(depth),'--seed','11','--method','pca-compress','--dim','64','--canary']
   canaries.append(run(cmd,f'canary-depth{depth}'))
  (out/'canary-gate.json').write_text(json.dumps({'status':'complete','excluded':True,'canaries':canaries},indent=2)+'\n')
  # Keep all boundaries and methods fixed; no outcome-based adaptation.
  grid=[(d,s,m,k) for d in p['internal_depths'] for s in p['seeds'] for m in p['methods'] for k in p['dims']]
  random.Random(350917).shuffle(grid)
  (out/'execution-order.json').write_text(json.dumps(grid,indent=2)+'\n')
  for d,s,m,k in grid:
   run([sys.executable,'-m','experiments.whitebox35.grid','--root',str(out/'stable-grid'),'--depth',str(d),'--seed',str(s),'--method',m,'--dim',str(k)],f'stable-d{d}-s{s}-{m}-k{k}')
  cells=[check_cell(x) for x in (out/'stable-grid').glob('whitebox-e2e/split*/seed-*/*/cell.json')]
  assert len(cells)==198
  (out/'stable-grid-complete.json').write_text(json.dumps({'status':'complete','cells':198,'depths':p['internal_depths']},indent=2)+'\n')
  repaired=[(d,s,k) for d in p['internal_depths'] for s in p['seeds'] for k in p['dims']]
  random.Random(350918).shuffle(repaired)
  for d,s,k in repaired:
   dest=out/'cotrain-final-inference'/f'depth{d}-seed{s}-dim{k}'
   run([sys.executable,'-m','experiments.whitebox35.cotrain','--output',str(dest),'--depth',str(d),'--seed',str(s),'--dim',str(k),'--device','cuda','--threads','4','--model',str(Path.home()/('.cache/huggingface/hub/models--gpt2/snapshots/'+p['model_revision']))],f'cotrain-d{d}-s{s}-k{k}')
  (out/'cotrain-grid-complete.json').write_text(json.dumps({'status':'complete','cells':99,'observation':'new final-state inference; historical18 stay excluded'},indent=2)+'\n')
  run([sys.executable,'-m','experiments.whitebox35.package','--root',str(out)],'analyse-package')
  state('complete',stable_cells=198,cotrain_cells=99)
 except Exception as e:
  state('failed',error=repr(e));raise
if __name__=='__main__':main()
