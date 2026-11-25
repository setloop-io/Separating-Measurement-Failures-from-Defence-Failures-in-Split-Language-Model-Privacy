"""Correct the original 36 attacks without rewriting any historical results."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
from experiments import run_whitebox_sweep as wb
from attacker.evidence import verify_evidence
ROOT=Path(__file__).resolve().parents[2]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def check_cell(p):
 d=json.loads(p.read_text())
 if d['status']!='COMPLETE':raise ValueError('Incomplete '+str(p))
 ok,errors=verify_evidence(str(p)+'.evidence.json')
 if not ok:raise ValueError(errors)
 a=d['attacks']['adaptive_e2e'];w=a['optimization']['work'];steps=a['optimization']['steps']
 assert 'head(x)' in a['objective']
 assert w['head_forward_calls']==w['head_backward_calls']==steps
 assert w['head_forward_token_positions']==w['head_backward_token_positions']==steps*a['snap']['n_tokens']
 assert w['head_forward_sequences']==steps*a['snap']['n_sequences']
 assert a['snap']['work']['head_forward_calls']==0
 return d

def wait_predecessors():
 dep=Path.home()/'llm-attacker-phase2-pending-20260917/results/phase2-pending-v1/koula/status.json'
 qwen=Path.home()/'llm-attacker-p6-qwen-20260917/results/qwen-queue-status.json'
 while True:
  try:
   a=json.loads(dep.read_text());q=json.loads(qwen.read_text())
   if a['status'] in ['complete','completed_with_failures','failed'] and q['status'] in ['complete','failed','budget_blocked']:return
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(30)

def main():
 os.chdir(ROOT);p=json.loads((ROOT/'experiments/whitebox35/protocol.json').read_text())
 output=ROOT/'results/whitebox35-original36-v1'
 if output.exists():raise ValueError('Fresh root required; no automatic retries')
 wait_predecessors()
 source=Path.home()/p['historical_root'];original=json.loads((source/'plan.json').read_text())
 for key,value in p['budgets'].items():
  if original['budgets'][key]!=value:raise ValueError('Original budget mismatch '+key)
 assert original['splits']==p['original_depths'] and original['seeds']==p['seeds']
 assert original['dims']==p['dims']
 assert sha(ROOT/p['corpus_path'])==p['corpus_sha256']
 # Copy metadata bytes and hardlink immutable tensors/captures; output tier is new.
 def copy(a,b):
  if str(a).endswith('.pt'):os.link(a,b)
  else:shutil.copy2(a,b)
 shutil.copytree(source,output,copy_function=copy,ignore=shutil.ignore_patterns('whitebox','whitebox-e2e','__pycache__'))
 originals={str(x.relative_to(output)):sha(x) for x in output.rglob('*') if x.is_file()}
 (output/'ORIGINAL-INPUT-HASHES.json').write_text(json.dumps(originals,indent=2)+'\n')
 (output/'HISTORICAL-INVALIDATION.md').write_text('The original 36 adaptive-e2e results used encoder(x) without the client head. They are INVALID, not a leakage bound. Original results remain untouched at '+str(source)+'. This new tier reruns the exact original captures, checkpoints and budgets using PR35. The 18 historical co-trained cells remain NOT_EVALUABLE.\n')
 args=wb.build_parser(['--root',str(output),'--model','gpt2','--device','cuda','--threads','4','--adaptive-e2e','--optimize-lr',str(p['optimize_lr'])])
 started=time.monotonic();wb.run(args)
 cells=[check_cell(x) for x in output.glob('whitebox-e2e/split*/seed-*/*/cell.json') if json.loads(x.read_text())['status']=='COMPLETE']
 excluded=[x for x in output.glob('whitebox-e2e/split*/seed-*/*/cell.json') if json.loads(x.read_text())['status']=='NOT_EVALUABLE']
 assert len(cells)==36 and len(excluded)==18
 for name,digest in originals.items():
  if sha(output/name)!=digest:raise ValueError('Historical input mutated: '+name)
 report={'status':'complete','corrected_cells':36,'historical_exclusions':18,'wall_s':time.monotonic()-started,'source':str(source),'protocol_sha256':sha(ROOT/'experiments/whitebox35/protocol.json'),'inputs_unchanged':True}
 (output/'verified-complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
