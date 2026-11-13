"""Budgeted sequential Qwen canaries, grid, integrity verification and analysis."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'experiments/crossed_span_protocol.p6.qwen3.v1.json'
def main():
 os.chdir(ROOT);root=ROOT/'results';root.mkdir(exist_ok=True)
 status=root/'qwen-queue-status.json';started=time.monotonic();deadline=started+24*3600
 def record(state,**extra):
  data={'status':state,'pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'wall_s':time.monotonic()-started,'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),**extra}
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,status);print(json.dumps(data),flush=True)
 def run(cmd,name):
  with (root/name).open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=max(1,deadline-time.monotonic()))
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise TimeoutError('24-hour queue budget exhausted; partial evidence retained')
  if rc:raise RuntimeError(f'{name} exited {rc}; no automatic retry')
 base=[sys.executable,'-m','experiments.run_p6_corrected']
 common=['--protocol',str(PROTOCOL),'--device','cuda','--threads','4']
 try:
  record('canary_running')
  t=time.monotonic();run(base+['grid','--span','6','--canary','--node','koula','--workdir',str(root/'qwen-canary-v1')]+common,'qwen-canary-v1.log');trained=time.monotonic()-t
  record('control_canary_running',trained_canary_s=trained)
  t=time.monotonic();run(base+['cell','--span','2','--start','0','--defense','D0','--seed','11','--kind','all_local','--workdir',str(root/'qwen-control-canary-v1')]+common,'qwen-control-canary-v1.log');control=time.monotonic()-t
  projection=1.5*(192*trained+8*control)
  budget={'trained_canary_s':trained,'control_canary_s':control,'projection_s':projection,'remaining_s':deadline-time.monotonic(),'factor':1.5,'canaries_excluded':True}
  (root/'qwen-budget-gate.json').write_text(json.dumps(budget,indent=2)+'\n')
  if projection>deadline-time.monotonic()-1800:record('budget_blocked',**budget);return
  record('grid_running',**budget)
  run(base+['grid','--span','0','--controls','--node','koula','--workdir',str(root/'qwen-grid-v1')]+common,'qwen-grid-v1.log')
  record('analysing')
  run([sys.executable,'-m','experiments.qwen_p6.analyze','--root',str(root/'qwen-grid-v1'),'--protocol',str(PROTOCOL),'--output',str(root/'qwen-grid-v1.analysis.json')],'qwen-analysis-v1.log')
  record('complete',grid=str(root/'qwen-grid-v1'),analysis=str(root/'qwen-grid-v1.analysis.json'))
 except Exception as error:
  record('failed',error=repr(error));raise
if __name__=='__main__':main()
