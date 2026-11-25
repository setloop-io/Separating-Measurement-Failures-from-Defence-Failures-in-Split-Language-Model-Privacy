"""Detached recovery/upload/verification worker for the reviewed 2026-09-17 release."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
STAGE=ROOT/'build/hf-release-2026-09-17';PY=ROOT/'.venv/bin/python'
ENV={**os.environ,'HF_HUB_DISABLE_XET':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','PYTHONUNBUFFERED':'1'}
def log(*x):print(time.strftime('%Y-%m-%dT%H:%M:%S'),*x,flush=True)
def run(*args):
 log('RUN',*map(str,args));subprocess.run(list(map(str,args)),env=ENV,check=True)
def wait_pid(pid):
 while True:
  p=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
  if p.returncode or not p.stdout.strip() or p.stdout.strip().startswith('Z'):return
  time.sleep(5)
def copytree(src,dst):
 shutil.copytree(src,dst,dirs_exist_ok=True,copy_function=shutil.copy2,ignore=shutil.ignore_patterns('.*','__pycache__','*.aux','*.log'))
def main():
 run(PY,'experiments/release_prepare.py')
 # Wait for the already-running small tiers and P6 collection before finalising.
 for pid in WAIT_PIDS:wait_pid(pid)
 for paper in ('paper-a','paper-b'):
  dst=STAGE/paper
  # Drop only rsync partial files left by interrupted recovery.
  for p in (dst/'binary-tier').rglob('.*'):
   if p.is_file() and '.pt.' in p.name:p.unlink()
  for name in ('release_prepare.py','release_manifest.py','release_upload.py','release_verify_remote.py'):
   shutil.copy2(ROOT/'experiments'/name,dst/'code/experiments'/name)
  shutil.copy2(ROOT/'docs/evidence-release-hf.md',dst/'REPRODUCE.md')
  shutil.copy2(ROOT/'results/phase2-status-2026-09-17/status.json',dst/'PHASE2-STATUS.json')
  shutil.copy2(ROOT/'results/phase2-status-2026-09-17/p6-evidence-verification.json',dst/'p6-evidence-verification.json')
  # Preserve manuscript bytes and satisfy their ../figures references.
  copytree(dst/'paper/figures',dst/'figures')
  pending=dst/'PENDING.md';text=pending.read_text()
  text=text.replace('all three GPT-2 Mode-2 grids completed (192 cells, zero execution failures); collection and analysis in progress','all three GPT-2 Mode-2 grids completed (192 cells, zero execution failures); full outputs packaged under p6-grid-v1; frozen-endpoint analysis and cluster-aware checks pending')
  text=text.replace('- Full binary reference recovery and remote release verification are in progress. Do not treat this snapshot as complete until verification reports pass.','- All 860 merged binary references have been recovered and hash-verified. Remote verification is recorded separately after upload; external evidence gaps above remain open.')
  if paper=='paper-b':text += '\n- BiSR fidelity JSON records and the reference aggregate are included under bisr-fidelity-v1 and splitllm-reference. The 79 GB full capture directory/checkpoint tier is not included in this snapshot; replay of that run still requires those original assets.\n'
  pending.write_text(text)
  readme=dst/'README.md';text=readme.read_text().replace('Release upload is in progress. MANIFEST.sha256 is published after all staged files.','MANIFEST.sha256 covers this snapshot. External evidence gaps remain listed in PENDING.md.');readme.write_text(text)
  with (dst/'INDEX.md').open('a') as f:
   f.write('\n## Phase 2 update (17 September 2026)\n\nSee PHASE2-STATUS.json for execution status and descriptive counts. These do not replace the frozen scientific analyses.\n')
  if paper=='paper-a':
   for span in (2,4,6):
    source=ROOT/'results'/f'p6-grid-s{span}-v1'
    state=json.loads((source/'run-index.json').read_text())
    if state['n_cell_records']!=64 or state['n_failed']!=0:raise RuntimeError('Incomplete P6 grid')
    copytree(source,dst/'p6-grid-v1'/f'span{span}')
   with (dst/'INDEX.md').open('a') as f:f.write('\nP6 Mode-2: p6-grid-v1/span{2,4,6}/run-index.json and corresponding artifacts/captures.\n')
  else:
   for name in ('bisr-fidelity-v1','splitllm-reference'):
    copytree(ROOT/'results/phase2-status-2026-09-17'/name,dst/name)
   with (dst/'INDEX.md').open('a') as f:f.write('\nBiSR preliminary records: bisr-fidelity-v1/our-bisr-attack.json; splitllm-reference/attacks.jsonl.aggregate.json. No fidelity pass is asserted.\n')
  run(PY,'experiments/release_manifest.py',dst)
 # Upload each repo and verify against one immutable remote revision.
 reports=[]
 for paper in ('paper-a','paper-b'):
  run(PY,'experiments/release_upload.py',paper)
  run(PY,'experiments/release_verify_remote.py',paper)
  reports.append(json.loads((STAGE/(paper+'-remote-verification.json')).read_text()))
 (STAGE/'release-complete.json').write_text(json.dumps(reports,indent=2)+'\n')
 log('RELEASE_COMPLETE_VERIFIED')
import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wait-pid", type=int, action="append", default=[])
WAIT_PIDS=ap.parse_args().wait_pid
try:main()
except Exception as exc:
 log('RELEASE_FAILED',type(exc).__name__,str(exc));raise
