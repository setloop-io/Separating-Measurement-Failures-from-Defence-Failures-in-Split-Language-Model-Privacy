"""Finish the reviewed Paper B release after complete immutable HF evidence verification.

No model execution. Refuses to overwrite concurrent edits or publish a PDF whose
render differs from the reviewed draft. All state transitions are persisted.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import release_paper_a_revision as publication

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pixels(p):
 from PIL import Image
 with Image.open(p) as im:
  im=im.convert('RGB')
  return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
 config=json.loads(a.config.read_text());base=Path(config['base']);draft=base/'reviewed-draft'
 def state(status,**extra):
  d={'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra}
  publication.save(base/'completion-status.json',d);print(json.dumps(d),flush=True)
 try:
  state('WAITING_FOR_COMPLETE_EVIDENCE_VERIFICATION')
  deadline=time.time()+24*3600
  while True:
   p=subprocess.run(['ssh','-o','ConnectTimeout=15','koula','cat /home/geo/bisr-fidelity-release-20260919/final-verification.json'],capture_output=True,text=True)
   if p.returncode==0:
    evidence=json.loads(p.stdout)
    if evidence.get('ok') is not True:raise ValueError('Final evidence verification did not pass')
    break
   if time.time()>deadline:raise TimeoutError('Evidence publication not verified within 24 hours')
   time.sleep(30)
  publication.save(base/'final-verification.json',evidence)
  pin=evidence['revision'];repo=evidence['repo']
  if len(pin)!=40 or repo!='Setloop/llm-attacker-paper-b' or evidence['prefix']!='fidelity-20260919/evidence':raise ValueError('Unexpected evidence target')
  # Recheck remote HF objects independently from this machine against the published manifest.
  from huggingface_hub import HfApi,hf_hub_download
  api=HfApi();prefix=evidence['prefix']
  manifest=Path(hf_hub_download(repo,prefix+'/MANIFEST.sha256',repo_type='dataset',revision=pin))
  if sha(manifest)!=evidence['manifest_sha256']:raise ValueError('Published manifest digest')
  expected={}
  for line in manifest.read_text().splitlines():
   digest,rel=line.split('  ',1);expected[prefix+'/'+rel]=digest
  remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',revision=pin,path_in_repo=prefix,recursive=True) if hasattr(x,'blob_id')}
  if set(remote)!=set(expected)|{prefix+'/MANIFEST.sha256'}:raise ValueError('Pinned evidence coverage changed')
  for name,want in expected.items():
   e=remote[name]
   if e.lfs:
    if e.lfs.sha256!=want:raise ValueError('LFS SHA mismatch '+name)
   else:
    if sha(Path(hf_hub_download(repo,name,repo_type='dataset',revision=pin)))!=want:raise ValueError('Blob SHA mismatch '+name)
  for rel,want in config['expected_workspace_sha256'].items():
   if sha(ROOT/rel)!=want:raise ValueError('Concurrent workspace edit: '+rel)
  state('EVIDENCE_VERIFIED_BUILDING_REVIEWED_MANUSCRIPT',revision=pin)
  stage=base/'manuscript';stage.mkdir(exist_ok=False)
  for name in ['main.tex','main.bib','main.bbl','response-to-review-2026-09-17.md']:
   p=draft/name
   if p.exists():(stage/name).write_text(p.read_text().replace('IMMUTABLE_FIDELITY_REVISION',pin))
  for _ in range(2):
   p=subprocess.run(['/Library/TeX/texbin/pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=stage,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   (base/'final-pdf-build.log').write_text(p.stdout)
   if p.returncode:raise ValueError('Final PDF failed to compile')
  log=(stage/'main.log').read_text()
  if 'undefined references' in log or 'undefined citations' in log or 'Overfull \\hbox' in log:raise ValueError('Final PDF has unresolved references or overfull boxes')
  rendered=base/'final-render';rendered.mkdir()
  subprocess.run(['/opt/homebrew/bin/pdftoppm','-r','72','-png',str(stage/'main.pdf'),str(rendered/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  expected_pixels=config['reviewed_pixels_sha256'];actual={p.name:pixels(p) for p in sorted(rendered.glob('page-*.png'))}
  if actual!=expected_pixels:raise ValueError('Final PDF render differs from visually reviewed draft')
  publication.save(base/'pdf-verification.json',{'ok':True,'pages':len(actual),'method':'Each final RGB page pixel hash equals visually reviewed draft at 72 dpi','pdf_sha256':sha(stage/'main.pdf')})
  report=(ROOT/'docs/bisr-fidelity-completed-2026-09-19.md').read_text().replace('IMMUTABLE_FIDELITY_REVISION',pin)
  (stage/'bisr-fidelity-completed-2026-09-19.md').write_text(report)
  current=json.loads((ROOT/'experiments/bisr_fidelity_current.json').read_text());current.update(publication_status='VERIFIED',evidence_hf=evidence)
  (stage/'bisr_fidelity_current.json').write_text(json.dumps(current,indent=2)+'\n')
  shutil.copy2(base/'final-verification.json',stage/'evidence-verification.json')
  shutil.copy2(base/'pdf-verification.json',stage/'pdf-verification.json')
  (stage/'README.md').write_text('# Paper B: qualified BiSR C-A revision\n\nThe revised PDF/LaTeX and reviewer response report single-seam implementation conformance, not original-setting reproduction. Complete evidence and reproduction: https://huggingface.co/datasets/'+repo+'/tree/'+pin+'/'+prefix+'\n\nThe final PDF matches every visually reviewed draft page pixel-for-pixel. Verify MANIFEST.sha256. Historical raw tiers remain at their separately cited immutable revisions.\n')
  # Keep build intermediates outside the publication inventory.
  logs=base/'final-tex-intermediates';logs.mkdir()
  for suffix in ['aux','log','out','blg','toc']:
   p=stage/('main.'+suffix)
   if p.exists():p.rename(logs/p.name)
  publication.REPO=repo;publication.PREFIX='fidelity-20260919'
  state('PUBLISHING_VERIFIED_MANUSCRIPT',evidence_revision=pin)
  manuscript=publication.publish(stage,'manuscript',lambda s,**kw:state(s,**kw),publication_label='Paper B BiSR conformance')
  publication.save(base/'manuscript-verification.json',manuscript)
  # Check again immediately before updating shared working files.
  for rel,want in config['expected_workspace_sha256'].items():
   if sha(ROOT/rel)!=want:raise ValueError('Concurrent workspace edit after publication: '+rel)
  for name in ['main.tex','main.pdf','response-to-review-2026-09-17.md']:shutil.copy2(stage/name,ROOT/'paper/paper_b'/name)
  (ROOT/'docs/bisr-fidelity-completed-2026-09-19.md').write_text(report)
  current['manuscript_hf']=manuscript
  publication.save(ROOT/'experiments/bisr_fidelity_current.json',current)
  outcome={'status':'COMPLETE','evidence':evidence,'manuscript':manuscript,'pdf_pages':len(actual),'scope':current['scope'],'frozen_v4_admitted':False,'published_setting_reproduced':False,'remaining_requirements':current['remaining_requirements']}
  publication.save(base/'release-complete.json',outcome);state('COMPLETE',**{k:v for k,v in outcome.items() if k!='status'})
  notes=Path('/Users/geo/.codex/scheduled-checks/bisr-20260919/outcome.md')
  with notes.open('a') as f:f.write('\n## Publication completed\n\nEvidence verified at '+pin+'; manuscript verified at '+manuscript['revision']+'. The tested C-A blocker is closed. Original-setting reproduction, frozen-v4 B1 and separate sasa analysis remain open. See '+str(base/'release-complete.json')+'.\n')
 except Exception as e:
  state('STOPPED',error_type=type(e).__name__,error=str(e));raise
if __name__=='__main__':main()
