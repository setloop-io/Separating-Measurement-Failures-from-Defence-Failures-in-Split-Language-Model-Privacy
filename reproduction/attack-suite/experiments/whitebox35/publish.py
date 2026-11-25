"""Collect verified packages from koula and publish pinned HF releases.

Uses local cached HF credentials only; no credentials are copied to nodes.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
from experiments.whitebox35.rerun import sha
ROOT=Path(__file__).resolve().parents[2]
REPO='Setloop/llm-attacker-paper-b'
REMOTE='llm-attacker-whitebox35-20260917/results/'

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,required=True);args=ap.parse_args()
 root=ROOT/'build/whitebox35-publication';root.mkdir(parents=True,exist_ok=False)
 def record(status,**extra):
  data={'status':status,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra}
  temp=root/'status.tmp';temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,root/'status.json');print(json.dumps(data),flush=True)
 def remote_json(path):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','koula','cat',path],capture_output=True,text=True)
  return json.loads(p.stdout) if p.returncode==0 else None
 def api_call(fn):
  for attempt in range(120):
   try:return fn()
   except Exception as e:
    record('api_retry',attempt=attempt+1,error=type(e).__name__)
    if attempt==119:raise
    time.sleep(60)
 try:
  for stage in ['original36','full']:
   source=REMOTE+'whitebox35-release-'+stage
   record('waiting_for_verified_package',stage=stage)
   deadline=time.monotonic()+7*86400
   while True:
    ready=remote_json(source+'/PACKAGE-READY.json')
    if ready:break
    state=remote_json(REMOTE+'whitebox35-followup-v1/status.json')
    if stage=='full' and state and state['status']=='failed':raise RuntimeError('Full grid failed; complete HF publication blocked')
    if time.monotonic()>deadline:raise TimeoutError('Package not ready after seven days')
    time.sleep(60)
   dest=root/stage;dest.mkdir()
   subprocess.run(['scp','koula:'+source+'/manifest.json',str(dest/'manifest.json')],check=True)
   assert sha(dest/'manifest.json')==ready['manifest_sha256']
   manifest=json.loads((dest/'manifest.json').read_text())
   record('collecting',stage=stage,files=len(manifest['files']))
   for name,meta in manifest['files'].items():
    if Path(name).name!=name:raise ValueError('Unsafe manifest name')
    subprocess.run(['rsync','-a','--partial','koula:'+source+'/'+name,str(dest/name)],check=True)
    if sha(dest/name)!=meta['sha256']:raise ValueError('Downloaded package digest mismatch')
   record('waiting_for_baseline_verification',stage=stage)
   while not args.baseline.exists():time.sleep(60)
   baseline=json.loads(args.baseline.read_text());assert len(baseline)==2 and all(x['ok'] for x in baseline)
   api=HfApi();prefix='whitebox35-20260917/'+stage+'/'
   files=sorted(dest.iterdir(),key=lambda p:p.name=='manifest.json')
   for path in files:
    record('uploading',stage=stage,file=path.name)
    commit=api_call(lambda:api.create_commit(REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=prefix+path.name,path_or_fileobj=str(path))],commit_message='Corrected PR35 '+stage+': '+path.name,num_threads=1))
   # At the final immutable commit verify every Git/LFS object digest.
   notice = ('# PR35 corrected white-box evidence\n\n'
      'The old adaptive-e2e results omitted the client head and are INVALID. '
      'The corrected original36 release preserves the original input hashes and budgets. '
      'Historical18 co-trained training cells remain NOT_EVALUABLE.\n\n'
      '[Corrected original36 evidence](whitebox35-20260917/original36/README.md)\n\n'
      + ('[Full internal-depth sweep and separate final-state co-training tier](whitebox35-20260917/full/README.md)\n' if stage=='full'
         else 'The full internal-depth sweep and repaired final-state co-training tier are not yet published.\n'))
   notice_path=root/'WHITEBOX35-UPDATE.md';notice_path.write_text(notice)
   commit=api_call(lambda:api.create_commit(REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo='WHITEBOX35-UPDATE.md',path_or_fileobj=str(notice_path))],commit_message='Index corrected PR35 evidence and invalidate historical missing-head results'))
   revision=commit.oid;remote={x.path:x for x in api_call(lambda:api.get_paths_info(REPO,[prefix+p.name for p in files]+['WHITEBOX35-UPDATE.md'],repo_type='dataset',revision=revision))}
   import hashlib
   for path in files:
    info=remote[prefix+path.name]
    if info.lfs:
     assert info.lfs.sha256==sha(path)
    else:
     data=path.read_bytes();assert info.blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
   data=notice_path.read_bytes();assert remote['WHITEBOX35-UPDATE.md'].blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
   report={'status':'verified','repo':REPO,'revision':revision,'stage':stage,'files':len(files)}
   (root/(stage+'-remote-verification.json')).write_text(json.dumps(report,indent=2)+'\n');record('stage_published',**{k:v for k,v in report.items() if k!='status'})
  record('complete')
 except Exception as error:
  record('failed',error=repr(error));raise
if __name__=='__main__':main()
