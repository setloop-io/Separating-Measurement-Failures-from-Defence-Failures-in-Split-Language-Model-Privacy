"""Apply reviewed reproduction/attribution additions, then verify the complete HF tier."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release_upload import inventory,hashes,matches
from huggingface_hub import HfApi,CommitOperationAdd

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['stage','amendments','previous_verification','output']:ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();prior=json.loads(a.previous_verification.read_text())
 if prior.get('ok') is not True:raise ValueError('Initial complete verification required')
 repo=prior['repo'];prefix=prior['prefix'];api=HfApi();info=api.repo_info(repo,repo_type='dataset')
 before=inventory(a.stage)
 pinned={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',path_in_repo=prefix,revision=info.sha,recursive=True) if hasattr(x,'blob_id')}
 for rel,p in before.items():
  if prefix+'/'+rel not in pinned or not matches(pinned[prefix+'/'+rel],hashes(p)):raise ValueError('Published tier changed before amendment: '+rel)
 allowed={'REPRODUCE.md','verify_and_reproduce.py','THIRD-PARTY-NOTICES.md','LICENSE-SplitLLM.txt','GPT2-MODEL-CARD.md','WIKITEXT-DATASET-CARD.md'}
 changes=inventory(a.amendments)
 if set(changes)!=allowed:raise ValueError('Unexpected or missing amendment files')
 for rel,p in changes.items():shutil.copy2(p,a.stage/rel)
 files=inventory(a.stage);manifest=a.stage/'MANIFEST.sha256'
 manifest.write_text(''.join(f'{hashes(p)[0]}  {rel}\n' for rel,p in sorted(files.items()) if rel!='MANIFEST.sha256'))
 files=inventory(a.stage);digests={rel:hashes(p) for rel,p in files.items()}
 operations=[CommitOperationAdd(path_in_repo=prefix+'/'+rel,path_or_fileobj=str(files[rel])) for rel in sorted(set(changes)|{'MANIFEST.sha256'})]
 c=api.create_commit(repo,repo_type='dataset',operations=operations,parent_commit=info.sha,commit_message='Complete BiSR C-A reproduction instructions and upstream attribution')
 remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',path_in_repo=prefix,revision=c.oid,recursive=True) if hasattr(x,'blob_id')}
 expected={prefix+'/'+rel for rel in files}
 errors=[rel for rel in files if prefix+'/'+rel not in remote or not matches(remote[prefix+'/'+rel],digests[rel])]
 if errors or set(remote)!=expected:raise ValueError('Final pinned hash/coverage verification failed')
 report={'repo':repo,'revision':c.oid,'prefix':prefix,'files':len(files),'bytes':sum(p.stat().st_size for p in files.values()),'ok':True,'errors':[],'private':info.private,'manifest_sha256':digests['MANIFEST.sha256'][0],'previous_verified_revision':prior['revision'],'method':'Every Git blob/LFS SHA digest and exact prefix coverage verified at immutable revision'}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
