import json,os,sys,shutil
from pathlib import Path
sys.path.insert(0,'/home/geo/paper-a-p6-release-20260920/code/experiments')
import release_paper_a_revision as release
root=Path('/home/geo/qwen-release-20260921')
a=json.loads((root/'archive/ARCHIVE.json').read_text())
r=json.loads((root/'supplement/cpu/REPRODUCTION.json').read_text())
assert a['status']=='ARCHIVED_PENDING_CPU_REPRODUCTION'
assert r['status']=='PASS' and r['verified_cells']==200
assert a['raw_inventory_sha256']==r['raw_inventory_sha256']
assert a['analysis_sha256']==r['reference_analysis_sha256']
stage=root/'evidence-stage';stage.mkdir(exist_ok=True)
for item in a['parts']:
 src=root/'archive'/item['path'];assert src.stat().st_size==item['bytes'];
 if not (stage/item['path']).exists(): os.link(src,stage/item['path'])
for n in ['ARCHIVE.json','RAW_FILES.jsonl']:
 if not (stage/n).exists(): os.link(root/'archive'/n,stage/n)
for n in ['metadata','cpu']:shutil.copytree(root/'supplement'/n,stage/n,dirs_exist_ok=True)
shutil.copytree(root/'supplement/collected',stage/'primary',dirs_exist_ok=True)
shutil.copy2(stage/'metadata/README.md',stage/'README.md')
release.PREFIX='qwen-fp64-20260921'
def state(status,**kw):release.save(root/'publisher-status.json',{'status':status,**kw})
state('HASHING_AND_PUBLISHING',archive_bytes=sum(x['bytes'] for x in a['parts']))
try:
 report=release.publish(stage,'evidence',state,'Qwen FP64 complete qualified evidence')
 release.save(root/'publisher-verification.json',report)
 state('COMPLETE_PINNED_HF_VERIFIED',**report)
except Exception as e:
 state('FAILED',error_type=type(e).__name__)
 raise
