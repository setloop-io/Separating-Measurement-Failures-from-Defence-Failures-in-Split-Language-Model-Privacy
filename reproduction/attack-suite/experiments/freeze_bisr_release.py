"""Freeze and independently audit a completed BiSR C-A campaign; no GPU work."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import sha256
from experiments.bisr_campaign_score import verified_rows

def read(p): return json.loads(p.read_text())
def save(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def check(ok,msg):
    if not ok: raise ValueError(msg)

def freeze(root, inputs, tokenizer, out):
    out.mkdir(parents=True,exist_ok=False)
    campaign=root/'campaign-v1'; prep=root/'run-v1/checkpoints'
    plan=read(root/'experiments/bisr_full_campaign.v1.json')
    status=read(campaign/'status.json'); result=read(campaign/'analysis/result.json')
    check(status['status']==result['status']=='CONFORMANCE_ESTABLISHED_FOR_DECLARED_C_A','Final verdict')
    check(sha256(campaign/'analysis/result.json')==status['analysis_sha256'],'Analysis digest')
    frozen=read(inputs/'manifest.json'); inputsha=sha256(inputs/'manifest.json')
    check(inputsha==plan['input_manifest_sha256']==result['input_manifest_sha256'],'Input identity')
    for name,h in frozen['outputs'].items(): check(sha256(inputs/name)==h,'Input '+name)
    for name,h in plan['source_sha256'].items(): check(sha256(root/name)==h,'Source '+name)
    check(sha256(root/'environment-packages.json')==plan['environment_sha256'],'Environment')
    check(sha256(root/'metric-gate-v1.json')==plan['metric_gate_sha256'],'Metric gate digest')
    check(read(root/'metric-gate-v1.json')['status']=='METRIC_PARITY_PASS','Metric gate')
    check(sha256(root/'nltk-data/corpora/wordnet.zip')==plan['wordnet_sha256'],'Wordnet')
    prepared=read(prep/'result.json')
    for name,h in prepared['files'].items(): check(sha256(prep/name)==h,'Checkpoint '+name)
    for name,h in prepared['model_files'].items(): check(sha256(tokenizer/name)==h,'Base model '+name)
    launch=read(campaign/'records/launch.json')
    check(sha256(campaign/'records/launch.json')==result['launch_manifest_sha256'],'Launch digest')
    check(sha256(prep/'result.json')==launch['preparation_result_sha256'],'Preparation anchor')
    commit=subprocess.check_output(['git','-C',str(campaign/'records'),'rev-parse','HEAD'],text=True).strip()
    check(commit==status['manifest_commit'],'Manifest commit')
    check(not subprocess.check_output(['git','-C',str(campaign/'records'),'status','--porcelain'],text=True).strip(),'Dirty record repository')
    events=[]; previous='0'*64
    for line in (campaign/'journal.jsonl').read_text().splitlines():
        e=json.loads(line); digest=e.pop('hash')
        check(e['previous']==previous and hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()==digest,'Campaign journal chain')
        previous=digest;events.append(e)
    check(events[-1]['event']=='campaign_complete' and events[-1]['verdict']==status['status'],'Terminal journal event')
    expected={f's{s}-d{d}-{g}' for s in plan['seeds'] for d in plan['depths'] for g in plan['knowledge']}
    check(set(launch['cells'])==expected and status['completed_prediction_cells']==24,'Complete campaign')
    counters=defaultdict(lambda:{'documents':0,'stage_wall_seconds':0.0,'peak_allocated_bytes':0,'layer_token_positions':defaultdict(int)})
    token_identical=0
    for cell,anchor in launch['cells'].items():
        manifest=campaign/'records'/f'{cell}.json'; gate=campaign/'gates'/cell/'result.json'
        check(sha256(manifest)==anchor['manifest_sha256'] and sha256(gate)==anchor['gate_sha256'],'Cell anchor '+cell)
        check(read(gate)['status']=='TRAINED_MECHANICAL_PASS','Trained gate '+cell)
        paired={}
        for engine in plan['engines']:
            rows=verified_rows(campaign/'predictions'/cell/engine,frozen['validation_ids'],anchor['manifest_sha256'])
            paired[engine]=rows
            for row in rows:
                for stage,w in row['work'].items():
                    c=counters[engine+'/'+stage];c['documents']+=1;c['stage_wall_seconds']+=w['wall_seconds']
                    c['peak_allocated_bytes']=max(c['peak_allocated_bytes'],w['peak_allocated_bytes'])
                    for k,n in w['layer_work'].items():c['layer_token_positions'][k]+=n
        token_identical+=sum(a['arms']==b['arms'] for a,b in zip(paired['candidate'],paired['reference']))
    check(len(result['strata'])==4 and all(v['statistical_status']=='EQUIVALENT' and len(v['endpoints'])==8 and all(e['status']=='EQUIVALENT' for e in v['endpoints'].values()) for v in result['strata'].values()),'Equivalence endpoints')
    prepstatus=read(root/'run-v1/status.json')
    report={'schema':'bisr.release_integrity.v1','ok':True,'status':status['status'],'analysis_sha256':status['analysis_sha256'],'input_manifest_sha256':inputsha,'source_commit':status['source_commit'],'manifest_commit':commit,'prediction_cells':24,'prediction_documents':6144,'paired_documents_with_identical_arm_tokens':token_identical,'trained_gates':12,'equivalence_endpoints':32,'completed_unix':events[-1]['time_unix'],'preparation_wall_seconds':prepared['wall_seconds'],'preparation_through_scoring_wall_seconds':events[-1]['time_unix']-prepstatus['started_unix'],'compute_by_engine_stage':dict(counters),'scope':result['scope'],'published_setting_reproduced':False,'compute_caveat':'Stage wall times and peak allocated memory are measured; layer token-position counters are not FLOPs. Shared GM in b/bf counted once. Offline training is separate; total elapsed includes overhead.'}
    save(out/'INTEGRITY.json',report);shutil.copy2(campaign/'analysis/result.json',out/'RESULT.json')
    # Keep every original evidence byte; omit only rebuildable Python caches and transient lock files.
    entries={}
    for prefix,directory in [('run',root),('inputs',inputs),('base-model',tokenizer)]:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and not p.name.endswith('.lock'):
                if prefix=='base-model' and p.name not in prepared['model_files']:continue
                entries[prefix+'/'+p.relative_to(directory).as_posix()]=p
    # Capture source/exclusion references used by the input selector without rewriting the frozen manifest.
    pathmap={}
    for name,v in frozen['source_pins'].items():
        p=Path(v['path'])
        if not p.is_absolute():p=inputs.parents[1]/p
        check(p.is_file() and sha256(p)==v['sha256'],'Input source pin '+name)
        rel='input-provenance/'+name.replace('/','_')+'/'+p.name
        entries[rel]=p;pathmap[v['path']]=rel
    save(out/'INPUT-PATH-MAP.json',pathmap)
    manifest={rel:sha256(p) for rel,p in entries.items()}
    save(out/'FILES.sha256.json',manifest)
    # Split raw tar at 4 GiB; hashes of both archive parts and inner files are published.
    from experiments.release_paper_a_revision import PartWriter
    writer=PartWriter(out)
    try:
        with tarfile.open(fileobj=writer,mode='w|',dereference=True) as tar:
            for rel,p in entries.items():tar.add(p,arcname=rel,recursive=False)
    finally:writer.close()
    for rel,p in entries.items():check(sha256(p)==manifest[rel],'Source changed while archiving '+rel)
    print(json.dumps({'ok':True,'files':len(entries),'bytes':sum(p.stat().st_size for p in entries.values()),'output':str(out)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['root','inputs','tokenizer','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();freeze(a.root,a.inputs,a.tokenizer,a.output)
