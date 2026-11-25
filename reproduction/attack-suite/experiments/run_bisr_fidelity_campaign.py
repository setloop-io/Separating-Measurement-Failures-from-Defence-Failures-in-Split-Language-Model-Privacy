"""Wait for preparation, gate trained weights, commit pins, run paired validation.

The worker never retries failed cells or reduces coverage. Predictions run in
separate label-free subprocesses; scoring starts only after all pairs finish.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import sha256,write_predictions
from experiments.bisr_upstream_preflight import source


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','inputs','tokenizer','upstream','root','plan'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--source-commit',required=True)
    args=ap.parse_args();code_root=Path(__file__).resolve().parents[1]
    plan=json.loads(args.plan.read_text())
    for name,digest in plan['source_sha256'].items():
        if sha256(code_root/name)!=digest:raise ValueError('Source changed: '+name)
    if sha256(args.inputs/'manifest.json')!=plan['input_manifest_sha256']:raise ValueError('Wrong inputs')
    if sha256(code_root/'environment-packages.json')!=plan['environment_sha256']:
        raise ValueError('Environment lock changed')
    installed=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True))
    expected_env=json.loads((code_root/'environment-packages.json').read_text())
    if {p['name']:p['version'] for p in installed}!={p['name']:p['version'] for p in expected_env}:
        raise ValueError('Installed package versions differ from the lock')
    if sha256(code_root/'nltk-data/corpora/wordnet.zip')!=plan['wordnet_sha256']:
        raise ValueError('Metric resource changed')
    args.root.mkdir(parents=True,exist_ok=False)
    status={'status':'WAITING_FOR_CHECKPOINTS','pid':os.getpid(),'resource_ceiling_hours':36,
            'source_commit':args.source_commit,'full_prediction_started':False,'completed_prediction_cells':0}
    previous='0'*64
    def event(name,**kw):
        nonlocal previous
        rec={'time_unix':time.time(),'event':name,'previous':previous,**kw}
        previous=hashlib.sha256(json.dumps(rec,sort_keys=True).encode()).hexdigest();rec['hash']=previous
        with (args.root/'journal.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        p=args.root/'status.tmp';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(args.root/'status.json')
        print(json.dumps(rec),flush=True)
    event('waiting_for_preparation')
    def remaining():return deadline-time.time()
    def run(cmd,log,env=None):
        if remaining()<=0:raise TimeoutError('36-hour campaign ceiling reached')
        with log.open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            status['active_child_pid']=child.pid;event('child_started',command=cmd,log=str(log))
            try:ret=child.wait(timeout=remaining())
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise TimeoutError('36-hour campaign ceiling reached')
        status.pop('active_child_pid',None)
        if ret:raise RuntimeError(f'Child exit {ret}: {log}')
    try:
        while True:
            prep_status=json.loads((args.preparation/'status.json').read_text())
            deadline=prep_status['started_unix']+(36-.1)*3600
            if remaining()<=0:raise TimeoutError('Campaign ceiling reached while waiting')
            if prep_status['status']=='PREPARATION_FAILED':raise RuntimeError('Preparation failed; no automatic retry')
            if prep_status['status']=='CHECKPOINTS_READY_PENDING_TRAINED_GATES':break
            time.sleep(min(30,remaining()))
        lock=(Path.home()/'.llm-attacker-bisr-gpu.lock').open('a')
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if remaining()<=0:raise TimeoutError('GPU lock remained busy')
                time.sleep(2)
        checkpoints=args.preparation/'checkpoints'
        prepared=json.loads((checkpoints/'result.json').read_text())
        if prepared['status']!='CHECKPOINTS_PREPARED_REPLAY_VERIFIED':raise ValueError('Unqualified checkpoints')
        if prepared['input_manifest_sha256']!=plan['input_manifest_sha256']:raise ValueError('Wrong checkpoint inputs')
        required={'public.pt','aux-rows-d3.pt','aux-rows-d6.pt'}
        for seed in (31415,27182,16180):
            required.add(f'victim-s{seed}.pt')
            for depth in (3,6):
                required.update(f'{kind}-s{seed}-d{depth}.pt' for kind in ('sip','head','observations'))
        if set(prepared['files'])!=required:raise ValueError('Missing/extra prepared checkpoint files')
        for name,digest in prepared['files'].items():
            if sha256(checkpoints/name)!=digest:raise ValueError('Changed checkpoint: '+name)
        # Independently verified metric gate is required before any prediction.
        metric=json.loads((code_root/'metric-gate-v1.json').read_text())
        if metric['status']!='METRIC_PARITY_PASS' or sha256(code_root/'metric-gate-v1.json')!=plan['metric_gate_sha256']:
            raise ValueError('Metric gate absent or changed')
        records=args.root/'records';records.mkdir()
        refs=args.root/'reference-sources';refs.mkdir()
        reference_paths={}
        for name,path in {'dlg':'sfl/model/attacker/dlg_attacker.py','sma':'sfl/model/attacker/sma_attacker.py',
                          'model':'sfl/utils/model.py','sip':'sfl/model/attacker/sip/inversion_models.py'}.items():
            p=refs/(name+'.py');p.write_text(source(args.upstream,path))
            if sha256(p)!=plan['upstream_source_sha256'][name]:raise ValueError('Reference source changed')
            reference_paths[name]={'path':str(p),'sha256':sha256(p)}
        frozen=json.loads((args.inputs/'manifest.json').read_text())
        def asset(name):return {'path':str(checkpoints/name),'sha256':prepared['files'][name]}
        launch={'schema':'bisr.complete_campaign.v1','input_manifest_sha256':plan['input_manifest_sha256'],
                'source_commit':args.source_commit,'source_sha256':plan['source_sha256'],
                'preparation_result_sha256':sha256(checkpoints/'result.json'),
                'resource_ceiling_hours':36,'deadline_unix':deadline,'cells':{},
                'metric_gate_sha256':plan['metric_gate_sha256'],'environment_sha256':plan['environment_sha256']}
        status['status']='TRAINED_CHECKPOINT_GATES';event('all_checkpoints_verified')
        for seed in (31415,27182,16180):
            for depth in (3,6):
                for grant in ('public','exact'):
                    key=f's{seed}-d{depth}-{grant}'
                    m={'schema':'bisr.campaign_prediction.v1','public_model':asset('public.pt'),
                        'inverter':asset(f'sip-s{seed}-d{depth}.pt'),
                        'observations':asset(f'observations-s{seed}-d{depth}.pt'),
                        'head':asset(f'head-s{seed}-d{depth}.pt') if grant=='exact' else None,
                        'boundary':depth,'knowledge':grant,'sources':reference_paths,
                        'source_sha256':plan['source_sha256'],'documents':256}
                    path=records/(key+'.json');write_predictions(path,m)
                    gate=args.root/'gates'/key;gate.parent.mkdir(exist_ok=True)
                    run([sys.executable,str(code_root/'experiments/bisr_campaign_predict.py'),
                         '--engine','gate','--manifest',str(path),'--output',str(gate)],args.root/(key+'-gate.log'))
                    gate_result=json.loads((gate/'result.json').read_text())
                    if gate_result['status']!='TRAINED_MECHANICAL_PASS':raise ValueError('Trained gate failed')
                    launch['cells'][key]={'manifest_sha256':sha256(path),'gate_sha256':sha256(gate/'result.json')}
                    event('trained_gate_pass',cell=key)
        write_predictions(records/'launch.json',launch)
        # Separate run-record repository: does not edit the working source branch.
        subprocess.run(['git','init','-q',str(records)],check=True)
        subprocess.run(['git','-C',str(records),'add','.'],check=True)
        subprocess.run(['git','-C',str(records),'-c','user.name=Codex','-c','user.email=codex@localhost',
                        'commit','-q','-m','Freeze shared checkpoints and complete BiSR prediction manifests'],check=True)
        status['manifest_commit']=subprocess.check_output(['git','-C',str(records),'rev-parse','HEAD'],text=True).strip()
        status['status']='MATCHED_PREDICTIONS_RUNNING';status['full_prediction_started']=True
        event('prediction_manifests_committed',commit=status['manifest_commit'])
        for key in launch['cells']:
            for engine in ('candidate','reference'):
                output=args.root/'predictions'/key/engine;output.parent.mkdir(parents=True,exist_ok=True)
                status.update(active_cell=key,active_engine=engine)
                run([sys.executable,str(code_root/'experiments/bisr_campaign_predict.py'),
                     '--manifest',str(records/(key+'.json')),'--engine',engine,'--output',str(output)],
                    args.root/(key+'-'+engine+'.log'))
                result=json.loads((output/'result.json').read_text())
                if result['status']!='PREDICTIONS_COMPLETE' or result['documents']!=256:raise ValueError('Incomplete cell')
                status['completed_prediction_cells']+=1;event('prediction_cell_complete',cell=key,engine=engine)
        status['status']='SCORING_COMPLETE_PAIRED_SET';event('all_predictions_complete')
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
        run([sys.executable,str(code_root/'experiments/bisr_campaign_score.py'),'--campaign',str(args.root),
             '--inputs',str(args.inputs),'--tokenizer',str(args.tokenizer),'--output',str(args.root/'analysis')],
            args.root/'analysis.log',env)
        analysis=json.loads((args.root/'analysis/result.json').read_text())
        status['status']=analysis['status'];status['analysis_sha256']=sha256(args.root/'analysis/result.json')
        event('campaign_complete',verdict=analysis['status'])
    except Exception as exc:
        status['status']='CAMPAIGN_STOPPED';status['error']=repr(exc);event('stopped',error=repr(exc))


if __name__=='__main__':main()
