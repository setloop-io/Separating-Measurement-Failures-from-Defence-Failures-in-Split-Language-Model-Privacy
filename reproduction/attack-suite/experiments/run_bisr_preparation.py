"""Supervise the authorised checkpoint stage with an exclusive lock and deadline."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('admission','model','inputs','root'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();plan=json.loads(args.admission.read_text())
    if plan['resource_ceiling_hours']!=36 or not plan['checkpoint_preparation_admitted']:
        raise ValueError('Missing approved resource amendment')
    source=Path(__file__).resolve().parents[1]
    for name,digest in plan['source_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Source mismatch: '+name)
    if sha(args.inputs/'manifest.json')!=plan['input_manifest_sha256']:raise ValueError('Wrong inputs')
    for name,digest in plan['model_files'].items():
        if sha(args.model/name)!=digest:raise ValueError('Model mismatch: '+name)
    args.root.mkdir(parents=True,exist_ok=False)
    lock=(Path.home()/'.llm-attacker-bisr-gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    status={'status':'PREPARING_CHECKPOINTS','supervisor_pid':os.getpid(),
            'started_unix':started,'admission_sha256':sha(args.admission),
            'resource_ceiling_hours':36,'full_prediction_started':False}
    def write():
        p=args.root/'status.tmp';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(args.root/'status.json')
    write()
    cmd=[sys.executable,str(source/'experiments/bisr_checkpoint_prepare.py'),'prepare',
        '--model',str(args.model),'--inputs',str(args.inputs),'--output',str(args.root/'checkpoints'),
        '--deadline-seconds',str(plan['preparation_ceiling_hours']*3600-30)]
    try:
        with (args.root/'preparation.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            status['worker_pid']=child.pid;write()
            try:code=child.wait(timeout=plan['preparation_ceiling_hours']*3600)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise TimeoutError('Preparation ceiling reached')
        result=json.loads((args.root/'checkpoints/result.json').read_text())
        if code or result['status']!='CHECKPOINTS_PREPARED_REPLAY_VERIFIED':
            raise RuntimeError('Preparation did not complete successfully')
        status.update(status='CHECKPOINTS_READY_PENDING_TRAINED_GATES',
                      result_sha256=sha(args.root/'checkpoints/result.json'))
    except Exception as exc:
        status.update(status='PREPARATION_FAILED',error=repr(exc))
    finally:
        status['elapsed_seconds']=time.time()-started;write()


if __name__=='__main__':main()
