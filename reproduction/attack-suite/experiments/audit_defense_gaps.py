"""Reproduce defense-gap engineering batteries without touching held-out data."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.attacks import load
from attacker.evidence import EvidenceLogger,file_sha256,verify_evidence
from attacker.security_assumptions import assess


def run(output,seeds):
    if output.exists(): raise ValueError('fresh output required')
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('distinct seeds required')
    output.mkdir(parents=True)
    artifact=output/'audit.json'
    result={'schema':'llm_attacker.defense_gap_audit.v1','status':'RUNNING','evidence_tier':'engineering_synthetic',
            'scope_sha256':file_sha256(Path(__file__).with_name('defense_gap_scope.json')),
            'results':[],'security_arguments':[],'limits':['No deployed defender or trained MINE/fragmentation reproduction.',
            'No production TLS/KDF verification; conditions are explicit assertions.',
            'No use of frozen held-out evaluation inputs; no new population scientific claims.']}
    with EvidenceLogger(artifact) as evidence:
        try:
            for seed in seeds:
                for name,mode,extra in [('replica-integrity','training',['--evidence']),('view-privacy','inference',[]),
                   ('representation-privacy','training',['--arms','bottleneck','gram_flattening','private_phase','--nonlinear-probe','--evidence']),
                   ('gradient-hardening','training',['--seeds',str(seed),'--evidence'])]:
                    module=load(name);path=output/f'{name}-{seed}.json'
                    args=module.build_parser().parse_args(['--toy','--quick','--seed',str(seed),'--output',str(path)]+extra)
                    args.mode=mode
                    with contextlib.redirect_stdout(io.StringIO()) as captured: code=module.run(args)
                    (output/f'{name}-{seed}.log').write_text(captured.getvalue())
                    if code not in (None,0):raise ValueError('attack returned nonzero')
                    ok,problems=verify_evidence(str(path)+'.evidence.json')
                    if not ok:raise ValueError(str(problems))
                    row={'attack':name,'mode':mode,'seed':seed,'artifact':path.name,'sha256':file_sha256(path),
                         'manifest_sha256':file_sha256(str(path)+'.evidence.json'),'module_sha256':file_sha256(module.__file__)}
                    result['results'].append(row);evidence.event('completed',row)
                    artifact.write_text(json.dumps(result,indent=2));print(name,seed,'verified',flush=True)
            tls=dict(tls13=True,authenticated_peers=True,uncompromised_keys=True,trusted_implementation=True)
            result['security_arguments']=[assess('tls13','passive_network',tls),assess('tls13','endpoint',tls),
                assess('kdf','passive_seam',dict(standard_hmac_kdf=False,high_entropy_secret=True,domain_separation=True,secret_uncompromised=True)),
                assess('ratchet','passive_seam',dict(separated_epoch_keys=True,enforced_epoch_budget=True,same_key_pooling_only=True,master_retained=True))]
            result['status']='COMPLETE'
        except BaseException as error:
            result.update(status='FAILED',error=f'{type(error).__name__}: {error}');raise
        finally:artifact.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    args=ap.parse_args();run(args.output,args.seeds)
