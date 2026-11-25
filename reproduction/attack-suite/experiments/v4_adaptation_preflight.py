"""Read-only fleet admission audit; never launches or scores candidate attacks."""
import hashlib
import json
import platform
import socket
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    host=socket.gethostname().split('.')[0].lower()
    plan=json.loads((ROOT/'experiments/v4_adaptation.v1.json').read_text())
    expected=json.loads((ROOT/'experiments/v4_adaptation_expected.json').read_text())
    report={'host':host,'utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
            'plan_sha256':digest(ROOT/'experiments/v4_adaptation.v1.json'),'candidate_predictions_generated':False,
            'checks':{},'files':[],'full_launch_admitted':False}
    checks=report['checks']
    checks['original_v4_unchanged']=digest(ROOT/'experiments/validation_protocol.v4.json')==plan['original_v4_sha256']
    checks['fleet_host']=host in plan['fleet']
    checks['matrix_complete']=len(plan['matrix']['victim_seeds'])*len(plan['matrix']['depths'])*len(plan['matrix']['knowledge'])*len(plan['matrix']['attack_seeds'])==80
    checks['expected_binary_coverage']=set(expected)=={f'{kind}-{seed}-{depth}.pt' for kind in ('public','exact','observations') for seed in plan['matrix']['victim_seeds'] for depth in (3,6)}
    checks['scoring_complete_set']=plan['statistics']['bootstrap_replicates']==100000 and plan['statistics']['bootstrap_seed']==93281
    if host in ('toula','koula'):
        base=Path('/home/geo/llm-attacker-pilots/gpt2-wikitext2-v01/code/results')
        shards=range(8) if host=='toula' else range(8,16)
        for name,sha in sorted(expected.items()):
            paths=[base/f'validation-heldout-v3-r3-shard{i}'/name for i in shards]
            path=next((p for p in paths if p.is_file()),None)
            got=digest(path) if path else None
            report['files'].append({'name':name,'path':str(path) if path else None,'expected':sha,'actual':got,'pass':got==sha})
        checks['original_binary_hashes']=all(x['pass'] for x in report['files'])
        py=Path('/home/geo/llm-attacker/.venv/bin/python')
        env=subprocess.run([str(py),'-c','import json,torch,transformers;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.cuda.is_available())))'],capture_output=True,text=True)
        report['environment']=env.stdout.strip();checks['torch_environment']=env.returncode==0
        report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    else:
        # CPU hosts audit the frozen declarations now; binary/scoring admission
        # remains pending until their specific data dependencies are staged.
        report['pending']=['Stage data for '+plan['fleet'].get(host,{}).get('role','unknown')]
    report['status']='PREFLIGHT_PASS' if all(checks.values()) else 'PREFLIGHT_FAILED'
    Path('preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':
    from analyze_v4_adaptation import analyze
    analyze(Path('/home/geo/v4-analysis-data-20260920/predictions'), Path.cwd()/'analysis')
