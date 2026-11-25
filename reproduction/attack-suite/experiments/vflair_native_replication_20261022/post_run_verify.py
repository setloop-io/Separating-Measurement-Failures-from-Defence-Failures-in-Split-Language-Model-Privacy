"""Include retained-output verification in bounded-study ORX success."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['phase'] != 'bounded-study':
        return
    from bounded_study import assert_launch_authorized
    assert_launch_authorized(cfg)
    run_id = os.environ.get('ORX_RUN_ID', '')
    if not re.fullmatch(r'[0-9a-f-]{36}', run_id):
        raise RuntimeError('Bounded-study verification requires its recorded ORX run')
    kit = Path(__file__).resolve().parent
    out = kit.parents[1] / cfg['output_root'] / run_id
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    command = [str(Path(cfg['host']['private_environment']) / 'bin/python'),
               str(kit / 'verify_run.py'), '--output', str(out)]
    subprocess.run(command, env=env, check=True, timeout=600)
    result = json.loads((out / 'verification.json').read_text())
    if (result.get('status') != 'PASS' or result.get('phase') != 'bounded-study'
            or result.get('utility_rows_verified') != 5153
            or result.get('prediction_rows_verified') != 400
            or result.get('capture_files_verified') != 400):
        raise RuntimeError('Full bounded-study retained-evidence verification did not pass')
    print(json.dumps({'event': 'bounded_retained_evidence_verified',
                      'run_id': run_id, 'status': result['status'],
                      'model_or_gpu_execution_by_verifier': False}), flush=True)


if __name__ == '__main__':
    main()
