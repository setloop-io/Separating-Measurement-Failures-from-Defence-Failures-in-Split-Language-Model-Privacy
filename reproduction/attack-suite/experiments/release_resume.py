"""Resume the frozen release after a reviewed remote cleanup; never restage."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'build/hf-release-2026-09-17'


def main():
    os.chdir(ROOT)
    env = {**os.environ, 'HF_HUB_DISABLE_XET': '1',
           'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'PYTHONUNBUFFERED': '1'}
    reports = []
    for paper in ('paper-a', 'paper-b'):
        scripts = ['release_verify_remote.py'] if paper == 'paper-a' else [
            'release_upload.py', 'release_verify_remote.py']
        for script in scripts:
            cmd = [sys.executable, 'experiments/'+script, paper]
            print(time.strftime('%Y-%m-%dT%H:%M:%S'), 'RUN', *cmd, flush=True)
            subprocess.run(cmd, env=env, check=True)
        report = json.loads((STAGE/(paper+'-remote-verification.json')).read_text())
        if not report['ok']:
            raise RuntimeError('Remote verification failed: '+paper)
        reports.append(report)
    target = STAGE/'release-complete.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(reports, indent=2)+'\n')
    os.replace(tmp, target)
    print('RELEASE_COMPLETE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
