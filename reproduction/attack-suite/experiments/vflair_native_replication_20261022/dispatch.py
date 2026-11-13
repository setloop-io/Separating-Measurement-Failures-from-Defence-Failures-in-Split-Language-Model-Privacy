"""Standard-library launcher; all model work remains in the native child process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def setup_command(command, logpath, timeout):
    """Bounded setup, with both a durable log and immediate stdout progress."""
    print(json.dumps({'event':'setup_command','command':command,'timeout_seconds':timeout}),flush=True)
    with logpath.open('w') as log:
        child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,start_new_session=True)
        def timed_out(_sig,_frame):
            os.killpg(child.pid,signal.SIGKILL)
            raise TimeoutError(f'Setup exceeded {timeout}s: {command[0]}')
        signal.signal(signal.SIGALRM,timed_out)
        signal.alarm(timeout)
        try:
            for line in child.stdout:
                print(line,end='',flush=True)
                log.write(line); log.flush()
            rc = child.wait()
            if rc:
                raise subprocess.CalledProcessError(rc,command)
        finally:
            signal.alarm(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    # Overrides are recorded in the run manifest; freeze run.json for ORX runs.
    p.add_argument('--phase', choices=['admission', 'canary', 'full', 'bounded-study'])
    a = p.parse_args()
    cfgpath = Path(a.config).resolve()
    cfg = json.loads(cfgpath.read_text())
    if a.phase:
        cfg['phase'] = a.phase
    from bounded_study import assert_launch_authorized
    assert_launch_authorized(cfg)
    task = Path(__file__).resolve().parent
    repo = task.parents[1]
    base = Path(os.path.expanduser(cfg['host']['base_python']))
    envdir = Path(os.path.expanduser(cfg['host']['private_environment']))
    envpy = envdir / 'bin/python'
    outbase = repo / cfg['output_root']
    runid = os.environ.get('ORX_RUN_ID') or ('local-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    out = outbase / runid
    out.mkdir(parents=True, exist_ok=False)
    (out / 'effective-run.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (out / 'launcher-provenance.json').write_text(json.dumps({
        'argv': sys.argv, 'run_json_sha256': sha(cfgpath),
        'task_files': {str(x.relative_to(task)): sha(x) for x in task.rglob('*') if x.is_file() and '__pycache__' not in x.parts},
        'run_id': runid, 'started_unix': time.time(), 'repository': str(repo)
    }, indent=2) + '\n')
    if cfg.get('bootstrap'):
        if not base.is_file():
            raise RuntimeError(f'Base Python unavailable: {base}')
        if not envpy.exists():
            subprocess.run([str(base), '-m', 'venv', '--system-site-packages', str(envdir)], check=True)
        # venv-from-venv does not inherit the parent venv's site-packages.
        # Append that directory with a .pth file, leaving private packages first.
        baseinfo = json.loads(subprocess.check_output([str(base), '-c',
            'import json,sysconfig,torch; print(json.dumps({"site":sysconfig.get_path("purelib"),"torch_file":torch.__file__,"version":torch.__version__}))'], text=True))
        private_site = subprocess.check_output([str(envpy), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
        Path(private_site, 'vflair_inherited_cuda.pth').write_text(baseinfo['site'] + '\n')
        writeinfo = out / 'inherited-torch.json'
        writeinfo.write_text(json.dumps(baseinfo, indent=2) + '\n')
        before_private = json.loads(subprocess.check_output([str(envpy), '-c',
            'import json,torch; print(json.dumps({"torch_file":torch.__file__,"version":torch.__version__}))'], text=True))
        if before_private['torch_file'] != baseinfo['torch_file'] or before_private['version'] != baseinfo['version']:
            raise RuntimeError('Private environment does not inherit the exact host CUDA torch')
        # Protect inherited CUDA torch: pip must satisfy this exact version.
        torchver = subprocess.check_output([str(base), '-c', 'import torch; print(torch.__version__)'], text=True).strip()
        constraints = out / 'torch-constraint.txt'
        constraints.write_text('torch===' + torchver + '\n')
        setup_command([str(envpy), '-m', 'pip', 'install', '--disable-pip-version-check',
                       '-c', str(constraints), '-r', str(task / 'requirements-private.txt')],
                      out/'bootstrap.log',int(cfg.get('setup_wall_seconds',1800)))
        after = subprocess.check_output([str(base), '-c', 'import torch; print(torch.__version__)'], text=True).strip()
        if after != torchver:
            raise RuntimeError('Base CUDA torch changed during private bootstrap')
        after_private = json.loads(subprocess.check_output([str(envpy), '-c',
            'import json,torch; print(json.dumps({"torch_file":torch.__file__,"version":torch.__version__}))'], text=True))
        if after_private != before_private:
            raise RuntimeError('Private bootstrap replaced the inherited CUDA torch')
    selected = envpy if envpy.exists() else base
    if not selected.is_file():
        raise RuntimeError(f'Host Python unavailable: {selected}')
    if cfg.get('prepare_inputs'):
        setup_command([str(selected),'-u',str(task/'prepare_inputs.py'),'lambada',
                       '--metadata',str(task/'inputs/lambada-metadata.json'),
                       '--tree',str(task/'inputs/lambada-tree.json'),
                       '--output-dir',str(Path(cfg['host']['dataset_manifest']).parent)],
                      out/'input-preparation.log',int(cfg.get('setup_wall_seconds',1800)))
    command = [str(selected), '-u', str(task / 'runner.py'), '--config', str(out / 'effective-run.json'), '--output', str(out)]
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', MPLBACKEND='Agg',
               CUDA_VISIBLE_DEVICES=str(cfg['gpu']), PYTHONHASHSEED=str(cfg['seed']))
    started = time.monotonic()
    status = 'failed'
    rc = 1
    with (out / 'stdout.log').open('w') as log:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        # A separate process alarm enforces the budget even when stdout is silent.
        def timeout(_sig, _frame):
            nonlocal status
            status = 'wall_budget_exceeded'
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(int(cfg['guards']['wall_seconds']))
        for line in child.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        rc = child.wait()
        signal.alarm(0)
        if rc == 0:
            status = 'complete'
    result = {'status': status, 'exit_code': rc, 'wall_seconds': time.monotonic()-started,
              'command': command, 'output': str(out), 'phase': cfg['phase']}
    (out / 'terminal.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'event': 'native_terminal', **result}), flush=True)
    return rc if rc else (0 if status == 'complete' else 124)


if __name__ == '__main__':
    sys.exit(main())
