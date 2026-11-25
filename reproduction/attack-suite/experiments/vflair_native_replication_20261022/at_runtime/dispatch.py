"""Run a recorded CPU compatibility reference in the frozen AT environment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def emit(event, **values):
    print(json.dumps({'event': event, 'time_unix': time.time(), **values}, allow_nan=False), flush=True)


def validate(cfg):
    if cfg.get('schema') != 'native-llama-runtime-reference-v1' or cfg.get('phase') != 'cpu-reference':
        raise ValueError('Only the separately labelled CPU reference is admitted by this entry')
    if cfg.get('private_environment') != '/home/geo/.venvs/vflair-native-at-20261022':
        raise ValueError('Reference requires the admitted separate AT environment')
    if cfg.get('dependency_contract_sha256') != '7aba20ed2948b78a1f58bb2a40ba5289be7502b6ec842aed2dd33ad05a9e56ab':
        raise ValueError('Reference dependency contract changed')
    if cfg.get('wall_seconds') != 600 or cfg.get('reference_seconds') != 420 or cfg.get('threads') != 4:
        raise ValueError('CPU reference limits changed')
    output = Path(cfg['output_root'])
    if output.is_absolute() or '..' in output.parts:
        raise ValueError('Output must remain inside the experiment archive')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    validate(cfg)
    if args.validate_only:
        emit('llama_reference_static_config_passed', model_execution=False)
        return
    run_id = os.environ.get('ORX_RUN_ID', '')
    if not re.fullmatch(r'[0-9a-f-]{36}', run_id):
        raise RuntimeError('CPU model references require a recorded ORX run')
    out = REPO / cfg['output_root'] / run_id
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'effective-run.json', cfg)
    started = time.monotonic()
    status, error = 'FAILED', None
    try:
        setup_dir = KIT / 'at_setup'
        sys.path.insert(0, str(setup_dir))
        spec = importlib.util.spec_from_file_location('frozen_at_setup', setup_dir / 'setup.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        setup_cfg = json.loads((setup_dir / 'run.json').read_text())
        contract = setup.validate(setup_cfg)
        if contract['contract_sha256'] != cfg['dependency_contract_sha256']:
            raise RuntimeError('Frozen AT source contract differs from the reference admission')
        env = setup.software_env(setup_cfg)
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        receipt = setup.check_receipt(setup_cfg, contract, env)
        write(out / 'dependency-admission.json', receipt)
        paths = [HERE / 'dispatch.py', HERE / 'entry.sh', HERE / 'run.json',
                 KIT / 'llama_reference_gate.py', KIT / 'llama_runtime_compat.py',
                 KIT / 'import_compat_llama.py']
        write(out / 'reference-source-files.json', [
            {'path': str(path.relative_to(REPO)), 'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in paths])
        command = [str(Path(cfg['private_environment']) / 'bin/python'), '-u',
                   str(KIT / 'llama_reference_gate.py'), '--output', str(out),
                   '--upstream-root', str(setup_dir / 'vendor/upstream')]
        emit('llama_cpu_reference_start', command=command, wall_limit_seconds=cfg['wall_seconds'],
             gpu_visible=False, pretrained_weights=False, corpus_training=False)
        remaining = min(cfg['reference_seconds'], cfg['wall_seconds'] - (time.monotonic() - started))
        if remaining <= 0:
            raise TimeoutError('CPU reference admission exhausted its wall allowance')
        subprocess.run(command, env=env, check=True, timeout=remaining)
        summary = json.loads((out / 'reference-summary.json').read_text())
        if (summary.get('status') != 'PASS_TINY_CPU_NUMERICAL_AND_THREE_STEP_GENERATION'
                or summary.get('checks_passed') != 8
                or len(summary.get('checks', [])) != 8
                or not all(row.get('passed') is True for row in summary['checks'])
                or summary.get('pretrained_weights_or_dataset_used') is not False
                or summary.get('native_party_routing_or_AT_training_validated') is not False):
            raise RuntimeError('CPU reference did not retain the complete declared numerical/generation gate')
        emit('llama_cpu_reference_accepted', checks_passed=summary['checks_passed'],
             reference_summary_sha256=sha(out / 'reference-summary.json'),
             native_party_or_AT_training_validated=False)
        status = 'CPU_REFERENCE_COMPLETED'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        terminal = {'status': status, 'error': error, 'run_id': run_id,
                    'wall_seconds': time.monotonic() - started,
                    'pretrained_model_or_gpu_execution': False}
        write(out / 'terminal.json', terminal)
        emit('llama_cpu_reference_terminal', **terminal)


if __name__ == '__main__':
    main()
