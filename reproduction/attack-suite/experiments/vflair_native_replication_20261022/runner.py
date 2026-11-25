"""Pinned native VFLAIR admission and orchestration. No local replacement models."""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

from import_compat import prepare


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False, default=str) + '\n')


def event(name, **v):
    print(json.dumps({'event': name, 'time_unix': time.time(), **v}, default=str), flush=True)


def verified_sources(repo, cfg):
    root = repo / cfg['upstream_root']
    mp = repo / cfg['upstream_manifest']
    manifest = json.loads(mp.read_text())
    if manifest['revision'] != cfg['upstream_revision']:
        raise RuntimeError('Upstream revision mismatch')
    checks = []
    for f in manifest['files']:
        p = root / f['path']
        actual = sha(p)
        if actual != f['sha256'] or p.stat().st_size != f['bytes']:
            raise RuntimeError(f'Pinned upstream file mismatch: {p}')
        checks.append({'path': f['path'], 'sha256': actual, 'bytes': p.stat().st_size})
    return root, {'revision': manifest['revision'], 'manifest_sha256': sha(mp), 'files': checks}


def inventory(cfg, out):
    import torch
    versions = {}
    for name in ['torch','transformers','peft','accelerate','datasets','numpy','pandas','pyarrow','scipy','scikit-learn','protobuf','tensorboard']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'versions': versions,
              'disk_free_bytes': shutil.disk_usage(out).free, 'cuda_available': torch.cuda.is_available(),
              'torch_cuda': torch.version.cuda, 'gpus': []}
    if result['cuda_available']:
        for n in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(n)
            result['gpus'].append({'index': n, 'name': p.name, 'memory_bytes': p.total_memory, 'capability': [p.major,p.minor]})
    write(out/'environment.json', result)
    return result


def check_inputs(cfg, out):
    host = cfg['host']
    result = {'model': None, 'dataset': None, 'ready': False, 'errors': []}
    if host.get('model_path'):
        model = Path(host['model_path']).expanduser().resolve()
        files = [p for p in model.iterdir() if p.is_file()] if model.is_dir() else []
        selected = [p for p in files if p.suffix in ['.json','.txt','.model','.bin','.safetensors']]
        result['model'] = {'path': str(model), 'revision': host.get('model_revision'),
                           'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(selected)]}
        if not (model/'config.json').is_file() or not any(p.suffix in ['.bin','.safetensors'] for p in files):
            result['errors'].append('Model config or weights are missing')
        if not host.get('model_revision'):
            result['errors'].append('Model revision is not pinned')
    else:
        result['errors'].append('host.model_path is unset')
    dm = host.get('dataset_manifest')
    if dm and Path(dm).expanduser().is_file():
        import pyarrow.parquet as pq
        dmp = Path(dm).expanduser().resolve()
        data = json.loads(dmp.read_text())
        ds = {'manifest_path': str(dmp), 'manifest_sha256': sha(dmp), 'revision': data.get('revision'), 'splits': {}}
        for split in ['train','test']:
            checked = []
            for item in data.get('splits',{}).get(split,[]):
                p = Path(item['path']).expanduser().resolve()
                h = sha(p)
                if h != item['sha256']:
                    raise RuntimeError(f'Data hash mismatch: {p}')
                parquet = pq.ParquetFile(p)
                columns = parquet.schema_arrow.names
                if not {'text','domain'}.issubset(columns):
                    raise RuntimeError(f'Native Lambada requires text/domain columns: {p}')
                checked.append({'path': str(p), 'sha256': h, 'bytes': p.stat().st_size,
                                'rows': parquet.metadata.num_rows, 'columns': columns})
            ds['splits'][split] = checked
            if not checked:
                result['errors'].append(f'Missing {split} parquet files')
        ds['row_counts'] = {k: sum(x['rows'] for x in v) for k,v in ds['splits'].items()}
        if not data.get('revision'):
            result['errors'].append('Dataset revision is not pinned')
        result['dataset'] = ds
    else:
        result['errors'].append('host.dataset_manifest is unset or absent')
    result['ready'] = not result['errors']
    write(out/'inputs.json', result)
    return result


def native_imports(runtime, out):
    os.chdir(runtime/'src')
    sys.path[:0] = [str(runtime/'src'), str(runtime)]
    try:
        main = importlib.import_module('main_pipeline_llm')
        result = {'ok': True, 'module': main.__file__}
    except Exception as e:
        result = {'ok': False, 'exception': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()}
        main = None
    write(out/'native-imports.json', result)
    event('native_imports', **{k:v for k,v in result.items() if k != 'traceback'})
    return main, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    from bounded_study import assert_launch_authorized
    assert_launch_authorized(cfg)
    out = Path(a.output).resolve()
    repo = Path(__file__).resolve().parents[2]
    source, provenance = verified_sources(repo,cfg)
    write(out/'upstream-verification.json', provenance)
    env = inventory(cfg,out)
    inputs = check_inputs(cfg,out)
    if cfg['phase']=='bounded-study' and inputs['ready']:
        from bounded_study import validate_inputs
        validate_inputs(cfg,inputs)
    runtime = out/'runtime-upstream'
    shutil.copytree(source,runtime)
    prepare(runtime,out/'compatibility.json')
    native, im = native_imports(runtime,out) if cfg.get('native_import_check',True) else (None,{'ok':False,'skipped':True})
    errors = list(inputs['errors'])
    if env['disk_free_bytes'] < cfg['guards']['minimum_free_gib']*2**30:
        errors.append('Insufficient free disk under configured admission guard')
    if not env['cuda_available']:
        errors.append('CUDA device unavailable')
    if not im['ok']:
        errors.append('Native pipeline import did not pass')
    if inputs['ready'] and cfg.get('count_native_windows'):
        from count_inputs import count_native_windows
        counts = count_native_windows(cfg,inputs,out)
        event('native_window_count_complete',**counts)
        if counts['test_over_model_1024']:
            errors.append('Native untruncated test input exceeds GPT2 positional capacity; preserve/report before benchmark admission')
    admission = {'status':'ADMITTED' if not errors else 'BLOCKED', 'errors':errors,
                 'source_files_verified': len(provenance['files']), 'phase':cfg['phase'],
                 'model_loaded':False, 'training_executed':False, 'attack_executed':False}
    write(out/'admission.json',admission)
    event('admission',**admission)
    if cfg['phase'] == 'admission':
        return 0 if not errors else 2
    if errors:
        return 2
    from native_pipeline import execute
    execute(native,cfg,inputs,runtime,out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
