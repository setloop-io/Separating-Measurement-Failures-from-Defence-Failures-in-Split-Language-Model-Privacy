"""Read imported module versions and duplicate distributions after a run.

This is a post-run environment inspection, not instrumentation of the original
process. It explains shadowed packages without modifying frozen evidence.
"""
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import socket
import sys
from datetime import datetime, timezone


def main():
    names = ['torch', 'transformers', 'scipy', 'huggingface_hub']
    imported = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__)
        imported[name] = {'version': module.__version__, 'path': str(path),
                          'module_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    installed = []
    for dist in metadata.distributions():
        name = dist.metadata.get('Name', '').lower().replace('-', '_')
        if name in names:
            installed.append({'name': name, 'version': dist.version, 'root': str(dist.locate_file(''))})
    torch = importlib.import_module('torch')
    print(json.dumps({'inspection': 'post-run; same interpreter and task environment',
                      'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                      'hostname': socket.gethostname(), 'python': platform.python_version(),
                      'executable': sys.executable, 'imported': imported,
                      'installed_distributions_in_search_order': installed,
                      'torch_cuda': torch.version.cuda,
                      'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                      'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
