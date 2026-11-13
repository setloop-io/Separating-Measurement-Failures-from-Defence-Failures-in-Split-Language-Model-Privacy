#!/usr/bin/env python3
"""Wait for an existing corrected grid, then verify and analyse it once.

Run from the isolated checkout with its existing Python environment. This
worker does not launch experiments, retry failed cells, or accept partial grids.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--pid', type=int, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    status = root.parent / (root.name + '.analysis-status.json')

    def record(state, **extra):
        value = {'status': state, 'grid': str(root), 'driver_pid': args.pid,
                 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
        status.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    record('waiting_for_grid')
    deadline = time.monotonic() + 24 * 3600
    try:
        while not (root / 'grid-complete.json').exists():
            if list(root.glob('*.failure.json')):
                raise RuntimeError('Grid has preserved failed cells; analysis blocked')
            if time.monotonic() > deadline:
                raise TimeoutError('Grid did not finish within 24 hours')
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError:
                raise RuntimeError('Driver exited without a complete grid')
            time.sleep(30)
        output = root.parent / (root.name + '.analysis.json')
        record('verifying_and_analysing')
        subprocess.run([sys.executable, '-m', 'experiments.analyze_p6_corrected',
                        '--root', str(root), '--output', str(output)], check=True)
        record('complete', analysis=str(output))
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
