#!/usr/bin/env python3
"""Read local HF worker journals every 30s without spending Hub API quota."""
import datetime
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'build/phase2'
STAGE = ROOT/'build/hf-release-2026-09-17'
SUPPLEMENT = ROOT/'build/hf-audit-supplement-2026-09-17'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def alive(pid_file):
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid, True
    except (OSError, ValueError):
        return None, False


def snapshot():
    log = WORK/'recovery-worker.log'
    lines = log.read_text(errors='replace').splitlines()
    current = None
    papers = {}
    for paper in ('paper-a', 'paper-b'):
        total = len((STAGE/paper/'MANIFEST.sha256').read_text().splitlines())+1
        papers[paper] = {'staged_files': total, 'status': 'bulk_upload_queued',
                         'remaining_at_last_commit': None}
    for line in lines:
        if ' RUN ' in line and 'release_verify_remote.py' in line:
            paper = line.rsplit(' ', 1)[-1]
            if paper in papers:
                papers[paper]['status'] = 'verifying'
                papers[paper]['verification_started_epoch'] = datetime.datetime.fromisoformat(
                    line.split(' RUN ', 1)[0]).timestamp()
        if ' RUN ' in line and 'release_upload.py' in line:
            current = line.rsplit(' ', 1)[-1]
            if current in papers:
                papers[current]['status'] = 'uploading'
        if current not in papers:
            continue
        match = re.search(r'COMMIT ([0-9a-f]+).*remaining=(\d+)', line)
        if match:
            remaining = int(match[2])
            papers[current].update(remaining_at_last_commit=remaining,
                last_commit=match[1], status='uploading',
                file_percent=round(100*(1-remaining/papers[current]['staged_files']), 2))
        if line.startswith('RETRY '):
            papers[current]['status'] = 'retrying_rate_limit' if '429 ' in line else 'retrying_upload'
        if line == 'UPLOAD_DONE':
            papers[current]['status'] = 'uploaded_awaiting_verification'
    for paper in papers:
        report_path = STAGE/(paper+'-remote-verification.json')
        verified = read_json(report_path)
        if verified and report_path.stat().st_mtime >= papers[paper].get('verification_started_epoch', 0):
            papers[paper].update(status='verified' if verified['ok'] else 'verification_failed',
                                 verification_revision=verified['revision'])
    pid, running = alive(WORK/'recovery-worker.pid')
    supplement_pid, supplement_running = alive(WORK/'audit-supplement-worker.pid')
    complete = read_json(STAGE/'release-complete.json')
    supplement_complete = read_json(SUPPLEMENT/'supplement-complete.json')
    baseline_ok = bool(complete) and all(x['ok'] for x in complete)
    supplement_ok = bool(supplement_complete) and all(x['ok'] for x in supplement_complete)
    return {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'worker_pid': pid, 'worker_alive': running, 'papers': papers,
            'log_age_seconds': round(time.time()-log.stat().st_mtime),
            'baseline_verified': baseline_ok, 'supplement_verified': supplement_ok,
            'supplement_worker_pid': supplement_pid, 'supplement_worker_alive': supplement_running,
            'paper_a_audit_verified': bool((read_json(SUPPLEMENT/'paper-a-immediate-verification.json') or {}).get('ok')),
            'attention_required': (not running and not baseline_ok) or
                (not supplement_running and not supplement_ok),
            'scope': 'Local worker reports. File counts are not byte progress or an ETA. Verification status comes from pinned remote digest checks.'}


def main():
    while True:
        state = snapshot()
        path = WORK/'hf-upload-progress.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        os.replace(temp, path)
        lines = ['# HF upload progress', '', 'Updated UTC: '+state['checked_utc'], '',
                 '| Repository | Status | Files remaining at last commit | File progress |',
                 '|---|---|---:|---:|']
        for name, p in state['papers'].items():
            lines.append(f"| {name} | {p['status']} | {p['remaining_at_last_commit'] if p['remaining_at_last_commit'] is not None else 'not yet reported'} | {str(p['file_percent'])+'%' if 'file_percent' in p else 'not yet reported'} |")
        lines += ['', 'Baseline fully verified: '+str(state['baseline_verified']),
                  'Both audit supplements verified: '+str(state['supplement_verified']),
                  'Paper A recovered-evidence audit verified: '+str(state['paper_a_audit_verified']),
                  'Attention required: '+str(state['attention_required']), '', state['scope']]
        (WORK/'hf-upload-progress.md').write_text('\n'.join(lines)+'\n')
        if state['attention_required'] or (state['baseline_verified'] and state['supplement_verified']):
            print(json.dumps(state), flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
