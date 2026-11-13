#!/usr/bin/env python3
"""Publish isolated Paper B manuscript/metadata and verified Phase 2 raw tiers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import release_paper_a_revision as publication
from release_upload import hashes
from attacker.evidence import verify_evidence


def verify_raw(root):
    checked=0
    for node in ('soula','voula','koula'):
        directory=root/node
        state=json.loads((directory/'status.json').read_text())
        if state['status']!='complete' or state['failed']:raise ValueError('Incomplete '+node)
        for p in directory.rglob('*.evidence.json'):
            ok,errors=verify_evidence(str(p))
            if not ok:raise ValueError(str(errors))
        for p in directory.glob('*/experiment.json'):
            experiment=json.loads(p.read_text())
            if experiment['status']!='COMPLETE':raise ValueError('Incomplete experiment')
            for item in experiment['files']:
                f=p.parent/item['path']
                if not f.is_file() or hashes(f)[0]!=item['sha256']:raise ValueError('Raw digest mismatch: '+str(f))
                checked+=1
        for p in directory.rglob('generation.json'):
            for rel,want in json.loads(p.read_text())['files'].items():
                if hashes(p.parent/rel)[0]!=want:raise ValueError('Synthetic bundle mismatch')
                checked+=1
    return checked


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tier',choices=['manuscript','phase2-raw']);ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--status',type=Path,required=True);ap.add_argument('--raw',type=Path)
    ap.add_argument('--wait-pid',type=int,action='append',default=[]);args=ap.parse_args()
    def state(status,**extra):
        record={'status':status,'utc':datetime.now(timezone.utc).isoformat(),'tier':args.tier,**extra}
        publication.save(args.status,record);print(json.dumps(record),flush=True)
    try:
        if args.tier=='phase2-raw':
            if args.raw is None:raise ValueError('--raw required')
            state('waiting_for_copy',pids=args.wait_pid)
            deadline=time.monotonic()+48*3600
            for pid in args.wait_pid:
                while time.monotonic()<deadline:
                    try:os.kill(pid,0)
                    except ProcessLookupError:break
                    time.sleep(30)
                else:raise TimeoutError('Copy still active after 48 hours')
            state('verifying_raw_files')
            checked=verify_raw(args.raw)
            args.stage.mkdir(parents=True,exist_ok=True)
            if list(args.stage.glob('evidence.tar.part*')):raise ValueError('Existing parts require inspection')
            state('archiving',verified_source_references=checked)
            writer=publication.PartWriter(args.stage)
            try:
                with tarfile.open(fileobj=writer,mode='w|') as archive:
                    archive.add(args.raw,arcname='phase2')
            finally:writer.close()
            (args.stage/'README.md').write_text(
                '# Verified Phase 2 raw evidence\n\nCompleted soula/voula/koula queues only; sasa excluded.\n'
                'Canaries are retained as engineering records, excluded from reported scientific summaries.\n'
                'All experiment-listed binary/metadata digests and synthetic bundle digests were checked.\n'
                f'Checked source references: {checked}.\n\n'
                'Verify MANIFEST.sha256, then `cat evidence.tar.part* | tar -xf -`.\n'
                'Original evidence paths and bytes are preserved. The sibling manuscript tier includes\n'
                'the frozen source, protocol, inputs and descriptive reanalysis instructions.\n')
        publication.REPO='Setloop/llm-attacker-paper-b'
        publication.PREFIX='revision-20260917'
        state('publishing')
        report=publication.publish(args.stage,args.tier,state)
        publication.save(args.status.with_name(args.status.stem+'-verification.json'),report)
        state('verified',**report)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__,error=str(exc)[:1000]);raise

if __name__=='__main__':main()
