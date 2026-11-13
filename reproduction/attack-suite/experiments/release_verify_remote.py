#!/usr/bin/env python3
"""Verify every staged byte against HF Git/LFS object digests at a pinned commit."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi
from release_upload import inventory, hashes, matches


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('paper',choices=['paper-a','paper-b'])
    ap.add_argument('--root',type=Path,default=Path('build/hf-release-2026-09-17'))
    args=ap.parse_args()
    root=args.root/args.paper
    repo='Setloop/llm-attacker-'+args.paper
    api=HfApi(); info=api.repo_info(repo,repo_type='dataset')
    remote={e.path:e for e in api.list_repo_tree(repo,repo_type='dataset',recursive=True,revision=info.sha) if hasattr(e,'blob_id')}
    files=inventory(root)
    missing=sorted(set(files)-set(remote))
    # An explicitly staged audit supplement may land before the large baseline.
    # Accept only its known paths whose remote bytes match the reviewed staging.
    supplement_root=args.root.parent/'hf-audit-supplement-2026-09-17'/args.paper
    supplement=inventory(supplement_root) if supplement_root.is_dir() else {}
    if set(supplement) & set(files):raise ValueError('Supplement overlaps baseline')
    verified_supplement=[r for r,p in supplement.items()
                         if r in remote and matches(remote[r],hashes(p))]
    extra=sorted(set(remote)-set(files)-set(verified_supplement)-{'.gitattributes'})
    mismatch=[r for r,p in files.items() if r in remote and not matches(remote[r],hashes(p))]
    manifest={}
    for line in (root/'MANIFEST.sha256').read_text().splitlines():
        digest,rel=line.split('  ',1);manifest[rel]=digest
    manifest_errors=sorted(set(manifest) ^ (set(files)-{'MANIFEST.sha256'}))
    manifest_errors += [r for r,d in manifest.items() if r in files and hashes(files[r])[0]!=d]
    report=dict(repo=repo,revision=info.sha,private=info.private,staged_files=len(files),remote_files=len(remote),missing=missing,extra=extra,mismatch=mismatch,manifest_errors=manifest_errors)
    report['separately_verified_supplement_files']=verified_supplement
    report['ok']=not any((missing,extra,mismatch,manifest_errors))
    (args.root/(args.paper+'-remote-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing','extra','mismatch','manifest_errors')},indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__=='__main__':main()
