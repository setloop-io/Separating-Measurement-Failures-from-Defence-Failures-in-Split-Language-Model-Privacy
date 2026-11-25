"""Publish an additive BiSR evidence/manuscript tier and verify immutable HF digests."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import release_paper_a_revision as publication

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--status',type=Path,required=True)
 ap.add_argument('--tier',choices=['evidence','manuscript'],required=True);a=ap.parse_args()
 publication.REPO='Setloop/llm-attacker-paper-b';publication.PREFIX='fidelity-20260919'
 def state(status,**extra):
  d={'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra}
  publication.save(a.status,d);print(json.dumps(d),flush=True)
 try:
  if a.tier=='evidence':
   if json.loads((a.stage/'INTEGRITY.json').read_text()).get('ok') is not True:raise ValueError('Integrity gate required')
   if json.loads((a.stage/'REPRODUCTION-VERIFIED.json').read_text()).get('ok') is not True:raise ValueError('Verified re-score required')
  report=publication.publish(a.stage,a.tier,state,publication_label="Paper B BiSR conformance")
  publication.save(a.status.with_name(a.status.stem+'-verification.json'),report)
  state('verified',**report)
 except Exception as e:
  state('failed',error_type=type(e).__name__,error=str(e)[:1500]);raise
if __name__=='__main__':main()
