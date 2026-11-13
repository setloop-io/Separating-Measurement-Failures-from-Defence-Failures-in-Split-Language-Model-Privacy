"""Publish and verify the additive v4 adaptation evidence/manuscript supplement."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
import release_paper_a_revision as publication

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--tier',choices=['evidence','manuscript'],required=True);p.add_argument('--status',type=Path,required=True);a=p.parse_args()
 publication.REPO='Setloop/llm-attacker-paper-b';publication.PREFIX='v4-adaptation-20260920'
 if a.tier=='evidence':
  for name in ('INTEGRITY.json','REPRODUCTION-VERIFIED.json','DEPENDENCIES-VERIFIED.json','RAW-ARCHIVE-VERIFIED.json'):
   if json.loads((a.stage/name).read_text()).get('ok') is not True:raise ValueError('Unpassed gate: '+name)
  if json.loads((a.stage/'REPRODUCTION-VERIFIED.json').read_text()).get('analysis_reproduced') is not True:raise ValueError('Full CPU analysis reproduction required')
 def state(status,**extra):
  record={'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra};publication.save(a.status,record);print(json.dumps(record),flush=True)
 try:
  report=publication.publish(a.stage,a.tier,state,publication_label='Paper B single-seam adapted comparison')
  publication.save(a.status.with_name(a.status.stem+'-verification.json'),report);state('VERIFIED',**report)
 except Exception as e:
  state('FAILED',error_type=type(e).__name__,error=str(e)[:800]);raise
if __name__=='__main__':main()
