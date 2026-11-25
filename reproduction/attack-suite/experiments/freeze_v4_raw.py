"""Archive completed v4 raw evidence and verify every member by streaming extraction."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
 source=a.root/'evidence';stage=a.root/'stage';stage.mkdir(exist_ok=True)
 for host in ('toula','koula'):
  r=json.loads((source/'predictions'/host/'result.json').read_text());assert r['status']=='HOST_PREDICTIONS_COMPLETE' and len(r['cells'])==40
 assert json.loads((source/'preparation'/'result.json').read_text())['gate_count']==80
 entries={str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file()}
 files={rel:sha(p) for rel,p in sorted(entries.items())};(stage/'RAW-FILES.sha256.json').write_text(json.dumps(files,indent=2)+'\n')
 archive=stage/'raw-evidence.tar.zst'
 with archive.open('xb') as out:
  z=subprocess.Popen(['zstd','-T4','-5','--long=27','-q','-c'],stdin=subprocess.PIPE,stdout=out)
  with tarfile.open(fileobj=z.stdin,mode='w|',dereference=True) as tar:
   for rel,p in sorted(entries.items()):tar.add(p,arcname=rel,recursive=False)
  z.stdin.close()
  if z.wait():raise RuntimeError('Compression failed')
 z=subprocess.Popen(['zstd','-q','-dc',str(archive)],stdout=subprocess.PIPE);seen=set()
 with tarfile.open(fileobj=z.stdout,mode='r|') as tar:
  for member in tar:
   if not member.isfile() or member.name not in files or member.name in seen:raise ValueError('Archive coverage')
   h=hashlib.sha256();stream=tar.extractfile(member)
   for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
   if h.hexdigest()!=files[member.name]:raise ValueError('Archive digest: '+member.name)
   seen.add(member.name)
 # Drain any stream padding before waiting for zstd.
 while z.stdout.read(8<<20):pass
 if z.wait() or seen!=set(files):raise ValueError('Incomplete stream verification')
 result={'ok':True,'members':len(seen),'uncompressed_bytes':sum(p.stat().st_size for p in entries.values()),'compressed_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'manifest_sha256':sha(stage/'RAW-FILES.sha256.json'),'method':'Every extracted member hashed against frozen source inventory; no filesystem extraction required','comparative_analysis_verified':False}
 (stage/'RAW-ARCHIVE-VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
