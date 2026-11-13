"""Verify and package corrected evidence, with bounded-size archive parts."""
import argparse,itertools,json,shutil,tarfile
from pathlib import Path
from experiments.whitebox35.rerun import check_cell,sha
ROOT=Path(__file__).resolve().parents[2]

class Parts:
 def __init__(self,root):self.root=root;self.index=0;self.used=0;self.f=None
 def write(self,data):
  total=len(data)
  while data:
   if self.f is None:self.f=(self.root/f'artifacts.tar.gz.part{self.index:03d}').open('xb')
   n=min(len(data),4_000_000_000-self.used);self.f.write(data[:n]);self.used+=n;data=data[n:]
   if self.used==4_000_000_000:self.f.close();self.f=None;self.used=0;self.index+=1
  return total
 def flush(self):
  if self.f:self.f.flush()
 def close(self):
  if self.f:self.f.close()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
 ap.add_argument('--stage',choices=['original36','full'],default='full');ap.add_argument('--output',type=Path)
 args=ap.parse_args();root=args.root.resolve();p=json.loads((ROOT/'experiments/whitebox35/protocol.json').read_text())
 output=args.output or ROOT/f'results/whitebox35-release-{args.stage}'
 output.mkdir(parents=True,exist_ok=False)
 if args.stage=='original36':
  assert json.loads((root/'verified-complete.json').read_text())['corrected_cells']==36
  cellroot=root;depths=p['original_depths']
 else:
  assert json.loads((root/'stable-grid-complete.json').read_text())['cells']==198
  assert json.loads((root/'cotrain-grid-complete.json').read_text())['cells']==99
  cellroot=root/'stable-grid';depths=p['internal_depths']
 rows=[];keys=set()
 for path in cellroot.glob('whitebox-e2e/split*/seed-*/*/cell.json'):
  data=json.loads(path.read_text())
  if data['status']=='NOT_EVALUABLE':continue
  d=check_cell(path);a=d['attacks']['adaptive_e2e'];key=(d['split'],d['seed'],d['method'],d['dim'])
  assert key not in keys;keys.add(key)
  rows.append({'depth':d['split'],'seed':d['seed'],'method':d['method'],'dim':d['dim'],'accuracy':a['snap']['accuracy'],'n_tokens':a['snap']['n_tokens'],'optimization_work':a['optimization']['work'],'readout_work':a['snap']['work'],'cell_sha256':sha(path)})
 expected=set(itertools.product(depths,p['seeds'],p['methods'],p['dims']));assert keys==expected
 cotrain=[]
 if args.stage=='full':
  from experiments.whitebox35.cotrain import verify_cell
  for path in (root/'cotrain-final-inference').glob('*/cell.json'):
   d=verify_cell(path.parent);e=d['entry'];a=d['attack'];cotrain.append({'depth':e['split'],'seed':e['seed'],'dim':e['dim'],'accuracy':a['accuracy'],'n_tokens':a['n_tokens'],'work':a['work'],'cell_sha256':sha(path)})
  assert {(d['depth'],d['seed'],d['dim']) for d in cotrain}==set(itertools.product(depths,p['seeds'],p['dims']))
 summary={'stage':args.stage,'status':'verified','stable_cells':rows,'cotrain_final_inference':cotrain,'historical_cotrain_exclusions':18,'depth12':'not part of internal-boundary grid','scope':'Descriptive corrected attack evidence. Original missing-head results invalid. New cotrain tier changes phase and grants final head; do not pool with historical public-head training observations.'}
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 for name in ['protocol.json'] :shutil.copy2(ROOT/'experiments/whitebox35'/name,output/name)
 readme='''# Corrected PR35 white-box evidence\n\nThe original 36 adaptive-e2e results omitted the client head and are invalid. This release uses token embeddings -> client head -> true encoder -> observed wire. Recorded head forward/backward work is checked for every complete cell. Historical outputs are retained at their source.\n\nThe original36 tier reuses unchanged original captures/checkpoints at depths3/6, seeds11/22/33, PCA/frozen bottlenecks64/128/256. The full tier covers internal GPT2 boundaries1–11 with the same seeds/dimensions. Depth12 is excluded, and per-depth canaries are excluded from scientific summaries.\n\nThe18 historical co-trained training cells remain NOT_EVALUABLE. Any new cotrain-final-inference records use a matched final victim head and final encoder on newly captured held-out inference observations. This changes the observation phase and grants stronger access; these results must remain separate.\n\nVerify each file against manifest.json, concatenate artifacts.tar.gz.part* in lexical order, then extract the resulting gzip tar archive. Archives preserve project-relative paths and include code, pinned corpus, checkpoints, captures, provenance, journals and measured work. Install from archived pyproject.toml using requirements-research.lock. Use the pinned GPT2 revision in protocol.json for fresh captures.\n\nTo regenerate corrected36 use `python -m experiments.run_whitebox_sweep --root results/whitebox35-original36-v1 --adaptive-e2e --device cuda --threads 4` on a disposable copy (the legacy runner overwrites its output tier). To regenerate fresh layers use commands in the full tier journal. Co-trained CLI is `python -m experiments.whitebox35.cotrain --help`. Hash-only cell verification is available as verify_cell in cotrain.py.\n\nSource code Apache-2.0. Corpus/model assets retain upstream licences. Repositories remain private. No global privacy guarantee is asserted.\n'''
 (output/'README.md').write_text(readme)
 paths=[root,ROOT/'attacker',ROOT/'experiments',ROOT/'pyproject.toml',ROOT/'requirements-research.lock',ROOT/'LICENSE',ROOT/p['corpus_path']]
 if (ROOT/'whitebox35-source-commit.txt').exists():paths.append(ROOT/'whitebox35-source-commit.txt')
 def filt(info):
  return None if '__pycache__' in info.name.split('/') or info.name.endswith('.pyc') else info
 parts=Parts(output)
 try:
  with tarfile.open(fileobj=parts,mode='w|gz',compresslevel=1,dereference=True) as archive:
   for path in paths:archive.add(path,arcname=str(path.relative_to(ROOT)),filter=filt)
 finally:parts.close()
 files={x.name:{'sha256':sha(x),'bytes':x.stat().st_size} for x in sorted(output.iterdir()) if x.is_file()}
 (output/'manifest.json').write_text(json.dumps({'stage':args.stage,'files':files,'scope':summary['scope']},indent=2)+'\n')
 (output/'PACKAGE-READY.json').write_text(json.dumps({'status':'verified','manifest_sha256':sha(output/'manifest.json')})+'\n')
 print(output,flush=True)
if __name__=='__main__':main()
