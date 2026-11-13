"""Fresh all-boundary stable-defence captures and corrected PR35 attack cells."""
import argparse,json,random,time
from pathlib import Path
from experiments import run_bottleneck_sweep as base
from experiments import run_whitebox_sweep as wb
from experiments.whitebox35.rerun import check_cell,sha
ROOT=Path(__file__).resolve().parents[2]

def run(args):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from attacker.attacks import seam_inversion as si
 p=json.loads((ROOT/'experiments/whitebox35/protocol.json').read_text())
 corpus=ROOT/p['corpus_path'];assert sha(corpus)==p['corpus_sha256']
 base.assert_development_corpus_disjoint(corpus,ROOT/'experiments/validation_corpus_manifest.json',require_excluded_pilot=True)
 root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
 model_path=Path.home()/('.cache/huggingface/hub/models--gpt2/snapshots/'+p['model_revision'])
 cfg=base.build_parser(['--output',str(root),'--corpus',str(corpus),'--model',str(model_path),'--device',args.device,'--threads','4','--splits',str(args.depth),'--seeds',str(args.seed),'--methods',args.method,'--dims',str(args.dim)])
 for k,v in p['budgets'].items():setattr(cfg,k,v)
 torch.set_num_threads(4);torch.manual_seed(args.seed)
 model=AutoModelForCausalLM.from_pretrained(model_path)
 tokenizer=AutoTokenizer.from_pretrained(model_path)
 docs=[json.loads(x) for x in corpus.read_text().splitlines()]
 partitions=base._partition(docs,cfg.data_seed)
 encoded={k:[torch.tensor([tokenizer.encode(d['text'],add_special_tokens=False)[:cfg.seq_len]],dtype=torch.long) for d in rows] for k,rows in partitions.items()}
 assert all(t.shape==(1,cfg.seq_len) for rows in encoded.values() for t in rows)
 head=root/f'split{args.depth}'/f'public-head-{args.seed}.pt';head.parent.mkdir(parents=True,exist_ok=True)
 if not head.exists():si.save_head_pt(str(head),model,args.depth)
 access=base.matched_access_spec(cfg)
 # Capture/utility only: do not spend or claim unrelated baseline attacks.
 original_baselines=base.seam_baselines;base.seam_baselines=lambda *a,**k:{}
 try:
  control_dir=root/base.cell_path(args.depth,args.seed,'none',0)
  if not control_dir.exists():
   control=base.run_cell(cfg,model,args.depth,args.seed,'none',0,encoded,control_dir,access)
  else:
   control=json.loads((control_dir/'cell.json').read_text())
   from attacker.evidence import verify_evidence
   ok,errors=verify_evidence(str(control_dir/'cell.json.evidence.json'));assert ok,errors
   assert control['status']=='COMPLETE'
  relative=base.cell_path(args.depth,args.seed,args.method,args.dim)
  if (root/relative).exists():raise ValueError('Fresh defended cell required')
  base.run_cell(cfg,model,args.depth,args.seed,args.method,args.dim,encoded,root/relative,access,control_ppl_after=control['utility_after']['perplexity'])
 finally:base.seam_baselines=original_baselines
 del model
 budget={**p['budgets']}
 if args.canary:budget.update(sipit_steps=8,attack_sequences=2)
 entry={'split':args.depth,'seed':args.seed,'method':args.method,'dim':args.dim,'path':relative,'checkpoint':str((Path(relative)/'captures'/wb.CHECKPOINT_NAMES[args.method])),'whitebox_e2e_path':wb.whitebox_e2e_cell_relpath(args.depth,args.seed,args.method,args.dim)}
 attack=wb.build_parser(['--root',str(root),'--model',str(model_path),'--device',args.device,'--threads','4','--adaptive-e2e','--optimize-lr',str(p['optimize_lr'])])
 wb.run_e2e_cell(attack,root,entry,budget)
 result=root/entry['whitebox_e2e_path']/'cell.json';check_cell(result)
 (result.parent/'execution-scope.json').write_text(json.dumps({'canary':args.canary,'input_scope':'fresh training captures, frozen pretrained public head, stable true defence encoder','other_baseline_attacks':'not executed','protocol_sha256':sha(ROOT/'experiments/whitebox35/protocol.json')},indent=2)+'\n')


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',required=True);ap.add_argument('--depth',type=int,required=True)
 ap.add_argument('--seed',type=int,required=True);ap.add_argument('--method',choices=['pca-compress','tb-frozen'],required=True);ap.add_argument('--dim',type=int,required=True)
 ap.add_argument('--canary',action='store_true');ap.add_argument('--device',default='cuda',choices=['cpu','cuda'])
 run(ap.parse_args())
if __name__=='__main__':main()
