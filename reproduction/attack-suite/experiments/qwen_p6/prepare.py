"""Retokenise the unchanged document partitions for pinned Qwen3; no outcomes."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from transformers import AutoTokenizer
 from huggingface_hub import snapshot_download
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 root=Path(__file__).resolve().parents[2];out=root/a.output
 if out.exists():raise ValueError('fresh output required')
 source=root/'results/validation-corpus-v1/partitioned-documents.json'
 manifest=json.loads((root/'experiments/validation_corpus_manifest.json').read_text())
 if sha(source)!=manifest['raw_partitioned_sha256']:raise ValueError('Source text manifest hash mismatch')
 raw=json.loads(source.read_text());baseline=json.loads((root/'results/validation-corpus-v1/validation-input.json').read_text())
 revision='c1899de289a04d12100db370d81485cdf75e47ca'
 model=snapshot_download('Qwen/Qwen3-0.6B',revision=revision,local_files_only=True)
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 result={'provenance':{'source_sha256':sha(source),'original_partition_input_sha256':sha(root/'results/validation-corpus-v1/validation-input.json'),'model':'Qwen/Qwen3-0.6B','model_revision':revision,'tokenization':'source text, add_special_tokens=False, first 32 Qwen tokens; same source-document assignments as GPT2'}}
 seen=set()
 for part in ['calibration','train','evaluation']:
  assert [r['source_document_id'] for r in raw[part]]==[r['source_document_id'] for r in baseline[part]]
  result[part]=[]
  for row in raw[part]:
   assert row['source_document_id'] not in seen;seen.add(row['source_document_id'])
   ids=tok(row['text'],add_special_tokens=False,truncation=True,max_length=32)['input_ids']
   if len(ids)!=32:raise ValueError('Document too short under Qwen tokenizer: '+row['id'])
   result[part].append({'id':row['id'],'source_document_id':row['source_document_id'],'tokens':ids})
 counts=Counter(t for row in result['calibration'] for t in set(row['tokens']))
 selected=sorted(counts,key=lambda t:(abs(counts[t]-16),t))[:3]
 result['provenance']['calibration_property_tokens']=selected
 result['provenance']['calibration_property_document_counts']={str(t):counts[t] for t in selected}
 result['provenance']['tokenizer_files_sha256']={p.name:sha(p) for p in Path(model).iterdir() if p.name.startswith('tokenizer') and p.is_file()}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'input':a.output,'sha256':sha(out),'property_tokens':selected,'prevalences':result['provenance']['calibration_property_document_counts'],'num_hidden_layers':json.loads((Path(model)/'config.json').read_text())['num_hidden_layers']}))
if __name__=='__main__':main()
