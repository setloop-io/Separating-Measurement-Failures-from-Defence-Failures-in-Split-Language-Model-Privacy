"""Download only the pinned public assets required to reproduce the pilot."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    args=ap.parse_args()
    p=json.loads(Path(__file__).with_name('pilot_gpt2_wikitext2.json').read_text())
    args.inputs.mkdir(parents=True,exist_ok=True)
    parquet=args.inputs/'train.parquet'
    if not parquet.exists():
        url=f"https://huggingface.co/datasets/{p['dataset']}/resolve/{p['dataset_revision']}/{p['dataset_configuration']}/train-00000-of-00001.parquet"
        temp=parquet.with_suffix('.download')
        urllib.request.urlretrieve(url,temp);temp.replace(parquet)
    model=snapshot_download(p['model'],revision=p['model_revision'],local_dir=args.inputs/'model',
                            allow_patterns=['config.json','generation_config.json','model.safetensors',
                                            'tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'])
    AutoTokenizer.from_pretrained(model).save_pretrained(args.inputs/'tokenizer')
    print(args.inputs)


if __name__=='__main__':main()
