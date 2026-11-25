"""CPU-only preflight counter for the exact retained LAMBADA window loop."""
import json
import math
import time


def native_window_count(token_count,max_content_tokens,doc_stride=512):
    start_offset = 0
    windows = 0
    while start_offset<token_count:
        length = token_count-start_offset-1
        if length>max_content_tokens:
            length = max_content_tokens
        windows += 1
        if start_offset+doc_stride+1>=token_count or start_offset+length+1>=token_count:
            break
        start_offset += min(length,doc_stride)
    return windows


def count_native_windows(cfg,inputs,out):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(inputs['model']['path'],local_files_only=True,trust_remote_code=True)
    prompt = 'Please complete the passages with the correct next word.'
    prompt_tokens = tokenizer.tokenize(prompt)
    max_seq_length = 512-len(prompt_tokens)
    stride = 512
    totals = {'train_documents':0,'train_tokens':0,'train_windows':0,'train_empty_documents':0,
              'train_empty_document_indices':[],'test_rows':0,
              'test_min_input_tokens':None,'test_max_input_tokens':0,'test_over_model_1024':0,
              'prompt_token_count':len(prompt_tokens),'max_content_tokens':max_seq_length,
              'counter':'literal native start_offset/length/break loop, using admitted GPT2 tokenizer'}
    started = time.monotonic()
    perdoc = (out/'window-counts.jsonl').open('w')
    for split in ['train','test']:
        index = 0
        for item in inputs['dataset']['splits'][split]:
            for batch in pq.ParquetFile(item['path']).iter_batches(batch_size=1,columns=['text']):
                text = batch.column(0)[0].as_py()
                tokens = tokenizer.tokenize(text)
                if not tokens and split=='test':
                    raise RuntimeError(f'Native LAMBADA would index an empty token list: {split}/{index}')
                n = len(tokens)
                if split=='train':
                    # The original while loop emits zero windows for empty
                    # training books; the source rows themselves stay intact.
                    windows = native_window_count(n,max_seq_length,stride)
                    if n==0:
                        totals['train_empty_documents'] += 1
                        totals['train_empty_document_indices'].append(index)
                    totals['train_documents'] += 1
                    totals['train_tokens'] += n
                    totals['train_windows'] += windows
                    row = {'split':split,'index':index,'tokens':n,'native_windows':windows}
                else:
                    native_text = tokenizer.convert_tokens_to_string(prompt_tokens+tokens[:-1])
                    count = len(tokenizer(native_text)['input_ids'])
                    totals['test_rows'] += 1
                    totals['test_min_input_tokens'] = count if totals['test_min_input_tokens'] is None else min(totals['test_min_input_tokens'],count)
                    totals['test_max_input_tokens'] = max(totals['test_max_input_tokens'],count)
                    totals['test_over_model_1024'] += int(count>1024)
                    row = {'split':split,'index':index,'tokens':n,'native_input_tokens':count}
                perdoc.write(json.dumps(row)+'\n')
                index += 1
                if index%25==0:
                    perdoc.flush()
                    print(json.dumps({'event':'native_window_count','split':split,'documents':index,'train_windows':totals['train_windows'],'wall_seconds':time.monotonic()-started}),flush=True)
    perdoc.close()
    totals['batches_per_epoch'] = math.ceil(totals['train_windows']/16)
    totals['maximum_30_epoch_steps'] = 30*totals['batches_per_epoch']
    totals['wall_seconds'] = time.monotonic()-started
    (out/'native-window-counts.json').write_text(json.dumps(totals,indent=2)+'\n')
    return totals
