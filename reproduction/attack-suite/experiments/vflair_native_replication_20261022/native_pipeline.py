"""Observe the original main/task/party/defence/attack implementations unchanged."""
import argparse
import copy
import contextlib
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import time


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def emit(name, **value):
    print(json.dumps({'event':name,'time_unix':time.time(),**value},default=str),flush=True)


def plain(value):
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(v) for v in value]
    if hasattr(value,'item'):
        return value.item()
    return value


def cpu(value):
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [cpu(v) for v in value]
    return value


def module_inventory(parties,hash_weights=False):
    rows = []
    for party_id,party in enumerate(parties):
        for name in ['local_model','local_model_tail','global_model','head_mid_model','head_adversarial_model','head_imagined_adversary']:
            module = getattr(party,name,None)
            if module is not None:
                row = {'party':party_id,'name':name,'native_class':type(module).__module__+'.'+type(module).__name__,
                    'parameters':sum(p.numel() for p in module.parameters()),
                    'trainable_parameters':sum(p.numel() for p in module.parameters() if p.requires_grad),
                    'parameter_dtypes':sorted({str(p.dtype) for p in module.parameters()})}
                if hash_weights:
                    h = hashlib.sha256()
                    for key,tensor in sorted(module.state_dict().items()):
                        h.update(key.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
                        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
                    row['state_sha256'] = h.hexdigest()
                rows.append(row)
    return rows


def optimizer_inventory(parties,include_state=False):
    seen = set()
    rows = []
    for party_id,party in enumerate(parties):
        # Native property global_model_optimizer unconditionally indexes key1,
        # including on passive parties whose optimizer dictionary has only key0.
        # The dictionary is authoritative; do not evaluate those alias properties.
        candidates = [(f'optimizers[{key}]',value) for key,value in sorted(getattr(party,'optimizers',{}).items(),key=lambda item:str(item[0]))]
        candidates.extend((name,getattr(party,name,None)) for name in ['head_imagined_adversary_optimizer','tail_imagined_adversary_optimizer'])
        for name,optimizer in candidates:
            if optimizer is not None and id(optimizer) not in seen:
                seen.add(id(optimizer))
                row = {'party':party_id,'name':name,'native_class':type(optimizer).__module__+'.'+type(optimizer).__name__,
                       'parameter_groups':[{k:plain(v) for k,v in group.items() if k!='params'} for group in optimizer.param_groups]}
                if include_state:
                    row['state_dict'] = cpu(optimizer.state_dict())
                rows.append(row)
    return rows


def scheduler_inventory(parties,include_state=False):
    rows = []
    seen = set()
    for party_id,party in enumerate(parties):
        for key,scheduler in sorted(getattr(party,'lr_schedulers',{}).items(),key=lambda item:str(item[0])):
            if scheduler is not None and id(scheduler) not in seen:
                seen.add(id(scheduler))
                row = {'party':party_id,'name':f'lr_schedulers[{key}]',
                       'native_class':type(scheduler).__module__+'.'+type(scheduler).__name__,
                       'last_epoch':scheduler.last_epoch,'last_lr':plain(scheduler.get_last_lr())}
                if include_state:
                    row['state_dict'] = cpu(scheduler.state_dict())
                rows.append(row)
    return rows


def training_context(torch,mode):
    """An explicit numerical protocol port; default executes without autocast."""
    if mode=='none':
        return contextlib.nullcontext()
    if mode=='bfloat16':
        return torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    raise ValueError('training_autocast must be none or bfloat16')


def training_dtype_inventory(parties):
    """Read tensor metadata only; no copies, reductions or optimizer operations."""
    result = {'parameters':{},'gradients':{},'optimizer_state':{}}
    seen_optimizers,seen_parameters = set(),set()
    def add(group,tensor):
        dtype = str(tensor.dtype)
        record = result[group].setdefault(dtype,{'tensors':0,'elements':0})
        record['tensors'] += 1
        record['elements'] += tensor.numel()
    for party in parties:
        optimizers = list(getattr(party,'optimizers',{}).values())
        optimizers.extend(getattr(party,name,None) for name in ['head_imagined_adversary_optimizer','tail_imagined_adversary_optimizer'])
        for optimizer in optimizers:
            if optimizer is None or id(optimizer) in seen_optimizers:
                continue
            seen_optimizers.add(id(optimizer))
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if id(parameter) in seen_parameters:
                        continue
                    seen_parameters.add(id(parameter))
                    add('parameters',parameter)
                    if parameter.grad is not None:
                        add('gradients',parameter.grad)
            for state in optimizer.state.values():
                for value in state.values():
                    if hasattr(value,'dtype') and hasattr(value,'numel'):
                        add('optimizer_state',value)
    return result


def stage_model_inputs(inputs,out):
    """Copy resolved HF payloads into independent, verified run-owned files."""
    modeldir = out/'model-input'
    modeldir.mkdir()
    receipt = []
    for item in inputs['model']['files']:
        source_entry = Path(inputs['model']['path'])/item['path']
        src = source_entry.resolve(strict=True)
        dst = modeldir/item['path']
        shutil.copy2(src,dst)
        method = 'copy2-resolved-payload'
        if not dst.is_file() or dst.is_symlink():
            raise RuntimeError(f'Staged model file is missing or remains a symlink: {dst}')
        h = hashlib.sha256()
        with dst.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):
                h.update(block)
        actual = h.hexdigest()
        if actual!=item['sha256'] or dst.stat().st_size!=item['bytes']:
            raise RuntimeError(f'Staged model payload hash/size mismatch: {dst}')
        receipt.append({'entry':str(source_entry),'resolved_source':str(src),'path':str(dst),
                        'method':method,'sha256':actual,'bytes':dst.stat().st_size})
    write(out/'model-staging.json',receipt)
    return modeldir


def native_config(cfg,runtime,modeldir):
    src = runtime/'src/configs/llm_configs/with_mia_and_defense/2-slice/lambada_gpt_mid.json'
    if cfg['phase']=='bounded-study':
        from bounded_study import validate_original_config
        validate_original_config(cfg,src)
    original = json.loads(src.read_text())
    result = copy.deepcopy(original)
    for party in ['0','1']:
        result['model_list'][party]['path'] = str(modeldir)
        result['model_list'][party]['train_set_file'] = '__PINNED_LAMBADA_MANIFEST__'
        result['model_list'][party]['test_set_file'] = '__PINNED_LAMBADA_MANIFEST__'
    condition = cfg['condition']
    if condition == 'none':
        result.pop('defense')
    elif condition == 'at':
        atsrc = runtime/'src/configs/llm_configs/with_mia_and_defense/2-slice/codealpaca_at.json'
        result['defense'] = copy.deepcopy(json.loads(atsrc.read_text())['defense'])
        result['defense']['parameters']['seq_length'] = 512
    elif condition != 'mid':
        raise ValueError(f'Unsupported native condition: {condition}')
    if not cfg['full'].get('include_bisr',False):
        del result['attack_list']['2']
    if cfg['phase'] == 'canary':
        # Native epochs==1 is a one-minibatch shortcut. A declared two-epoch
        # engineering profile exercises the ordinary full minibatch loop.
        native_epochs = cfg['canary'].get('native_epochs',1)
        if type(native_epochs) is not int or native_epochs not in [1,2]:
            raise ValueError('Canary native_epochs must be explicitly one or two')
        result['epochs'] = native_epochs
        for attack in result['attack_list'].values():
            attack['parameters']['epochs'] = cfg['canary']['attack_epochs']
            attack['parameters']['attack_sample_num'] = cfg['canary']['attack_samples']
        if cfg['full'].get('include_bisr'):
            raise RuntimeError('BiSR has a separate architecture/training admission; it is excluded from pipeline canary')
    else:
        expected = {'epochs':30,'batch_size':16,'sequence_length':512,'learning_rate':0.001,'native_early_stop_threshold':1}
        if any(cfg['full'][k] != v for k,v in expected.items()):
            raise RuntimeError('Full native profile differs from the retained original MID configuration')
    return original,result


def prepare_data_io(cfg,inputs,out):
    import datasets
    import load.LoadDataset as native_data
    native_loader = native_data.load_dataset
    files = {s:[x['path'] for x in f] for s,f in inputs['dataset']['splits'].items()}
    pinned = datasets.load_dataset('parquet',data_files=files,cache_dir=str(out/'dataset-cache'))
    original_rows = {s:len(d) for s,d in pinned.items()}
    selected = {s:list(range(len(d))) for s,d in pinned.items()}
    if cfg['phase']=='bounded-study':
        from bounded_study import select_fixed_books
        pinned,receipt = select_fixed_books(pinned,cfg,inputs)
        selected['train'] = receipt['source_indices']
        write(out/'bounded-study-selection-receipt.json',receipt)
    if cfg['phase'] == 'canary':
        for split,key in [('train','train_rows'),('test','test_rows')]:
            n = min(cfg['canary'][key],len(pinned[split]))
            selected[split] = list(range(n))
            pinned[split] = pinned[split].select(selected[split])
    def admitted_loader(path,*args,**kwargs):
        if path == '__PINNED_LAMBADA_MANIFEST__':
            return pinned
        return native_loader(path,*args,**kwargs)
    native_data.load_dataset = admitted_loader
    write(out/'data-selection.json',{'format':'exact pinned parquet data I/O adapter',
        'original_rows':original_rows,'selected_indices':selected,
        'canary_subset':cfg['phase']=='canary',
        'native_windowing_prompt_tokenization_labels_unchanged':True})
    return pinned


class Observer:
    def __init__(self,cfg,out):
        self.cfg,self.out = cfg,out
        self.training_autocast = cfg.get('training_autocast','none')
        if self.training_autocast not in ['none','bfloat16']:
            raise ValueError('training_autocast must be none or bfloat16')
        if self.training_autocast=='bfloat16' and cfg['phase']!='canary':
            raise ValueError('The BF16 training port is admitted only for a separately labelled engineering canary')
        self.stage = 'setup'
        self.step = 0
        self.attack = None
        self.predictions = []
        self.attack_summaries = []
        self.utility = []
        self.capture_index = []
        self.steps = (out/'steps.jsonl').open('w')
        self.predfile = (out/'attack-predictions.jsonl').open('w')
        (out/'captures').mkdir()
        self.capture_bytes = 0

    def install(self,cls):
        import torch
        observer = self
        old_batch = cls.train_batch
        old_infer = cls.inference
        old_assess = cls.generate_assessment
        old_pred = cls.pred_transmit
        old_attack = cls.evaluate_attack
        def train_batch(top,*args,**kwargs):
            maximum = observer.cfg['guards']['maximum_optimizer_steps']
            if maximum is not None and observer.step >= maximum:
                raise RuntimeError('Optimizer-step guard reached before native training completed')
            started = time.monotonic()
            # Context is restricted to the original native training call.
            # Utility, captures outside training, and inversion attacks retain
            # their original precision path; there is no GradScaler.
            with training_context(torch,observer.training_autocast):
                result = old_batch(top,*args,**kwargs)
            torch.cuda.synchronize()
            observer.step += 1
            row = {'step':observer.step,'epoch':getattr(top,'current_epoch',None),
                   'loss':plain(result[0]),'train_accuracy':plain(result[1]),
                   'wall_seconds':time.monotonic()-started,
                   'cuda_allocated_bytes':torch.cuda.memory_allocated(),
                   'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
                   'loss_finite':math.isfinite(float(result[0])),
                   'training_autocast':observer.training_autocast,
                   'training_tensor_dtypes':training_dtype_inventory(top.parties),
                   'optimizers':optimizer_inventory(top.parties),
                   'schedulers':scheduler_inventory(top.parties)}
            observer.steps.write(json.dumps(row,default=str)+'\n'); observer.steps.flush()
            emit('native_train_step',**row)
            if not row['loss_finite']:
                raise FloatingPointError('Native training loss is nonfinite under the declared numerical protocol')
            return result
        def inference(top,*args,**kwargs):
            before = observer.stage
            observer.stage = 'utility'
            try:
                return old_infer(top,*args,**kwargs)
            finally:
                observer.stage = before
        def assessment(top,pred,label):
            result = old_assess(top,pred,label)
            if observer.stage == 'utility':
                row = {'predicted_ids':plain(pred),'target_ids':plain(label),'native_result':plain(result),
                       'metric':observer.cfg.get('utility_metric','native LAMBADA best_pred final-BPE-token accuracy'),
                       'evaluation_stochasticity':'MID remains stochastic in native eval mode'}
                observer.utility.append(row)
                write(observer.out/'utility-predictions.json',observer.utility)
                emit('native_utility',n=len(pred),result=plain(result))
            return result
        def pred_transmit(top,*args,**kwargs):
            value = old_pred(top,*args,**kwargs)
            if observer.stage == 'attack':
                idx = len(observer.capture_index)
                path = observer.out/'captures'/f'{idx:05d}-{observer.attack}.pt'
                payload = {'native_transmitted_result':cpu(value),
                           'native_local_data_input':cpu(getattr(top.parties[0],'local_data_input',None)),
                           'attention_mask':cpu(getattr(top.parties[0],'local_batch_attention_mask',None)),
                           'attack':observer.attack,'capture_index':idx}
                torch.save(payload,path)
                observer.capture_bytes += path.stat().st_size
                if observer.capture_bytes > observer.cfg['guards']['maximum_capture_bytes']:
                    raise RuntimeError('Capture-byte guard exceeded')
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                observer.capture_index.append({'path':str(path.relative_to(observer.out)),'sha256':h,'bytes':path.stat().st_size,
                                               'attack':observer.attack,'index':idx})
                write(observer.out/'capture-index.json',observer.capture_index)
            return value
        def attack(top,*args,**kwargs):
            observer.stage = 'attack'
            observer.attack = top.args.attack_name
            original_decode = top.args.tokenizer.decode
            original_sample = random.sample
            selections = []
            before_count = len(observer.predictions)
            def selected(population,k,*a,**kw):
                result = original_sample(population,k,*a,**kw)
                selections.append({'population_size':len(population),'k':k,'selected':plain(result)})
                return result
            def decode(ids,*a,**kw):
                result = original_decode(ids,*a,**kw)
                frame = inspect.currentframe().f_back
                # Native source's pred_text assignment is the last decode call.
                source_line = inspect.getframeinfo(frame).code_context
                if frame.f_code.co_filename.endswith(observer.attack+'.py') and source_line and 'pred_text =' in source_line[0]:
                    local = frame.f_locals
                    truth = plain(local['sample_origin_data']).copy()
                    while isinstance(truth,list) and len(truth)==1 and isinstance(truth[0],list):
                        truth = truth[0]
                    predicted = plain(local['predicted_indexs'])
                    pad = top.args.tokenizer.pad_token_id
                    active = [i for i,t in enumerate(truth) if t != pad]
                    clean = [truth[i] for i in active]
                    precision = sum(t in clean for t in predicted)/len(predicted)
                    recall = sum(t in predicted for t in clean)/len(clean)
                    same_length = len(truth)==len(predicted)
                    matched = sum(truth[i]==predicted[i] for i in active) if same_length else None
                    row = {'attack':observer.attack,'sample_ordinal':len(observer.predictions)-before_count,
                           'original_ids':truth,'predicted_ids':predicted,'pad_token_id':pad,
                           'native_membership_precision_recomputed':precision,'native_membership_recall_recomputed':recall,
                           'position_matches_excluding_pad':matched,'position_count_excluding_pad':len(active),
                           'position_accuracy_excluding_pad':matched/len(active) if same_length else None,
                           'exact_document_excluding_pad':all(truth[i]==predicted[i] for i in active) if same_length else None,
                           'position_metric_defined':same_length,'capture_ordinal':len(observer.capture_index)-1}
                    observer.predictions.append(row)
                    observer.predfile.write(json.dumps(row)+'\n'); observer.predfile.flush()
                    emit('native_attack_sample',attack=observer.attack,index=row['sample_ordinal'],precision=precision,recall=recall)
                return result
            top.args.tokenizer.decode = decode
            random.sample = selected
            started = time.monotonic()
            try:
                result = old_attack(top,*args,**kwargs)
            finally:
                top.args.tokenizer.decode = original_decode
                random.sample = original_sample
                observer.stage = 'train'
            rows = observer.predictions[before_count:]
            if not rows:
                raise RuntimeError('Native attack completed without retained prediction IDs')
            expected_rows = min(top.args.attack_configs['attack_sample_num'],len(top.final_state['test_data'][0]))
            if len(rows) != expected_rows:
                raise RuntimeError(f'Native attack prediction coverage incomplete: {len(rows)} != {expected_rows}')
            p = sum(x['native_membership_precision_recomputed'] for x in rows)/len(rows)
            r = sum(x['native_membership_recall_recomputed'] for x in rows)/len(rows)
            if not (abs(p-float(result[0])) < 1e-10 and abs(r-float(result[1])) < 1e-10):
                raise RuntimeError('Independent prediction-ID scoring disagrees with native attack metrics')
            summary = {'attack':observer.attack,'native_return':plain(result),'prediction_rows':len(rows),
                       'recomputed_precision':p,'recomputed_recall':r,'sample_selection_calls':selections,
                       'wall_seconds':time.monotonic()-started,'native_metrics_verified':True}
            observer.attack_summaries.append(summary)
            write(observer.out/'attack-summaries.json',observer.attack_summaries)
            emit('native_attack_complete',**summary)
            return result
        cls.train_batch = train_batch
        cls.inference = inference
        cls.generate_assessment = assessment
        cls.pred_transmit = pred_transmit
        cls.evaluate_attack = attack


def execute(native,cfg,inputs,runtime,out):
    from bounded_study import assert_launch_authorized
    assert_launch_authorized(cfg)
    import torch
    import party.party_utils as party_utils
    modeldir = stage_model_inputs(inputs,out)
    original,config = native_config(cfg,runtime,modeldir)
    write(out/'native-original-config.json',original)
    write(out/'native-effective-config.json',config)
    config_name = '__replication_native__'
    write(runtime/'src/configs'/f'{config_name}.json',config)
    prepare_data_io(cfg,inputs,out)
    # Upstream joins by raw string concatenation, so the separator is required.
    party_utils.Trained_Models_Save_Path = str(out/'checkpoints')+'/'
    args = argparse.Namespace(device='cuda',gpu=0,seed=cfg['seed'],current_seed=cfg['seed'],prefix='native-replication',
                              configs=config_name,save_model=1,save_defense_model=True,attack_only=0)
    native.set_seed(args.current_seed)
    args = native.load_basic_configs_llm(args.configs,args)
    args.need_auxiliary = 0
    args.dataset = args.dataset_split['dataset_name']
    torch.cuda.set_device(0)
    dirs = native.create_exp_dir_and_file(args)
    for key,value in zip(['exp_res_dir','exp_res_path','model_folder','defense_model_folder','trained_model_folder'],dirs):
        setattr(args,key,value)
    # Ensure native destination directories exist; they are fresh run-owned paths.
    for folder in [args.defense_model_folder,args.trained_model_folder]:
        Path(folder).mkdir(parents=True,exist_ok=True)
    args = native.load_attack_configs(args.configs,args,-1)
    started = time.monotonic()
    args = native.load_parties_llm(args)
    if cfg['phase']=='bounded-study':
        mid_model = getattr(args.parties[0],'head_mid_model',None)
        defaults = {'condition':cfg['condition'], 'victim_seed':cfg['seed'],
                    'apply_mid':bool(args.apply_mid),
                    'mid_class':type(mid_model).__name__ if mid_model is not None else None,
                    'mid_std_shift':getattr(mid_model,'std_shift',None)}
        write(out/'native-defense-runtime.json',defaults)
        if cfg['condition']=='mid':
            if not defaults['apply_mid'] or defaults['mid_class']!='MIDModel_Linear' or defaults['mid_std_shift']!=5:
                raise RuntimeError('Bounded-study native MID implementation/defaults differ from the frozen protocol')
        elif defaults['apply_mid'] or mid_model is not None:
            raise RuntimeError('Matched bounded-study NONE unexpectedly contains native MID')
        emit('bounded_native_defense_defaults',**defaults)
    trainability = module_inventory(args.parties,hash_weights=True)
    write(out/'native-trainability.json',trainability)
    write(out/'native-initial-optimizers.json',optimizer_inventory(args.parties))
    write(out/'native-initial-schedulers.json',scheduler_inventory(args.parties))
    counts = {'train_windows':len(args.parties[0].train_dst),'test_rows':len(args.parties[0].test_dst)}
    counts['native_batches_per_epoch'] = math.ceil(counts['train_windows']/args.batch_size)
    counts['native_max_training_steps'] = counts['native_batches_per_epoch']*args.main_epochs
    if cfg['phase']=='canary' and args.main_epochs==1:
        counts['native_max_training_steps'] = 1
    write(out/'native-data-counts.json',counts)
    emit('native_data_ready',**counts)
    if cfg['phase']=='bounded-study':
        from bounded_study import validate_native_counts
        write(out/'bounded-study-count-validation.json',validate_native_counts(cfg,counts))
    for key,actual in [('maximum_training_windows',counts['train_windows']),('maximum_test_rows',counts['test_rows'])]:
        limit = cfg['guards'][key]
        if limit is not None and actual>limit:
            raise RuntimeError(f'{key} admission guard exceeded before training: {actual}>{limit}')
    if cfg['phase']=='full' and any(cfg['guards'][k] is None for k in ['maximum_training_windows','maximum_test_rows','maximum_optimizer_steps']):
        raise RuntimeError('Full native run requires explicit frozen window/test/optimizer-step guard limits')
    ancestor = native.get_cls_ancestor(args.config.model_type,args.model_architect)
    cls = native.create_main_task(ancestor)
    observer = Observer(cfg,out)
    observer.install(cls)
    native.MainTaskVFL_LLM = cls
    args.basic_vfl = None
    args.main_acc_noattack = None
    observer.stage = 'train'
    args.basic_vfl,args.main_acc_noattack = native.evaluate_no_attack_finetune(args)
    trained = time.monotonic()
    write(out/'native-final-weights.json',module_inventory(args.parties,hash_weights=True))
    import numpy as np
    torch.save({'optimizers':optimizer_inventory(args.parties,include_state=True),
                'schedulers':scheduler_inventory(args.parties,include_state=True),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'native_final_epoch':args.basic_vfl.final_epoch,'observer_training_steps':observer.step},
               out/'native-final-optimizer-rng.pt')
    native.evaluate_inversion_attack(args)
    checkpoints = []
    for path in sorted((out/'checkpoints').rglob('*')):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(8*1024*1024),b''):
                    h.update(block)
            checkpoints.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    if not checkpoints:
        raise RuntimeError('Native training completed without checkpoint files')
    write(out/'checkpoint-index.json',checkpoints)
    summary = {'status':'complete','profile':cfg['phase'],'condition':cfg['condition'],'seed':cfg['seed'],
               'native_training_steps':observer.step,'final_epoch':args.basic_vfl.final_epoch,
               'native_utility':plain(args.main_acc_noattack),'utility_metric':cfg.get('utility_metric','final-BPE-token best_pred accuracy'),
               'native_reported_training_time':getattr(args.basic_vfl,'training_time',None),
               'setup_and_training_wall_seconds':trained-started,'total_pipeline_wall_seconds':time.monotonic()-started,
               'attack_summaries':observer.attack_summaries,'capture_files':len(observer.capture_index),
               'canary_has_benchmark_claim':False,'class_function_bodies_modified':False,
               'result_scope':'engineering canary excluded from benchmark results' if cfg['phase']=='canary' else 'native configuration workload; selected attack coverage recorded',
               'native_maximum_epochs':args.main_epochs,
               'training_autocast':cfg.get('training_autocast','none'),
               'training_precision_scope':'autocast context around native train_batch only; original utility/attack precision; no GradScaler',
               'training_precision_protocol':'declared BF16 training port; FP32-equivalence not asserted' if cfg.get('training_autocast','none')=='bfloat16' else 'original native training precision path',
               'native_epochs_one_minibatch_shortcut':args.main_epochs==1,
               'at_scope':'native CodeAlpaca AT mechanism transferred to GPT2/LAMBADA' if cfg['condition']=='at' else None,
               'native_early_stop_policy':f'maximum {args.main_epochs} epochs; threshold{args.early_stop_threshold} compares last minibatch loss and cumulative nonimprovement count'}
    if cfg['phase']=='bounded-study':
        summary['result_scope'] = 'native FP32 study with frozen eight-book training adaptation and complete5153testutility; selected nativeVMI/WBI budgets'
        summary['bounded_study_selection_receipt_sha256'] = hashlib.sha256((out/'bounded-study-selection-receipt.json').read_bytes()).hexdigest()
        summary['bounded_study_source_indices'] = json.loads((out/'bounded-study-selection-receipt.json').read_text())['source_indices']
    write(out/'native-summary.json',summary)
    emit('native_pipeline_complete',**summary)
