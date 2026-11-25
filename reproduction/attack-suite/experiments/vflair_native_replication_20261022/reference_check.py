"""No-training reference gate for the pinned native GPT-2/LAMBADA pipeline.

Called by the run-owned runner as execute(native, cfg, inputs, runtime, out).
This module does not run on import.  Its admitted run uses the original native
split=1 NONE condition and exactly 32 predeclared test indices; it never calls
train_vfl, train_batch, an optimizer step, or an attack.  The comparison point
is the original next-token utility path: Task.forward, followed by the last
position of CausalLMOutput.logits (MainTaskVFL_LLM.generate_result).

Required config: phase='reference', condition='none', reference.test_indices
is a list of 32 unique nonnegative integers, transformers_version='4.39.0'.
Optional predeclared reference tolerances default to max_abs_error=1e-4 and
rms_error=1e-5.  Both limits and all 32 top-1 agreements must pass.  These are
FP32 engineering acceptance limits, not statistical or privacy guarantees.
Only a separately registered run may call execute; local syntax/static checks
must not load models.  The runner remains responsible for source/input hashes,
the isolated native import compatibility receipt and resource admission.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _tensor_receipt(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'raw_bytes_sha256': hashlib.sha256(raw).hexdigest()}


def _rng_snapshot(torch, np):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().clone(),
            'cuda': [v.clone() for v in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else []}


def _rng_restore(state, torch, np):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def _rng_receipt(state):
    numpy_state = state['numpy']
    return {'python_sha256': hashlib.sha256(repr(state['python']).encode()).hexdigest(),
            'numpy_sha256': hashlib.sha256(
                repr((numpy_state[0], numpy_state[2:])).encode() +
                numpy_state[1].tobytes()).hexdigest(),
            'torch': _tensor_receipt(state['torch']),
            'cuda': [_tensor_receipt(v) for v in state['cuda']]}


def _reference_options(cfg):
    if cfg.get('phase') != 'reference' or cfg.get('condition') != 'none':
        raise ValueError('Reference gate requires phase=reference and condition=none')
    ref = cfg['reference']
    indices = ref['test_indices']
    if (not isinstance(indices, list) or len(indices) != 32 or
            any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != 32):
        raise ValueError('Exactly 32 unique predeclared nonnegative test indices are required')
    if ref.get('transformers_version') != '4.39.0':
        raise ValueError('The reference protocol pins transformers_version=4.39.0')
    result = dict(ref)
    for key, default in [('max_abs_error', 1e-4), ('rms_error', 1e-5)]:
        value = ref.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < value < float('inf')):
            raise ValueError(f'{key} must be a finite positive predeclared tolerance')
        result[key] = float(value)
    if cfg.get('training_autocast', 'none') != 'none':
        raise ValueError('The untrained reference gate uses FP32 without autocast')
    return result


def _prepare_reference_data(inputs, out, indices):
    """Change data I/O only; leave native prompt/tokenization/labels untouched."""
    import datasets
    import load.LoadDataset as native_data
    original_loader = native_data.load_dataset
    files = {split: [item['path'] for item in records]
             for split, records in inputs['dataset']['splits'].items()}
    pinned = datasets.load_dataset('parquet', data_files=files,
                                   cache_dir=str(out / 'dataset-cache'))
    original_rows = {split: len(value) for split, value in pinned.items()}
    if max(indices) >= len(pinned['test']):
        raise ValueError('Predeclared test index is outside the pinned test split')
    # Zero training rows avoids 503,227 unused native training windows.  The
    # original native preprocessing accepts empty train text/domain arrays.
    pinned['train'] = pinned['train'].select([])
    pinned['test'] = pinned['test'].select(indices)

    def admitted_loader(path, *args, **kwargs):
        if path == '__PINNED_LAMBADA_MANIFEST__':
            return pinned
        return original_loader(path, *args, **kwargs)

    native_data.load_dataset = admitted_loader
    _write(out / 'data-selection.json', {
        'format': 'exact pinned parquet data I/O adapter',
        'original_rows': original_rows,
        'selected_indices': {'train': [], 'test': indices},
        'training_preprocessing_skipped': True,
        'native_windowing_prompt_tokenization_labels_unchanged': True,
        'reference_subset': True})
    return native_data, original_loader


def _check_native_profile(config):
    model = config['model_list']
    if (config['k'] != 2 or config['model_architect'] != 'CLM' or
            model['model_type'] != 'GPT2' or model['vfl_model_slice_num'] != 2 or
            model['local_encoders_num'] != 1 or model['local_tail_encoders_num'] != 0 or
            model['0']['head']['trainable'] != 1 or model['1']['tail']['trainable'] != 1 or
            config['dataset']['dataset_name'] != 'Lambada' or
            model['task']['task_type'] != 'CausalLM' or 'defense' in config or
            'attack_list' in config):
        raise RuntimeError('Reference profile must retain the native untrained split=1 NONE model')


def execute(native, cfg, inputs, runtime, out):
    """Run only inside a separately admitted ORX reference child; no training."""
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel
    import party.party_utils as party_utils
    from native_pipeline import (stage_model_inputs, module_inventory,
                                 optimizer_inventory, plain)

    ref = _reference_options(cfg)
    if importlib.metadata.version('transformers') != ref['transformers_version']:
        raise RuntimeError('Installed transformers version differs from the pinned 4.39.0 reference')
    runtime, out = Path(runtime).resolve(), Path(out).resolve()
    started = time.monotonic()
    caller_rng = _rng_snapshot(torch, np)
    caller_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    caller_cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    caller_hashseed = os.environ.get('PYTHONHASHSEED')
    old_model_base = party_utils.Trained_Models_Save_Path
    data_module, old_loader = None, None
    summary = None
    try:
        modeldir = stage_model_inputs(inputs, out)
        source = runtime / 'src/configs/llm_configs/with_mia_and_defense/2-slice/lambada_gpt_mid.json'
        original = json.loads(source.read_text())
        config = copy.deepcopy(original)
        config.pop('defense')
        config.pop('attack_list')
        for party in ['0', '1']:
            config['model_list'][party]['path'] = str(modeldir)
            config['model_list'][party]['train_set_file'] = '__PINNED_LAMBADA_MANIFEST__'
            config['model_list'][party]['test_set_file'] = '__PINNED_LAMBADA_MANIFEST__'
        _check_native_profile(config)
        _write(out / 'native-original-config.json', original)
        _write(out / 'native-effective-config.json', config)
        name = '__replication_reference__'
        _write(runtime / 'src/configs' / (name + '.json'), config)
        data_module, old_loader = _prepare_reference_data(inputs, out, ref['test_indices'])
        party_utils.Trained_Models_Save_Path = str(out / 'checkpoints') + '/'
        device = ref.get('device', 'cuda')
        # The unchanged native passive-party constructor unconditionally calls
        # torch.cuda.current_device().  A CPU model port is a separate protocol.
        if device != 'cuda' or not torch.cuda.is_available():
            raise ValueError('The unchanged native constructor requires reference.device=cuda')
        torch.cuda.set_device(0)
        args = argparse.Namespace(device=device, gpu=0, seed=cfg['seed'],
                                  current_seed=cfg['seed'], prefix='native-reference',
                                  configs=name, save_model=0, save_defense_model=False,
                                  attack_only=0)
        native.set_seed(args.current_seed)
        args = native.load_basic_configs_llm(args.configs, args)
        args.need_auxiliary = 0
        args.dataset = args.dataset_split['dataset_name']
        directories = native.create_exp_dir_and_file(args)
        for key, value in zip(['exp_res_dir', 'exp_res_path', 'model_folder',
                               'defense_model_folder', 'trained_model_folder'], directories):
            setattr(args, key, value)
        args = native.load_attack_configs(args.configs, args, -1)
        args = native.load_parties_llm(args)
        if (args.max_new_tokens != 1 or args.metric_type != 'best_pred' or
                len(args.parties[0].train_dst) != 0 or len(args.parties[0].test_dst) != 32):
            raise RuntimeError('Reference requires 32 native next-token test inputs and no training windows')
        before = module_inventory(args.parties, hash_weights=True)
        if any(row['parameter_dtypes'] != ['torch.float32'] for row in before):
            raise RuntimeError('Native reference parameters must all be FP32')
        _write(out / 'native-trainability.json', before)
        optimizers_before = optimizer_inventory(args.parties, include_state=True)
        if any(row['state_dict']['state'] for row in optimizers_before):
            raise RuntimeError('Reference native optimizer state was already populated')
        cls = native.create_main_task(native.get_cls_ancestor(args.config.model_type, args.model_architect))
        top = cls(args)
        top.init_communication()
        top.eval()
        # Load the monolithic HF checkpoint from the exact same independently
        # staged bytes.  No checkpoint download, fitted map or custom split.
        reference_model = GPT2LMHeadModel.from_pretrained(
            str(modeldir), local_files_only=True, torch_dtype=torch.float32).to(device)
        reference_model.eval()
        reference_model.config.pad_token_id = args.pad_id
        native_modules = [getattr(party, name, None) for party in args.parties
                          for name in ['local_model', 'local_model_tail', 'global_model']]
        if any(module.training for module in native_modules if module is not None):
            raise RuntimeError('Native reference modules did not enter eval mode')
        measurement_rng = _rng_snapshot(torch, np)
        packets, rows, native_predictions, native_targets = [], [], [], []
        squared_error, logit_elements = 0.0, 0
        for ordinal, dataset_index in enumerate(ref['test_indices']):
            sample, label = args.parties[0].test_dst[ordinal]
            # Exactly the stack operation used by native predict() for test
            # batch size one.  Retain masks and IDs before either model runs.
            native_inputs = {key: torch.stack([value.clone()]) if isinstance(value, torch.Tensor)
                             else [value] for key, value in sample.items()}
            ids = native_inputs['input_ids']
            mask = native_inputs['attention_mask']
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != mask.shape:
                raise RuntimeError('Native context IDs/attention masks are not aligned [1,L] tensors')
            if ids.shape[1] > reference_model.config.n_positions:
                raise RuntimeError(f'Pinned test context {dataset_index} exceeds native GPT2 positional capacity')
            hf_inputs = {key: value.clone().to(device) for key, value in native_inputs.items()
                         if key in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']}
            pair_rng = _rng_snapshot(torch, np)
            top._clear_past_key_values()
            _rng_restore(pair_rng, torch, np)
            if device == 'cuda':
                torch.cuda.synchronize()
            native_started = time.monotonic()
            with torch.no_grad():
                output = top.forward(**native_inputs)
                native_logits = output.logits[:, -1, :].detach().cpu().clone()
                target, pred, count = top.generate_result(output, [label])
            if device == 'cuda':
                torch.cuda.synchronize()
            native_seconds = time.monotonic() - native_started
            top._clear_past_key_values()
            _rng_restore(pair_rng, torch, np)
            hf_started = time.monotonic()
            with torch.no_grad():
                hf_logits = reference_model(**hf_inputs, use_cache=False).logits[:, -1, :].detach().cpu().clone()
            if device == 'cuda':
                torch.cuda.synchronize()
            hf_seconds = time.monotonic() - hf_started
            _rng_restore(pair_rng, torch, np)
            if (count != 1 or native_logits.shape != hf_logits.shape or
                    native_logits.dtype != torch.float32 or hf_logits.dtype != torch.float32 or
                    not bool(torch.isfinite(native_logits).all()) or not bool(torch.isfinite(hf_logits).all())):
                raise RuntimeError('Reference logits must be aligned, finite FP32 single-example outputs')
            native_id = int(native_logits.argmax(dim=-1).item())
            hf_id = int(hf_logits.argmax(dim=-1).item())
            if int(pred[0]) != native_id or int(target[0]) != int(label):
                raise RuntimeError('Native generate_result differs from the retained final-logit argmax/label')
            native_predictions.extend(pred)
            native_targets.extend(target)
            delta = native_logits.double() - hf_logits.double()
            row_sum = float(delta.square().sum().item())
            squared_error += row_sum
            logit_elements += delta.numel()
            packet = {'ordinal': ordinal, 'source_test_index': dataset_index,
                      'input_ids': ids.detach().cpu().clone(),
                      'attention_mask': mask.detach().cpu().clone(),
                      'target_id': int(label), 'native_final_logits': native_logits,
                      'hf_final_logits': hf_logits}
            packets.append(packet)
            row = {'ordinal': ordinal, 'source_test_index': dataset_index,
                   'input_ids': plain(packet['input_ids']),
                   'attention_mask': plain(packet['attention_mask']),
                   'target_id': int(label), 'native_top1_id': native_id, 'hf_top1_id': hf_id,
                   'token_agreement': native_id == hf_id,
                   'max_abs_error': float(delta.abs().max().item()),
                   'rms_error': (row_sum / delta.numel()) ** 0.5,
                   'native_forward_seconds': native_seconds, 'hf_forward_seconds': hf_seconds,
                   'tensors': {key: _tensor_receipt(value) for key, value in packet.items()
                               if isinstance(value, torch.Tensor)}}
            row['passed'] = (row['token_agreement'] and row['max_abs_error'] <= ref['max_abs_error']
                             and row['rms_error'] <= ref['rms_error'])
            rows.append(row)
            _write(out / 'reference-predictions.json', rows)
            print(json.dumps({'event': 'reference_example_complete', **{key: row[key] for key in
                             ['ordinal', 'source_test_index', 'max_abs_error', 'rms_error', 'token_agreement', 'passed']}}), flush=True)
        # Expose any unexpected state update despite no_grad and no training.
        after = module_inventory(args.parties, hash_weights=True)
        optimizers_after = optimizer_inventory(args.parties, include_state=True)
        weights_unchanged = before == after
        optimizer_state_empty = all(not row['state_dict']['state'] for row in optimizers_after)
        measurement_rng_restored = _rng_receipt(measurement_rng) == _rng_receipt(_rng_snapshot(torch, np))
        assessment = plain(top.generate_assessment(native_predictions, native_targets))
        independent_native_accuracy = sum(row['native_top1_id'] == row['target_id'] for row in rows) / 32
        native_metric_verified = abs(assessment['acc'] - independent_native_accuracy) <= 1e-12
        torch.save({'schema': 'native-hf-reference-tensors-v1', 'examples': packets},
                   out / 'reference-tensors.pt')
        _write(out / 'native-final-weights.json', after)
        _write(out / 'reference-native-utility.json', assessment)
        summary = {
            'schema': 'native-hf-reference-v1', 'status': 'complete', 'profile': 'reference',
            'passed': all(row['passed'] for row in rows) and weights_unchanged and
                      optimizer_state_empty and measurement_rng_restored and native_metric_verified,
            'condition': 'none', 'native_split': 1, 'training_steps': 0,
            'attack_executed': False, 'model_loaded': True, 'inference_executed': True,
            'examples': len(rows), 'native_forward_calls': len(rows), 'hf_forward_calls': len(rows),
            'test_indices': ref['test_indices'], 'seed': cfg['seed'], 'device': device,
            'transformers_version': importlib.metadata.version('transformers'),
            'tolerances': {'max_abs_error': ref['max_abs_error'], 'per_example_rms_error': ref['rms_error'],
                           'top1_agreement_required': 32},
            'maximum_absolute_logit_error': max(row['max_abs_error'] for row in rows),
            'maximum_per_example_rms_error': max(row['rms_error'] for row in rows),
            'pooled_rms_logit_error': (squared_error / logit_elements) ** 0.5,
            'logit_elements_compared': logit_elements,
            'top1_agreements': sum(row['token_agreement'] for row in rows),
            'native_weights_unchanged': weights_unchanged,
            'native_optimizer_state_empty': optimizer_state_empty,
            'measurement_rng_restored': measurement_rng_restored,
            'native_utility_metric_verified': native_metric_verified,
            'native_utility': assessment, 'native_utility_recomputed': independent_native_accuracy,
            'hf_utility_recomputed': sum(row['hf_top1_id'] == row['target_id'] for row in rows) / 32,
            'result_scope': 'untrained native split versus identical-checkpoint monolithic HF numerical reference; no defence efficacy or training convergence claim',
            'native_methods_modified': False,
            'comparison_point': 'native MainTaskVFL_LLM.forward; generate_result CausalLMOutput.logits[:, -1, :] before argmax',
            'source': {'reference_module_sha256': _sha_file(__file__),
                       'native_config_sha256': _sha_file(source),
                       'upstream_revision': cfg['upstream_revision'],
                       'model_revision': inputs['model']['revision'],
                       'dataset_revision': inputs['dataset']['revision']},
            'output_files': {name: {'sha256': _sha_file(out / name), 'bytes': (out / name).stat().st_size}
                             for name in ['reference-predictions.json', 'reference-tensors.pt',
                                          'native-trainability.json', 'native-final-weights.json',
                                          'model-staging.json', 'data-selection.json']},
            'wall_seconds': time.monotonic() - started}
        _write(out / 'reference-summary.json', summary)
    finally:
        if data_module is not None:
            data_module.load_dataset = old_loader
        party_utils.Trained_Models_Save_Path = old_model_base
        _rng_restore(caller_rng, torch, np)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = caller_cudnn
        if caller_cuda_device is not None:
            torch.cuda.set_device(caller_cuda_device)
        if caller_hashseed is None:
            os.environ.pop('PYTHONHASHSEED', None)
        else:
            os.environ['PYTHONHASHSEED'] = caller_hashseed
        restoration = {
            'before': _rng_receipt(caller_rng),
            'after': _rng_receipt(_rng_snapshot(torch, np)),
            'restored': _rng_receipt(caller_rng) == _rng_receipt(_rng_snapshot(torch, np))}
        _write(out / 'reference-rng-restoration.json', restoration)
        if summary is not None:
            summary['caller_rng_restored'] = restoration['restored']
            summary['passed'] = summary['passed'] and restoration['restored']
            path = out / 'reference-rng-restoration.json'
            summary['output_files'][path.name] = {'sha256': _sha_file(path), 'bytes': path.stat().st_size}
            _write(out / 'reference-summary.json', summary)
    if summary is None or not summary['passed']:
        raise RuntimeError('Native/HF reference acceptance gate failed; retained outputs were not altered')
    print(json.dumps({'event': 'native_reference_complete', **summary}), flush=True)
    return summary
