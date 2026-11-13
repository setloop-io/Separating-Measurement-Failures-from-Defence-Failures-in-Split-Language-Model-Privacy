"""CPU-only derivative audit of 400 retained native GPT-2 wire packets.

No models are imported, constructed or executed.  Run after full retention:
  .venv/bin/python bounded_capture_audit.py --output RUNOUT --receipt-out AUDIT
Success requires all 400 indexed packets, including actual finite FP32 wires
and native input/mask consistency. Active/frozen runs are never changed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ATTACKS = {'VanillaModelInversion_WhiteBox', 'WhiteBoxInversion'}
EXPECTED_CAPTURES = 400
MAX_CAPTURE_BYTES = 8 * 2**20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _tensor_receipt(tensor):
    import torch
    value = tensor.detach().contiguous()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'raw_sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def validate_packet(payload, original_ids, expected_attack, expected_index):
    """Validate a primitive saved packet, never invoking a model or tokenizer."""
    import torch
    require(isinstance(payload, dict), 'Capture payload is not a primitive dictionary')
    require(payload.get('attack') == expected_attack, 'Payload attack identity differs from its prediction/index')
    require(type(payload.get('capture_index')) is int and payload['capture_index'] == expected_index,
            'Payload capture index differs from its prediction/index')
    require(isinstance(original_ids, list) and original_ids
            and all(type(token) is int and 0 <= token < 50257 for token in original_ids),
            'Prediction original IDs are not native GPT2 vocabulary IDs')
    local = payload.get('native_local_data_input')
    native = payload.get('native_transmitted_result')
    require(isinstance(local, dict) and isinstance(native, dict), 'Native input/wire dictionary is absent')
    ids, input_mask = local.get('input_ids'), local.get('attention_mask')
    wire, wire_mask = native.get('inputs_embeds'), native.get('attention_mask')
    require(all(isinstance(value, torch.Tensor) for value in [ids, input_mask, wire, wire_mask]),
            'Native IDs, masks or transmitted seam tensor is missing')
    require(all(value.device.type == 'cpu' for value in [ids, input_mask, wire, wire_mask]),
            'Saved packet audit accepts CPU tensors only')
    length = len(original_ids)
    require(1 <= length <= 1024, 'Native GPT2 input length exceeds its retained positional capacity')
    require(ids.dtype == torch.int64 and tuple(ids.shape) == (1, length), 'Native input IDs must be int64[1,T]')
    require(ids.tolist() == [original_ids], 'Prediction truth IDs differ from actual captured native input IDs')
    require(input_mask.dtype == torch.int64 and tuple(input_mask.shape) == (1, length),
            'Native input mask must be int64[1,T]')
    require(bool(((input_mask == 0) | (input_mask == 1)).all()) and bool(input_mask.any()),
            'Native input mask is not a nonempty binary mask')
    require(wire.dtype == torch.float32 and tuple(wire.shape) == (1, length, 768),
            'Native transmitted seam must be FP32[1,T,768]')
    require(bool(torch.isfinite(wire).all()), 'Native transmitted seam contains a nonfinite value')
    require(wire_mask.dtype == torch.float32 and tuple(wire_mask.shape) == (1, 1, 1, length),
            'Native GPT2 additive mask must be FP32[1,1,1,T]')
    require(bool(torch.isfinite(wire_mask).all()), 'Native additive mask contains a nonfinite value')
    expected_mask = (1.0 - input_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
    require(torch.equal(wire_mask, expected_mask), 'Transmitted additive mask disagrees with original binary mask')
    outer = payload.get('attention_mask')
    if outer is not None:
        require(isinstance(outer, torch.Tensor) and outer.device.type == 'cpu', 'Outer mask is not a CPU tensor/None')
        if tuple(outer.shape) == tuple(input_mask.shape):
            require(outer.dtype in [torch.int64, torch.float32, torch.bool] and torch.equal(outer, input_mask),
                    'Outer binary mask disagrees with captured native input')
        else:
            require(outer.dtype == torch.float32 and tuple(outer.shape) == tuple(wire_mask.shape)
                    and torch.equal(outer, expected_mask), 'Outer additive mask disagrees with captured native input')
    return {'capture_index': expected_index, 'attack': expected_attack, 'tokens': length,
            'wire_shape': list(wire.shape), 'wire_dtype': str(wire.dtype), 'wire_finite': True,
            'input_ids_match_prediction': True, 'binary_additive_masks_consistent': True,
            'outer_mask_kind': 'none' if outer is None else 'binary' if outer.ndim == 2 else 'additive',
            'tensors': {name: _tensor_receipt(value) for name, value in
                        [('input_ids', ids), ('input_mask', input_mask), ('wire', wire), ('wire_mask', wire_mask)]}}


def _write_receipt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def audit_run(out, receipt_out):
    """Return/write PASS only after every retained native packet is verified."""
    out, receipt_out = Path(out).resolve(), Path(receipt_out).resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = {'schema': 'bounded-native-capture-audit-v1', 'status': 'FAIL',
              'output': str(out), 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(Path(__file__)), 'capture_files_verified': 0,
              'wire_tensors_verified': 0, 'input_ids_masks_verified': 0,
              'expected_captures': EXPECTED_CAPTURES, 'checks': [], 'captures': [],
              'model_imported_constructed_or_executed': False,
              'training_inference_or_attacks_executed': False, 'device': 'cpu'}
    source_files = {}
    try:
        import torch
        require(not torch.cuda.is_initialized(), 'CUDA was already initialized; use a fresh CPU-only auditor process')
        torch.set_num_threads(2)
        def read(name):
            path = out / name
            require(path.is_file() and not path.is_symlink(), 'Missing/nonregular source metadata: ' + name)
            source_files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
            return json.loads(path.read_text())
        terminal, cfg = read('terminal.json'), read('effective-run.json')
        require(terminal['status'] == 'complete' and terminal['exit_code'] == 0,
                'Source run terminal metadata is not complete')
        require(cfg['phase'] == 'bounded-study' and cfg['condition'] in ['mid', 'none'],
                'Derivative capture gate accepts only the fixed bounded MID/NONE study')
        captures = read('capture-index.json')
        prediction_path = out / 'attack-predictions.jsonl'
        require(prediction_path.is_file() and not prediction_path.is_symlink(), 'Missing/nonregular prediction file')
        source_files['attack-predictions.jsonl'] = {'sha256': sha(prediction_path), 'bytes': prediction_path.stat().st_size}
        predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
        require(len(captures) == len(predictions) == EXPECTED_CAPTURES, 'Expected all 400 captures and predictions')
        require(all(type(item.get('index')) is int for item in captures)
                and [item['index'] for item in captures] == list(range(EXPECTED_CAPTURES)),
                'Capture index is not an ordered one-to-one 0..399 mapping')
        ordinals = [row.get('capture_ordinal') for row in predictions]
        require(all(type(index) is int for index in ordinals)
                and sorted(ordinals) == list(range(EXPECTED_CAPTURES)),
                'Prediction-to-capture ordinals are not a one-to-one 0..399 mapping')
        require({row['attack'] for row in predictions} == ATTACKS
                and all(sum(row['attack'] == attack for row in predictions) == 200 for attack in ATTACKS),
                'Expected both native inversion families with 200 predictions each')
        paths = [item['path'] for item in captures]
        require(len(set(paths)) == EXPECTED_CAPTURES, 'Duplicate indexed capture paths')
        require(sum(item['bytes'] for item in captures) <= cfg['guards']['maximum_capture_bytes'],
                'Captured artifacts exceed the run-declared byte allowance')
        prediction_by_capture = {row['capture_ordinal']: row for row in predictions}
        resolved_paths = set()
        for item in captures:
            ordinal = item['index']
            relative = Path(item['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'Capture path escaped the retained run')
            path = out / relative
            resolved = path.resolve()
            require(resolved.is_relative_to(out) and not path.is_symlink() and path.is_file(), 'Missing/unsafe capture path')
            require(resolved not in resolved_paths, 'Multiple index paths resolve to the same capture file')
            resolved_paths.add(resolved)
            require(type(item['bytes']) is int and 0 < item['bytes'] <= MAX_CAPTURE_BYTES,
                    'Capture packet exceeds the bounded native schema file-size limit')
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Actual capture hash/size mismatch')
            row = prediction_by_capture[ordinal]
            require(item['attack'] == row['attack'], 'Capture/prediction attack identity mismatch')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            result = validate_packet(payload, row['original_ids'], row['attack'], ordinal)
            result.update(path=item['path'], file_sha256=item['sha256'], bytes=item['bytes'], status='PASS')
            report['captures'].append(result)
            report['capture_files_verified'] += 1
            report['wire_tensors_verified'] += 1
            report['input_ids_masks_verified'] += 1
        require(not torch.cuda.is_initialized(), 'CPU artifact validation unexpectedly initialized CUDA')
        require(not any(name in sys.modules for name in ['transformers', 'datasets']), 'Model/dataset runtime imported in auditor process')
        report.update(status='PASS', phase=cfg['phase'], condition=cfg['condition'], seed=cfg['seed'],
                      capture_bytes_verified=sum(item['bytes'] for item in captures),
                      minimum_tokens=min(item['tokens'] for item in report['captures']),
                      maximum_tokens=max(item['tokens'] for item in report['captures']),
                      actual_wire_dtype='torch.float32', cuda_initialized=False)
        report['checks'] = [{'id': key, 'status': 'PASS'} for key in
                            ['400_bijective_index_prediction_paths', 'two_families_200_each',
                             'all_actual_file_hashes', 'all_actual_native_input_ids',
                             'all_binary_additive_masks', 'all_fp32_wire_shapes_finite', 'cpu_only_no_models']]
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        report['checks'].append({'id': 'complete_capture_gate', 'status': 'FAIL', 'message': str(exc)})
    report['source_artifacts'] = source_files
    _write_receipt(receipt_out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt-out', type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.output, args.receipt_out)
    print(json.dumps({key: result[key] for key in ['status', 'capture_files_verified', 'wire_tensors_verified', 'source_sha256']}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
