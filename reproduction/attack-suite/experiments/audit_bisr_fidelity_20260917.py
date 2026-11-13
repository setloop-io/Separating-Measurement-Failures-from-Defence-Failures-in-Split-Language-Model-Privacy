#!/usr/bin/env python3
"""Recompute descriptive B1 evidence and a fail-closed fidelity verdict; no torch."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/phase2-status-2026-09-17'

def main():
    records = [json.loads(s) for s in (BASE/'splitllm-reference/attacks.jsonl').read_text().splitlines()]
    aggregate = json.loads((BASE/'splitllm-reference/attacks.jsonl.aggregate.json').read_text())
    reference = {}
    for name in sorted({r['name'] for r in records}):
        cells = [r for r in records if r['name'] == name]
        values = []
        correct = tokens = 0
        for c in cells:
            assert len(c['attacked_ids']) == len(c['true_ids'])
            hit = count = 0
            for p, t in zip(c['attacked_ids'], c['true_ids']):
                assert len(p) == len(t)
                hit += sum(a == b for a, b in zip(p, t))
                count += len(t)
            values.append(hit/count)
            correct += hit
            tokens += count
        assert abs(statistics.mean(values)-aggregate[name]['posacc']['mean']) < 1e-12
        reference[name] = {'batches':len(cells), 'steps':[c['local_step'] for c in cells],
                           'n_correct':correct,'n_tokens':tokens,'positional_accuracy':correct/tokens,
                           'rouge_l_f':statistics.mean(c['rouge_l_f'] for c in cells),
                           'token_set_f1':statistics.mean(c['tokacc'] for c in cells)}
    ours = json.loads((BASE/'bisr-fidelity-v1/our-bisr-attack.json').read_text())
    tokens = sum(c['n_tokens'] for c in ours['results'])
    own = {'batches':len(ours['results']), 'steps':[c['step'] for c in ours['results']], 'n_tokens':tokens,
           'beam_positional_accuracy':sum(c['n_correct'] for c in ours['results'])/tokens,
           'bre_bf_positional_accuracy':sum(c['bisr_opt']['n_correct'] for c in ours['results'])/tokens,
           'immutable_token_predictions_saved':False}
    files = [ROOT/'experiments/validation_protocol.v4.json', ROOT/'attacker/attacks/bisr_baseline.py', Path(__file__),
             BASE/'splitllm-reference/attacks.jsonl',BASE/'splitllm-reference/attacks.jsonl.aggregate.json',
             BASE/'bisr-fidelity-v1/our-bisr-attack.json',BASE/'bisr-fidelity-v1/our-bisr-attack.json.evidence.json']
    files += sorted((BASE/'bisr-audit-sources').glob('*'))
    verdict = {'schema':'llm_attacker.bisr_fidelity_audit.v1','date':'2026-09-17','verdict':'NOT_ESTABLISHED',
        'allows_v4_frozen_capture_evaluation':False,'predeclared_fidelity_tolerance':None,
        'tolerance_search_scope':['frozen v4','open commitments plan','phase2 fleet runbook','module docstring','koula SplitLLM FIDELITY-NOTES.md and run scripts'],
        'reason_codes':['NO_OPERATIONAL_PREDECLARED_TOLERANCE','REFERENCE_AUX_SUBSTITUTED','REFERENCE_TRAINING_SHRUNK','SINGLE_REFERENCE_SEED','UNMATCHED_SEAMS_DECODERS_AND_VICTIMS','UNMATCHED_BATCHES_AND_METRICS','LOCAL_PREDICTIONS_NOT_RETAINED','V4_BF_ORDER_PROSE_CONFLICT'],
        'reference':reference,'local':own,
        'upstream_commit':'c376590227f9718077765824af4088665bb78687',
        'local_run_commit':ours['provenance']['dtraining_commit'],
        'protocol_commit':'adf4cf688373b3983ebc3d6cbe2c1eebc99a97d2',
        'primary_sources':['https://arxiv.org/html/2409.00960v2','https://github.com/StupidTrees/SplitLLM/tree/c376590227f9718077765824af4088665bb78687'],
        'remediation':['Predeclare an operational prospective fidelity gate, with scope, tolerance justification, seeds, metrics and missing-data rule in an additive protocol amendment; preserve v4.',
                       'Obtain original SensiReplaced or explicitly amend the target to a substituted-data implementation comparison; do not call it exact published replication.',
                       'Align GRU/MLP scope, dual cuts, victim snapshots, sample IDs, auxiliary training budgets, GM/SM order and numerical environment before running.',
                       'Retain predictions and inputs with hashes; compute reference ROUGE/METEOR/token-set F1 and positional accuracy on both implementations.',
                       'Run the predeclared independent validation once, retain every failure, then resolve B1 before matched v3 evaluation.'],
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out=BASE/'bisr-fidelity-verdict.json'
    out.write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps({'verdict':verdict['verdict'],'reference':reference['BiSR(b+f)'],'local':own},indent=2))

if __name__ == '__main__':
    main()
