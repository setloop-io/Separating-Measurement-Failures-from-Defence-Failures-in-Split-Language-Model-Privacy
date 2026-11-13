"""Matched, state-preserving measurements for the corrected development pilot."""
import hashlib
import json
import math
from contextlib import contextmanager


@contextmanager
def evaluation_state(split, state):
    import torch
    models = [split.model, *state.get('replica_models', [])]
    modes = [(m, m.training) for model in models for m in model.modules()]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        for model in models:
            model.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode


def utility(split, cfg, state, batches, warmup):
    """Same evaluation documents and actual current trained state in every arm.

    Mixup uses a throwaway pool warmed only on public auxiliary documents;
    evaluation never changes the training pool, replica journal, or RNG.
    """
    import torch
    import torch.nn.functional as F
    from attacker import splitbench as sb
    from attacker.trained_defenses import SecretMixup, coordinate_median
    local = dict(state)
    if 'mixup' in state:
        local['mixup'] = SecretMixup(state['mixup'].config)
    losses, count = 0., 0
    with evaluation_state(split, state), torch.no_grad():
        if 'mixup' in local:
            for ids in warmup:
                ids = ids.to(split.device)
                local['mixup'].apply(split.head_forward(ids), tokens=ids)
        for ids in batches:
            ids = ids.to(split.device)
            h = split.head_forward(ids)
            if 'replica' in local:
                y = coordinate_median([f(h) for f in local['replica'].channel_forwards])
                logits = split.lm_head(split.ln_f(y))
            else:
                local['mixup_ids'] = ids
                wire = sb._apply_seam_defense(h, cfg, local)
                logits = sb._server_rest_forward(split, wire, cfg, local)
            n = ids[:, 1:].numel()
            losses += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                           ids[:, 1:].reshape(-1), reduction='sum'))
            count += n
    ce = losses / count
    return {'cross_entropy': ce, 'perplexity': math.exp(min(700., ce)),
            'tokens': count, 'documents': len(batches), 'partition': 'utility_eval',
            'evaluation_contract': 'matched-final-state-v2'}


def representation_bundle(split, state, encoded, document_ids):
    """Public labeled auxiliary releases; disjoint victim documents scored later.

    Fresh post-training releases, one per source document, not training-step IDs.
    Encoder access is a declared auxiliary release oracle, not granted weights.
    """
    import torch
    from attacker.attacks.representation_privacy import validate_bundle
    bundle = {'schema': 'representation_privacy.v1', 'provenance': {
        'source': 'document-disjoint public auxiliary release oracle',
        'defender': 'in-repo trained bottleneck', 'release_stage': 'post-training encoder output'},
        'calibration': [], 'evaluation': []}
    seen_tokens = set()
    with evaluation_state(split, state), torch.no_grad():
        for target, source in [('calibration', 'calibration'), ('evaluation', 'victim_train')]:
            for ident, ids in zip(document_ids[source], encoded[source], strict=True):
                digest = hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()
                if digest in seen_tokens:
                    raise ValueError('duplicate token window in representation partitions')
                seen_tokens.add(digest)
                h = split.head_forward(ids.to(split.device))
                z = state['tb'].encode(h)[0].cpu().tolist()
                bundle[target].append({'document_id': ident, 'released': z, 'label': int(ids[0, 0])})
    validate_bundle(bundle)
    return bundle
