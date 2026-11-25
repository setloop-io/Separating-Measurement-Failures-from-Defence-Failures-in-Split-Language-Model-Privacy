"""ORX-only tiny CPU Llama numerical/generation admission; no pretrained assets."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("ORX_RUN_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Tiny reference execution requires recorded ORX and CUDA_VISIBLE_DEVICES=''")
    def deadline(signum, frame):
        raise TimeoutError("Tiny CPU reference exceeded its 120-second internal guard")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    import torch
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM
    from import_compat_llama import prepare
    import llama_runtime_compat as compat
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.random.default_generator.manual_seed(31861)
    upstream = args.upstream_root.resolve()
    proof = json.loads(Path(__file__).with_name("at_setup").joinpath("vendor/upstream-manifest.json").read_text())
    for item in proof["files"]:
        path = upstream / item["path"]
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Pinned upstream source differs: " + item["path"])
    runtime = out / "runtime-upstream"
    shutil.copytree(upstream, runtime)
    prepare(runtime, out / "import-compatibility.json")
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    from models.llm_models import llama as native
    compat.install(native, out / "runtime-compatibility.json")
    config = LlamaConfig(vocab_size=37, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         use_cache=False, pad_token_id=0, bos_token_id=1, eos_token_id=None,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    reference = LlamaForCausalLM(copy.deepcopy(config)).cpu().float().eval()

    class SplitReference(native.LlamaTailForCausalLM):
        """External test bridge; each selected native forward body is unchanged."""
        def __init__(self):
            super().__init__(copy.deepcopy(config))
            self.load_state_dict(reference.state_dict(), strict=True)
            self.reference_head = native.LlamaModelHead(copy.deepcopy(config))
            self.reference_body = native.LlamaModelBody(copy.deepcopy(config))
            for part in [self.reference_head, self.reference_body]:
                loaded = part.load_state_dict(reference.model.state_dict(), strict=False)
                if loaded.missing_keys or loaded.unexpected_keys != ["norm.weight"]:
                    raise RuntimeError("Unexpected native split weight mapping")
            self.reference_head.vfl_split(range(0, 1))
            self.reference_body.vfl_split(range(1, 2))
            self.vfl_split(range(2, 4))
            compat.configure_cache_disabled_generation(self)
            self.prepared_positions = []

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, use_cache=False, output_attentions=None, output_hidden_states=None,
                    return_dict=None, cache_position=None, **kwargs):
            if use_cache is not False or past_key_values is not None:
                raise RuntimeError("Reference bridge is full-prefix/cache-disabled only")
            length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            if cache_position is None:
                cache_position = torch.arange(length, device="cpu")
            self.prepared_positions.append(cache_position.tolist())
            shared = dict(position_ids=position_ids, use_cache=False, cache_position=cache_position)
            head = self.reference_head(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                       attention_mask=attention_mask, **shared)
            body = self.reference_body(**head, **shared)
            return super().forward(**body, **shared, return_dict=True)

    split = SplitReference().cpu().float().eval()
    if any(parameter.device.type != "cpu" for model in [reference, split] for parameter in model.parameters()):
        raise RuntimeError("Tiny reference tensors must remain on CPU")
    cases = [
        ("unpadded", [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]], [[1]*5, [1]*5], False),
        ("left_padding", [[0, 0, 4, 5, 6], [0, 9, 10, 11, 12]], [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], False),
        ("right_padding", [[4, 5, 6, 0, 0], [9, 10, 11, 12, 0]], [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], False),
        ("explicit_position_offset", [[4, 5, 6, 7, 8]], [[1]*5], True),
    ]
    checks, raw_forward, raw_generation = [], [], []
    with torch.no_grad():
        for name, ids, mask, offset in cases:
            ids, mask = torch.tensor(ids, device="cpu"), torch.tensor(mask, device="cpu")
            positions = mask.long().cumsum(-1)-1
            positions.masked_fill_(mask == 0, 1)
            if offset:
                positions += 3
            shared = dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                          cache_position=torch.arange(ids.shape[1], device="cpu"), use_cache=False, return_dict=True)
            expected, actual = reference(**shared).logits, split(**shared).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            checks.append({"case": name, "max_absolute_logit_error": float((actual-expected).abs().max()), "passed": True})
            raw_forward.append({"case": name, "input_ids": ids.tolist(), "attention_mask": mask.tolist(),
                                "position_ids": positions.tolist(), "reference_logits": expected.tolist(), "native_split_logits": actual.tolist()})
        for name, ids, mask, _ in cases[:2]:
            ids, mask = torch.tensor(ids, device="cpu"), torch.tensor(mask, device="cpu")
            prompt = ids.clone()
            expected_ids, working_mask, manual_logits = ids.clone(), mask.clone(), []
            for step in range(3):
                pos = working_mask.long().cumsum(-1)-1
                pos.masked_fill_(working_mask == 0, 1)
                logits = reference(input_ids=expected_ids, attention_mask=working_mask, position_ids=pos,
                                   cache_position=torch.arange(expected_ids.shape[1], device="cpu"), use_cache=False).logits[:, -1, :]
                manual_logits.append(logits.tolist())
                expected_ids = torch.cat([expected_ids, logits.argmax(-1, keepdim=True)], dim=1)
                working_mask = torch.cat([working_mask, torch.ones((len(ids), 1), dtype=working_mask.dtype)], dim=1)
            split.prepared_positions = []
            generated = split.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=3,
                                       do_sample=False, use_cache=False, return_dict_in_generate=True, output_scores=True,
                                       eos_token_id=None, pad_token_id=0)
            if not torch.equal(generated.sequences, expected_ids) or len(generated.scores) != 3:
                raise RuntimeError("Native split multi-step greedy generation differs from monolithic full-prefix reference")
            for step, (expected_logits, scores) in enumerate(zip(manual_logits, generated.scores)):
                torch.testing.assert_close(scores, torch.tensor(expected_logits), rtol=1e-5, atol=1e-6)
                if split.prepared_positions[step] != list(range(prompt.shape[1]+step)):
                    raise RuntimeError("Generation failed complete-prefix cache-position coverage")
            checks.append({"case": name+"_three_step_generation", "steps": 3, "passed": True})
            raw_generation.append({"case": name, "prompt_ids": prompt.tolist(), "attention_mask": mask.tolist(),
                                   "reference_ids": expected_ids.tolist(), "native_split_ids": generated.sequences.tolist(),
                                   "reference_next_logits": manual_logits, "native_split_next_logits": [row.tolist() for row in generated.scores],
                                   "prepared_cache_positions": split.prepared_positions})
        for mode in ["use_cache_true", "past_values"]:
            try:
                split.prepare_inputs_for_generation(torch.tensor([[4, 5]]), attention_mask=torch.ones((1, 2), dtype=torch.long),
                                                    use_cache=mode == "use_cache_true", past_key_values=() if mode == "past_values" else None)
            except RuntimeError:
                checks.append({"case": "reject_"+mode, "passed": True})
            else:
                raise RuntimeError("Unsupported cached generation did not fail closed")
    write(out / "forward-logits.json", raw_forward)
    write(out / "generation-logits.json", raw_generation)
    summary = {"schema": "native-llama-tiny-reference-v1", "status": "PASS_TINY_CPU_NUMERICAL_AND_THREE_STEP_GENERATION",
               "checks": checks, "checks_passed": len(checks), "transformers": transformers.__version__, "torch": torch.__version__,
               "cpu_threads": torch.get_num_threads(), "model_config": config.to_dict(), "fresh_random_tiny_model": True,
               "pretrained_weights_or_dataset_used": False, "native_party_routing_or_AT_training_validated": False,
               "scope": "Native head/body/tail numeric bridge and external no-cache generation port only; actual seven-billion-parameter native task requires a separate canary",
               "wall_seconds": time.monotonic()-started,
               "artifact_hashes": {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ["forward-logits.json", "generation-logits.json", "runtime-compatibility.json", "import-compatibility.json"]}}
    write(out / "reference-summary.json", summary)
    signal.alarm(0)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
