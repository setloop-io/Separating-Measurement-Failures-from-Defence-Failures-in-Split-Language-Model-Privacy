"""Explicit HF 4.39 ports for the native cache-disabled Llama path only.

Native method bodies and source files remain unchanged. This changes dependency
helper dispatch and wraps generation preparation outside the native class body.
"""
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

PROFILE = "native-llama-hf439-cache-disabled-v1"
HF_SOURCE_SHA256 = "1ca8e709720f00eb107b98588cf5b421dd23fa3f3b771e7e49ad79d7ea312bd6"
NATIVE_SOURCE_SHA256 = "61ab9b00c7f3e1462cd45b9f8e26243d2f9f6c065d78e9f219428c9769f954a0"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_cache_disabled_generation(model):
    """Explicit external policy; needed before native generation, not training."""
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    return {"profile": PROFILE, "config_use_cache": False,
            "generation_config_use_cache": getattr(getattr(model, "generation_config", None), "use_cache", None)}


def install(native, receipt_path, original_source=None):
    import torch
    from transformers.models.llama import modeling_llama as hf
    if importlib.metadata.version("transformers") != "4.39.0":
        raise RuntimeError("Llama compatibility profile requires exact Transformers 4.39.0")
    if _sha(hf.__file__) != HF_SOURCE_SHA256:
        raise RuntimeError("HF 4.39 Llama source bytes differ from reviewed package")
    original_source = Path(original_source) if original_source else Path(__file__).with_name("at_setup") / "vendor/upstream/src/models/llm_models/llama.py"
    if _sha(original_source) != NATIVE_SOURCE_SHA256:
        raise RuntimeError("Original native Llama source differs from its pinned bytes")
    names = {"LlamaModelHead", "LlamaModelBody", "LlamaModelTail", "LlamaTailForCausalLM"}
    def bodies(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body
                if isinstance(node, ast.ClassDef) and node.name in names}
    if bodies(original_source) != bodies(native.__file__):
        raise RuntimeError("Native Llama class bodies changed outside the declared external port")
    old_mask = hf.LlamaModel._update_causal_mask
    old_prepare = native.LlamaTailForCausalLM.prepare_inputs_for_generation
    if getattr(old_mask, "_native_llama_compat", None) or getattr(old_prepare, "_native_llama_compat", None):
        raise RuntimeError("Llama compatibility profile must be installed exactly once in a fresh process")
    if list(inspect.signature(old_mask).parameters) != ["self", "attention_mask", "input_tensor", "cache_position"]:
        raise RuntimeError("Unexpected dependency causal-mask helper API")

    def compatible_mask(self, attention_mask, input_tensor, cache_position, past_key_values=None, output_attentions=False):
        if past_key_values is not None:
            raise RuntimeError("Compatibility profile excludes KV caches")
        if any(getattr(layer.self_attn, "past_key_value", None) is not None for layer in self.layers):
            raise RuntimeError("Compatibility profile excludes static/internal KV caches")
        length = input_tensor.shape[1]
        expected = torch.arange(length, device=input_tensor.device)
        if cache_position is None or cache_position.ndim != 1 or not torch.equal(cache_position, expected):
            raise RuntimeError("Cache-disabled native mask requires complete-prefix positions 0..L-1")
        return old_mask(self, attention_mask, input_tensor, cache_position)

    def full_prefix_prepare(self, input_ids, past_key_values=None, attention_mask=None,
                            inputs_embeds=None, cache_position=None, **kwargs):
        requested = kwargs.get("use_cache", getattr(getattr(self, "generation_config", None), "use_cache", None))
        if requested is not False or past_key_values is not None or self.config.use_cache is not False:
            raise RuntimeError("Generation port requires explicit cache-disabled configuration and no past values")
        length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
        if attention_mask is not None and (attention_mask.ndim != 2 or attention_mask.shape[-1] != length):
            raise RuntimeError("Generation port requires the complete-prefix 2D attention mask")
        full_positions = torch.arange(length, device=device)
        kwargs["use_cache"] = False
        # HF4.39 GenerationMixin supplies only the newest cache_position after
        # step one even with use_cache=False. Native slices still consume all
        # prefix tokens, so regenerate complete-prefix positions every step.
        result = old_prepare(self, input_ids, past_key_values=None, attention_mask=attention_mask,
                             inputs_embeds=inputs_embeds, cache_position=full_positions, **kwargs)
        if not torch.equal(result["cache_position"], full_positions):
            raise RuntimeError("Native preparation did not retain complete-prefix positions")
        return result

    compatible_mask._native_llama_compat = PROFILE
    full_prefix_prepare._native_llama_compat = PROFILE
    hf.LlamaModel._update_causal_mask = compatible_mask
    native.LlamaTailForCausalLM.prepare_inputs_for_generation = full_prefix_prepare
    receipt = {"schema": "native-llama-runtime-compatibility-v1", "profile": PROFILE,
               "transformers": "4.39.0", "hf_source_sha256": _sha(hf.__file__),
               "native_original_source_sha256": _sha(original_source), "native_runtime_source_sha256": _sha(native.__file__),
               "native_class_AST_preserved": True, "adapter_source_sha256": _sha(__file__),
               "mask_port": "Native five-argument call delegates to exact HF4.39 three-argument helper; only full-prefix no-cache positions admitted",
               "generation_port": "External native preparation wrapper rebuilds full-prefix cache_position; native position_ids/mask preparation retained",
               "excluded": ["KV cache", "static cache", "incremental input", "different Transformers source/version"],
               "numerical_validation_performed_by_install": False}
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
