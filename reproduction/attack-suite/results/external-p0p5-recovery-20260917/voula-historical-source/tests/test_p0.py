"""P0 instrument-qualification self-tests (RESEARCH_PLAN.md section 3).

These tests qualify the instrument blind to pilot outcomes: parameter-update
verification, pairing bit-identity, hand-computed FLOPs, schema compliance,
quantization byte accounting, and seed separation. They use a tiny CPU model;
no pilot outcomes are inspected anywhere in this file.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from jsonschema import ValidationError

from pcr.data import (
    load_corpus,
    make_eval_batches,
    make_probe_batch,
    make_train_sampler,
    split_corpus,
)
from pcr.instrument import RemoteInstrument, count_block_flops, remote_step_flops
from pcr.manifest import build_manifest, validate_manifest
from pcr.modes import remote_parameters, trainable_parameters
from pcr.model import BackboneConfig
from pcr.split import AffineQuantizer, SplitBackbone, SplitConfig
from pcr.train import OptimConfig, build_initialized_model, train_model
from pcr.transcript import (
    COUNTED_CHANNELS,
    TranscriptCapture,
    build_inventory,
    validate_inventory,
)
from pcr.utils import parameters_checksum, state_dict_checksum

DEVICE = torch.device("cpu")
TINY_BACKBONE = BackboneConfig(vocab_size=256, d_model=64, n_heads=2, n_layers=4, d_ff=256, max_seq_len=32)
TINY_SPLIT = SplitConfig(start=1, span=2, width=32, q_bits=8)
SEED = 101
BLOCK_ID = "pb-101"


@pytest.fixture(scope="module")
def corpus_regions():
    train_region, heldout = split_corpus(load_corpus())
    return train_region, heldout


def tiny_sampler(region):
    return make_train_sampler(region, batch_size=8, seq_len=32, seed=SEED)


def run_tiny(mode, corpus_regions, steps=20, seed=SEED, block_id=BLOCK_ID):
    train_region, heldout = corpus_regions
    split = None if mode == "all_local" else TINY_SPLIT
    model = build_initialized_model(TINY_BACKBONE, mode, split, seed=seed, pairing_block_id=block_id)
    capture = TranscriptCapture(AffineQuantizer(TINY_SPLIT.q_bits)) if split else None
    instrument = RemoteInstrument(TINY_SPLIT.width if split else 0, 32, mode)
    eval_batches = make_eval_batches(heldout, 2, 8, 32, seed)
    result = train_model(
        model,
        tiny_sampler(train_region),
        OptimConfig(steps=steps),
        DEVICE,
        capture=capture,
        instrument=instrument,
        eval_batches=eval_batches,
        eval_every=steps,
    )
    return model, capture, instrument, result


class TestParameterUpdateVerification:
    def test_trained_mode_updates_remote_and_backbone(self, corpus_regions):
        model = build_initialized_model(TINY_BACKBONE, "trained", TINY_SPLIT, SEED, BLOCK_ID)
        remote_before = parameters_checksum(remote_parameters(model))
        backbone_before = state_dict_checksum(model.backbone_state_dict())
        train_model(model, tiny_sampler(corpus_regions[0]), OptimConfig(steps=20), DEVICE)
        assert parameters_checksum(remote_parameters(model)) != remote_before
        assert state_dict_checksum(model.backbone_state_dict()) != backbone_before

    def test_frozen_remote_bit_identical_backbone_updates(self, corpus_regions):
        model = build_initialized_model(TINY_BACKBONE, "frozen_random", TINY_SPLIT, SEED, BLOCK_ID)
        remote_before = parameters_checksum(remote_parameters(model))
        backbone_before = state_dict_checksum(model.backbone_state_dict())
        frozen = [p for p in remote_parameters(model) if p.requires_grad]
        assert not frozen, "frozen_random must not expose trainable remote params"
        train_model(model, tiny_sampler(corpus_regions[0]), OptimConfig(steps=20), DEVICE)
        assert parameters_checksum(remote_parameters(model)) == remote_before  # bit-identical
        assert state_dict_checksum(model.backbone_state_dict()) != backbone_before

    def test_removed_remote_has_zero_remote_trainable_params(self, corpus_regions):
        model, _, _, _ = run_tiny("removed_remote", corpus_regions)
        assert remote_parameters(model) == []
        remote_prefix = f"blocks.{TINY_SPLIT.start}.remote."
        assert not any(
            name.startswith(remote_prefix)
            for name, p in model.named_parameters()
            if p.requires_grad
        )
        # encoder/decoder (trusted) must still be present and trainable
        assert any("encoder." in name for name, p in model.named_parameters() if p.requires_grad)
        assert any("decoder." in name for name, p in model.named_parameters() if p.requires_grad)

    def test_backbone_updates_in_every_mode(self, corpus_regions):
        for mode in ("all_local", "trained", "frozen_random", "removed_remote", "oracle"):
            split = None if mode == "all_local" else TINY_SPLIT
            model = build_initialized_model(TINY_BACKBONE, mode, split, SEED, BLOCK_ID)
            key = "backbone_state_dict" if isinstance(model, SplitBackbone) else "state_dict"
            before = state_dict_checksum(getattr(model, key)())
            train_model(model, tiny_sampler(corpus_regions[0]), OptimConfig(steps=20), DEVICE)
            assert state_dict_checksum(getattr(model, key)()) != before, mode


class TestPairing:
    def test_paired_trained_frozen_bit_identical_init_and_batches(self, corpus_regions):
        train_region, _ = corpus_regions
        trained = build_initialized_model(TINY_BACKBONE, "trained", TINY_SPLIT, SEED, BLOCK_ID)
        frozen = build_initialized_model(TINY_BACKBONE, "frozen_random", TINY_SPLIT, SEED, BLOCK_ID)
        assert state_dict_checksum(trained.backbone_state_dict()) == state_dict_checksum(
            frozen.backbone_state_dict()
        )
        assert parameters_checksum(remote_parameters(trained)) == parameters_checksum(
            remote_parameters(frozen)
        )
        sampler_a = make_train_sampler(train_region, 8, 32, seed=SEED)
        sampler_b = make_train_sampler(train_region, 8, 32, seed=SEED)
        for step in range(5):
            xa, ya = sampler_a.batch(step)
            xb, yb = sampler_b.batch(step)
            assert torch.equal(xa, xb) and torch.equal(ya, yb)

    def test_different_block_id_changes_remote_init_only(self):
        a = build_initialized_model(TINY_BACKBONE, "trained", TINY_SPLIT, SEED, "pb-101")
        b = build_initialized_model(TINY_BACKBONE, "trained", TINY_SPLIT, SEED, "pb-202")
        assert parameters_checksum(remote_parameters(a)) != parameters_checksum(remote_parameters(b))
        assert state_dict_checksum(a.backbone_state_dict()) == state_dict_checksum(
            b.backbone_state_dict()
        )


class TestFlopCounter:
    def test_hand_computed_case(self):
        # dim=2, d_ff=4, seq_len=3:
        #   projections 8*2*2 = 32; attention 4*3*2 = 24; mlp 4*2*4 = 32 -> 88
        assert count_block_flops(dim=2, d_ff=4, seq_len=3) == 88

    def test_step_flops_by_mode(self):
        fwd = remote_step_flops(width=32, seq_len=8, tokens=10, mode="trained")
        assert fwd["fwd_only"] == count_block_flops(32, 128, 8) * 10
        assert fwd["fwd_bwd"] == 3 * fwd["fwd_only"]
        frozen = remote_step_flops(width=32, seq_len=8, tokens=10, mode="frozen_random")
        assert frozen["fwd_bwd"] == 2 * frozen["fwd_only"]
        assert remote_step_flops(32, 8, 10, "removed_remote") == {"fwd_only": 0, "fwd_bwd": 0}
        assert remote_step_flops(32, 8, 10, "all_local") == {"fwd_only": 0, "fwd_bwd": 0}


class TestTranscriptSchema:
    def test_generated_inventory_validates(self, corpus_regions):
        _, capture, _, result = run_tiny("trained", corpus_regions)
        inventory = build_inventory(capture.total_bytes, result["remote_state_checksums"], True)
        validate_inventory(inventory)  # must not raise
        for ch in COUNTED_CHANNELS:
            assert inventory["channels"][ch] is True
            assert inventory["accumulated_bytes"][ch] > 0
        assert inventory["history"] == {"per_step": True, "accumulated": True}
        assert inventory["omissions"] == []

    def test_tampered_inventory_fails(self, corpus_regions):
        _, capture, _, result = run_tiny("trained", corpus_regions)
        inventory = build_inventory(capture.total_bytes, result["remote_state_checksums"], True)
        tampered = copy.deepcopy(inventory)
        del tampered["channels"]["outbound_gradients"]
        with pytest.raises(ValidationError):
            validate_inventory(tampered)


class TestManifestSchema:
    def test_generated_manifest_validates(self, corpus_regions, tmp_path):
        run_tiny("trained", corpus_regions, steps=5)
        artifact = tmp_path / "curve.jsonl"
        artifact.write_text('{"step": 0}\n')
        resolved = {
            "run_id": "p0-selftest-manifest",
            "cohort": "exploratory",
            "protocol_version": "p1-pilot-0.1",
            "mode": "trained",
            "seed": SEED,
            "pairing_block_id": BLOCK_ID,
            "backbone": TINY_BACKBONE.to_dict(),
            "split": TINY_SPLIT.to_dict(),
        }
        manifest = build_manifest(
            resolved,
            status="completed",
            artifact_paths=[artifact],
            captured_channels=list(COUNTED_CHANNELS) + ["remote_state"],
        )
        validate_manifest(manifest)
        assert len(manifest["git_commit"]) >= 7
        assert len(manifest["resolved_config_sha256"]) == 64
        assert manifest["artifacts"][0]["sha256"] and len(manifest["artifacts"][0]["sha256"]) == 64


class TestQuantization:
    def test_serialized_size_and_counter_match_measured(self):
        quantizer = AffineQuantizer(q_bits=8)
        torch.manual_seed(0)
        batch, seq, width = 8, 32, 32
        x = torch.randn(batch, seq, width)
        payload_per_position = quantizer.payload_nbytes(x) // (batch * seq)
        assert payload_per_position == width  # 8 bits -> 1 byte per position per unit width
        codes = quantizer.integer_codes(x)
        assert codes.dtype == torch.uint8 and codes.shape == x.shape

        capture = TranscriptCapture(quantizer)
        measured_total = 0
        for step in range(3):
            capture.begin_step()
            capture.on_forward_values(torch.randn(batch, seq, width))
            measured_total += quantizer.serialized_nbytes(torch.zeros(batch, seq, width))
            capture.end_step()
        assert capture.total_bytes["forward_values"] == measured_total

    def test_round_trip_error_bounded(self):
        quantizer = AffineQuantizer(q_bits=8)
        torch.manual_seed(1)
        x = torch.randn(4, 16, 32) * 3.0
        x_hat = quantizer.quantize_dequantize(x)
        x_range = (x.max() - x.min()).item()
        bound = x_range / (2 * (2**8 - 1))
        assert (x - x_hat).abs().max().item() <= bound + 1e-6

    def test_ste_gradient_passes_through(self):
        quantizer = AffineQuantizer(q_bits=8)
        x = torch.randn(2, 4, 8, requires_grad=True)
        z = quantizer.ste(x)
        z.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))


class TestSeedSeparation:
    def test_same_seed_reproduces_exactly(self, corpus_regions):
        train_region, _ = corpus_regions
        a = make_train_sampler(train_region, 8, 32, seed=101)
        b = make_train_sampler(train_region, 8, 32, seed=101)
        for step in (0, 3, 17):
            assert torch.equal(a.batch(step)[0], b.batch(step)[0])

    def test_different_seeds_differ(self, corpus_regions):
        train_region, _ = corpus_regions
        a = make_train_sampler(train_region, 8, 32, seed=101)
        b = make_train_sampler(train_region, 8, 32, seed=202)
        assert not torch.equal(a.batch(0)[0], b.batch(0)[0])

    def test_eval_batches_identical_across_runs_of_a_seed(self, corpus_regions):
        _, heldout = corpus_regions
        a = make_eval_batches(heldout, 4, 8, 32, seed=101)
        b = make_eval_batches(heldout, 4, 8, 32, seed=101)
        assert all(torch.equal(xa[0], xb[0]) for xa, xb in zip(a, b))
        c = make_eval_batches(heldout, 4, 8, 32, seed=202)
        assert not torch.equal(a[0][0], c[0][0])

    def test_probe_batch_identical_for_all_seeds(self, corpus_regions):
        _, heldout = corpus_regions
        p1 = make_probe_batch(heldout, 4, 32)
        p2 = make_probe_batch(heldout, 4, 32)
        assert torch.equal(p1[0], p2[0])

    def test_init_generators_separated_by_key(self):
        m1 = build_initialized_model(TINY_BACKBONE, "trained", TINY_SPLIT, 101, "pb-101")
        m2 = build_initialized_model(TINY_BACKBONE, "trained", TINY_SPLIT, 202, "pb-101")
        assert state_dict_checksum(m1.backbone_state_dict()) != state_dict_checksum(
            m2.backbone_state_dict()
        )
        # remote init keyed by block id only: same block id -> identical remote init
        assert parameters_checksum(remote_parameters(m1)) == parameters_checksum(
            remote_parameters(m2)
        )
