# Original capture replay correction — 19 September 2026

The failed eager replay was a kernel mismatch, demonstrated by a controlled attention-backend comparison on the hash-pinned seed101/depth3 first8 documents. Eager reproduced the previous error (max absolute 0.000244140625, RMS 1.4923332400940126e-6;1376 tolerance violations), whereas SDPA reproduced the saved activations bit-for-bit, with zero error.

The original v3 source loads GPT2LMHeadModel.from_pretrained without an eager override. The original and current split head-forward implementations are identical in the relevant operations. The diagnostic changes only attention execution while retaining exact checkpoint, ordered tokens, FP32, batch8, model architecture and device. Diagnostic ORX run:02bb0655-90d8-4ecd-bc2d-1f4e182ab02d; source:f8904c82f53561fd1322edc7807de604f44e6481.

Correction commit63a88c1 uses SDPA for mechanical capture replay only and requires torch.equal over every element, a stricter requirement than the previous tolerance. It retains the qualified strict exact-head loader. Public downstream placeholders are never executed by this activation-only check. Original h/g tensors and checkpoint files remain unchanged; conversion adds only the label-isolated schema and documented factor256 gradient loss-normalisation metadata.

The candidate attack remains the C-A-qualified eager implementation because its optimisation requires second derivatives. This correction does not assert bitwise equality between eager and SDPA. The adaptation must explicitly record capture_attention=sdpa and attack_attention=eager. No full-victim gradient replay, original published-setting reproduction, or full campaign admission follows from this activation replay result.

Full coverage verification runs: toula a4393a65-fccf-4ab0-ae56-833dc4acaba0 (seeds101–505), koula44979686-5ac4-4868-b4e6-d08a706dce8a (seeds606–1010), depths3/6,256documents each. Both runs completed successfully.


## Verified outcome

All 20 seed/depth captures, covering 5,120 document-depth rows, reproduce bit-for-bit on the original SDPA path. Maximum activation error is zero on both hosts. The exact-head activation-replay blocker is closed. Digest-bound per-host reports and summary are committed in `experiments/v4-replay-verification-20260919/`. The earlier failed attempts remain in ORX.

Full adaptation admission remains separate: fresh attack-seed 17/29 GRUs and their mechanical gates, complete source/input/control/environment bindings, explicit capture-versus-attack kernel declaration, and resource admission. The 40 GPU-hour request remains pending; no full comparison has launched. This result verifies activation identity, not full downstream gradient reproduction.
