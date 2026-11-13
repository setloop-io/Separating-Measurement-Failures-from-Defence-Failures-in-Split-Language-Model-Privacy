#!/usr/bin/env python3
"""Required attack metadata (THREAT_MODEL_REVIEW Release 1, item 2).

The attack contract (``attacker.attacks.validate_contract``) enforces
``MODES`` and ``REQUIRES_LABELS`` but says nothing machine-checkable about
the documented threat axes: actor intent (passive/active/malicious),
persistence (per-/cross-epoch ...), observation surface, security
property, lifecycle stage, and framework mappings (OWASP / NIST / ATLAS).
This registry makes those coordinates first-class, one entry per attack,
validated as a whole by ``validate_registry`` (CI-enforced in tests and in
the registry self-test).

The registry is deliberately a single declarative table — metadata is a
claim about the attack, reviewed with the same doctrine as the frozen
gate: changing an entry is an explicit, reviewable edit.

Surfaces extend the paper's S1-S9 model with the stores/control-plane
surfaces the review found unmodeled (S10-S14); see README.md.

Pure stdlib; torch-free (repo rule).
"""

from . import threat_model

#: v2 (Lane B): adds the optional, CLOSED-vocab owasp_agentic_2026 mapping
#: (v1 had no agentic mapping). A named, version-bumped schema change.
SCHEMA = "llm_attacker.attack_metadata.v2"

#: Observation-surface identifiers (README.md).  S1-S9 are the
#: paper's split-LLM surface model; S10-S14 are the surfaces the threat
#: model review found unmodeled (control plane, stores, router, supply
#: chain, application context); S15-S18 are the confidential-compute
#: boundary. Trusted diagnostic truth is a control role, not an attacker
#: observation surface.
SURFACES = {
    "S1": "seam activations, forward (h* crossing to the body)",
    "S2": "seam gradients, backward (dL/dh* returned by the body)",
    "S3": "body weights (held by the cloud)",
    "S4": "outputs / logits / answers",
    "S5": "GPU memory residue (freed buffers, allocator state)",
    "S6": "cache / timing (KV/prefix cache, contention, acceptance)",
    "S7": "traffic shape (packet sizes/timing on the link)",
    "S8": "API reasoning blobs (encrypted CoT envelopes)",
    "S9": "labels / properties",
    "S10": "protocol and control-plane metadata (negotiation, epochs, keys)",
    "S11": "persistent stores and telemetry (captures, logs, checkpoints, "
           "crash dumps, backups)",
    "S12": "MoE router and scheduling metadata",
    "S13": "AI artifacts and supply chain (models, tokenizers, adapters, "
           "datasets, plugins)",
    "S14": "application context (prompts, system instructions, RAG "
           "documents, tool schemas, memory)",
    "S15": "attestation evidence, verifier policy, endpoint binding and "
           "key-release decisions",
    "S16": "confidential-host traces (page/access events, counters, crash "
           "and restart observations)",
    "S17": "FHE ciphertext layout, context, errors, status, timing and "
           "verification feedback",
    "S18": "remote computation and result binding (weights, operators, "
           "gradients, request/step identity)",
}

#: Security properties an attack can threaten.
SECURITY_PROPERTIES = ("confidentiality", "integrity", "availability",
                       "authenticity", "authorization", "privacy", "safety")

#: Lifecycle stages (aligned with threat_model scenario.lifecycle_stage).
LIFECYCLE_STAGES = ("training", "fine-tuning", "inference", "deployment",
                    "operations", "supply-chain")

OWASP_LLM_2026 = tuple(f"LLM{i:02d}" for i in range(1, 11))

#: The OWASP Top 10 for Agentic Applications 2026 (a CLOSED set: ASI01-ASI10).
#: An optional mapping validated against this vocabulary — so "ASI99" is a
#: descriptive error, never a silent pass.
OWASP_AGENTIC_2026 = tuple(f"ASI{i:02d}" for i in range(1, 11))

_FIELDS = {"actor_intent": threat_model.INTENTS,
           "persistence": threat_model.PERSISTENCE,
           "surfaces": tuple(SURFACES),
           "security_properties": SECURITY_PROPERTIES,
           "lifecycle_stages": LIFECYCLE_STAGES,
           "owasp_llm_2026": OWASP_LLM_2026}

# name -> {actor_intent, persistence, surfaces, security_properties,
#          lifecycle_stages, owasp_llm_2026, mitre_atlas, nist_ai_100_2}
ATTACK_METADATA = {
    # --- boundary / rotation family ---
    "accumulation": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "known-prefix": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "stale-key": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S1", "S10"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "seed-disclosure": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S10",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "deployment"),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "sharded": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "wire-eval": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "alignment-search": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "subspace-joint": {
        "actor_intent": "passive", "persistence": "cross-epoch",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "leak-accumulation": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "max-effort": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    # --- training-side ---
    "gradient-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S2",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "fine-tuning"),
        "owasp_llm_2026": ("LLM02",)},
    "membership": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("privacy",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02",)},
    "lira-membership": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1",), "security_properties": ("privacy",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02",)},
    "label-scoring": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S2", "S9"), "security_properties": ("privacy",),
        "lifecycle_stages": ("training", "fine-tuning"),
        "owasp_llm_2026": ("LLM02",)},
    "batch-label-recon": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S2", "S9"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02",)},
    "malicious-server": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S2", "S9"),
        "security_properties": ("confidentiality", "integrity"),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM05")},
    "active-cloud": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality", "integrity"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02", "LLM05")},
    # --- inference / output side ---
    "output-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S4",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02",)},
    "kv-timing": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S6",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02",)},
    "reasoning-replay": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S8",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "trace-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S8", "S4"), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    # --- latent / embedding ---
    "latent-probe": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "latent-sensitivity": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "latent-matching": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training",),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "seam-inversion": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "vec2text-inversion": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1", "S4"),
        "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "body-extraction": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S3",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM02",)},
    # --- systems / hardware seam ---
    "gpu-memory": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S5",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    "wire-shape": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S7",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    # --- deployment shapes ---
    "remote-fetch": {
        "actor_intent": "passive", "persistence": "single-request",
        "surfaces": ("S11",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("operations",),
        "owasp_llm_2026": ("LLM02",)},
    "mitm-proxy": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S7", "S10"),
        "security_properties": ("confidentiality", "integrity"),
        "lifecycle_stages": ("training", "inference", "deployment"),
        "owasp_llm_2026": ("LLM02",)},
    "ica-bss": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1",), "security_properties": ("confidentiality",),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    # --- threat-model review Release 2 ---
    "protocol-integrity": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S10", "S7"),
        "security_properties": ("integrity", "availability", "authenticity"),
        "lifecycle_stages": ("deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM05", "LLM06")},
    "multi-tenant-isolation": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S4", "S5", "S6"),
        "security_properties": ("confidentiality", "authorization"),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM06", "LLM08", "LLM09")},
    "artifact-telemetry-leak": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S11",),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM08")},
    "key-session-abuse": {
        "actor_intent": "malicious", "persistence": "cross-epoch",
        "surfaces": ("S10",),
        "security_properties": ("confidentiality", "authenticity"),
        "lifecycle_stages": ("deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM04", "LLM08")},
    "secure-compute-audit": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S7", "S10", "S11", "S15", "S16", "S17", "S18"),
        "security_properties": ("confidentiality", "integrity",
                                "authenticity", "authorization"),
        "lifecycle_stages": ("training", "inference", "deployment", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM05", "LLM06", "LLM08")},
    "resource-exhaustion": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S6", "S10"),
        "security_properties": ("availability",),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM06",)},
    # --- threat-model review Release 3 ---
    "poisoning-backdoor": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S13", "S2"),
        "security_properties": ("integrity", "safety"),
        "lifecycle_stages": ("training", "fine-tuning"),
        "owasp_llm_2026": ("LLM04", "LLM05")},
    "malicious-client": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S1", "S2", "S3"),
        "security_properties": ("confidentiality", "integrity",
                                "availability"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02", "LLM05", "LLM06")},
    "supply-chain": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S13",),
        "security_properties": ("integrity", "authenticity"),
        "lifecycle_stages": ("supply-chain", "deployment"),
        "owasp_llm_2026": ("LLM04", "LLM05")},
    "router-side-channel": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S12",),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM02", "LLM09")},
    "privacy-composition": {
        "actor_intent": "passive", "persistence": "cross-epoch",
        "surfaces": ("S1", "S2", "S6", "S7"),
        "security_properties": ("privacy", "confidentiality"),
        "lifecycle_stages": ("training", "inference", "operations"),
        "owasp_llm_2026": ("LLM02",)},
    # --- threat-model review Release 4 (GenAI application profile) ---
    "prompt-injection": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization",
                                "confidentiality"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01", "LLM02", "LLM03", "LLM08")},
    "hidden-context-exposure": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S14", "S4"),
        "security_properties": ("confidentiality",),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM08", "LLM02")},
    "rag-vector-abuse": {
        "actor_intent": "active", "persistence": "cross-session",
        "surfaces": ("S14", "S9"),
        "security_properties": ("confidentiality", "integrity",
                                "authorization"),
        "lifecycle_stages": ("inference", "operations"),
        "owasp_llm_2026": ("LLM01", "LLM02", "LLM05", "LLM09")},
    "output-sink-injection": {
        "actor_intent": "active", "persistence": "session",
        "surfaces": ("S14", "S4"),
        "security_properties": ("integrity", "safety", "authorization"),
        "lifecycle_stages": ("inference", "deployment"),
        "owasp_llm_2026": ("LLM10",)},
    "misinformation-integrity": {
        "actor_intent": "passive", "persistence": "single-request",
        "surfaces": ("S4", "S14"),
        "security_properties": ("safety", "integrity"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM07",)},
    # --- threat-model review Lane B (agentic / OWASP ASI profile) ---
    # owasp_llm_2026=("LLM03",) (Excessive Agency) satisfies the mandatory
    # LLM mapping AND is a genuine analogue of tool misuse; owasp_agentic_2026
    # maps the closed ASI02 vocabulary (validated against OWASP_AGENTIC_2026).
    "tool-misuse": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI02",)},
    # owasp_llm_2026=("LLM01",) (Prompt Injection): the hijack is driven by an
    # injected directive in untrusted tool-output / retrieved-document content
    # — a genuine prompt-injection channel — so LLM01 is the correct LLM
    # analogue; owasp_agentic_2026 maps the closed ASI01 vocabulary; S14
    # (application context: tool outputs, RAG documents) is the surface.
    "agent-goal-hijack": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI01",)},
    # owasp_llm_2026=("LLM07",) (Misinformation): a misleading approval summary
    # that misrepresents the executed effect is misinformation delivered to the
    # human — the correct LLM analogue for the display-vs-execution mismatch;
    # owasp_agentic_2026 maps the closed ASI09 vocabulary; S14 (application
    # context: the approval-prompt summary) is the surface; CWE-451 is the
    # UI-misrepresentation-of-critical-information weakness.
    "human-agent-trust": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM07",),
        "owasp_agentic_2026": ("ASI09",),
        "cwe": ("CWE-451",)},
    # owasp_llm_2026=("LLM01",) (Prompt Injection): a directive planted in the
    # agent's persistent memory and adopted in a later session is INDIRECT
    # prompt injection — the correct LLM analogue for cross-session memory
    # poisoning; owasp_agentic_2026 maps the closed ASI06 vocabulary; S14
    # (application context: agent memory) is the surface; CWE-501 is the
    # trust-boundary-violation weakness (untrusted persisted memory crossing
    # into the trusted planner). persistence=cross-session captures write-now/
    # trigger-later, the dimension that distinguishes it from ASI01.
    "memory-context-poisoning": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI06",),
        "cwe": ("CWE-501",)},
    # owasp_llm_2026=("LLM01",) (Prompt Injection): the toy models the compromise
    # as the planner ADOPTING a refund-routing binding declared by an untrusted/
    # unverified component — indirect prompt injection through a supply-chain
    # channel — so LLM01 is the correct LLM analogue, consistent with how the
    # sibling agentic modules that adopt an injected directive (ASI01, ASI06) map
    # to LLM01 (it is NOT the split-LLM artifact supply chain, S13/LLM04, which
    # the distinct "supply-chain" battery covers). owasp_agentic_2026 maps the
    # closed ASI04 vocabulary; S14 (application context: tool/connector schemas
    # the agent pulls) is the surface; CWE-1357 is reliance on an insufficiently
    # trustworthy component. persistence=cross-session captures the tampered
    # component persisting across the sessions that adopt it.
    "agentic-supply-chain": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI04",),
        "cwe": ("CWE-1357",)},
    # owasp_llm_2026=("LLM03",) (Excessive Agency): README.md
    # names "privilege" and "confused-deputy" tests as LLM03's gap — identity
    # and privilege abuse IS excessive agency (the agent wields a principal or
    # credential the session never bound to it). It is deliberately NOT LLM01:
    # the tool result is only the DELIVERY channel; the modeled weakness (and
    # the load-bearing control) is the missing invocation-identity binding at
    # the gateway, not the planner's adoption of a directive (ASI01/ASI06/
    # ASI04 map to LLM01 on that adoption mechanism); consistent with
    # tool-misuse (ASI02), the other agentic module whose weakness is an
    # authorization-layer gap rather than instruction adoption. S14
    # (application context: tool outputs/results) is the surface; CWE-863 is
    # incorrect authorization (CWE-441 is the arm-1 confused-deputy special
    # case). persistence=session: the abuse is confined to the session whose
    # authenticated principal is spoofed.
    "identity-privilege-abuse": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI03",),
        "cwe": ("CWE-863",)},
    # owasp_llm_2026=("LLM01",) (Prompt Injection): the toy models the cascade as
    # the planner ADOPTING a value that propagated through an un-revalidated
    # pipeline hop — indirect prompt injection through a multi-hop value-provenance
    # channel — so LLM01 is the correct LLM analogue, consistent with the sibling
    # agentic modules that adopt an injected/propagated directive (ASI01, ASI06,
    # ASI04) mapping to LLM01. LLM05 is DELIBERATELY NOT used: it is the
    # split-integrity-availability.v1 poisoning surface (mapped by the integrity
    # attacks in attack_metadata) and adding cascade to it would disturb that
    # frozen integrity-profile calibration; LLM01 is not in that profile's threat
    # set, so it does not collide. owasp_agentic_2026 maps the closed ASI08
    # vocabulary; S14 (application context: the multi-hop pipeline the value
    # propagates through) is the surface; CWE-345 is insufficient verification of
    # data authenticity (a downstream hop trusting an upstream value without
    # re-validating it against authoritative state). persistence=cross-session
    # captures the corruption propagating across the derive/consume hops.
    "cascading-failure": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("integrity", "authorization", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM01",),
        "owasp_agentic_2026": ("ASI08",),
        "cwe": ("CWE-345",)},
    # owasp_llm_2026=("LLM03",) (Excessive Agency): README.md
    # names "tool-call", "approval-bypass" and "capability-budget" tests as
    # LLM03's gap — unexpected code execution IS excessive agency (the agent
    # wields a computation the operator never approved for it; the
    # approval-swap arm IS approval-bypass). It is deliberately NOT LLM01:
    # the injected digest is only the DELIVERY channel for the injection arms
    # (and the approval-swap arm involves no adoption at all); the modeled
    # weakness (and the load-bearing control) is the missing code-digest
    # approval binding at the execution gateway, not the planner's adoption
    # of a directive (ASI01/ASI06/ASI04/ASI08 map to LLM01 on that adoption
    # mechanism); consistent with tool-misuse (ASI02) and
    # identity-privilege-abuse (ASI03), the other agentic modules whose
    # weakness is an authorization-layer gap rather than instruction adoption.
    # S14 (application context: tool outputs/results, retrieved documents) is
    # the surface; CWE-94 is the improper-control-of-code-generation/execution
    # weakness (the ASI05-name weakness; the approval-swap arm is its CWE-367
    # TOCTOU special case, and CWE-863 is deliberately left to ASI03's
    # incorrect-authorization mapping). persistence=session: the abuse is
    # confined to the session whose sanctioned computation is subverted.
    "unexpected-code-execution": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI05",),
        "cwe": ("CWE-94",)},
    # owasp_llm_2026=("LLM03",) (Excessive Agency): README.md
    # names "tool-call", "approval-bypass" and "capability-budget" tests as
    # LLM03's gap — a rogue agent IS excessive agency (the agent exercises
    # lifecycle capabilities — spawn / self-modify / persist — BEYOND its
    # operator mandate; an unattested lifecycle call IS a capability-budget
    # breach). It is deliberately NOT LLM01: the steered envelope is only the
    # DELIVERY channel (and the persistence arm's scored effect is the
    # lifecycle WRITE itself, not a content adoption — ASI01/ASI06/ASI04/ASI08
    # map to LLM01 on that adoption mechanism); consistent with tool-misuse
    # (ASI02), identity-privilege-abuse (ASI03) and unexpected-code-execution
    # (ASI05), the other agentic modules whose weakness is an
    # authorization-layer gap at a gateway rather than instruction adoption.
    # S14 (application context: tool outputs/results, retrieved documents, the
    # agent's own goal/config and long-term store) is the surface; CWE-693 is
    # the protection-mechanism-failure weakness (the lifecycle surface ships
    # with NO operator-attestation mechanism — deliberately not CWE-345, which
    # ASI08 maps to insufficient verification of DATA authenticity across
    # hops, and not CWE-863, ASI03's incorrect-authorization-of-a-principal
    # mapping: the acting principal here IS the session's; the failure is the
    # missing lifecycle-attestation mechanism, not a mis-authorized identity).
    # persistence=cross-session: the persistence-reload arm's raison d'être
    # is continuity across session end (write now, resume later — the
    # dimension that distinguishes a rogue agent from a session-scoped
    # hijack), matching memory-context-poisoning's cross-session choice.
    "rogue-agent": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI10",),
        "cwe": ("CWE-693",)},
    # owasp_llm_2026=("LLM03",) (Excessive Agency): README.md
    # names "tool-call", "approval-bypass" and "confused-deputy" tests as
    # LLM03's gap — an agent that EXECUTES an unauthenticated peer
    # instruction IS excessive agency on the inter-agent channel (its
    # tool-call capability is exercised on an instruction whose origin was
    # never verified — the confused-deputy analogue across agents). It is
    # deliberately NOT LLM01: the spoofed/tampered/replayed message is only
    # the DELIVERY channel, and the modeled weakness + the load-bearing
    # control is the missing per-message authenticity/freshness verification
    # at the receiving endpoint (an AUTHENTICATION mechanism), not
    # instruction adoption — consistent with tool-misuse (ASI02),
    # identity-privilege-abuse (ASI03), unexpected-code-execution (ASI05) and
    # rogue-agent (ASI10), the other agentic modules whose weakness is an
    # authorization-layer gap at a gateway/endpoint rather than instruction
    # adoption (ASI01/ASI06/ASI04/ASI08 map to LLM01 on that adoption
    # mechanism). S14 (application context: the inter-agent message channel)
    # is the surface. CWE-294 (authentication bypass by capture-replay): the
    # battery's distinguishing arms beyond per-message data authenticity are
    # sender-spoofing and REPLAY — the nonce-freshness clause and the replay
    # arm (a valid, already-consumed instruction re-delivered and executed a
    # second time) are the canonical capture-replay mechanism; deliberately
    # NOT CWE-345 (ASI08's mapping for insufficient verification of DATA
    # authenticity across hops — keeping the CWE mappings distinct per module
    # mirrors the distinctness doctrine) and not CWE-319 (cleartext
    # transmission — a confidentiality weakness the battery does not model;
    # the scored effects are integrity/authorization violations).
    # persistence=cross-session: the replay arm's raison d'être is a stale
    # instruction surviving session end (delivered yesterday, re-delivered
    # and executed today), matching memory-context-poisoning's and
    # cascading-failure's cross-session choice.
    "insecure-agent-communication": {
        "actor_intent": "malicious", "persistence": "cross-session",
        "surfaces": ("S14",),
        "security_properties": ("authenticity", "authorization", "integrity"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "owasp_agentic_2026": ("ASI07",),
        "cwe": ("CWE-294",)},
    # owasp_llm_2026=("LLM03",) (Excessive Agency): README.md names
    # "confused-deputy" tests as LLM03's gap — a CROSS-AGENT confused deputy
    # IS excessive agency (the deputy exercises a held authority its
    # session's authenticated principal lacks, on an authentic peer's
    # say-so). It is deliberately NOT LLM01: the forwarded message is only
    # the DELIVERY channel and it is AUTHENTIC (no authority-label adoption
    # check would drop a trusted peer's properly signed request — ASI01/
    # ASI06/ASI04/ASI08 map to LLM01 on that adoption mechanism); the modeled
    # weakness and the load-bearing control is the gateway's
    # invocation-identity binding (principal_binding), consistent with
    # tool-misuse (ASI02), identity-privilege-abuse (ASI03),
    # unexpected-code-execution (ASI05), rogue-agent (ASI10) and
    # insecure-agent-communication (ASI07), the other agentic modules whose
    # weakness is an authorization/authentication-layer gap at a
    # gateway/endpoint rather than instruction adoption.
    # owasp_agentic_2026 is DELIBERATELY NOT set: the closed ASI01-ASI10 set
    # is already completely mapped (exactly one module each since
    # benchmark.agentic.v11), and coverage marks a threat COVERED only when
    # ALL of its mapped attacks are measured — so mapping this scenario to
    # ASI03 or ASI07 would silently demote those threats from COVERED to
    # PARTIAL on every existing evidence bundle. The scenario rides the
    # inter-agent bus like ASI07 but its mechanism is the AUTHORITY
    # confusion of ASI03 (an authentic forwarded request, not an
    # unauthentic message); it is surfaced under LLM03 + S14 instead —
    # exactly the "confused-deputy-across-agents" analogue the ASI07 LLM03
    # mapping comment anticipates. S14 (application context: the inter-agent
    # message channel and the session/principal context) is the surface;
    # CWE-441 is THE confused-deputy weakness (Unintended Proxy or
    # Intermediary — named in ASI03's mapping comment as the confused-deputy
    # special case but mapped by no module's metadata entry yet).
    # persistence=session: the forwarded request is consumed within the
    # session it targets.
    "cross-agent-deputy": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S14",),
        "security_properties": ("authorization", "integrity", "safety"),
        "lifecycle_stages": ("inference",),
        "owasp_llm_2026": ("LLM03",),
        "cwe": ("CWE-441",)},
    "replica-integrity": {
        "actor_intent": "malicious", "persistence": "session",
        "surfaces": ("S1", "S2", "S10"),
        "security_properties": ("integrity", "availability", "confidentiality"),
        "lifecycle_stages": ("training", "inference"), "owasp_llm_2026": ("LLM02", "LLM05")},
    "view-privacy": {
        "actor_intent": "active", "persistence": "cross-epoch",
        "surfaces": ("S1", "S4", "S10"),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "inference"), "owasp_llm_2026": ("LLM02",)},
    "representation-privacy": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1", "S9"), "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "fine-tuning", "inference"), "owasp_llm_2026": ("LLM02",)},
    "gradient-hardening": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"), "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "fine-tuning"), "owasp_llm_2026": ("LLM02",)},
    "two-seam-delegation": {
        "actor_intent": "passive", "persistence": "session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
    "bisr-baseline": {
        "actor_intent": "passive", "persistence": "cross-session",
        "surfaces": ("S1", "S2"),
        "security_properties": ("confidentiality", "privacy"),
        "lifecycle_stages": ("training", "inference"),
        "owasp_llm_2026": ("LLM02",)},
}

#: Optional framework mappings beyond OWASP LLM (validated as strlists).
#: mitre_atlas / nist_ai_100_2 / cwe are free-form; owasp_agentic_2026 is
#: closed-vocab (validated against OWASP_AGENTIC_2026, below).
_EXTRA_MAPPINGS = ("mitre_atlas", "nist_ai_100_2", "cwe", "owasp_agentic_2026")

#: Extra mappings whose VALUES are constrained to a closed vocabulary.
_CLOSED_EXTRA_MAPPINGS = {"owasp_agentic_2026": OWASP_AGENTIC_2026}


def metadata_for(name):
    """The metadata dict for attack `name`; KeyError naming the registry
    when the attack has none (a registry-completeness bug — CI-enforced)."""
    try:
        return ATTACK_METADATA[name]
    except KeyError:
        raise KeyError(f"attack {name!r} has no entry in "
                       f"attacker.attack_metadata.ATTACK_METADATA; add one "
                       f"(THREAT_MODEL_REVIEW Release 1 item 2)") from None


def validate_metadata(name, md):
    """Raise a descriptive ValueError when metadata `md` for attack `name`
    violates the metadata schema; return `md` on success."""
    problems = []
    if not isinstance(md, dict):
        raise ValueError(f"attack {name!r}: metadata must be a dict")
    unknown = sorted(set(md) - (set(_FIELDS) | set(_EXTRA_MAPPINGS)))
    if unknown:
        problems.append(f"unknown fields {unknown}")
    for field, vocab in _FIELDS.items():
        if field not in md:
            problems.append(f"missing required field {field!r}")
            continue
        value = md[field]
        if field in ("actor_intent", "persistence"):
            if value not in vocab:
                problems.append(f"{field} must be one of {list(vocab)}, "
                                f"got {value!r}")
        else:
            if not (isinstance(value, (tuple, list)) and value
                    and all(isinstance(x, str) for x in value)):
                problems.append(f"{field} must be a non-empty list of str")
                continue
            bad = sorted(set(value) - set(vocab))
            if bad:
                problems.append(f"{field} values {bad} not in "
                                f"{sorted(vocab)}")
    for field in _EXTRA_MAPPINGS:
        if field in md and not (isinstance(md[field], (tuple, list))
                                and all(isinstance(x, str)
                                        for x in md[field])):
            problems.append(f"{field} must be a list of str")
    # closed-vocab extra mappings: values must be in their vocabulary
    for field, vocab in _CLOSED_EXTRA_MAPPINGS.items():
        if field in md and isinstance(md[field], (tuple, list)):
            bad = sorted(set(md[field]) - set(vocab))
            if bad:
                problems.append(f"{field} values {bad} not in "
                                f"{sorted(vocab)}")
    if problems:
        raise ValueError(f"attack {name!r} metadata invalid: "
                         + "; ".join(problems))
    return md


def validate_registry(names=None):
    """Validate every entry (and, when `names` is given — e.g. the attack
    registry keys — that each name has metadata and no entry is orphaned).
    Returns the sorted list of covered attack names; raises ValueError with
    all problems listed."""
    problems = []
    for name, md in ATTACK_METADATA.items():
        try:
            validate_metadata(name, md)
        except ValueError as e:
            problems.append(str(e))
    if names is not None:
        missing = sorted(set(names) - set(ATTACK_METADATA))
        orphaned = sorted(set(ATTACK_METADATA) - set(names))
        if missing:
            problems.append(f"attacks without metadata: {missing}")
        if orphaned:
            problems.append(f"metadata entries for unknown attacks: "
                            f"{orphaned}")
    if problems:
        raise ValueError("attack metadata registry invalid: "
                         + "; ".join(problems))
    return sorted(ATTACK_METADATA)


def attacks_for_owasp(risk):
    """Attack names whose metadata maps to OWASP LLM 2026 `risk`
    (e.g. "LLM02"), sorted."""
    if risk not in OWASP_LLM_2026:
        raise ValueError(f"unknown OWASP LLM 2026 risk {risk!r} "
                         f"(known: {list(OWASP_LLM_2026)})")
    return sorted(n for n, md in ATTACK_METADATA.items()
                  if risk in md.get("owasp_llm_2026", ()))


def attacks_for_agentic(risk):
    """Attack names whose metadata maps to OWASP Agentic 2026 `risk`
    (e.g. "ASI02"), sorted. Mirrors attacks_for_owasp for the closed
    agentic vocabulary."""
    if risk not in OWASP_AGENTIC_2026:
        raise ValueError(f"unknown OWASP Agentic 2026 risk {risk!r} "
                         f"(known: {list(OWASP_AGENTIC_2026)})")
    return sorted(n for n, md in ATTACK_METADATA.items()
                  if risk in md.get("owasp_agentic_2026", ()))


def attacks_for_surface(surface):
    """Attack names observing `surface` (e.g. "S1"), sorted."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r} "
                         f"(known: {sorted(SURFACES)})")
    return sorted(n for n, md in ATTACK_METADATA.items()
                  if surface in md.get("surfaces", ()))


# ---------------------------------------------------------------------------
def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    def _closed_md(**kw):
        base = {"actor_intent": "passive", "persistence": "session",
                "surfaces": ("S1",), "security_properties": ("confidentiality",),
                "lifecycle_stages": ("training",), "owasp_llm_2026": ("LLM02",)}
        base.update(kw)
        return base

    print("attack metadata registry (pure python):")
    try:
        covered = validate_registry()
        check("every registry entry is valid", bool(covered))
    except ValueError as e:
        check("every registry entry is valid", False)
        print(f"      {e}")
    try:
        validate_metadata("fake", {"actor_intent": "sneaky"})
        check("bad intent is a descriptive error", False)
    except ValueError as e:
        check("bad intent is a descriptive error",
              "fake" in str(e) and "actor_intent" in str(e))
    try:
        validate_metadata("fake", {
            "actor_intent": "passive", "persistence": "session",
            "surfaces": ("S99",), "security_properties": ("confidentiality",),
            "lifecycle_stages": ("training",), "owasp_llm_2026": ("LLM02",)})
        check("unknown surface rejected", False)
    except ValueError as e:
        check("unknown surface rejected", "S99" in str(e))
    try:
        validate_registry(["accumulation", "no-such-attack"])
        check("metadata gap vs registry is a descriptive error", False)
    except ValueError as e:
        check("metadata gap vs registry is a descriptive error",
              "no-such-attack" in str(e))
    check("attacks_for_owasp inverts the mapping",
          "accumulation" in attacks_for_owasp("LLM02"))
    check("attacks_for_surface inverts the mapping",
          "wire-shape" in attacks_for_surface("S7"))
    check("schema is v2 (closed agentic mapping added)",
          SCHEMA == "llm_attacker.attack_metadata.v2")
    try:
        validate_metadata("fake", _closed_md(owasp_agentic_2026=("ASI99",)))
        check("out-of-vocab agentic id rejected", False)
    except ValueError as e:
        check("out-of-vocab agentic id rejected", "ASI99" in str(e))
    check("valid agentic id accepted",
          validate_metadata("fake", _closed_md(
              owasp_agentic_2026=("ASI02",))) is not None)
    check("attacks_for_agentic inverts the closed mapping",
          "tool-misuse" in attacks_for_agentic("ASI02"))
    try:
        attacks_for_agentic("ASI99")
        check("attacks_for_agentic rejects an unknown risk", False)
    except ValueError as e:
        check("attacks_for_agentic rejects an unknown risk", "ASI99" in str(e))
    try:
        metadata_for("no-such-attack")
        check("metadata_for unknown attack raises naming the registry", False)
    except KeyError as e:
        check("metadata_for unknown attack raises naming the registry",
              "attack_metadata" in str(e))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
