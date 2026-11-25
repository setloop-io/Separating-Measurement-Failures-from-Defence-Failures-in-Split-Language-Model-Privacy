"""Attack registry + plugin API.

Each attack module exposes:
  EXPERIMENT_ID    str, used in artifacts/journals (collection/ledger conv.)
  MODES            tuple of "training" / "inference" it applies to
  REQUIRES_LABELS  bool (or short str) — labeled (oracle/known-plaintext)
                   vs label-free
  DESCRIPTION      one-liner for --list-attacks
  build_parser()   argparse.ArgumentParser for the attack's own flags
                   (MUST work torch-less: import-time torch is forbidden)
  run(args)        executes; returns exit code
  self_test()      optional; pure-python fixtures (torch parts guarded)

validate_contract(module) enforces the contract and runs over every
built-in at import. Third-party attacks plug in two ways:

  - entry points: group "llm_attacker.attacks", one entry point per attack,
    discovered at import; each entry point resolves to a module (or
    module-like object) or a callable returning one;
  - register_attack(name, module): programmatic hook for embedders.

Discovered/registered attacks merge into ATTACKS (name -> module path str
for built-ins, module-like object for plugins). A name collision with an
already-registered attack is a descriptive error, never a silent override.
"""

import importlib.metadata
from importlib import import_module

ENTRY_POINT_GROUP = "llm_attacker.attacks"
VALID_MODES = ("training", "inference")

ATTACKS = {
    # --- superset of the historical scripts ---
    "accumulation": "attacker.attacks.accumulation",        # E-R1a
    "known-prefix": "attacker.attacks.known_prefix",        # E-R3
    "stale-key": "attacker.attacks.stale_key",              # E-R4
    "seed-disclosure": "attacker.attacks.seed_disclosure",  # protocol state
    "sharded": "attacker.attacks.sharded",                  # E-R5
    "wire-eval": "attacker.attacks.wire_eval",              # e9 / er_train
    "gradient-inversion": "attacker.attacks.gradient_inversion",  # DLG++
    "membership": "attacker.attacks.membership",            # E-A4
    "output-inversion": "attacker.attacks.output_inversion",
    "latent-probe": "attacker.attacks.latent_probe",
    "latent-sensitivity": "attacker.attacks.latent_sensitivity",
    "latent-matching": "attacker.attacks.latent_matching",
    # --- the six new capabilities ---
    "alignment-search": "attacker.attacks.alignment_search",   # E-R8
    "active-cloud": "attacker.attacks.active_cloud",
    "subspace-joint": "attacker.attacks.subspace_joint",
    "ica-bss": "attacker.attacks.ica_bss",
    "leak-accumulation": "attacker.attacks.leak_accumulation",
    "max-effort": "attacker.attacks.max_effort",
    # --- systems/hardware seam (Phase B) ---
    "gpu-memory": "attacker.attacks.gpu_memory",        # LeftoverLocals line
    "kv-timing": "attacker.attacks.kv_timing",          # prefix-cache timing
    "wire-shape": "attacker.attacks.wire_shape",        # traffic-shape
    # --- deployment shapes: remote / man-in-the-middle ---
    "remote-fetch": "attacker.attacks.remote_fetch",
    "mitm-proxy": "attacker.attacks.mitm_proxy",
    # --- standard-attack adapters (Phase C) ---
    "lira-membership": "attacker.attacks.lira_membership",
    "seam-inversion": "attacker.attacks.seam_inversion",  # VMI/WBI line
    "label-scoring": "attacker.attacks.label_scoring",    # norm-based (NS)
    "batch-label-recon": "attacker.attacks.batch_label_recon",  # BLI line
    "vec2text-inversion": "attacker.attacks.vec2text_inversion",  # vec2text
    "body-extraction": "attacker.attacks.body_extraction",  # body-slice IP
    "malicious-server": "attacker.attacks.malicious_server",  # crafted body
    # --- reasoning-theft family (output-side, API surface) ---
    "reasoning-replay": "attacker.attacks.reasoning_replay",
    "trace-inversion": "attacker.attacks.trace_inversion",
    # --- threat-model review Release 2: integrity / availability / isolation ---
    "protocol-integrity": "attacker.attacks.protocol_integrity",
    "multi-tenant-isolation": "attacker.attacks.multi_tenant_isolation",
    "artifact-telemetry-leak": "attacker.attacks.artifact_telemetry_leak",
    "key-session-abuse": "attacker.attacks.key_session_abuse",
    "secure-compute-audit": "attacker.attacks.secure_compute_audit",
    "resource-exhaustion": "attacker.attacks.resource_exhaustion",
    # --- threat-model review Release 3: training / artifact integrity ---
    "poisoning-backdoor": "attacker.attacks.poisoning_backdoor",
    "malicious-client": "attacker.attacks.malicious_client",
    "supply-chain": "attacker.attacks.supply_chain",
    "router-side-channel": "attacker.attacks.router_side_channel",
    "privacy-composition": "attacker.attacks.privacy_composition",
    # --- threat-model review Release 4: GenAI application profile ---
    "prompt-injection": "attacker.attacks.prompt_injection",
    "hidden-context-exposure": "attacker.attacks.hidden_context_exposure",
    "rag-vector-abuse": "attacker.attacks.rag_vector_abuse",
    "output-sink-injection": "attacker.attacks.output_sink_injection",
    "misinformation-integrity": "attacker.attacks.misinformation_integrity",
    # --- Lane B: the agentic (OWASP ASI) modules ---
    "tool-misuse": "attacker.attacks.tool_misuse",
    "agent-goal-hijack": "attacker.attacks.agent_goal_hijack",
    "human-agent-trust": "attacker.attacks.human_agent_trust",
    "memory-context-poisoning": "attacker.attacks.memory_context_poisoning",
    "agentic-supply-chain": "attacker.attacks.agentic_supply_chain",
    "identity-privilege-abuse": "attacker.attacks.identity_privilege_abuse",
    "cascading-failure": "attacker.attacks.cascading_failure",
    "unexpected-code-execution": "attacker.attacks.unexpected_code_execution",
    "rogue-agent": "attacker.attacks.rogue_agent",
    "insecure-agent-communication":
        "attacker.attacks.insecure_agent_communication",
    "cross-agent-deputy": "attacker.attacks.cross_agent_deputy",
    # Assumption-scoped defense gap audits (separate from frozen study).
    "replica-integrity": "attacker.attacks.replica_integrity",
    "view-privacy": "attacker.attacks.view_privacy",
    "representation-privacy": "attacker.attacks.representation_privacy",
    "gradient-hardening": "attacker.attacks.gradient_hardening",
    "two-seam-delegation": "attacker.attacks.two_seam_delegation",
    # Published-baseline gap development module (scoped adaptation).
    "bisr-baseline": "attacker.attacks.bisr_baseline",
}


def validate_contract(module, name=None):
    """Raise ValueError with a clear per-requirement message if `module`
    violates the attack module contract; return the module on success."""
    label = name or getattr(module, "__name__", None) or repr(module)
    problems = []
    if not isinstance(getattr(module, "EXPERIMENT_ID", None), str) \
            or not module.EXPERIMENT_ID:
        problems.append("EXPERIMENT_ID must be a non-empty str")
    modes = getattr(module, "MODES", None)
    if not isinstance(modes, (tuple, list)) or not modes \
            or any(m not in VALID_MODES for m in modes):
        problems.append(f"MODES must be a non-empty tuple/list drawn from "
                        f"{VALID_MODES}")
    labels = getattr(module, "REQUIRES_LABELS", None)
    if not (isinstance(labels, bool)
            or (isinstance(labels, str) and labels)):
        problems.append("REQUIRES_LABELS must be a bool or a non-empty str")
    if not isinstance(getattr(module, "DESCRIPTION", None), str):
        problems.append("DESCRIPTION must be a str")
    if not callable(getattr(module, "build_parser", None)):
        problems.append("build_parser must be callable (and work torch-less)")
    if not callable(getattr(module, "run", None)):
        problems.append("run must be callable")
    if problems:
        raise ValueError(f"attack {label!r} violates the module contract: "
                         + "; ".join(problems))
    return module


def load(name):
    target = ATTACKS[name]
    module = import_module(target) if isinstance(target, str) else target
    return validate_contract(module, name)


def for_mode(mode):
    return [n for n in ATTACKS if mode in load(n).MODES]


def register_attack(name, module):
    """Programmatic plugin hook for embedders: validate the contract and
    merge `module` into ATTACKS. Refuses to override an existing name."""
    if name in ATTACKS:
        raise ValueError(f"attack name {name!r} is already registered; "
                         f"refusing to override (rename the plugin)")
    validate_contract(module, name)
    ATTACKS[name] = module
    return module


def _entry_points():
    """Third-party attack entry points; [] when the metadata
    infrastructure has none (or cannot answer) — never raises."""
    try:
        return list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
    except Exception:
        return []


def discover_plugins(entry_points=None):
    """Merge entry-point attacks into ATTACKS; returns the names added.
    Each entry point resolves to a module(-like) or a callable returning
    one. Collisions and load/contract failures are descriptive errors."""
    if entry_points is None:
        entry_points = _entry_points()
    added = []
    for ep in entry_points:
        if ep.name in ATTACKS:
            raise RuntimeError(
                f"attack entry point {ep.name!r} (group {ENTRY_POINT_GROUP!r})"
                f" collides with an already-registered attack; rename the "
                f"plugin")
        try:
            obj = ep.load()
        except Exception as e:
            raise RuntimeError(
                f"attack entry point {ep.name!r} (group {ENTRY_POINT_GROUP!r})"
                f" failed to load: {e}") from e
        module = obj
        if not (hasattr(obj, "run") and hasattr(obj, "build_parser")):
            if callable(obj):
                module = obj()
        register_attack(ep.name, module)
        added.append(ep.name)
    return added


# Validate lazily when selected; importing a planner must not import torch.
discover_plugins()


# ---------------------------------------------------------------------------
def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import argparse
    import types

    print("attack registry / plugin API (pure python):")
    try:
        for n in list(ATTACKS):
            validate_contract(load(n), n)
        check("all registered built-ins satisfy the module contract", True)
    except ValueError as e:
        check("all registered built-ins satisfy the module contract", False)
        print(f"      {e}")

    # THREAT_MODEL_REVIEW Release 1 item 2: attack metadata is required —
    # every registered built-in must have a valid entry.
    from .. import attack_metadata
    try:
        attack_metadata.validate_registry(list(ATTACKS))
        check("every registered attack has valid threat metadata", True)
    except ValueError as e:
        check("every registered attack has valid threat metadata", False)
        print(f"      {e}")

    bad = types.SimpleNamespace(EXPERIMENT_ID="bad", MODES=("training",),
                                REQUIRES_LABELS=True, DESCRIPTION="x")
    try:
        validate_contract(bad, "bad-fake")
        check("contract violation is a descriptive error", False)
    except ValueError as e:
        check("contract violation is a descriptive error",
              "bad-fake" in str(e) and "build_parser" in str(e)
              and "run" in str(e))
    bad_modes = types.SimpleNamespace(
        EXPERIMENT_ID="bad", MODES=("sideways",), REQUIRES_LABELS=True,
        DESCRIPTION="x", build_parser=lambda: None, run=lambda a: 0)
    try:
        validate_contract(bad_modes, "bad-modes")
        check("invalid MODES rejected", False)
    except ValueError as e:
        check("invalid MODES rejected", "MODES" in str(e))

    fake = types.SimpleNamespace(
        EXPERIMENT_ID="selftest_fake", MODES=("training",),
        REQUIRES_LABELS=False, DESCRIPTION="self-test fake attack",
        build_parser=lambda: argparse.ArgumentParser(), run=lambda args: 0)

    class FakeEP:
        def __init__(self, name, obj):
            self.name, self._obj = name, obj

        def load(self):
            return self._obj

    added = []
    try:
        check("empty discovery is a no-op", discover_plugins([]) == [])
        added = discover_plugins([FakeEP("selftest-plugin", fake),
                                  FakeEP("selftest-plugin-factory",
                                         lambda: fake)])
        check("entry points merge into ATTACKS",
              sorted(added) == ["selftest-plugin", "selftest-plugin-factory"])
        check("module entry point registers as-is",
              ATTACKS.get("selftest-plugin") is fake)
        check("callable entry point resolves to its module-like object",
              load("selftest-plugin-factory") is fake)
        check("plugin participates in for_mode",
              "selftest-plugin" in for_mode("training"))
        try:
            discover_plugins([FakeEP("accumulation", fake)])
            check("name collision with a built-in is a descriptive error",
                  False)
        except RuntimeError as e:
            check("name collision with a built-in is a descriptive error",
                  "accumulation" in str(e) and ENTRY_POINT_GROUP in str(e))
        try:
            register_attack("wire-eval", fake)
            check("register_attack refuses to override", False)
        except ValueError as e:
            check("register_attack refuses to override",
                  "wire-eval" in str(e))
    finally:
        for n in added:
            ATTACKS.pop(n, None)
    check("registry left pristine after self-test",
          "selftest-plugin" not in ATTACKS
          and "selftest-plugin-factory" not in ATTACKS)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
