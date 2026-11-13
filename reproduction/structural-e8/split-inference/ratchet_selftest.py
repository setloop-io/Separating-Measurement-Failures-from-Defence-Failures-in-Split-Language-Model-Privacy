#!/usr/bin/env python3
"""[ER] Torch-free self-test for the ratcheted per-epoch rotation spec
(E-R1b/E-R2/E-R4) implemented in jacobi_server.py / cloud_server_kv.py.

Stdlib only, so it runs in CI BEFORE torch is installed (same convention as
draft_spec.py --self-test). It pins, as a spec oracle:
  1. epoch_seed derivation against frozen SHA256 vectors (S=77000, t=0..3);
  2. fixed-N epoch arithmetic (epoch = served // N) on a frozen token stream;
  3. budget-trigger transitions on a frozen event stream (served tokens +1,
     repeated first-64-token prefix +S, advance+reset at evidence >= B);
  4. the wire-capture sidecar schema keys (must include "epoch").

The reference implementations below are deliberately small duplicates of the
server logic; if the servers drift from these pinned vectors, this test fails.
"""
import hashlib
import json
import sys


def epoch_seed(seed_base, epoch):
    """Reference: int.from_bytes(sha256(f"{S}:{t}")[:8], "little")."""
    return int.from_bytes(
        hashlib.sha256(f"{seed_base}:{epoch}".encode()).digest()[:8],
        "little")


# Frozen vectors for S=77000, t=0..3 (computed once from the reference above;
# the servers must agree with these, not the other way around).
FROZEN_EPOCH_SEEDS = {
    0: 15148736061989184457,
    1: 17865006442405182368,
    2: 11879499459178023863,
    3: 15456178592288874741,
}


def fixed_n_epochs(served_increments, n):
    """Reference fixed-mode arithmetic: epoch = served // N after each
    increment (prefill adds the prompt length, each decode step adds 1)."""
    served = 0
    out = []
    for inc in served_increments:
        served += inc
        out.append(served // n)
    return out


def budget_epochs(events, budget, seed_base):
    """Reference budget-mode arithmetic. events: list of ("token", None) or
    ("prefix_hit", None). Served tokens add 1, prefix hits add S; the counter
    advances evidence // B epochs and CARRIES the remainder. Returns
    (epoch_per_event, epochs_advanced).

    The carry is load-bearing (#89): zeroing on overshoot discards evidence
    and under-rotates, and a prefix hit adds S at once, so overshoot is the
    normal case on this path rather than the exception. Must stay identical
    to er_ratchet.TokenRatchet.advance and
    jacobi_server._ratchet_advance."""
    epoch = 0
    evidence = 0
    out = []
    for kind, _ in events:
        evidence += 1 if kind == "token" else seed_base
        advances, evidence = divmod(evidence, budget)
        epoch += advances
        out.append(epoch)
    return out, epoch


SIDECAR_KEYS = {"session_id", "request_seq", "phase", "position", "epoch",
                "timestamp"}


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("ER1 epoch_seed matches the frozen SHA256 vectors (S=77000, t=0..3):")
    for t, want in FROZEN_EPOCH_SEEDS.items():
        check(f"t={t}: {want}", epoch_seed(77000, t) == want)
    check("distinct epochs derive distinct seeds",
          len({epoch_seed(77000, t) for t in range(8)}) == 8)
    check("distinct seed bases derive distinct epoch-0 seeds",
          epoch_seed(77000, 0) != epoch_seed(77001, 0))

    print("ER2 fixed-N epoch arithmetic (N=64, prefill 40 + decode steps):")
    ep = fixed_n_epochs([40] + [1] * 90, 64)
    check("epoch 0 while served < 64 (served = 40 + step)",
          ep[:24] == [0] * 24)
    check("epoch flips to 1 exactly when served crosses 64 (step 24)",
          ep[24] == 1 and ep[23] == 0)
    check("epoch 2 when served crosses 128 (step index 88)",
          ep[88] == 2 and ep[87] == 1)

    print("ER3 budget-trigger transitions (S=300, B=1000, frozen stream):")
    stream = (["token"] * 700 + ["prefix_hit"] + ["token"] * 800
              + ["prefix_hit"] + ["token"] * 500)
    events = [(k, None) for k in stream]
    ep, advanced = budget_epochs(events, 1000, 300)
    check("700 tokens alone do not trip the budget", ep[699] == 0)
    check("prefix hit at 700 (+300 = 1000) advances to epoch 1", ep[700] == 1)
    check("counter reset after the trip: next 699 tokens stay in epoch 1",
          ep[701] == 1 and ep[1399] == 1)
    check("second trip at 800 tokens + prefix hit (1100) -> epoch 2",
          ep[1501] == 2 and ep[1500] == 1)
    check("exactly 2 epochs advanced over the stream", advanced == 2)

    print("ER3b prefix-hit overshoot carries the remainder (#89):")
    # A prefix hit adds S at once, so overshoot is the normal case here, not
    # the exception. 900 tokens + a 300 hit reaches 1200: one epoch and 200
    # carried. The old rule zeroed that 200, so the following 800 tokens fell
    # 200 short of the next rotation and the stream under-rotated by one.
    stream = ["token"] * 900 + ["prefix_hit"] + ["token"] * 800
    ep, advanced = budget_epochs([(k, None) for k in stream], 1000, 300)
    check("the 300 hit at 900 trips epoch 1 (1200 >= 1000)", ep[900] == 1)
    check("the carried 200 lets the next 800 tokens trip epoch 2",
          ep[1700] == 2)
    check("2 epochs over the stream, not the 1 the old rule gave",
          advanced == 2)

    print("ER4 wire-capture sidecar schema keys:")
    sample = {"session_id": "s", "request_seq": 3, "phase": "decode",
              "position": 41, "epoch": 2, "timestamp": 0.0}
    check("sidecar key set matches (incl. 'epoch')",
          set(sample.keys()) == SIDECAR_KEYS)
    check("'epoch' is null when the ratchet is off",
          set({"session_id": "s", "request_seq": 0, "phase": "prefill",
               "position": 0, "epoch": None, "timestamp": 0.0}.keys())
          == SIDECAR_KEYS)
    check("sidecar round-trips through JSON unchanged",
          json.loads(json.dumps(sample)) == sample)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    print(__doc__)
