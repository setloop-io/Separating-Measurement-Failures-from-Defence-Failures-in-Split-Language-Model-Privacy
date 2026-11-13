#!/usr/bin/env python3
"""Fail-loud E-R9 emulated TEE handshake across separate processes."""

import json
import os
import secrets
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "split-training"))
sys.path.insert(0, ROOT)

from tee_enclave import GPUConfidentialCompute, TEEEncryptedChannel


def main():
    client = GPUConfidentialCompute(mode="emulated")
    client_pub = client.generate_ecdh_keypair()
    nonce = secrets.token_bytes(16)
    server_code = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from tee_enclave import GPUConfidentialCompute
probe = GPUConfidentialCompute(mode="emulated")
report, key = probe.get_attestation_report(bytes.fromhex(sys.argv[2]),
                                            bytes.fromhex(sys.argv[3]))
print(json.dumps({"report": report, "key": key.hex()}))
"""
    raw = subprocess.check_output(
        [sys.executable, "-c", server_code, ROOT, nonce.hex(), client_pub.hex()],
        text=True)
    server = json.loads(raw.strip().splitlines()[-1])
    client_key = client.verify_remote_report(
        server["report"], nonce, client._private_key)
    if client_key.hex() != server["key"]:
        raise AssertionError("cross-process ECDH keys differ")
    sender = TEEEncryptedChannel(client_key)
    receiver = TEEEncryptedChannel(bytes.fromhex(server["key"]))
    message = b"er9-cross-process-attestation"
    if receiver.decrypt_payload(sender.encrypt_payload(message)) != message:
        raise AssertionError("cross-process encrypted payload round-trip failed")
    print("TEE cross-process attestation/encryption: PASS")


if __name__ == "__main__":
    main()
