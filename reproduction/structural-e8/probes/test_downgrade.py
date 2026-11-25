import asyncio
import websockets
import json
import secrets
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "split-training")))

async def test_downgrade():
    url = "ws://10.10.10.2:5003"
    try:
        async with websockets.connect(url, max_size=None) as ws:
            hello = {
                "op": "hello",
                "split_after": 8,
                "resume_after": 23,
                "lr": 0.0,
                "trainable": False,
                "wire_dtype": "fp16",
                "tee_attest": True,
                "tee_emulated": True,
                "tee_nonce": secrets.token_hex(16),
            }
            from tee_enclave import GPUConfidentialCompute
            probe = GPUConfidentialCompute(mode="emulated")
            hello["tee_client_pub"] = probe.generate_ecdh_keypair().hex()
            
            await ws.send(json.dumps(hello))
            ack = json.loads(await ws.recv())
            if ack.get("op") == "error":
                print(f"Failed to hello: {ack}")
                return False
                
            print("[test] TEE Negotiated successfully.")
            
            # Send an UNENCRYPTED forward frame
            import struct
            def pack_frame(header, payload=b""):
                hdr_bytes = json.dumps(header).encode("utf-8")
                return struct.pack(">I", len(hdr_bytes)) + hdr_bytes + payload

            def unpack_frame(data):
                hdr_len = struct.unpack(">I", data[:4])[0]
                hdr = json.loads(data[4:4 + hdr_len].decode("utf-8"))
                payload = data[4 + hdr_len:]
                return hdr, payload

            header = {"op": "forward_with_graph", "step": 0, "mb_id": 0, "dtype": "fp16", "hidden_shape": [1, 16, 4096]}
            payload = b"\x00" * (1 * 16 * 4096 * 2)
            frame = pack_frame(header, payload)
            
            await ws.send(frame)
            resp = await ws.recv()
            if isinstance(resp, str):
                rh = json.loads(resp)
            else:
                rh, rp = unpack_frame(resp)
            if rh.get("op") == "error" and "downgrade" in rh.get("error", ""):
                print("[test] PASSED: Server rejected plaintext payload under active TEE!")
                return True
            else:
                print(f"[test] FAILED: Server did not reject downgrade. Response: {rh}")
                return False
    except Exception as e:
        print(f"[test] FAILED: Connection failed (is cloud_trainer_server running?): {e}")
        return False

if __name__ == "__main__":
    if not asyncio.run(test_downgrade()):
        sys.exit(1)
