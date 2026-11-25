"""R1 GB10 environment acceptance smoke: real CUDA ops + Stage A timing projection.

Engineering evidence only; no R1 run, training cohort, or scientific observation.
"""
import json
import os
import tempfile
import time

t0 = time.monotonic()
import torch  # noqa: E402

t1 = time.monotonic()
torch.cuda.init()
x = torch.randn(8, 32, 32, device="cuda")
t2 = time.monotonic()

import torch.nn as nn  # noqa: E402

m = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, batch_first=True),
    num_layers=2,
).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
for _ in range(8):
    loss = m(x).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
t3 = time.monotonic()

p = tempfile.mktemp(suffix=".pt")
torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, p)
_ = torch.load(p, weights_only=False)
sz = os.path.getsize(p)
os.unlink(p)
t4 = time.monotonic()

print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "import_seconds": round(t1 - t0, 3),
    "cuda_init_seconds": round(t2 - t1, 3),
    "eight_step_train_seconds": round(t3 - t2, 3),
    "checkpoint_seconds": round(t4 - t3, 3),
    "checkpoint_bytes": sz,
    "stage_a_projected_process_overhead_seconds": round(14 * 2 * ((t1 - t0) + (t2 - t1)), 1),
    "stage_a_projection_note": "per-run overhead ~= 2x(import+cuda_init) for client+remote; "
                               "14 runs must fit the 300s whole-stage cap",
}))
