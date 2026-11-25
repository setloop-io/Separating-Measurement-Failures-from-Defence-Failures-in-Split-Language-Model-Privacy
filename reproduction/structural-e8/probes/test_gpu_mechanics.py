import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "split-training")))

def test_dp_sgd_clipping():
    from dp_sgd import LocalDPSGD
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block = torch.nn.Linear(16, 16).to(device)
    
    # Initialize DP-SGD wrapper with correct signature:
    # params, max_grad_norm, noise_multiplier, accumulation, device
    dp = LocalDPSGD(list(block.parameters()), max_grad_norm=1.0, noise_multiplier=0.1, accumulation=1, device=device)
    
    # Step 1: accumulate gradients
    block.weight.grad = torch.ones_like(block.weight) * 100.0
    block.bias.grad = torch.ones_like(block.bias) * 100.0
    
    # Clip and accumulate
    grad_norm, clip_scale = dp.clip_and_accumulate()
    
    # Check that scale is strictly applying clipping
    assert clip_scale < 1.0, f"DP-SGD scale failed, clip_scale is {clip_scale}"
    
    # Check buffers has accumulated scaled gradients
    buf = dp.buffers[0]
    assert buf is not None and buf.abs().mean() > 0, "DP-SGD buffer is empty"
    
    # Step 2: materialize noisy average
    dp.materialize_noisy_average()
    
    # Check gradients are assigned back and buffers reset
    assert dp.buffers[0] is None, "DP-SGD buffers were not reset"
    
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in block.parameters()]))
    
    print(f"[test] DP-SGD GPU clipping PASSED (final noisy norm: {total_norm:.4f}).")

def test_rotation_mechanics():
    from er_ratchet import derive_epoch_W
    
    # Generate rotation for epoch 0
    R = derive_epoch_W(77000, 0, 16)
    R_inv = R.T
    
    identity = torch.eye(16, dtype=torch.float64, device=R.device)
    diff = (R.double() @ R_inv.double()) - identity
    max_err = diff.abs().max().item()
    
    assert max_err < 1e-5, f"Rotation matrix inverse is not orthogonal, max error: {max_err}"
    
    # Test tensor rotation
    h = torch.randn(2, 16)
    h_rot = (h.double() @ R.double()).float()
    h_unrot = (h_rot.double() @ R_inv.double()).float()
    
    rec_err = (h - h_unrot).abs().max().item()
    assert rec_err < 1e-5, f"Rotation/Inverse rotation failed, max error: {rec_err}"
    print("[test] Rotation mechanics (orthogonal inverse) PASSED.")

if __name__ == "__main__":
    try:
        test_dp_sgd_clipping()
        test_rotation_mechanics()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
