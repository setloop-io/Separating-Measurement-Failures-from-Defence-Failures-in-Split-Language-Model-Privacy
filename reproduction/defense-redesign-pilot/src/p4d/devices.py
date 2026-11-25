"""Explicit device selection; unknown names and unavailable devices are errors."""
import re

import torch


def resolve_device(name):
    text = str(name)
    if text == "cpu":
        return torch.device("cpu")
    if text == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Device 'mps' requested but not available")
        return torch.device("mps")
    if text == "cuda" or re.fullmatch(r"cuda:\d+", text):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Device '{text}' requested but CUDA is not available")
        device = torch.device(text)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Device '{text}' requested but only {torch.cuda.device_count()} CUDA devices exist")
        return device
    raise ValueError(f"Unknown device '{text}'; expected 'cpu', 'mps', or 'cuda[:N]'")


def device_record(device):
    record = {"device": str(device)}
    if device.type == "cuda":
        record["device_name"] = torch.cuda.get_device_name(device)
        record["capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(device))
    return record
