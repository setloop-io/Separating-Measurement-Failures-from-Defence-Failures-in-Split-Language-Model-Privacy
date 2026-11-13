"""Reuse pinned pcr initialization/controls with a Paper 4 boundary adapter."""
from torch import nn
from pcr.split import RemotePath
from .defenses import CausalAffineQuantizer, PropertyNoiseDefense


class BoundaryCapture:
    def __init__(self):
        self.values = {}

    def on_forward_values(self, z):
        self.values["forward_values"] = z.detach().cpu().clone()
        if z.requires_grad:
            z.register_hook(lambda g: self._grad("outbound_gradients", g))

    def on_returned_values(self, y):
        self.values["returned_values"] = y.detach().cpu().clone()
        if y.requires_grad:
            y.register_hook(lambda g: self._grad("inbound_gradients", g))

    def _grad(self, key, g):
        self.values[key] = g.detach().cpu().clone()


class DefendedRemotePath(RemotePath):
    def __init__(self, original, defense, basis=None, seed=0):
        nn.Module.__init__(self)
        self.cfg = original.cfg
        for key in ("encoder_ln", "encoder", "remote", "decoder"):
            setattr(self, key, getattr(original, key))
        self.quantizer = CausalAffineQuantizer(self.cfg.q_bits)
        if defense["name"] == "property_noise":
            self.defense = PropertyNoiseDefense(self.cfg.width, defense["noise_scale"], basis, seed)
        elif defense["name"] in ("none", "aggressive_rate"):
            self.defense = nn.Identity()
        else:
            raise ValueError("Unimplemented defense; refusing silent fallback")
        if self.cfg.grad_defense is not None:
            raise ValueError("Backward defense requires a separate Paper 4 protocol")

    def forward(self, x, capture=None):
        pre = self.encoder(self.encoder_ln(x))
        z, metadata = self.quantizer(self.defense(pre))
        if capture is not None:
            capture.on_forward_values(z)
            capture.values["quantizer_metadata"] = {k: v.detach().cpu().clone()
                                                     for k, v in metadata.items()}
        y = self.remote(z)
        if capture is not None:
            capture.on_returned_values(y)
        return x + self.decoder(y)


def attach_defense(model, defense, basis=None, seed=0):
    """Preserve paired initialization; replace only the boundary implementation."""
    original = model.remote_path
    model.blocks[model.split.start] = DefendedRemotePath(original, defense, basis, seed)
    return model
