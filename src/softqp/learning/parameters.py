"""Source-stable initialization, numeric parameter serialization and validation."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import math

import numpy as np


def require_version(package, expected):
    if version(package) != expected:
        raise ImportError(f"learning requires {package}=={expected}")


require_version("torch", "2.15.0+cpu")
import torch


SEEDS = (41001, 41002, 41003)
HEADS = tuple(sorted((
    (100, 180), (12, 44), (128, 224), (130, 194), (2, 3), (3, 4), (32, 56),
    (40, 88), (6, 14), (6, 8), (60, 140), (68, 132), (80, 176), (98, 162),
), key=lambda shape: f"H-{shape[0]}-{shape[1]}"))


def head_id(n, m):
    if (n, m) not in HEADS:
        raise ValueError(f"unsupported-input: unknown L2WS shape ({n}, {m})")
    return f"H-{n}-{m}"


def cpu64(value):
    return torch.as_tensor(value, dtype=torch.float64, device="cpu")


def stream(method, seed, head, parameter):
    if method not in ("L-1", "L-2") or type(seed) is not int or seed not in SEEDS:
        raise ValueError("unknown learning method or seed")
    key = f"softqp-learning-r2|{method}|{seed}|{head}|{parameter}"
    number = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "little") % 2**63
    return number, torch.Generator(device="cpu").manual_seed(number)


def tensor_bytes(tensor):
    if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
        raise ValueError("numeric learning tensors must be float64 on CPU")
    return tensor.detach().numpy().astype("<f8", copy=False).tobytes()


@dataclass(frozen=True)
class ParameterReceipt:
    name: str
    shape: tuple
    initialization_rule: str
    initialization_seed: int
    sha256: str


def initialize(model, prior=None):
    records = []
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            number = 0
            if name == "output_layer.lin3.bias":
                parameter.fill_(math.log(prior / (1 - prior)))
                rule = "train-log-positive-odds"
            elif name.endswith(".bias"):
                parameter.zero_()
                rule = "zeros"
            elif name.startswith("norm"):
                parameter.fill_(1)
                rule = "ones"
            else:
                number, generator = stream(model.method, model.seed, model.head, name)
                output = name.split(".")[0] in ("output", "output_layer")
                if output:
                    gain, rule = 1, "xavier-output-gain1"
                elif model.method == "L-1":
                    gain, rule = math.sqrt(2), "xavier-relu-sqrt2"
                else:
                    gain, rule = math.sqrt(2 / (1 + 0.1**2)), "xavier-leakyrelu-slope0.1"
                torch.nn.init.xavier_uniform_(parameter, gain=gain, generator=generator)
            records.append(ParameterReceipt(name, tuple(parameter.shape), rule, number,
                                            sha256(tensor_bytes(parameter)).hexdigest()))
    return tuple(records)


def weight_bytes(model):
    """Concatenate sorted, contiguous little-endian float64 parameter arrays."""
    return b"".join(tensor_bytes(p) for _, p in sorted(model.named_parameters()))


def load_weights(model, raw, *, expected_sha256):
    """Verify a numeric payload completely before loading it into the model."""
    if type(raw) is not bytes or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("numeric checkpoint weight digest mismatch")
    parameters = [p for _, p in sorted(model.named_parameters())]
    expected_size = sum(p.numel() for p in parameters) * 8
    if len(raw) != expected_size:
        raise ValueError("numeric checkpoint weight size mismatch")
    values = np.frombuffer(raw, dtype="<f8")
    if not np.isfinite(values).all():
        raise ValueError("numeric checkpoint contains nonfinite weights")
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            value = values[offset:offset + size].astype(np.float64, copy=True)
            parameter.copy_(cpu64(value).reshape(parameter.shape))
            offset += size


def bundle_sha256(models):
    """Hash concatenated head payloads in lexical head-ID order."""
    hasher = sha256()
    for _, model in sorted(models.items()):
        hasher.update(weight_bytes(model))
    return hasher.hexdigest()
