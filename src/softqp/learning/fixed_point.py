"""Registered float64 L2WS coordinates, initialization and differentiated loss."""

from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

from .parameters import cpu64, head_id, initialize, stream, tensor_bytes, torch


def coordinate_names(n, m):
    return tuple(
        [f"P:{i}:{j}" for i in range(n) for j in range(i, n)]
        + [f"A:{i}:{j}" for i in range(m) for j in range(n)]
        + [f"q:{j}" for j in range(n)]
        + [f"{name}:{i}" for name in ("lower", "upper", "lower_finite", "upper_finite")
           for i in range(m)]
    )


class _UninitializedLinear(torch.nn.Linear):
    def reset_parameters(self):
        """The registered per-tensor generator supplies all parameters."""


class L2WSHead(torch.nn.Module):
    method = "L-1"

    def __init__(self, n, m, seed):
        super().__init__()
        self.head = head_id(n, m)
        self.n, self.m, self.seed = n, m, seed
        stream(self.method, seed, self.head, "validate")
        columns = n * (n + 1) // 2 + n * m + n + 4 * m
        options = dict(dtype=torch.float64, device="cpu")
        self.hidden1 = _UninitializedLinear(columns, 128, **options)
        self.hidden2 = _UninitializedLinear(128, 128, **options)
        self.output = _UninitializedLinear(128, n + m, **options)
        self.initialization = initialize(self)

    def forward(self, features):
        return self.output(self.hidden2(self.hidden1(features).relu()).relu())


@dataclass(frozen=True)
class Normalizer:
    mean: torch.Tensor
    scale: torch.Tensor
    source_ids: tuple
    frame_weights: tuple

    @classmethod
    def from_admitted(cls, mean, scale, source_ids, frame_weights):
        """Load arrays reconstructed and source-verified by the artifact reader."""
        center, width = cpu64(mean).clone(), cpu64(scale).clone()
        if center.ndim != 1 or center.shape != width.shape or not bool(
            torch.isfinite(center).all() and torch.isfinite(width).all() and (width > 0).all()
        ):
            raise ValueError("invalid admitted normalizer")
        return cls(center, width, tuple(source_ids), tuple(frame_weights))

    @property
    def numeric_sha256(self):
        return sha256(tensor_bytes(self.mean) + tensor_bytes(self.scale)).hexdigest()

    @classmethod
    def fit(cls, weighted_train):
        """Fit on explicitly train-scoped SourceUnits of FixedPointProblem frames."""
        from .training import weighted_units

        units = weighted_units(weighted_train, "train", condition=True)
        features = [frame.features() for source, _ in units for frame in source.frames]
        weights = tuple(weight / len(source.frames) for source, weight in units
                        for _ in source.frames)
        shape = units[0][0].frames[0].shape
        head_id(*shape)
        if any(frame.shape != shape for source, _ in units for frame in source.frames):
            raise ValueError("normalizer input mixes distinct shape heads")
        values = torch.stack(features)
        w = cpu64(weights).unsqueeze(-1)
        mean = (w * values).sum(dim=0) / w.sum()
        # Detect exact constants before variance arithmetic and subtraction.
        constant = (values == values[0]).all(dim=0)
        mean = torch.where(constant, values[0], mean)
        deviations = values - mean
        magnitude = deviations.abs().amax(dim=0)
        denominator = torch.where(constant, 1.0, magnitude)
        scale = denominator * ((w * (deviations / denominator).square()).sum(dim=0)
                               / w.sum()).sqrt()
        scale = torch.where(constant, 1.0, scale)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("nonrepresentable-transform: L2WS normalizer")
        return cls(mean, scale, tuple(s.source_id for s, _ in units), weights)

    def transform(self, features):
        if self.mean.shape != features.shape or self.scale.shape != features.shape:
            raise ValueError("normalizer coordinate shape mismatch")
        values = (features - self.mean) / self.scale
        if not bool((self.scale > 0).all() and torch.isfinite(values).all()):
            raise ValueError("nonrepresentable-transform: normalized L2WS coordinates")
        return values


class State(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    w: torch.Tensor


class FixedPointProblem:
    """Prepare the scalar-rho operator from one dimensionless interval lift."""

    rho = 0.1
    sigma = 1e-6

    def __init__(self, P, q, A, lower, upper):
        # Tensor clones retain coefficient gradients while isolating the
        # caller's original arrays and the prepared canonical snapshot.
        self.P, self.q, self.A = (cpu64(v).clone() for v in (P, q, A))
        self.lower, self.upper = (cpu64(v).detach().clone() for v in (lower, upper))
        n, m = self.q.numel(), self.lower.numel()
        self.shape = (n, m)
        if not n or not m or self.q.shape != (n,) or self.P.shape != (n, n) or self.A.shape != (m, n) or (
            self.lower.shape != (m,) or self.upper.shape != (m,)
        ):
            raise ValueError("unsupported-input: malformed fixed-point dimensions")
        if not all(bool(torch.isfinite(v).all()) for v in (self.P, self.q, self.A)):
            raise ValueError("nonrepresentable-transform: nonfinite fixed-point coefficients")
        if not torch.equal(self.P, self.P.T):
            raise ValueError("fixed-point Hessian must be exactly symmetric")
        if not bool((self.lower <= self.upper).all()) or bool(
            torch.isposinf(self.lower).any() or torch.isneginf(self.upper).any()
        ):
            raise ValueError("invalid fixed-point intervals")
        self.system = self.P + self.sigma * torch.eye(n, dtype=torch.float64, device="cpu")
        self.system = self.system + self.A.T @ (self.rho * self.A)
        if not bool(torch.isfinite(self.system).all()):
            raise ValueError("nonrepresentable-transform: fixed-point Gram matrix")
        self.factor = torch.linalg.cholesky(self.system)

    def features(self):
        return mlp_features(self.P, self.q, self.A, self.lower, self.upper)

    def initial_state(self, prediction):
        if prediction.shape != (sum(self.shape),):
            raise ValueError("fixed-point prediction shape mismatch")
        x, y = prediction.split(self.shape)
        return State(x, y, self.A @ x)

    def transition(self, state):
        x, y, w = state
        rhs = self.sigma * x - self.q + self.A.T @ (self.rho * w - y)
        x_next = torch.cholesky_solve(rhs.unsqueeze(-1), self.factor).squeeze(-1)
        z = self.A @ x_next
        value = z + y / self.rho
        # Constant endpoints give exactly zero derivative at either kink.
        w_next = torch.where(value <= self.lower, self.lower,
                             torch.where(value >= self.upper, self.upper, value))
        return State(x_next, y + self.rho * (z - w_next), w_next)

    def residual_loss(self, prediction, *, steps=5):
        if steps not in (0, 5):
            raise ValueError("only K5 and diagnostic K0 are registered")
        state = self.initial_state(prediction)
        for _ in range(steps):
            state = self.transition(state)
        successor = self.transition(state)
        return sum((new - old).square().mean() for new, old in zip(successor, state)) / 3

    def loss(self, model, normalizer):
        if model.head != head_id(*self.shape):
            raise ValueError("fixed-point model head mismatch")
        return self.residual_loss(model(normalizer.transform(self.features())))


def mlp_features(P, q, A, lower, upper):
    """Assemble the ordered dimensionless coordinates without factoring a system."""
    P, q, A, lower, upper = (cpu64(value) for value in (P, q, A, lower, upper))
    row, col = torch.triu_indices(q.numel(), q.numel(), device="cpu")
    lf, uf = torch.isfinite(lower), torch.isfinite(upper)
    return torch.cat((P[row, col], A.flatten(), q, torch.where(lf, lower, 0.0),
                      torch.where(uf, upper, 0.0), lf.to(torch.float64), uf.to(torch.float64)))
