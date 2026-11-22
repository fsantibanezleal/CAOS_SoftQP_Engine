"""Split-scoped weighted epoch steps and numeric-only local resume."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Sequence

import numpy as np

from .parameters import load_weights, stream, tensor_bytes, torch, weight_bytes


@dataclass(frozen=True)
class SourceUnit:
    source_id: str
    family: str
    regime: int
    split: str
    frames: Sequence

    def __post_init__(self):
        if not self.source_id or not self.family or not len(self.frames):
            raise ValueError("source unit requires identity, family and at least one frame")
        if self.split not in ("train", "validation", "calibration", "test"):
            raise ValueError("unknown source phase")


def balance_sources(sources, split):
    """Assign method-level weights before conditioning an L2WS head."""
    sources = tuple(sources)
    cells = defaultdict(set)
    for source in sources:
        if source.split != split:
            raise ValueError("source accessed outside its admitted phase")
        cells[source.family].add(source.regime)
    counts = defaultdict(int)
    for source in sources:
        counts[source.family, source.regime] += 1
    units = [(s, 1 / (len(cells) * len(cells[s.family]) * counts[s.family, s.regime])) for s in sources]
    return weighted_units(units, split)


def weighted_units(units, split, *, condition=False):
    """Validate identity/phase before any frame is accessed."""
    units = tuple(units)
    if not units or len({u.source_id for u, _ in units}) != len(units):
        raise ValueError("empty or duplicate source population")
    if any(unit.split != split for unit, _ in units):
        raise ValueError("source accessed outside its admitted phase")
    if any(not math.isfinite(w) or w <= 0 for _, w in units):
        raise ValueError("source weights must be finite and positive")
    total = math.fsum(w for _, w in units)
    if not condition and not math.isclose(total, 1, rel_tol=1e-14):
        raise ValueError("complete method source weights must sum to one")
    return tuple(sorted(((unit, w / total if condition else w) for unit, w in units),
                        key=lambda item: item[0].source_id))


@dataclass(frozen=True)
class Epoch:
    epoch: int
    train_objective: float | None
    validation_objective: float
    optimizer_steps: int
    gradient_finite: bool
    gradient_nonzero: bool
    zero_mask_frames: int
    order_seed: int | None
    permutation_sha256: str | None


@dataclass(frozen=True)
class NumericResume:
    """Non-executable, identity-bound snapshot of one head at an epoch boundary."""

    metadata: bytes
    initial: bytes
    current: bytes
    best: bytes
    moments: bytes
    rng: bytes

    @property
    def sha256(self):
        digest = sha256()
        for data in (self.metadata, self.initial, self.current, self.best, self.moments, self.rng):
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
        return digest.hexdigest()


def _order(model, epoch, size):
    number, generator = stream(model.method, model.seed, model.head, f"epoch-{epoch}-order")
    permutation = torch.randperm(size, generator=generator, device="cpu")
    digest = sha256(permutation.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    return permutation.tolist(), number, digest, generator.get_state().numpy().tobytes()


def _objective(model, units, loss):
    value = 0.0
    with torch.no_grad():
        for unit, weight in units:
            for frame in unit.frames:
                value += float(loss(model, frame)) * (weight / len(unit.frames))
    if not math.isfinite(value):
        raise ValueError("nonfinite epoch objective")
    return value


class HeadTrainer:
    """Advance one prescribed learning stream only when explicitly supplied phase inputs.

    The source identity is the product's admitted dataset/split/features binding
    digest. This class has no file or cohort traversal and exposes no training CLI.
    """

    def __init__(self, model, train, validation, loss, *, source_sha256):
        if not _is_sha256(source_sha256):
            raise ValueError("invalid training source identity")
        self.source_sha256 = source_sha256
        self.model, self.loss = model, loss
        head_condition = model.method == "L-1"
        self.train = weighted_units(train, "train", condition=head_condition)
        self.validation = weighted_units(validation, "validation", condition=head_condition)
        if {unit.source_id for unit, _ in self.train} & {unit.source_id for unit, _ in self.validation}:
            raise ValueError("source overlap across train and validation")
        self.batch_size, self.patience, self.improvement, decay = (
            (32, 20, 1e-6, 0.0001) if head_condition else (8, 5, 0.001, 0.01))
        if not head_condition and not any(
            bool(frame.labels.mask.any()) for unit, _ in self.validation for frame in unit.frames
        ):
            raise ValueError("validation undefined: all graph masks are empty")
        self.parameters = tuple(p for _, p in sorted(model.named_parameters()))
        self.optimizer = torch.optim.AdamW(self.parameters, lr=0.001, weight_decay=decay,
                                           betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                                           foreach=False, fused=False)
        initial = weight_bytes(model)
        self.initial = self.best = initial
        initial_value = _objective(model, self.validation, loss)
        self.history = [Epoch(0, None, initial_value, 0, True, False, 0, None, None)]
        self.best_epoch = 0
        self._last_significant = 0
        self._rng = b""
        self._interrupted = False

    @property
    def stopped(self):
        epoch = self.history[-1].epoch
        return epoch >= 100 or epoch - self._last_significant >= self.patience

    def advance(self):
        if self._interrupted:
            raise ValueError("restore the last complete epoch before retrying")
        if self.stopped:
            raise ValueError("registered learning stream has stopped")
        self._interrupted = True
        epoch = self.history[-1].epoch + 1
        order, seed, permutation, rng = _order(self.model, epoch, len(self.train))
        steps = zero_masks = 0
        nonzero = False
        objective = 0.0
        for begin in range(0, len(order), self.batch_size):
            self.optimizer.zero_grad(set_to_none=True)
            for index in order[begin:begin + self.batch_size]:
                unit, weight = self.train[index]
                for frame in unit.frames:
                    if self.model.method == "L-2":
                        zero_masks += not bool(frame.labels.mask.any())
                    value = self.loss(self.model, frame) * (weight / len(unit.frames))
                    numeric_value = float(value.detach())
                    if not math.isfinite(numeric_value):
                        raise ValueError("nonfinite training loss")
                    objective += numeric_value
                    value.backward()
            for param in self.parameters:
                if param.grad is None or not bool(torch.isfinite(param.grad).all()):
                    raise ValueError("missing or nonfinite training gradient")
                nonzero |= bool(torch.count_nonzero(param.grad))
            self.optimizer.step()
            steps += 1
        validation = _objective(self.model, self.validation, self.loss)
        self.history.append(Epoch(epoch, objective, validation, steps, True, nonzero,
                                  zero_masks, seed, permutation))
        if validation < self.history[self.best_epoch].validation_objective:
            self.best_epoch = epoch
            self.best = weight_bytes(self.model)
        if self.history[self._last_significant].validation_objective - validation >= self.improvement:
            self._last_significant = epoch
        self._rng = rng
        self._interrupted = False
        return self.history[-1]

    def _identity(self):
        return dict(method=self.model.method, head=self.model.head, seed=self.model.seed,
                    source_sha256=self.source_sha256,
                    train=[[u.source_id, len(u.frames), w] for u, w in self.train],
                    validation=[[u.source_id, len(u.frames), w] for u, w in self.validation],
                    parameters=[[n, list(p.shape)] for n, p in sorted(self.model.named_parameters())])

    def snapshot(self):
        """Copy all optimizer arrays and immutable best weights in deterministic order."""
        if self._interrupted:
            raise ValueError("cannot snapshot an interrupted epoch")
        first, second, steps = [], [], []
        for parameter in self.parameters:
            state = self.optimizer.state.get(parameter)
            if state:
                steps.append(int(state["step"]))
                first.append(tensor_bytes(state["exp_avg"]))
                second.append(tensor_bytes(state["exp_avg_sq"]))
            else:
                steps.append(0)
        metadata = dict(format="softqp.learning.numeric-resume/1", identity=self._identity(),
                        history=[asdict(entry) for entry in self.history],
                        best_epoch=self.best_epoch, significant_epoch=self._last_significant,
                        optimizer_steps=steps)
        return NumericResume(_json(metadata), self.initial, weight_bytes(self.model), self.best,
                             b"".join((*first, *second)), self._rng)

    def restore(self, snapshot, *, expected_sha256):
        """Restore a trusted local epoch boundary, refusing stale or malformed state."""
        if type(snapshot) is not NumericResume or snapshot.sha256 != expected_sha256:
            raise ValueError("numeric resume digest mismatch")
        metadata = json.loads(snapshot.metadata)
        if metadata.get("format") != "softqp.learning.numeric-resume/1" or metadata.get(
            "identity"
        ) != self._identity() or self.initial != snapshot.initial:
            raise ValueError("numeric resume source or architecture mismatch")
        entries = [Epoch(**row) for row in metadata["history"]]
        if not entries or len(entries) > 101 or entries[0] != self.history[0]:
            raise ValueError("invalid numeric resume epoch zero")
        best = significant = 0
        rng = b""
        expected_batch_steps = math.ceil(len(self.train) / self.batch_size)
        for epoch, row in enumerate(entries[1:], 1):
            if epoch - 1 - significant >= self.patience:
                raise ValueError("resume contains epochs after registered stopping")
            _, order_seed, digest, rng = _order(self.model, epoch, len(self.train))
            if (row.epoch != epoch or row.optimizer_steps != expected_batch_steps
                or not math.isfinite(row.validation_objective) or row.train_objective is None
                or not math.isfinite(row.train_objective) or row.order_seed != order_seed
                or row.permutation_sha256 != digest or row.gradient_finite is not True
                or type(row.gradient_nonzero) is not bool):
                raise ValueError("invalid numeric resume history")
            if row.validation_objective < entries[best].validation_objective:
                best = epoch
            if entries[significant].validation_objective - row.validation_objective >= self.improvement:
                significant = epoch
        if (metadata["best_epoch"], metadata["significant_epoch"], snapshot.rng) != (best, significant, rng):
            raise ValueError("stale best epoch, stopping or RNG resume state")
        step = (len(entries) - 1) * expected_batch_steps
        if metadata["optimizer_steps"] != [step] * len(self.parameters):
            raise ValueError("invalid optimizer step inventory")
        parameter_size = len(self.initial)
        for weights in (snapshot.current, snapshot.best):
            if len(weights) != parameter_size or not np.isfinite(np.frombuffer(weights, dtype="<f8")).all():
                raise ValueError("invalid numeric resume weights")
        if not step and (snapshot.current != snapshot.initial or snapshot.best != snapshot.initial):
            raise ValueError("epoch-zero weights differ from initialization")
        required = parameter_size * 2 if step else 0
        if len(snapshot.moments) != required:
            raise ValueError("invalid optimizer moment dimensions")
        values = np.frombuffer(snapshot.moments, dtype="<f8")
        if not np.isfinite(values).all():
            raise ValueError("invalid optimizer moments")
        state, offset = {}, 0
        if step:
            for parameter in self.parameters:
                size = parameter.numel()
                first = values[offset:offset + size]
                shift = len(values) // 2
                second = values[shift + offset:shift + offset + size]
                if (second < 0).any():
                    raise ValueError("invalid negative AdamW second moment")
                state[parameter] = {
                    "step": torch.tensor(float(step), dtype=torch.float32, device="cpu"),
                    "exp_avg": torch.from_numpy(first.copy()).reshape(parameter.shape),
                    "exp_avg_sq": torch.from_numpy(second.copy()).reshape(parameter.shape),
                }
                offset += size
        load_weights(self.model, snapshot.current, expected_sha256=sha256(snapshot.current).hexdigest())
        self.optimizer.state.clear()
        self.optimizer.state.update(state)
        self.history, self.best = entries, snapshot.best
        self.best_epoch, self._last_significant, self._rng = best, significant, rng
        self._interrupted = False


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
