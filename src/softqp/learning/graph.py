"""Exact one-sided graph encoding, LEConv classifier, labels and calibration."""

from dataclasses import dataclass
import math

import numpy as np

from .parameters import cpu64, initialize, require_version, stream, torch

require_version("torch-geometric", "2.8.1")
from torch_geometric.data import Batch, Data
from torch_geometric.nn import LEConv


@dataclass(frozen=True)
class Side:
    canonical_row: int
    side: str
    sign: int
    equality: bool


@dataclass(frozen=True)
class OneSidedLift:
    P: np.ndarray
    q: np.ndarray
    G: np.ndarray
    b: np.ndarray
    row_map: tuple[Side, ...]

    @classmethod
    def from_interval(cls, P, q, A, lower, upper):
        """Use the admitted dimensionless interval lift and canonical side order."""
        rows, bounds, mapping = [], [], []
        for i, (lo, up) in enumerate(zip(lower, upper, strict=True)):
            if lo == up:
                sides = (("equality", 1, up),)
            else:
                sides = (("upper", 1, up), ("lower", -1, lo))
            for name, sign, bound in sides:
                if np.isfinite(bound):
                    rows.append(sign * A[i])
                    bounds.append(sign * bound)
                    mapping.append(Side(i, name, sign, name == "equality"))
        return cls(P.copy(), q.copy(), np.asarray(rows, dtype=np.float64).reshape(len(rows), len(q)),
                   np.array(bounds, dtype=np.float64), tuple(mapping))

    @property
    def equality(self):
        return np.array([row.equality for row in self.row_map], dtype=bool)

    def interval_dual(self, dual, rows):
        if np.shape(dual) != self.b.shape:
            raise ValueError("invalid one-sided dual dimensions")
        result = np.zeros(rows, dtype=np.float64)
        for side, multiplier in zip(self.row_map, dual, strict=True):
            result[side.canonical_row] += side.sign * multiplier
        return result

    def graph(self):
        n = self.q.size
        nodes = np.zeros((n + self.b.size, 4), dtype=np.float64)
        nodes[:n, 0] = self.q
        nodes[n:, 1:] = np.column_stack((self.b, self.equality, np.ones(self.b.size)))
        pi, pj = np.nonzero(self.P)
        gi, gj = np.nonzero(self.G)
        # Alternate the two orientations of each signed constraint edge.
        sources = np.column_stack((gj, n + gi)).ravel()
        targets = np.column_stack((n + gi, gj)).ravel()
        edges = np.array([np.r_[pi, sources], np.r_[pj, targets]], dtype=np.int64)
        edge_weights = np.r_[self.P[pi, pj], np.repeat(self.G[gi, gj], 2)]
        columns = nodes.T.tolist()
        normalized = np.empty_like(nodes)
        for j, column in enumerate(columns):
            mean = sum(column) / len(column)
            variance = sum((value - mean)**2 for value in column) / len(column)
            scale = math.sqrt(variance) if variance > 0 else 1
            normalized[:, j] = [(value - mean) / scale for value in column]
        if not np.isfinite(normalized).all():
            raise ValueError("nonrepresentable-transform: graph normalization")
        return Data(x=cpu64(normalized), edge_index=torch.as_tensor(edges, device="cpu"),
                    edge_weight=cpu64(edge_weights),
                    inequality_mask=torch.as_tensor(
                        np.r_[np.zeros(n, dtype=bool), ~self.equality], device="cpu"),
                    num_nodes=nodes.shape[0])


class ActiveSetGNN(torch.nn.Module):
    method = "L-2"
    head = "graph"

    def __init__(self, seed, positive_prior):
        super().__init__()
        if not 0 < positive_prior < 1:
            raise ValueError("GNN initialization requires both training classes")
        self.seed = seed
        stream(self.method, seed, self.head, "validate")
        # PyG's constructor allocates temporary parameters on meta, so its
        # default initializers never consume the CPU RNG. Then set every value.
        with torch.device("meta"):
            self.input_layer = LEConv(4, 128)
            self.inner_layer = LEConv(128, 128)
            self.output_layer = LEConv(128, 1)
            self.norm1 = torch.nn.LayerNorm(128)
            self.norm2 = torch.nn.LayerNorm(128)
        self.to_empty(device="cpu").double()
        if sum(p.numel() for p in self.parameters()) != 52_098:
            raise RuntimeError("pinned LEConv parameter count mismatch")
        self.initialization = initialize(self, positive_prior)

    def forward(self, graph):
        edges, weights = graph.edge_index, graph.edge_weight
        x = self.norm1(self.input_layer(graph.x, edges, weights))
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.norm2(self.inner_layer(x, edges, weights))
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        return self.output_layer(x, edges, weights).squeeze(-1)

    def probabilities(self, graph):
        return self(graph).sigmoid()


def batch_graphs(graphs):
    """Batch graphs whose input-local normalization was completed independently."""
    return Batch.from_data_list(list(graphs))


@dataclass(frozen=True)
class Labels:
    target: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.target.ndim != 1 or self.target.shape != self.mask.shape or self.mask.dtype != torch.bool:
            raise ValueError("invalid graph label shape or mask")
        if not bool(((self.target == 0) | (self.target == 1)).all()):
            raise ValueError("non-binary label")
        if bool((self.target[~self.mask] != 0).any()):
            raise ValueError("masked graph nodes may not have targets")


def robust_labels(lift, x, raw_dual):
    """Admit numeric robust dual-active labels after original-space acceptance."""
    if np.shape(x) != lift.q.shape or np.shape(raw_dual) != lift.b.shape:
        raise ValueError("invalid DAQP label result dimensions")
    with np.errstate(over="raise", invalid="raise"):
        slack = lift.b - lift.G @ x
    if not all(np.isfinite(v).all() for v in (x, raw_dual, slack)):
        raise ValueError("nonfinite DAQP label result")
    tau = 1e-7 * np.maximum(1, np.abs(lift.b))
    inequality = ~lift.equality
    positive = inequality & (np.abs(slack) <= tau) & (raw_dual > 1e-7)
    negative = inequality & (slack > tau) & (np.abs(raw_dual) <= 1e-7)
    variables = np.zeros(lift.q.size, dtype=bool)
    return Labels(cpu64(np.r_[variables, positive]),
                  torch.as_tensor(np.r_[variables, positive | negative], device="cpu"))


@dataclass(frozen=True)
class ClassStatistics:
    mass0: float
    mass1: float

    @property
    def positive_prior(self):
        return self.mass1 / (self.mass0 + self.mass1)

    @property
    def weights(self):
        total = self.mass0 + self.mass1
        return total / (2 * self.mass0), total / (2 * self.mass1)


def fit_classes(weighted_train):
    from .training import weighted_units

    mass = [0.0, 0.0]
    for unit, source_weight in weighted_units(weighted_train, "train"):
        for frame in unit.frames:
            labels = frame.labels
            active = labels.target[labels.mask]
            if active.numel():
                frame_weight = source_weight / len(unit.frames)
                for cls in (0, 1):
                    mass[cls] += frame_weight * int((active == cls).sum()) / active.numel()
    if min(mass) <= 0:
        raise ValueError("both graph training classes require positive weighted mass")
    return ClassStatistics(*mass)


def classification_loss(logits, labels, statistics):
    if logits.shape != labels.mask.shape:
        raise ValueError("invalid classification logit shape")
    values = logits[labels.mask]
    if not values.numel():
        # Maintain the graph and give exactly zero gradient for an empty mask.
        return values.sum()
    target = labels.target[labels.mask].to(torch.float64)
    w0, w1 = statistics.weights
    zero = torch.zeros_like(values)
    return torch.where(target == 1, w1 * torch.logaddexp(zero, -values),
                       w0 * torch.logaddexp(zero, values)).mean()


@dataclass(frozen=True)
class GraphFrame:
    graph: Data
    labels: Labels

    def __post_init__(self):
        if self.labels.mask.shape != (self.graph.num_nodes,) or bool(
            (self.labels.mask & ~self.graph.inequality_mask).any()
        ):
            raise ValueError("variable or equality node entered a graph loss mask")

    def loss(self, model, statistics):
        return classification_loss(model(self.graph), self.labels, statistics)


@dataclass(frozen=True)
class Observation:
    probabilities: torch.Tensor
    labels: Labels
    inequality_mask: torch.Tensor | None = None

    def __post_init__(self):
        p = self.probabilities
        if p.shape != self.labels.mask.shape or not bool(((0 <= p) & (p <= 1)).all()):
            raise ValueError("invalid graph probabilities")
        inequality = self.inequality_mask
        if inequality is not None and (inequality.shape != p.shape or inequality.dtype != torch.bool
                                       or bool((self.labels.mask & ~inequality).any())):
            raise ValueError("invalid graph inequality observation mask")


GRID = tuple(i / 10 for i in range(1, 10))


def _macro_f1(units, threshold):
    scores = []
    for unit, weight in units:
        tp = fp = fn = 0.0
        for frame in unit.frames:
            active = frame.labels.mask
            truth = frame.labels.target[active] == 1
            prediction = frame.probabilities[active] > threshold
            tp += int((prediction & truth).sum()) / len(unit.frames)
            fp += int((prediction & ~truth).sum()) / len(unit.frames)
            fn += int((~prediction & truth).sum()) / len(unit.frames)
        denominator = 2 * tp + fp + fn
        scores.append(weight * (2 * tp / denominator if denominator else 0.0))
    return sum(scores)


def _points(units):
    points = []
    counts = [0, 0]
    zero_mask_frames = 0
    for unit, weight in units:
        for frame in unit.frames:
            active = frame.labels.mask
            count = int(active.sum())
            if not count:
                zero_mask_frames += 1
                continue
            for p, y in zip(frame.probabilities[active].tolist(), frame.labels.target[active].tolist()):
                counts[int(y)] += 1
                points.append((p, y, (weight / len(unit.frames)) / count))
    return points, counts, zero_mask_frames


def _quality(points):
    mass = sum(w for _, _, w in points)
    if not mass:
        return None, None
    brier = sum(w * (p - y)**2 for p, y, w in points) / mass
    bins = [[] for _ in range(10)]
    for row in points:
        bins[min(int(row[0] * 10), 9)].append(row)
    ece = sum(abs(sum(w * p for p, _, w in selected) - sum(w * y for _, y, w in selected)) / mass
              for selected in bins if selected)
    return brier, ece


def calibrate(weighted_calibration):
    """Tune the fixed grid on calibration only, preferring a larger exact tie."""
    from .training import weighted_units

    units = weighted_units(weighted_calibration, "calibration")
    points, counts, zero_frames = _points(units)
    if not points:
        raise ValueError("calibration undefined: all graph masks are empty")
    scores = tuple(_macro_f1(units, threshold) for threshold in GRID)
    best = max(range(len(GRID)), key=lambda i: (scores[i], GRID[i]))
    brier, ece = _quality(points)
    return dict(threshold=GRID[best], scores=scores, f1=scores[best], brier=brier, ece=ece,
                class_counts=tuple(counts), mask_count=sum(counts), zero_mask_frames=zero_frames)


def classification_metrics(weighted_test, *, threshold):
    from .training import weighted_units

    if threshold not in GRID:
        raise ValueError("unregistered classification threshold")
    units = weighted_units(weighted_test, "test")
    points, counts, zero_frames = _points(units)
    inequality_count = 0
    coverage = coverage_mass = 0.0
    for unit, source_weight in units:
        frame_weight = source_weight / len(unit.frames)
        for frame in unit.frames:
            if frame.inequality_mask is None:
                raise ValueError("held-out classification requires its graph inequality mask")
            count = int(frame.inequality_mask.sum())
            inequality_count += count
            if count:
                coverage += frame_weight * int(frame.labels.mask.sum()) / count
                coverage_mass += frame_weight
    brier, ece = _quality(points)
    pos = math.fsum(w for _, y, w in points if y == 1)
    neg = math.fsum(w for _, y, w in points if y == 0)
    result = dict(roc=None, pr=None, f1=_macro_f1(units, threshold) if points else None,
                  brier=brier, ece=ece, class_counts=tuple(counts), mask_count=sum(counts),
                  inequality_count=inequality_count, zero_mask_frames=zero_frames,
                  coverage=coverage / coverage_mass if coverage_mass else None)
    if not (pos and neg):
        result["undefined_reason"] = "missing-class" if points else "all-masks-empty"
        return result
    by_score = {}
    for p, y, w in points:
        by_score.setdefault(p, [0.0, 0.0])[int(y)] += w
    tp = fp = 0.0
    roc, pr = [(0.0, 0.0)], [(0.0, 1.0)]
    for score in sorted(by_score, reverse=True):
        d_fp, d_tp = by_score[score]
        tp, fp = tp + d_tp, fp + d_fp
        roc.append((min(fp / neg, 1.0), min(tp / pos, 1.0)))
        pr.append((min(tp / pos, 1.0), tp / (tp + fp)))
    result.update(roc=tuple(roc), pr=tuple(pr), undefined_reason=None)
    return result
