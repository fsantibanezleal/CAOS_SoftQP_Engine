"""Real stock-OSQP and DAQP warm-start paths over the official original-QP lift."""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
import inspect
import time
from types import SimpleNamespace
from weakref import WeakKeyDictionary

import numpy as np
from scipy import sparse

from ..comparators.backend import BackendUnavailable
from ..comparators.oracle import REFERENCE_LIMITS, reference_objective
from ..comparators.vertical import ComparatorSession, _LIMITS
from .parameters import head_id, require_version, torch, weight_bytes
from .fixed_point import FixedPointProblem, L2WSHead, mlp_features


OSQP_SETTINGS = dict(
    eps_abs=1e-7, eps_rel=1e-7, max_iter=200_000, rho=0.1, rho_is_vec=False,
    sigma=1e-6, alpha=1.0, scaling=0, adaptive_rho=False, polishing=False,
    check_termination=1, verbose=False, warm_starting=True,
)
DAQP_SETTINGS = dict(
    primal_tol=1e-7, dual_tol=1e-7, zero_tol=1e-12,
    iter_limit=200_000, eps_prox=0.0, eq_reduction=-1,
)
DAQP_LABEL_SETTINGS = dict(
    DAQP_SETTINGS, primal_tol=1e-10, dual_tol=1e-10, iter_limit=1_000_000, time_limit=30,
)
_REFERENCE_VERSIONS = {"stock-osqp": "1.1.3", "clarabel": "0.11.1", "piqp": "0.6.4"}
_REFERENCE_STATUSES = {"stock-osqp": "solved", "clarabel": "Solved", "piqp": "Status.PIQP_SOLVED"}
_REFERENCES = WeakKeyDictionary()


class CheckpointUnavailable(ValueError):
    """A requested learned result has no matching numeric initializer."""


class UnsupportedInput(ValueError):
    """The canonical problem falls outside a learned method's domain."""


class ReferenceEvidence:
    """Opaque local result of the independent fixed three-backend reference."""

    __slots__ = ("__weakref__",)

    def as_dict(self):
        """Return a detached audit copy. The copy cannot authorize a solve."""
        return deepcopy(_REFERENCES[self])

    def __getitem__(self, key):
        return deepcopy(_REFERENCES[self][key])


def _reference_value(problem, reference):
    if type(reference) is not ReferenceEvidence or reference not in _REFERENCES:
        raise ValueError("independent three-backend reference evidence is required")
    receipt = _REFERENCES[reference]
    if receipt["transform_sha256"] != problem.transform_sha256:
        raise ValueError("independent reference transform mismatch")
    return receipt


@lru_cache
def _backend(name, expected):
    try:
        require_version(name, expected)
        return import_module(name)
    except ImportError as error:
        raise BackendUnavailable(f"learning requires {name}=={expected}") from error


def backend_settings():
    """Read the exact admitted binding defaults without constructing a QP."""
    osqp = _backend("osqp", "1.1.3")
    daqp = _backend("daqp", "0.10.3")
    settings = osqp.OSQP(algebra="builtin").ext.OSQPSettings()
    osqp.OSQP(algebra="builtin").ext.osqp_set_default_settings(settings)
    osqp_defaults = {}
    for key in dir(settings):
        if not key.startswith("_"):
            value = getattr(settings, key)
            osqp_defaults[key] = dict(name=value.name, value=value.value) if hasattr(value, "name") else value
    daqp_defaults = {name: p.default for name, p in inspect.signature(daqp.solve).parameters.items()
                     if p.default is None or type(p.default) in (bool, int, float)}
    daqp_defaults["break_points"] = np.asarray(
        inspect.signature(daqp.solve).parameters["break_points"].default).tolist()
    return {
        "osqp": {**osqp_defaults, **OSQP_SETTINGS},
        "daqp": {**daqp_defaults, **DAQP_SETTINGS},
        "daqp-label": {**daqp_defaults, **DAQP_LABEL_SETTINGS},
    }


def _exact_pd(matrix):
    """Check exact positive pivots within independent nonzero Hessian blocks."""
    if np.any(matrix.diagonal() <= 0):
        return False
    off_diagonal = matrix - sparse.diags(matrix.diagonal(), format="csc")
    off_diagonal.eliminate_zeros()
    if not off_diagonal.nnz:
        return True
    count, labels = sparse.csgraph.connected_components(off_diagonal, directed=False)
    for component in range(count):
        indices = np.flatnonzero(labels == component)
        if indices.size == 1:
            continue
        rows = matrix[indices][:, indices].toarray()
        exact = [[Fraction(float(value)) for value in row] for row in rows]
        for k, row in enumerate(exact):
            pivot = row[k]
            if pivot <= 0:
                return False
            for i in range(k + 1, len(exact)):
                if not row[i]:
                    continue
                multiple = row[i] / pivot
                for j in range(i, len(exact)):
                    exact[i][j] -= multiple * row[j]
    return True


def _digest(formulation, arrays):
    hasher = sha256()
    hasher.update(formulation)
    for value in arrays:
        if sparse.issparse(value):
            arrays = (np.asarray(value.shape, dtype="<i8"), value.indptr, value.indices, value.data)
        else:
            arrays = (np.asarray(value),)
        for a in arrays:
            dtype = "<i8" if a.dtype.kind in "iu" else "<f8"
            hasher.update(a.size.to_bytes(8, "little"))
            hasher.update(a.astype(dtype, copy=False).tobytes())
    return hasher.hexdigest()


class PreparedLearningProblem:
    """Prepare the original QP and retain the exact physical/normalized lift maps."""

    def __init__(self, problem, **units):
        started = time.perf_counter()
        self._session = ComparatorSession(problem, "clarabel", **units)
        session = self._session
        original = session.problem
        physical, _, _, dimensionless, dx, dr = session.data
        if session.peak:
            formulation = b"peak"
            semantics = original.layout()
        else:
            formulation = b"separable"
            p = original.penalty
            semantics = (p.soft, p.lower_linear, p.upper_linear, p.lower_quadratic, p.upper_quadratic)
        self._transform_sha256 = _digest(formulation, (
            original.P, original.q, original.A, original.lower, original.upper, *semantics,
            session.dx, session.dr, session.dt, [session.sf], *physical, *dimensionless, dx, dr,
        ))
        self.transform_seconds = time.perf_counter() - started

    @property
    def transform_sha256(self):
        return self._transform_sha256

    @property
    def shape(self):
        A = self._session.data[0][2]
        return A.shape[1], A.shape[0]

    def fixed_point(self):
        P, q, A, lower, upper = self._session.data[3]
        return FixedPointProblem(P.toarray(), q, A.toarray(), lower, upper)

    def features(self):
        P, q, A, lower, upper = self._session.data[3]
        return mlp_features(P.toarray(), q, A.toarray(), lower, upper)

    def one_sided(self):
        from .graph import OneSidedLift

        P, q, A, lower, upper = self._session.data[3]
        if not _exact_pd(P):
            raise UnsupportedInput("DAQP requires an exactly positive definite lift")
        return OneSidedLift.from_interval(P.toarray(), q, A.toarray(), lower, upper)

    def reference(self):
        """Run the fixed three-reference policy, under caller resource supervision."""
        p = self._session
        measured = reference_objective(p.problem, variable_units=p.dx, row_units=p.dr,
                                       peak_units=p.dt, objective_unit=p.sf)
        if measured["status"] == "oracle-agreed":
            references = measured["references"]
            if set(references) != set(_REFERENCE_VERSIONS):
                raise ValueError("incomplete independent reference backend inventory")
            objectives = []
            for name, data in references.items():
                metrics = data["metrics"]
                if (data["passed"] is not True or data["backend_version"] != _REFERENCE_VERSIONS[name]
                    or data["native_status"] != _REFERENCE_STATUSES[name]
                    or not all(0 <= metrics[k] <= value for k, value in REFERENCE_LIMITS.items())
                    or not metrics.get("epigraph", {}).get("accepted", True)
                    or not np.isfinite(metrics["objective"])):
                    raise ValueError(f"independent reference rejected: {name}")
                objectives.append(metrics["objective"])
            scale = max(p.sf, max(abs(value) for value in objectives))
            spread = max(objectives) / scale - min(objectives) / scale
            if (spread != measured["spread"] or spread > 1e-8
                or sorted(objectives)[1] != measured["objective"]):
                raise ValueError("independent reference spread or objective mismatch")
        evidence = ReferenceEvidence()
        _REFERENCES[evidence] = deepcopy(measured) | dict(transform_sha256=self.transform_sha256)
        return evidence


@dataclass(frozen=True)
class NativeAttempt:
    flags: tuple[int, ...]
    native_flag: int
    removed_row: int | None
    elapsed_seconds: float
    setup_seconds: float
    solve_seconds: float
    iterations: int


@dataclass(frozen=True)
class InferenceBinding:
    """Numeric bindings reconstructed by the product's admitted artifact reader."""

    transform_sha256: str
    weights_sha256: str
    normalizer_sha256: str | None = None
    threshold: float | None = None


def repair_active_set(lift, probabilities, threshold, attempt):
    """Invoke the callback with registered sense vectors and record every attempt."""
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("invalid active-set threshold")
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != lift.b.shape or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("invalid DAQP constraint probabilities")
    equality = lift.equality
    active = (scores > threshold) & ~equality
    flags = np.where(equality, 5, active).astype(np.intc)
    drop_order = sorted(np.flatnonzero(active),
                        key=lambda i: (scores[i], lift.row_map[i].canonical_row, i))
    traces = []
    for remove in (*drop_order, None):
        start = time.perf_counter()
        native = attempt(flags.copy())
        wall = time.perf_counter() - start
        _, _, flag, info = native
        removed = int(remove) if flag == -6 and remove is not None else None
        traces.append(NativeAttempt(tuple(int(f) for f in flags), int(flag), removed, wall,
                                    info["setup_time"], info["solve_time"], info["iterations"]))
        if removed is None:
            return native, tuple(traces)
        flags[remove] = 0


def _daq_native(lift, flags, settings):
    daqp = _backend("daqp", "0.10.3")
    # DAQP's native infinity sentinel is 1e30, matching its declared fval bound.
    if np.any(np.abs(lift.b) >= 1e25):
        raise UnsupportedInput("DAQP finite constraint exceeds sentinel-safe range")
    lower = np.where(lift.equality, lift.b, -1e30)
    return daqp.solve(H=lift.P, f=lift.q, A=lift.G, bupper=lift.b,
                      blower=lower, sense=flags, **settings)


def _lift_accepted(problem, x, y, metrics, limits):
    P, q, A, lower, upper = problem._session.data[3]
    px, aty, ax = P @ x, A.T @ y, A @ x
    inf = lambda values: np.max(np.abs(values), initial=0.0)
    denominator = max(1.0, inf(px), inf(q), inf(aty))
    stationarity = inf(px + q + aty) / denominator
    bound = np.where(y > 0, upper, lower)
    finite = np.isfinite(bound)
    y_ax, y_bound = y[finite] * ax[finite], y[finite] * bound[finite]
    complements = np.abs(y_ax - y_bound) / np.maximum(1, np.maximum(abs(y_ax), abs(y_bound)))
    lifted_objective = 0.5 * (x @ px) + q @ x
    objective = metrics["objective"] / problem._session.sf
    gap = abs(lifted_objective - objective) / max(1, abs(objective))
    return bool(
        all(np.isfinite(v).all() for v in (px, aty, ax, complements, stationarity, gap))
        and stationarity <= limits["K"]
        and np.max(lower - ax, initial=0) <= limits["H"]
        and np.max(ax - upper, initial=0) <= limits["H"]
        and np.max(np.abs(y[~finite]), initial=0) <= 1e-7
        and np.max(complements, initial=0) <= limits["F"]
        and gap <= limits["H"]
    )


def _checked(problem, x, y, native_flag, successes, reference, *, limits=_LIMITS,
             objective_gap_limit=1e-6):
    receipt = _reference_value(problem, reference) if reference is not None else None
    result = dict(native_flag=int(native_flag), status="native-failure", x=None, y=None,
                  metrics=None, objective_gap=None)
    if native_flag not in successes:
        return result
    try:
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ArithmeticError("nonfinite native result")
        x_lift, y_lift = x, y
        x, y, metrics = problem._session.recover_metrics(SimpleNamespace(x=x_lift, y=y_lift))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            metrics["canonical_accepted"] = _lift_accepted(problem, x_lift, y_lift, metrics, limits)
        accepted = (all(metrics[key] <= bound for key, bound in limits.items())
                    and metrics.get("epigraph", {}).get("accepted", True)
                    and metrics["canonical_accepted"])
        objective_gap = None
        if receipt is None:
            status = "candidate" if accepted else "accuracy-rejected"
        elif receipt["status"] != "oracle-agreed":
            status = "oracle-failed"
        else:
            f_ref = receipt["objective"]
            denominator = max(problem._session.sf, abs(f_ref))
            objective_gap = abs(metrics["objective"] - f_ref) / denominator
            status = "accepted" if accepted and objective_gap <= objective_gap_limit else "accuracy-rejected"
        result.update(status=status, x=x, y=y, metrics=metrics, objective_gap=objective_gap)
    except (ArithmeticError, ValueError) as error:
        result.update(status="accuracy-rejected", reason=str(error))
    return result


def _verify_model(problem, model, architecture, binding):
    if model is None or type(binding) is not InferenceBinding:
        raise CheckpointUnavailable("missing numeric learning checkpoint identity")
    if problem.transform_sha256 != binding.transform_sha256:
        raise CheckpointUnavailable("mismatched learning input transform identity")
    if type(model) is not architecture or sha256(weight_bytes(model)).hexdigest() != binding.weights_sha256:
        raise CheckpointUnavailable("mismatched numeric learning checkpoint identity")


def _stock_setup(matrices, **overrides):
    osqp = _backend("osqp", "1.1.3")
    P, q, A, lo, up = matrices
    if (abs(lo[np.isfinite(lo)]) >= 1e25).any() or (abs(up[np.isfinite(up)]) >= 1e25).any():
        raise UnsupportedInput("OSQP finite constraint exceeds sentinel-safe range")
    solver = osqp.OSQP(algebra="builtin")
    solver.setup(P=sparse.triu(P, format="csc"), q=q, A=A, l=lo, u=up,
                 **(OSQP_SETTINGS | overrides))
    return solver


def solve_osqp(problem, *, model=None, normalizer=None, binding=None,
               reference=None, mode="learned", previous=None):
    """Predict one physical lifted warm start or use the declared matched control.

    Previous-mode input is a previously accepted physical lifted x/y tuple.
    Pass None for frame zero or the prescribed failure reset.
    """
    if reference is not None:
        _reference_value(problem, reference)
    started = time.perf_counter()
    head = head_id(*problem.shape)
    physical, _, _, _, dx, dr = problem._session.data
    sf = problem._session.sf
    feature_seconds = inference_seconds = checkpoint_seconds = 0.0
    if mode == "learned":
        _verify_model(problem, model, L2WSHead, binding)
        if normalizer is None or model.head != head or (
            normalizer.numeric_sha256 != binding.normalizer_sha256 or binding.threshold is not None
        ):
            raise CheckpointUnavailable("missing normalizer or incorrect L2WS head")
        checkpoint_seconds = time.perf_counter() - started
        feature_start = time.perf_counter()
        features = normalizer.transform(problem.features())
        feature_seconds = time.perf_counter() - feature_start
        inference_start = time.perf_counter()
        with torch.no_grad():
            prediction = model(features).numpy()
        x0, y0 = prediction[:len(dx)] * dx, prediction[len(dx):] * (sf / dr)
        inference_seconds = time.perf_counter() - inference_start
    elif mode == "previous" and previous is not None:
        x0, y0 = previous
    elif mode in ("cold", "previous"):
        x0, y0 = np.zeros_like(dx), np.zeros_like(dr)
    else:
        raise ValueError("unsupported OSQP initialization mode")
    if np.shape(x0) != dx.shape or np.shape(y0) != dr.shape or not (
        np.isfinite(x0).all() and np.isfinite(y0).all()
    ):
        raise ValueError("nonrepresentable OSQP warm start")
    setup_start = time.perf_counter()
    solver = _stock_setup(physical)
    setup_wall = time.perf_counter() - setup_start
    warm_start = time.perf_counter()
    solver.warm_start(x=x0, y=y0)
    warm_seconds = time.perf_counter() - warm_start
    solve_start = time.perf_counter()
    raw = solver.solve(raise_error=False)
    native_wall = time.perf_counter() - solve_start
    evaluation_start = time.perf_counter()
    good = raw.info.status_val == 1
    x = raw.x / dx if good else None
    y = raw.y * dr / sf if good else None
    result = _checked(problem, x, y, raw.info.status_val, (1,), reference)
    result.update(native_status=raw.info.status, iterations=raw.info.iter,
                  setup_seconds=raw.info.setup_time, solve_seconds=raw.info.solve_time,
                  setup_wall_seconds=setup_wall, solve_wall_seconds=native_wall,
                  checkpoint_seconds=checkpoint_seconds, warm_start_seconds=warm_seconds,
                  initial_x=x0, initial_y=y0, feature_seconds=feature_seconds,
                  inference_seconds=inference_seconds, lifted_x=raw.x if good else None,
                  lifted_y=raw.y if good else None,
                  evaluation_seconds=time.perf_counter() - evaluation_start,
                  transform_seconds=problem.transform_seconds,
                  transform_sha256=problem.transform_sha256,
                  checkpoint_sha256=binding.weights_sha256 if mode == "learned" else None,
                  elapsed_seconds=time.perf_counter() - started)
    return result


def solve_daqp(problem, *, model=None, binding=None,
               reference=None, mode="learned", previous=None):
    """Infer constraint scores, inject the bounded repaired sense, and solve."""
    from .graph import ActiveSetGNN, GRID

    if reference is not None:
        _reference_value(problem, reference)
    started = time.perf_counter()
    lift = problem.one_sided()
    feature_seconds = time.perf_counter() - started
    inference_seconds = checkpoint_seconds = 0.0
    node_scores = None
    if mode == "learned":
        checkpoint_start = time.perf_counter()
        _verify_model(problem, model, ActiveSetGNN, binding)
        threshold = binding.threshold
        if threshold not in GRID or binding.normalizer_sha256 is not None:
            raise CheckpointUnavailable("missing registered graph calibration threshold")
        checkpoint_seconds = time.perf_counter() - checkpoint_start
        feature_start = time.perf_counter()
        graph = lift.graph()
        feature_seconds += time.perf_counter() - feature_start
        inference_start = time.perf_counter()
        with torch.no_grad():
            node_scores = model.probabilities(graph).numpy()
            if node_scores.shape != (graph.num_nodes,) or not np.isfinite(node_scores).all():
                raise ValueError("nonrepresentable graph probabilities")
            scores = node_scores[lift.q.size:]
        inference_seconds = time.perf_counter() - inference_start
    elif mode == "previous" and previous is not None:
        # previous_working_set supplies positive scores only on admitted active sides.
        scores, threshold = previous, 0.0
    elif mode in ("cold", "previous"):
        scores, threshold = np.zeros_like(lift.b), 0.0
    else:
        raise ValueError("unsupported DAQP initialization mode")
    repair_start = time.perf_counter()
    native, trace = repair_active_set(lift, scores, threshold, lambda flags: _daq_native(
        lift, flags, DAQP_SETTINGS))
    repair_seconds = time.perf_counter() - repair_start
    x, _, flag, info = native
    evaluation_start = time.perf_counter()
    good = flag in (1, 2)
    interval_dual = lift.interval_dual(info["lam"], problem.shape[1]) if good else None
    result = _checked(problem, x, interval_dual, flag, (1, 2), reference)
    result.update(attempts=trace, raw_side_dual=info["lam"] if good else None,
                  lifted_x=x if good else None, lifted_y=interval_dual, threshold=threshold,
                  scores=scores, node_scores=node_scores, feature_seconds=feature_seconds,
                  inference_seconds=inference_seconds,
                  checkpoint_seconds=checkpoint_seconds, repair_wall_seconds=repair_seconds,
                  setup_seconds=sum(row.setup_seconds for row in trace),
                  solve_seconds=sum(row.solve_seconds for row in trace),
                  iterations=sum(row.iterations for row in trace),
                  transform_seconds=problem.transform_seconds,
                  evaluation_seconds=time.perf_counter() - evaluation_start,
                  checkpoint_sha256=binding.weights_sha256 if mode == "learned" else None,
                  transform_sha256=problem.transform_sha256,
                  elapsed_seconds=time.perf_counter() - started)
    return result


def previous_working_set(previous_problem, previous_result, current_problem):
    """Map the last accepted DAQP state through the unchanged canonical row layout."""
    if previous_result is None or previous_result["status"] != "accepted":
        return None
    from .graph import robust_labels

    if previous_result["transform_sha256"] != previous_problem.transform_sha256:
        raise ValueError("previous DAQP input identity mismatch")
    old, new = previous_problem.one_sided(), current_problem.one_sided()
    if old.row_map != new.row_map or old.q.shape != new.q.shape:
        raise ValueError("previous DAQP canonical mapping changed")
    labels = robust_labels(old, previous_result["lifted_x"], previous_result["raw_side_dual"])
    dual = previous_result["raw_side_dual"]
    return np.where(labels.target[old.q.size:].numpy() == 1, dual / max(1.0, np.max(dual, initial=0.0)), 0.0)


def cold_labels(problem, *, reference):
    """Generate robust labels only from an accepted cold DAQP result and oracle."""
    from .graph import GraphFrame, robust_labels

    if reference is None or _reference_value(problem, reference)["status"] != "oracle-agreed":
        raise ValueError("oracle-failed: DAQP labels lack three-reference agreement")
    lift = problem.one_sided()
    flags = np.where(lift.equality, 5, 0).astype(np.intc)
    x, _, flag, info = _daq_native(lift, flags, DAQP_LABEL_SETTINGS)
    if flag not in (1, 2):
        raise ValueError(f"cold DAQP label rejected: native {flag}")
    y = lift.interval_dual(info["lam"], problem.shape[1])
    result = _checked(problem, x, y, flag, (1, 2), reference,
                      limits=REFERENCE_LIMITS, objective_gap_limit=1e-8)
    if result["status"] != "accepted":
        raise ValueError(f"cold DAQP label rejected: {result['status']} (native {flag})")
    labels = robust_labels(lift, x, info["lam"])
    return GraphFrame(lift.graph(), labels), result | dict(
        sense=flags, lifted_x=x, lifted_y=y, raw_side_dual=info["lam"],
        setup_seconds=info["setup_time"], solve_seconds=info["solve_time"],
        iterations=info["iterations"],
    )
