"""Directional coupled peak QPs, native original-row updates and exact baselines.

The native method is a restricted fixed-metric research implementation. It makes
no priority, runtime or controller-safety claim. Disjoint groups only.
"""

import copy
from dataclasses import dataclass, replace
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .problem import Penalty, Problem
from .solvers import UnsupportedProblemError


def _scalar(value, name):
    a = np.asarray(value)
    if a.shape or a.dtype.kind not in "fi" or not np.isfinite(a) or a < 0:
        raise ValueError(f"{name} must be finite, scalar and nonnegative")
    return float(a)


def _vector(value, size, name, *, positive=False, scalar=False):
    a = np.asarray(value)
    if scalar and a.shape == ():
        a = np.full(size, a)
    if a.shape != (size,) or a.dtype.kind not in "fi" or not np.isfinite(a).all():
        raise ValueError(f"{name} must have {size} finite numeric entries")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be positive")
    return a.astype(np.float64, copy=True)


def _certify_psd(matrix):
    # Never turn a genuinely negative curvature into a PSD model by shifting H.
    diagonal = matrix.diagonal()
    if np.any(diagonal < 0):
        raise ValueError("peak Hessian must be positive semidefinite")
    difference = matrix-matrix.T
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError("peak Hessian must be symmetric")
    off_diagonal = matrix-sparse.diags(diagonal)
    off_diagonal.eliminate_zeros()
    if not off_diagonal.nnz:
        return
    n = matrix.shape[0]
    magnitude = float(sparse.linalg.norm(matrix, ord=np.inf))
    if not np.isfinite(magnitude):
        raise UnsupportedProblemError("PSD numerical scale is not representable")
    try:
        minimum = (np.linalg.eigvalsh(matrix.toarray())[0] if n <= 128 else
                   sparse.linalg.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0])
    except (np.linalg.LinAlgError, sparse.linalg.ArpackError) as error:
        raise UnsupportedProblemError("PSD numerical certification did not converge") from error
    margin = 64*n*np.finfo(float).eps*magnitude
    if minimum > margin:
        return
    if minimum < -margin:
        raise ValueError("peak Hessian must be positive semidefinite")
    if n > 32:
        count, labels = sparse.csgraph.connected_components(off_diagonal, directed=False)
        if count > 1 and max(np.bincount(labels)) <= 32:
            for component in range(count):
                indices = np.flatnonzero(labels == component)
                _certify_psd(matrix[indices][:, indices])
            return
        raise UnsupportedProblemError("numerically ambiguous coupled PSD block exceeds the exact 32-variable budget")
    # Floats are exact dyadic rationals. This bounded Schur test proves PSD of
    # the submitted float64 matrix, including singular non-diagonal matrices.
    from fractions import Fraction
    exact = [[Fraction.from_float(float(value)) for value in row] for row in matrix.toarray()]
    for k in range(n):
        pivot = exact[k][k]
        if pivot < 0 or (pivot == 0 and any(exact[i][k] for i in range(k+1, n))):
            raise ValueError("peak Hessian must be positive semidefinite; exact negative curvature detected")
        if pivot > 0:
            for i in range(k+1, n):
                for j in range(i, n):
                    exact[j][i] -= exact[i][k]*exact[j][k]/pivot
                    exact[i][j] = exact[j][i]


@dataclass(frozen=True)
class PeakGroup:
    rows: object
    lower_scale: object = 1.0
    upper_scale: object = 1.0
    linear: float = 1.0
    quadratic: float = 0.0


@dataclass(frozen=True)
class PeakProblem:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    groups: tuple[PeakGroup, ...]

    def validate(self):
        if not sparse.issparse(self.P) or not sparse.issparse(self.A):
            raise ValueError("P and A must be sparse real matrices")
        n, m = self.P.shape[0], self.A.shape[0]
        if n <= 0 or self.P.shape != (n, n) or self.A.shape[1] != n:
            raise ValueError("invalid nonempty QP dimensions")
        for matrix in (self.P, self.A):
            if matrix.dtype.kind not in "fi" or not np.isfinite(matrix.data).all():
                raise ValueError("P and A must be finite real matrices")
        if not all(isinstance(value, np.ndarray) for value in (self.q, self.lower, self.upper)):
            raise ValueError("q and bound vectors must be NumPy arrays")
        _vector(self.q, n, "q")
        for bound in (self.lower, self.upper):
            a = np.asarray(bound)
            if a.shape != (m,) or a.dtype.kind not in "fi" or np.isnan(a).any():
                raise ValueError("invalid interval vectors")
        if np.any(self.lower > self.upper) or np.isposinf(self.lower).any() or np.isneginf(self.upper).any():
            raise ValueError("invalid interval endpoints")
        _certify_psd(sparse.csc_matrix(self.P))
        self.layout()

    def layout(self):
        m = self.A.shape[0]
        labels = np.full(m, -1, dtype=np.int32)
        a, b = np.zeros(m), np.zeros(m)
        cl, cu = np.ones(m), np.ones(m)
        for g, group in enumerate(self.groups):
            if not isinstance(group, PeakGroup):
                raise ValueError("every group must be a PeakGroup")
            rows = np.asarray(group.rows)
            if (rows.ndim != 1 or rows.size == 0 or rows.dtype.kind not in "iu"
                    or np.any(rows >= m) or np.any(rows < 0)
                    or len(np.unique(rows)) != rows.size):
                raise ValueError("group rows must be nonempty distinct in-range integer indices")
            if np.any(labels[rows] != -1):
                raise ValueError("overlapping peak groups are unsupported")
            labels[rows] = g
            a[rows] = _scalar(group.linear, "linear")
            b[rows] = _scalar(group.quadratic, "quadratic")
            cl[rows] = _vector(group.lower_scale, rows.size, "lower_scale", positive=True, scalar=True)
            cu[rows] = _vector(group.upper_scale, rows.size, "upper_scale", positive=True, scalar=True)
        return labels, a, b, cl, cu

    def evaluate(self, x, y=None):
        x = _vector(x, len(self.q), "x")
        labels, a, b, cl, cu = self.layout()
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z = np.asarray(self.A @ x)
            if not np.isfinite(z).all():
                raise ArithmeticError("nonfinite original constraint values")
            lo, up = np.maximum(self.lower-z, 0), np.maximum(z-self.upper, 0)
            hard = labels == -1
            group_metrics, penalty = [], 0.0
            for g in range(len(self.groups)):
                rows = labels == g
                peak = float(max(np.max(cl[rows]*lo[rows]), np.max(cu[rows]*up[rows])))
                first = np.flatnonzero(rows)[0]
                cost = float(a[first]*peak + (0.5*b[first]*peak)*peak)
                penalty += cost
                group_metrics.append(dict(group=g, peak=peak, penalty=cost))
            base = float(0.5*x @ (self.P @ x) + self.q @ x)
            answer = dict(objective=base+penalty, base_objective=base, penalty=penalty,
                          groups=group_metrics, lower_violation=lo.tolist(), upper_violation=up.tolist(),
                          hard_violation=float(np.max(np.maximum(lo[hard], up[hard]), initial=0)))
            if y is not None:
                y = _vector(y, len(self.lower), "y")
                repaired = y.copy()
                exact_domain, forbidden = True, 0.0
                for i, yi in enumerate(y):
                    endpoint = self.upper[i] if yi > 0 else self.lower[i]
                    if yi != 0 and not np.isfinite(endpoint):
                        exact_domain = False
                        forbidden = max(forbidden, abs(yi))
                        repaired[i] = 0
                domain_excess, normalized_domain = 0.0, 0.0
                for g in range(len(self.groups)):
                    rows = labels == g
                    first = np.flatnonzero(rows)[0]
                    original_polar = float(np.sum(np.maximum(y[rows], 0)/cu[rows]
                                                   + np.maximum(-y[rows], 0)/cl[rows]))
                    if b[first] == 0:
                        excess = max(original_polar-a[first], 0)
                        domain_excess = max(domain_excess, excess)
                        normalized_domain = max(normalized_domain, excess/max(original_polar, a[first])
                                                if max(original_polar, a[first]) > 0 else 0)
                        if excess > 0:
                            exact_domain = False
                        polar = float(np.sum(np.maximum(repaired[rows], 0)/cu[rows]
                                             + np.maximum(-repaired[rows], 0)/cl[rows]))
                        if polar > a[first]:
                            factor = np.nextafter(a[first]/polar, 0) if a[first] > 0 else 0
                            repaired[rows] *= factor
                support, conjugate, local_fenchel = 0.0, 0.0, 0.0
                for i, yi in enumerate(repaired):
                    if yi != 0:
                        endpoint = self.upper[i] if yi > 0 else self.lower[i]
                        support += yi*endpoint
                for g in range(len(self.groups)):
                    rows = labels == g
                    first = np.flatnonzero(rows)[0]
                    group_conjugate = 0.0
                    if b[first] > 0:
                        polar = float(np.sum(np.maximum(repaired[rows], 0)/cu[rows]
                                             + np.maximum(-repaired[rows], 0)/cl[rows]))
                        excess = max(polar-a[first], 0)
                        group_conjugate = (0.5*excess)*(excess/b[first])
                        conjugate += group_conjugate
                    group_support = sum(repaired[i]*(self.upper[i] if repaired[i]>0 else self.lower[i])
                                        for i in np.flatnonzero(rows) if repaired[i]!=0)
                    group_inner = float(repaired[rows] @ z[rows])
                    group_cost = group_metrics[g]['penalty']
                    local_fenchel = max(local_fenchel,
                        abs(group_cost+group_support+group_conjugate-group_inner)/
                        max(1, abs(group_cost), abs(group_support), abs(group_inner)))
                # Approximate hard feasibility can make its signed gap negative.
                # Never let it cancel a wrong subgradient in another row/group.
                for i in np.flatnonzero(hard):
                    if repaired[i]!=0:
                        endpoint_product = repaired[i]*(self.upper[i] if repaired[i]>0 else self.lower[i])
                        inner = repaired[i]*z[i]
                        local_fenchel = max(local_fenchel, abs(endpoint_product-inner)/
                                            max(1, abs(endpoint_product), abs(inner)))
                repair_norm = float(np.max(np.abs(repaired-y), initial=0))
                normalized_repair = repair_norm/max(1, float(np.max(np.abs(y), initial=0)))
                px = self.P @ x
                raw_stationarity = []
                for dual in (y, repaired):
                    aty = self.A.T @ dual
                    absolute = float(np.max(np.abs(px+self.q+aty), initial=0))
                    normalized = absolute/max(1, float(np.max(np.abs(px), initial=0)),
                                              float(np.max(np.abs(self.q), initial=0)),
                                              float(np.max(np.abs(aty), initial=0)))
                    raw_stationarity.append((absolute, normalized))
                # A tiny forbidden multiplier can have a large A.T effect.
                # Never combine repaired Fenchel with unchecked raw stationarity.
                answer.update(stationarity_residual=max(v[0] for v in raw_stationarity),
                              normalized_stationarity=max(v[1] for v in raw_stationarity),
                              raw_normalized_stationarity=raw_stationarity[0][1],
                              repaired_normalized_stationarity=raw_stationarity[1][1],
                              exact_dual_domain=exact_domain, dual_domain_excess=float(domain_excess),
                              normalized_dual_domain_excess=float(normalized_domain),
                              forbidden_side_dual_residual=float(forbidden), dual_repair_norm=repair_norm,
                              normalized_dual_repair=normalized_repair, repaired_y=repaired.tolist())
                usable = normalized_repair <= 1e-7
                fenchel = float(penalty+support+conjugate-repaired @ z)
                answer['approximate_fenchel_residual'] = fenchel if usable else None
                answer['normalized_aggregate_fenchel_residual'] = (
                    abs(fenchel)/max(1, abs(penalty), abs(support), abs(float(repaired @ z))) if usable else None)
                answer['normalized_fenchel_residual'] = float(local_fenchel) if usable else None
        if not np.isfinite(answer['objective']):
            raise ArithmeticError("nonfinite original peak objective")
        return answer


def _assemble(problem, method):
    """Compact sparse epigraph or exact one-sided published GSP formulation."""
    labels, a, b, cl, cu = problem.layout()
    m, n = problem.A.shape
    hard = np.flatnonzero(labels < 0)
    active = (labels >= 0) & ((a > 0) | (b > 0))
    if method == "row-split-gsp" and np.any(b[active] > 0):
        raise UnsupportedProblemError("published native NORMINF supports linear peaks, not squared peaks")
    upper = np.flatnonzero(active & np.isfinite(problem.upper))
    lower = np.flatnonzero(active & np.isfinite(problem.lower))
    live = np.unique(labels[np.r_[upper, lower]])
    remap = {int(g): k for k, g in enumerate(live)}
    npeak = len(live) if method == "epigraph" else 0
    rows = np.r_[hard, upper, lower, np.full(npeak, -1)].astype(int)
    kind = np.r_[np.zeros(len(hard)), np.ones(len(upper)), -np.ones(len(lower)), np.full(npeak, 2)].astype(int)
    coeff = np.r_[np.ones(len(hard)), cu[upper], -cl[lower], np.zeros(npeak)]
    original_rows = np.r_[hard, upper, lower]
    left = _scaled_rows(problem.A[original_rows], coeff[:len(original_rows)])
    if npeak:
        side_gids = np.array([remap[int(labels[i])] for i in np.r_[upper, lower]], dtype=int)
        right = sparse.csc_matrix((-np.ones(len(side_gids)),
                                   (np.arange(len(hard), len(original_rows)), side_gids)),
                                  shape=(len(original_rows), npeak))
        A = sparse.vstack([sparse.hstack([left, right]),
                           sparse.hstack([sparse.csc_matrix((npeak, n)), sparse.eye(npeak)])], format="csc")
    else:
        A = sparse.csc_matrix(left)
    firsts = np.array([np.flatnonzero(labels == g)[0] for g in live], dtype=int)
    P = sparse.block_diag([problem.P, sparse.diags(b[firsts])], format="csc") if npeak else problem.P
    q_tail = a[firsts] if npeak else np.empty(0)
    mapped = rows >= 0
    recover = sparse.csc_matrix((coeff[mapped], (rows[mapped], np.flatnonzero(mapped))), shape=(m, len(rows)))
    groups = np.full(len(rows), -1, dtype=np.int32)
    if method == "row-split-gsp":
        for j in range(len(hard), len(original_rows)):
            groups[j] = remap[int(labels[rows[j]])]
    weights = np.zeros(len(rows))
    if method == "row-split-gsp":
        for j in np.flatnonzero(groups >= 0):
            weights[j] = a[rows[j]]
    return dict(P=P, A=A, rows=rows, kind=kind, coeff=coeff, recover=recover,
                q_tail=q_tail, groups=groups, weights=weights, ngroups=len(live))


def _vectors(problem, assembly):
    rows, kind, coeff = (assembly[k] for k in ('rows', 'kind', 'coeff'))
    lower, upper = np.full(len(rows), -np.inf), np.full(len(rows), np.inf)
    hard, ups, lows, positive = kind == 0, kind == 1, kind == -1, kind == 2
    lower[hard], upper[hard] = problem.lower[rows[hard]], problem.upper[rows[hard]]
    with np.errstate(over='raise', under='raise', invalid='raise'):
        upper[ups] = coeff[ups]*problem.upper[rows[ups]]
        upper[lows] = coeff[lows]*problem.lower[rows[lows]]
    lower[positive] = 0
    return np.r_[problem.q, assembly['q_tail']], lower, upper


def _scaled_bounds(lower, upper, scale):
    with np.errstate(over="raise", under="raise", invalid="raise"):
        transformed_lower, transformed_upper = lower*scale, upper*scale
    if not (np.array_equal(np.isfinite(transformed_lower), np.isfinite(lower))
            and np.array_equal(np.isfinite(transformed_upper), np.isfinite(upper))):
        raise UnsupportedProblemError("row scaling changed a finite-side pattern")
    return transformed_lower, transformed_upper


def _scaled_rows(matrix, scale):
    canonical = sparse.csc_matrix(matrix, copy=True)
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    transformed = sparse.csc_matrix(sparse.diags(scale) @ canonical)
    transformed.sum_duplicates()
    transformed.eliminate_zeros()
    if not np.isfinite(transformed.data).all() or transformed.nnz != canonical.nnz:
        raise UnsupportedProblemError("nonrepresentable scaled matrix entries")
    return transformed


def _native_bounds(lower, upper):
    # OSQP's finite sentinel must never silently convert a finite side to free.
    for bounds in (lower, upper):
        if np.any(np.abs(bounds[np.isfinite(bounds)]) >= 1e25):
            raise UnsupportedProblemError("finite bounds exceed the native sentinel-safe range")


def _read_daqp_lp(problem, defaults):
    # Pinned Model.setup(None) has a documented/implemented dimensionality gap.
    # The published one-shot H=None API reaches DAQP's actual LP path.
    import daqp
    from .solvers import _daqp_controls
    controls = _daqp_controls(defaults, proximal=True)
    x, _, flag, info = daqp.solve(None, np.ascontiguousarray(problem.q),
                                 np.ascontiguousarray(problem.A.toarray()),
                                 np.ascontiguousarray(problem.upper), np.ascontiguousarray(problem.lower),
                                 np.where(problem.lower == problem.upper, 5, 0).astype(np.intc), **controls)
    status = {1: "optimal", 2: "soft optimal", 4: "optimal inexact", 5: "feasible"}.get(flag, f"DAQP status {flag}")
    result = SimpleNamespace(info=SimpleNamespace(status=status, status_val=flag,
                             iter=info['iterations'], setup_time=info['setup_time'],
                             solve_time=info['solve_time'], proximal_iterations=int(info.get('nodes', 0))))
    return result, np.asarray(x), np.asarray(info['lam']), dict(controls, lp_hessian='None')


class PreparedPeakSolver:
    """Owned fixed-shape workspace with validated atomic vector updates."""

    def __init__(self, problem, method="native", *, row_normalization=True, **settings):
        start = time.perf_counter()
        if method not in ("native", "epigraph", "row-split-gsp", "epigraph-daqp", "epigraph-daqp-prox"):
            raise ValueError("method must be native, epigraph, row-split-gsp, epigraph-daqp or epigraph-daqp-prox")
        self._problem = copy.deepcopy(problem)
        self._problem.validate()
        _native_bounds(self._problem.lower, self._problem.upper)
        self.method = method
        self._daqp_proximal = method == "epigraph-daqp-prox"
        if not isinstance(row_normalization, (bool, np.bool_)):
            raise ValueError("row_normalization must be boolean")
        self._assembly = None
        self._row_scale = np.ones(self._problem.A.shape[0])
        self.row_normalization = bool(row_normalization) if method == "native" else False
        defaults = dict(verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000,
                        polishing=False, alpha=1.6, adaptive_rho=False, scaling=0)
        defaults.update(settings)
        if method == "native":
            alpha = np.asarray(defaults['alpha'])
            if (alpha.shape or alpha.dtype.kind not in "fi" or not np.isfinite(alpha)
                    or not 0 < alpha < 2 or defaults['adaptive_rho']
                    or defaults['scaling'] or defaults['polishing']):
                raise UnsupportedProblemError("native peaks require fixed metric, finite scalar 0<alpha<2, scaling=0 and no polishing")
            defaults['alpha'] = float(alpha)
        if method in ("epigraph-daqp", "epigraph-daqp-prox"):
            allowed = {"eps_abs", "max_iter"}
            if self._daqp_proximal:
                allowed |= {"eps_prox", "eta_prox", "lp_mode"}
            if set(settings) - allowed:
                raise UnsupportedProblemError("DAQP uses absolute eps_abs, not eps_rel; accepted controls are eps_abs, max_iter and declared proximal/LP controls")
            from .solvers import _setup_daqp, _daqp_controls
            _daqp_controls(defaults, proximal=self._daqp_proximal)
            if self._daqp_proximal and defaults.get("lp_mode", "one-shot") not in ("one-shot", "model"):
                raise ValueError("lp_mode must be one-shot or model")
        if method == "epigraph":
            from osqp import OSQP
        else:
            from .interface import OSQP
        self._solver = OSQP() if method not in ("epigraph-daqp", "epigraph-daqp-prox") else None
        if method == "native":
            labels, _, _, cl, cu = self._problem.layout()
            if self.row_normalization:
                self._row_scale = np.where(labels >= 0, np.maximum(cl, cu), 1)
            P, A = self._problem.P, _scaled_rows(self._problem.A, self._row_scale)
            q = self._problem.q
            lower, upper = _scaled_bounds(self._problem.lower, self._problem.upper, self._row_scale)
        else:
            self._assembly = _assemble(self._problem, "epigraph" if method in ("epigraph-daqp", "epigraph-daqp-prox") else method)
            P, A = self._assembly['P'], self._assembly['A']
            q, lower, upper = _vectors(self._problem, self._assembly)
        _native_bounds(lower, upper)
        if not np.isfinite(P.data).all() or not np.isfinite(A.data).all():
            raise UnsupportedProblemError("nonrepresentable transformed matrices")
        self._defaults = defaults
        if method in ("epigraph-daqp", "epigraph-daqp-prox"):
            if len(q) > 2048 or A.shape[0] > 8192:
                raise UnsupportedProblemError("DAQP epigraph exceeds its declared 2048-variable / 8192-row dense budget")
            self._daqp_problem = Problem(P, q, A, lower, upper, Penalty.uniform(len(lower), soft=False))
            self._daqp_lp = self._daqp_proximal and not np.any(P.data) and defaults.get("lp_mode", "one-shot") == "one-shot"
            if self._daqp_lp:
                self._daqp_model, self._daqp_setup = None, 0.
            else:
                self._daqp_model, self._daqp_setup = _setup_daqp(self._daqp_problem, defaults, proximal=self._daqp_proximal)
        else:
            self._solver.setup(sparse.triu(P, format="csc"), q, A, lower, upper, **defaults)
        labels, a, b, cl, cu = self._problem.layout()
        if method == "native" and len(self._problem.groups):
            with np.errstate(over="raise", under="raise", invalid="raise"):
                cl_native, cu_native = cl/self._row_scale, cu/self._row_scale
            self._solver._solver.setup_directional_peaks(len(self._problem.groups), labels, a, b, cl_native, cu_native)
        elif method == "row-split-gsp" and self._assembly['ngroups']:
            from .interface import PenaltyType
            self._solver.setup_penalty(
                np.where(self._assembly['groups'] >= 0, int(PenaltyType.OSQP_PENALTY_NORMINF), 0),
                alpha1=self._assembly['weights'], group_id=self._assembly['groups'])
        self.dimensions = dict(variables=len(q), rows=A.shape[0], nnz=P.nnz+A.nnz)
        self.setup_wall_seconds = time.perf_counter()-start
        self.update_wall_seconds = 0.0
        self.update_count = 0
        self.reset_wall_seconds = 0.0
        self.reset_refactorized = False

    @property
    def problem(self):
        return copy.deepcopy(self._problem)

    def update(self, *, q=None, lower=None, upper=None):
        start = time.perf_counter()
        old = self._problem
        next_q = old.q.copy() if q is None else _vector(q, len(old.q), "q")
        # Bounds may contain the correctly directed IEEE infinity.
        bounds = []
        for value, current, name in ((lower, old.lower, "lower"), (upper, old.upper, "upper")):
            a = current.copy() if value is None else np.asarray(value)
            if a.shape != current.shape or a.dtype.kind not in "fi" or np.isnan(a).any():
                raise ValueError(f"{name}: invalid bound vector")
            if not np.array_equal(np.isfinite(a), np.isfinite(current)):
                raise ValueError("finite-side pattern must stay fixed")
            bounds.append(a.astype(float, copy=True))
        lo, up = bounds
        if np.any(lo > up) or np.isposinf(lo).any() or np.isneginf(up).any():
            raise ValueError("invalid intervals")
        hard = old.layout()[0] < 0
        from .interface import constant
        rho_tol = constant("OSQP_RHO_TOL")
        if not np.array_equal(((up-lo) < rho_tol)[hard], ((old.upper-old.lower) < rho_tol)[hard]):
            raise ValueError("hard equality classification must stay fixed")
        candidate = replace(old, q=next_q, lower=lo, upper=up)
        if self._assembly is None:
            solver_q = next_q
            solver_lo, solver_up = _scaled_bounds(lo, up, self._row_scale)
        else:
            solver_q, solver_lo, solver_up = _vectors(candidate, self._assembly)
        _native_bounds(solver_lo, solver_up)
        if self.method in ("epigraph-daqp", "epigraph-daqp-prox"):
            flag = 0 if self._daqp_lp else self._daqp_model.update(
                f=np.ascontiguousarray(solver_q), blower=np.ascontiguousarray(solver_lo),
                bupper=np.ascontiguousarray(solver_up))
            if flag < 0:
                from .solvers import _setup_daqp
                self._daqp_model, self._daqp_setup = _setup_daqp(self._daqp_problem, self._defaults, proximal=self._daqp_proximal)
                raise ValueError(f"DAQP rejected vector update: {flag}; previous problem restored")
            self._daqp_problem = replace(self._daqp_problem, q=solver_q, lower=solver_lo, upper=solver_up)
        else:
            self._solver.update(q=solver_q, l=solver_lo, u=solver_up)
        self._problem = candidate
        self.update_count += 1
        self.update_wall_seconds = time.perf_counter()-start

    def reset(self):
        start = time.perf_counter()
        if self.method in ("epigraph-daqp", "epigraph-daqp-prox"):
            # DAQP's pinned API has no reset operation. Rebuilding is explicit;
            # never report it as retaining a factorization or hide its cost.
            from .solvers import _setup_daqp
            if not self._daqp_lp:
                self._daqp_model, self._daqp_setup = _setup_daqp(self._daqp_problem, self._defaults, proximal=self._daqp_proximal)
            self.reset_refactorized = not self._daqp_lp
        else:
            self._solver.warm_start(x=np.zeros(self.dimensions['variables']), y=np.zeros(self.dimensions['rows']))
        self.reset_wall_seconds = time.perf_counter()-start

    def solve(self):
        start = time.perf_counter()
        backend_settings = None
        if self.method in ("epigraph-daqp", "epigraph-daqp-prox"):
            from .solvers import _read_daqp
            if self._daqp_lp:
                result, result_x, result_y, backend_settings = _read_daqp_lp(self._daqp_problem, self._defaults)
            else:
                result, result_x, result_y, _, backend_settings = _read_daqp(self._daqp_problem, self._daqp_model, self._daqp_setup)
            result.x, result.y = result_x, result_y
        else:
            result = self._solver.solve(raise_error=False)
        from .solvers import DAQP_CANDIDATE_FLAGS
        good = result.info.status_val in (DAQP_CANDIDATE_FLAGS if self.method.startswith("epigraph-daqp") else (1, 2))
        x, y, metrics = None, None, None
        if good:
            x = result.x[:len(self._problem.q)]
            y = result.y*self._row_scale if self._assembly is None else self._assembly['recover'] @ result.y
            metrics = self._problem.evaluate(x, y)
        output = dict(method=self.method, status=result.info.status, status_code=int(result.info.status_val),
                    x=x.tolist() if good else None, y=y.tolist() if good else None, metrics=metrics,
                    dimensions=self.dimensions.copy(), iterations=int(result.info.iter),
                    setup_seconds=float(result.info.setup_time), solve_seconds=float(result.info.solve_time),
                    setup_wall_seconds=self.setup_wall_seconds, update_wall_seconds=self.update_wall_seconds,
                    elapsed_seconds=time.perf_counter()-start, update_count=self.update_count)
        output.update(reset_wall_seconds=self.reset_wall_seconds, reset_refactorized=self.reset_refactorized)
        if self.method == "native":
            output.update(row_normalization=self.row_normalization, row_scale=self._row_scale.tolist())
        if backend_settings is not None:
            output['backend_settings'] = backend_settings
            output['proximal_iterations'] = int(getattr(result.info, 'proximal_iterations', 0))
            output['one_shot_lp'] = self._daqp_lp
            output['workspace_reused'] = not self._daqp_lp
            output['inexact_candidate'] = result.info.status_val in (4, 5)
            output['relative_tolerance_mapped'] = False
        return output


def solve_peak(problem, method="native", **settings):
    """Cold whole operation including validation, assembly, solve and recovery."""
    start = time.perf_counter()
    result = PreparedPeakSolver(problem, method, **settings).solve()
    result['elapsed_seconds'] = time.perf_counter()-start
    return result
