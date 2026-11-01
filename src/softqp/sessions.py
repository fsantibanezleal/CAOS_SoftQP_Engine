"""Reusable native QP workspaces with fixed matrices and penalty weights."""

import copy
from dataclasses import replace
import time
import numpy as np
from scipy import sparse
from .solvers import _setup_native, _setup_daqp, _read_daqp, UnsupportedProblemError
from .lifting import lift_problem


class PreparedSolver:
    """Reuse factorization and warm state for q/bound changes, with original-space recovery.

    P, A, soft policy, coefficients and finite-side pattern are fixed. Update timing
    includes Python validation and lifted-vector construction. No penalty update API
    or backend-specific mutable state is exposed.
    """

    def __init__(
        self, problem, method="asymmetric", *, reference_backend="native", **settings
    ):
        start = time.perf_counter()
        if method not in ("generalized", "asymmetric", "lifted"):
            raise UnsupportedProblemError(
                "prepared workspaces support generalized, asymmetric and lifted methods"
            )
        if reference_backend not in ("native", "stock"):
            raise ValueError("reference_backend must be native or stock")
        self._problem = copy.deepcopy(problem)
        self._problem.validate()
        self.method = method
        self.reference_backend = reference_backend
        defaults = dict(
            verbose=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=100000,
            polishing=False,
            warm_starting=True,
        )
        defaults.update(settings)
        self._lifted = None
        if method == "lifted":
            self._lifted = lift_problem(self._problem)
            if reference_backend == "native":
                from .interface import OSQP
            else:
                from osqp import OSQP
            self._solver = OSQP()
            data = self._lifted
            self._solver.setup(
                sparse.triu(data.P, format="csc"),
                data.q,
                data.A,
                data.lower,
                data.upper,
                **defaults,
            )
        else:
            self._solver = _setup_native(self._problem, method, defaults)
        self.setup_wall_seconds = time.perf_counter() - start
        self.update_wall_seconds = 0.0
        self.update_count = 0

    @property
    def problem(self):
        """An independent snapshot; mutating it cannot alter workspace metadata."""
        return copy.deepcopy(self._problem)

    def update(self, *, q=None, lower=None, upper=None):
        start = time.perf_counter()
        current = self._problem
        candidate = replace(
            current,
            q=current.q.copy() if q is None else np.array(q, copy=True),
            lower=current.lower.copy() if lower is None else np.array(lower, copy=True),
            upper=current.upper.copy() if upper is None else np.array(upper, copy=True),
        )
        candidate.validate()
        if not (
            np.array_equal(np.isfinite(candidate.lower), np.isfinite(current.lower))
            and np.array_equal(np.isfinite(candidate.upper), np.isfinite(current.upper))
        ):
            raise ValueError(
                "prepared workspace requires an unchanged finite-side pattern"
            )
        if self._lifted is not None:
            data = lift_problem(candidate)
            if (
                data.P.shape != self._lifted.P.shape
                or data.A.shape != self._lifted.A.shape
                or (data.P != self._lifted.P).nnz
                or (data.A != self._lifted.A).nnz
                or not np.array_equal(data.dual_rows, self._lifted.dual_rows)
            ):
                raise ValueError(
                    "bound edit changes the lifted matrix structure; prepare a new workspace"
                )
            self._solver.update(q=data.q, l=data.lower, u=data.upper)
            self._lifted = data
        else:
            self._solver.update(q=candidate.q, l=candidate.lower, u=candidate.upper)
        self._problem = candidate
        self.update_count += 1
        self.update_wall_seconds = time.perf_counter() - start

    def reset(self):
        """Reset primal and dual warm iterates while retaining the factorization."""
        n = self._lifted.q.size if self._lifted is not None else self._problem.q.size
        m = (
            self._lifted.lower.size
            if self._lifted is not None
            else self._problem.lower.size
        )
        self._solver.warm_start(x=np.zeros(n), y=np.zeros(m))

    def solve(self):
        start = time.perf_counter()
        result = self._solver.solve()
        p = self._problem
        if self._lifted is not None:
            x, y = self._lifted.recover(result.x, result.y)
            dimensions = self._lifted.dimensions
        else:
            x, y = result.x, result.y
            dimensions = dict(
                variables=p.q.size, rows=p.lower.size, nnz=p.P.nnz + p.A.nnz
            )
        metrics = p.evaluate(x) if result.info.status_val in (1, 2) else None
        if metrics is not None:
            metrics["stationarity_residual"] = float(
                np.max(np.abs(p.P @ x + p.q + p.A.T @ y), initial=0.0)
            )
        return dict(
            method=self.method,
            status=result.info.status,
            status_code=int(result.info.status_val),
            x=x.tolist() if metrics is not None else None,
            y=y.tolist() if metrics is not None else None,
            metrics=metrics,
            dimensions=dimensions,
            iterations=int(result.info.iter),
            setup_seconds=float(result.info.setup_time),
            solve_seconds=float(result.info.solve_time),
            elapsed_seconds=time.perf_counter() - start,
        )


class DaqpWorkspace:
    """Retain DAQP's transformed QP and active state for q/bound updates.

    Matrices, penalty weights, finite sides and hard equality classification are
    fixed. Bounds are passed in original units, and sense flags are not resent,
    so DAQP can reuse the previous active state as its documented update API does.
    """

    def __init__(self, problem, *, eps_abs=1e-7, eps_rel=1e-7, max_iter=100000):
        start = time.perf_counter()
        self._problem = copy.deepcopy(problem)
        self._problem.validate()
        self._defaults = dict(eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)
        self._model, self._setup_seconds = _setup_daqp(self._problem, self._defaults)
        self.setup_wall_seconds = time.perf_counter() - start
        self.update_wall_seconds = 0.0
        self.update_count = 0

    @property
    def problem(self):
        return copy.deepcopy(self._problem)

    def update(self, *, q=None, lower=None, upper=None):
        start = time.perf_counter()
        old = self._problem
        candidate = replace(
            old,
            q=old.q.copy() if q is None else np.array(q, copy=True),
            lower=old.lower.copy() if lower is None else np.array(lower, copy=True),
            upper=old.upper.copy() if upper is None else np.array(upper, copy=True),
        )
        candidate.validate()
        if not (
            np.array_equal(np.isfinite(old.lower), np.isfinite(candidate.lower))
            and np.array_equal(np.isfinite(old.upper), np.isfinite(candidate.upper))
        ):
            raise ValueError("DAQP workspace requires an unchanged finite-side pattern")
        if not np.array_equal(
            (old.lower == old.upper) & ~old.penalty.soft,
            (candidate.lower == candidate.upper) & ~old.penalty.soft,
        ):
            raise ValueError(
                "DAQP workspace requires unchanged hard equality classification"
            )
        flag = self._model.update(
            f=np.ascontiguousarray(candidate.q, dtype=np.float64),
            bupper=np.ascontiguousarray(candidate.upper, dtype=np.float64),
            blower=np.ascontiguousarray(candidate.lower, dtype=np.float64),
        )
        if flag < 0:
            self._model, self._setup_seconds = _setup_daqp(old, self._defaults)
            raise ValueError(
                f"DAQP rejected vector update: {flag}; previous problem restored"
            )
        self._problem = candidate
        self.update_count += 1
        self.update_wall_seconds = time.perf_counter() - start

    def solve(self):
        start = time.perf_counter()
        p = self._problem
        result, x, y, dimensions, settings = _read_daqp(
            p, self._model, self._setup_seconds
        )
        from .solvers import DAQP_CANDIDATE_FLAGS
        metrics = p.evaluate(x) if result.info.status_val in DAQP_CANDIDATE_FLAGS else None
        if metrics is not None:
            metrics["stationarity_residual"] = float(
                np.max(np.abs(p.P @ x + p.q + p.A.T @ y), initial=0)
            )
        return dict(
            method="daqp",
            status=result.info.status,
            status_code=int(result.info.status_val),
            x=x.tolist() if metrics is not None else None,
            y=y.tolist() if metrics is not None else None,
            metrics=metrics,
            dimensions=dimensions,
            iterations=int(result.info.iter),
            setup_seconds=float(result.info.setup_time),
            solve_seconds=float(result.info.solve_time),
            elapsed_seconds=time.perf_counter() - start,
            backend_settings=settings,
        )
