"""Full native generalized projection and independent explicit slack lifting."""

import time
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from .problem import Problem


DAQP_CANDIDATE_FLAGS = (1, 2, 4, 5)


class UnsupportedProblemError(ValueError):
    """The declared objective is outside a method's supported formulation."""


def _setup_native(problem, method, defaults):
    p = problem.penalty
    m = problem.lower.size
    if method == "generalized" and not (
        np.array_equal(p.lower_linear, p.upper_linear)
        and np.array_equal(p.lower_quadratic, p.upper_quadratic)
    ):
        raise UnsupportedProblemError(
            "generalized method requires symmetric penalties; use lifted for side-specific costs"
        )
    from .interface import OSQP, PenaltyType

    solver = OSQP()
    solver.setup(
        sparse.triu(problem.P, format="csc"),
        problem.q,
        problem.A,
        problem.lower,
        problem.upper,
        **defaults,
    )
    types = np.where(
        p.soft,
        int(PenaltyType.OSQP_PENALTY_L1L2),
        int(PenaltyType.OSQP_PENALTY_NONE),
    ).astype(np.int32)
    if m == 0:
        pass
    elif method == "asymmetric":
        solver._solver.setup_asymmetric_penalty(
            types,
            *[
                np.ascontiguousarray(a, dtype=np.float64)
                for a in (
                    p.lower_linear,
                    p.lower_quadratic,
                    p.upper_linear,
                    p.upper_quadratic,
                )
            ],
        )
    else:
        solver.setup_penalty(types, alpha1=p.lower_linear, alpha2=p.lower_quadratic)
    return solver


def solve(
    problem: Problem, method="generalized", *, reference_backend="stock", **settings
):
    if reference_backend not in ("stock", "native"):
        raise ValueError("reference_backend must be stock or native")
    problem.validate()
    start = time.perf_counter()
    defaults = dict(
        verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=100000, polishing=False
    )
    defaults.update(settings)
    n = problem.q.size
    m = problem.lower.size
    if method in ("generalized", "asymmetric"):
        solver = _setup_native(problem, method, defaults)
        result = solver.solve()
        x, y = result.x, result.y
        dimensions = dict(variables=n, rows=m, nnz=problem.P.nnz + problem.A.nnz)
    elif method == "lifted":
        if reference_backend == "stock":
            import osqp

            solver_class = osqp.OSQP
        else:
            from .interface import OSQP

            solver_class = OSQP

        from .lifting import lift_problem

        lifted = lift_problem(problem)
        solver = solver_class()
        solver.setup(
            sparse.triu(lifted.P, format="csc"),
            lifted.q,
            lifted.A,
            lifted.lower,
            lifted.upper,
            **defaults,
        )
        result = solver.solve()
        x, y = lifted.recover(result.x, result.y)
        dimensions = lifted.dimensions
    elif method == "daqp":
        result, x, y, dimensions, backend_settings = _solve_daqp(problem, defaults)
    else:
        raise ValueError(f"unknown method: {method}")
    elapsed = time.perf_counter() - start
    candidate_flags = DAQP_CANDIDATE_FLAGS if method == "daqp" else (1, 2)
    metrics = problem.evaluate(x) if result.info.status_val in candidate_flags else None
    if metrics is not None:
        metrics["stationarity_residual"] = float(
            np.max(np.abs(problem.P @ x + problem.q + problem.A.T @ y), initial=0.0)
        )
    output = dict(
        method=method,
        status=result.info.status,
        status_code=int(result.info.status_val),
        x=x.tolist() if metrics is not None else None,
        y=y.tolist() if metrics is not None else None,
        metrics=metrics,
        dimensions=dimensions,
        iterations=int(result.info.iter),
        setup_seconds=float(result.info.setup_time),
        solve_seconds=float(result.info.solve_time),
        elapsed_seconds=elapsed,
    )
    if method == "daqp":
        output["backend_settings"] = backend_settings
    return output


def _daqp_controls(defaults, *, proximal=False):
    for name in ("eps_abs", "eps_rel"):
        value = np.asarray(defaults[name])
        if value.shape or value.dtype.kind not in "fi" or not np.isfinite(value) or value < 0:
            raise ValueError(f"DAQP {name} must be finite and nonnegative")
    if defaults["eps_abs"] <= 0:
        raise ValueError("DAQP eps_abs must be strictly positive")
    limit = np.asarray(defaults["max_iter"])
    if limit.shape or limit.dtype.kind not in "iu" or limit <= 0 or limit > np.iinfo(np.intc).max:
        raise ValueError("DAQP max_iter must be a positive C integer")
    controls = dict(primal_tol=float(defaults["eps_abs"]), dual_tol=float(defaults["eps_abs"]),
                    iter_limit=int(limit), w_soft=0., eps_prox=0.)
    if proximal:
        eps, eta = defaults.get("eps_prox", -1e-6), defaults.get("eta_prox", -1.)
        for value in (eps, eta):
            candidate = np.asarray(value)
            if candidate.shape or candidate.dtype.kind not in "fi" or not np.isfinite(candidate):
                raise ValueError("proximal DAQP controls must be finite numeric scalars")
        if eps == 0:
            raise ValueError("proximal DAQP requires nonzero eps_prox")
        controls.update(eps_prox=float(eps), eta_prox=float(eta))
    return controls


def _setup_daqp(problem, defaults, *, proximal=False):
    p = problem.penalty
    n = problem.q.size
    m = problem.lower.size
    controls = _daqp_controls(defaults, proximal=proximal)
    import daqp

    if n > 2048 or m > 8192:
        raise UnsupportedProblemError(
            "DAQP dense reference exceeds its declared 2048-variable / 8192-row budget"
        )
    H = np.ascontiguousarray(problem.P.toarray(), dtype=np.float64)
    if proximal:
        if np.any(p.soft):
            raise UnsupportedProblemError("proximal DAQP adapter requires an explicit hard epigraph")
        eps_prox = defaults.get("eps_prox", -1e-6)
        eta_prox = defaults.get("eta_prox", -1.0)
        if not np.isfinite(eps_prox) or eps_prox == 0 or not np.isfinite(eta_prox):
            raise ValueError("proximal DAQP requires nonzero finite eps_prox and finite eta_prox")
    else:
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError as error:
            raise UnsupportedProblemError(
                "DAQP reference requires positive definite P; no regularization is inserted"
            ) from error
    if np.any(p.soft & np.isfinite(problem.lower) & (p.lower_quadratic <= 0)) or np.any(
        p.soft & np.isfinite(problem.upper) & (p.upper_quadratic <= 0)
    ):
        raise UnsupportedProblemError(
            "DAQP reference requires positive quadratic cost on each finite soft side"
        )
    model = daqp.Model()
    model.settings = controls
    sense = np.where(p.soft, 8, np.where(problem.lower == problem.upper, 5, 0)).astype(
        np.intc
    )
    setup_status, setup_seconds = model.setup(
        H,
        np.ascontiguousarray(problem.q, dtype=np.float64),
        np.ascontiguousarray(problem.A.toarray(), dtype=np.float64),
        np.ascontiguousarray(problem.upper, dtype=np.float64),
        np.ascontiguousarray(problem.lower, dtype=np.float64),
        sense,
    )
    if setup_status < 0:
        raise ValueError(f"DAQP setup failed: {setup_status}")
    if np.any(p.soft):
        rho_l = np.divide(
            1.0,
            p.lower_quadratic,
            out=np.zeros(m),
            where=p.soft & (p.lower_quadratic > 0),
        )
        rho_u = np.divide(
            1.0,
            p.upper_quadratic,
            out=np.zeros(m),
            where=p.soft & (p.upper_quadratic > 0),
        )
        model.soft_weights(
            rho_l=rho_l, rho_u=rho_u, w_l=p.lower_linear, w_u=p.upper_linear
        )
    return model, setup_seconds


def _solve_daqp(problem, defaults):
    model, setup_seconds = _setup_daqp(problem, defaults)
    return _read_daqp(problem, model, setup_seconds)


def _read_daqp(problem, model, setup_seconds):
    n = problem.q.size
    m = problem.lower.size
    x, _, flag, info = model.solve()
    y = np.asarray(info["lam"])
    result = SimpleNamespace(
        info=SimpleNamespace(
            status={1: "optimal", 2: "soft optimal", 4: "optimal inexact", 5: "feasible"}.get(flag, f"DAQP status {flag}"),
            status_val=flag,
            iter=info["iterations"],
            setup_time=setup_seconds,
            solve_time=info["solve_time"],
            proximal_iterations=int(info.get("nodes", 0)),
        )
    )
    dimensions = dict(variables=n, rows=m, nnz=problem.P.nnz + problem.A.nnz)
    return result, x, y, dimensions, model.settings
