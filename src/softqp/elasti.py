"""Compatible original-space adapter for the published ElastiQP dual active-set solver."""

import time
import numpy as np
from .solvers import UnsupportedProblemError


def solve_elastiqp(problem, *, eps_abs=1e-7, eps_rel=1e-7, max_iter=100000):
    import elastiqp

    start = time.perf_counter()
    problem.validate()
    p = problem.penalty
    finite_l = np.isfinite(problem.lower)
    finite_u = np.isfinite(problem.upper)
    hard = ~p.soft & (finite_l | finite_u)
    equality = hard & (problem.lower == problem.upper)
    if np.any(hard & ~equality):
        raise UnsupportedProblemError(
            "ElastiQP retains hard equalities only; hard inequalities are not softened"
        )
    if np.any(p.soft & finite_l & (p.lower_quadratic != 0)) or np.any(
        p.soft & finite_u & (p.upper_quadratic != 0)
    ):
        raise UnsupportedProblemError(
            "ElastiQP adapter requires pure L1 soft costs; quadratic costs are not removed"
        )
    upper = p.soft & finite_u
    lower = p.soft & finite_l
    rows = np.r_[np.flatnonzero(upper), np.flatnonzero(lower)]
    signs = np.r_[np.ones(upper.sum()), -np.ones(lower.sum())]
    weights = np.r_[p.upper_linear[upper], p.lower_linear[lower]]
    if np.any(weights <= 0):
        raise UnsupportedProblemError("ElastiQP requires positive finite-side L1 costs")
    if problem.q.size > 2048 or rows.size > 8192:
        raise UnsupportedProblemError(
            "ElastiQP adapter exceeds its dense reference budget"
        )
    H = np.asfortranarray(problem.P.toarray(), dtype=np.float64)
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError as error:
        raise UnsupportedProblemError(
            "ElastiQP comparison requires positive definite P; no regularizer is inserted"
        ) from error
    G = np.asfortranarray(problem.A[rows].toarray() * signs[:, None])
    h = np.ascontiguousarray(
        np.r_[problem.upper[upper], -problem.lower[lower]], dtype=np.float64
    )
    A = np.asfortranarray(problem.A[equality].toarray())
    b = np.ascontiguousarray(problem.lower[equality], dtype=np.float64)
    settings = elastiqp.Settings("das")
    settings.eps_abs = eps_abs
    settings.eps_rel = eps_rel
    settings.max_iter = max_iter
    settings.eps_prox = 0.0
    settings.warm_start = False
    solver = elastiqp.Solver("das")
    solver.settings = settings
    solver.setup(
        H,
        np.ascontiguousarray(problem.q),
        G,
        h,
        np.ascontiguousarray(weights),
        A=A,
        b=b,
    )
    setup = time.perf_counter() - start
    clock = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - clock
    good = result.status == elastiqp.Status.Solved
    x = np.asarray(result.x)
    y = np.zeros(problem.lower.size)
    np.add.at(y, rows, signs * np.asarray(result.z))
    y[equality] = np.asarray(result.y)
    metrics = problem.evaluate(x) if good else None
    if metrics is not None:
        metrics["stationarity_residual"] = float(
            np.max(np.abs(problem.P @ x + problem.q + problem.A.T @ y), initial=0)
        )
    actual = {
        name: float(getattr(settings, name))
        for name in dir(settings)
        if not name.startswith("_")
        and isinstance(getattr(settings, name), (int, float, bool))
    }
    return dict(
        method="elastiqp",
        status=str(result.status),
        status_code=int(result.status.value),
        x=x.tolist() if good else None,
        y=y.tolist() if good else None,
        metrics=metrics,
        dimensions=dict(
            variables=problem.q.size + rows.size,
            rows=2 * rows.size + int(equality.sum()),
            nnz=int(
                problem.P.nnz
                + np.count_nonzero(G)
                + 2 * rows.size
                + np.count_nonzero(A)
            ),
        ),
        iterations=int(result.iters),
        setup_seconds=setup,
        solve_seconds=elapsed,
        elapsed_seconds=setup + elapsed,
        backend_settings=actual,
    )
