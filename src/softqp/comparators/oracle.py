"""Independent fixed three-library reference for one prospectively registered QP."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .backend import solve_candidate
from .vertical import ComparatorSession, _SUCCESS


REFERENCE_LIMITS = dict(H=1e-8, K=1e-8, F=1e-8, F_all=1e-8, D=1e-9)


def _stock(data):
    from importlib.metadata import version
    import osqp

    if version("osqp") != "1.1.3":
        raise ImportError("oracle requires osqp==1.1.3")
    P, q, A, lower, upper = data
    if np.any(np.abs(np.r_[lower[np.isfinite(lower)], upper[np.isfinite(upper)]]) >= 1e25):
        raise ValueError("oracle OSQP finite bound exceeds sentinel-safe range")
    solver = osqp.OSQP(algebra="builtin")
    solver.setup(
        P=sparse.triu(P, format="csc"), q=q, A=A, l=lower, u=upper,
        eps_abs=1e-10, eps_rel=1e-10, max_iter=1000000, alpha=1.6, rho=0.1,
        rho_is_vec=False, sigma=1e-6, scaling=10, adaptive_rho=True,
        adaptive_rho_interval=25, polishing=True, check_termination=1, verbose=False,
        warm_starting=False,
    )
    result = solver.solve(raise_error=False)
    good = int(result.info.status_val) == 1
    return SimpleNamespace(
        native_status=result.info.status, x=result.x if good else None,
        y=result.y if good else None, version="1.1.3",
    )


def reference_objective(problem, **units):
    """Run all three references and return their independent metrics and agreement.

    The registered product runner supervises whole-process time and memory limits.
    No candidate evaluates or supplies any reference objective.
    """
    session = ComparatorSession(problem, "clarabel", **units)
    normalized = session.data[3]
    detail = {}
    for method in ("stock-osqp", "clarabel", "piqp"):
        try:
            result = _stock(normalized) if method == "stock-osqp" else solve_candidate(
                method, *normalized, oracle=True
            )
            native_success = result.x is not None and (
                method == "stock-osqp" or result.native_status == _SUCCESS[method]
            )
            row = dict(native_status=result.native_status, backend_version=result.version, passed=False)
            if native_success:
                x, y, metrics = session.recover_metrics(result)
                row.update(metrics=metrics, x=x.tolist(), y=y.tolist(), passed=(
                    all(metrics[key] <= limit for key, limit in REFERENCE_LIMITS.items())
                    and metrics.get("epigraph", {}).get("accepted", True)
                ))
            detail[method] = row
        except (ArithmeticError, ImportError, RuntimeError, ValueError) as error:
            detail[method] = dict(passed=False, reason=str(error))
    output = dict(status="oracle-failed", objective=None, spread=None, references=detail)
    if all(row["passed"] for row in detail.values()):
        costs = [row["metrics"]["objective"] for row in detail.values()]
        scale = max(session.sf, max(abs(f) for f in costs))
        spread = max(costs) / scale - min(costs) / scale
        output["spread"] = spread
        if spread <= 1e-8:
            output.update(status="oracle-agreed", objective=float(np.median(costs)))
    return output
