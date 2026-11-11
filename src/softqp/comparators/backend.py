"""Normalize a verified hard interval QP and recover signed canonical duals.

This internal layer reports raw solver candidates. The independent original-space
evaluator and the registered oracle determine whether a candidate is accepted.
"""

from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Literal

import numpy as np
from scipy import sparse


Method = Literal["clarabel", "proxqp", "piqp"]
_DEPENDENCIES = {
    "clarabel": ("clarabel", "0.11.1"),
    "proxqp": ("proxsuite", "0.7.2"),
    "piqp": ("piqp", "0.6.4"),
}


@dataclass(frozen=True)
class Candidate:
    method: Method
    version: str
    native_status: str
    x: np.ndarray | None
    y: np.ndarray | None


def _import_pinned(method):
    name, version = _DEPENDENCIES[method]
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"{method} requires {name}=={version} in the admitted runtime") from error
    if installed != version:
        raise RuntimeError(f"{method} requires {name}=={version}; found {installed}")
    return import_module(name), version


def _validate(P, q, A, lower, upper):
    for name, matrix in (("P", P), ("A", A)):
        if not sparse.isspmatrix_csc(matrix) or matrix.dtype != np.float64:
            raise ValueError(f"{name} must be float64 CSC")
        matrix.check_format(full_check=True)
        if not matrix.has_canonical_format or not np.isfinite(matrix.data).all():
            raise ValueError(f"{name} must have sorted unique indices and finite coefficients")
    m, n = A.shape
    if n == 0 or P.shape != (n, n) or (P != P.T).nnz:
        raise ValueError("P must be exactly symmetric with the declared variable dimension")
    for name, vector, length in (("q", q, n), ("lower", lower, m), ("upper", upper, m)):
        if not isinstance(vector, np.ndarray) or vector.dtype != np.float64 or vector.shape != (length,):
            raise ValueError(f"{name} must be a float64 vector of length {length}")
    if (not np.isfinite(q).all() or np.isnan(lower).any() or np.isnan(upper).any()
            or np.isposinf(lower).any() or np.isneginf(upper).any() or (lower > upper).any()):
        raise ValueError("invalid linear cost or interval bounds")


def _clarabel(lib, P, q, A, lower, upper, equality, inequality, *, oracle):
    settings = lib.DefaultSettings()
    tolerance = 1e-10 if oracle else 1e-7
    settings.max_iter = 1000000 if oracle else 200000
    for field in ("tol_gap_abs", "tol_gap_rel", "tol_feas", "tol_infeas_abs", "tol_infeas_rel"):
        setattr(settings, field, tolerance)
    settings.verbose = False
    side_rows, side_signs, side_bounds = [], [], []
    for row in inequality:
        if np.isfinite(upper[row]):
            side_rows.append(row)
            side_signs.append(1.0)
            side_bounds.append(upper[row])
        if np.isfinite(lower[row]):
            side_rows.append(row)
            side_signs.append(-1.0)
            side_bounds.append(-lower[row])
    sides = sparse.diags(side_signs, shape=(len(side_rows), len(side_rows))) @ A[side_rows]
    C = sparse.vstack([A[equality], sides], format="csc")
    b = np.r_[lower[equality], side_bounds]
    cones = []
    if equality.size:
        cones.append(lib.ZeroConeT(equality.size))
    if side_rows:
        cones.append(lib.NonnegativeConeT(len(side_rows)))
    result = lib.DefaultSolver(sparse.triu(P, format="csc"), q, C, b, cones, settings).solve()
    status = str(result.status)
    if status != "Solved":
        return status, None, None
    z = np.asarray(result.z, dtype=np.float64)
    if z.shape != (C.shape[0],):
        raise ArithmeticError("Clarabel returned a dual vector with the wrong dimension")
    y = np.zeros(A.shape[0])
    y[equality] = z[:equality.size]
    np.add.at(y, side_rows, np.asarray(side_signs) * z[equality.size:])
    return status, result.x, y


def _interval(lib, method, P, q, A, lower, upper, equality, inequality, *, oracle):
    E, C = A[equality].tocsc(), A[inequality].tocsc()
    b, low, up = lower[equality], lower[inequality], upper[inequality]
    if method == "proxqp":
        qp = lib.proxqp
        solver = qp.sparse.QP(q.size, equality.size, inequality.size)
        settings = solver.settings
        settings.eps_abs = settings.eps_rel = settings.eps_duality_gap_abs = settings.eps_duality_gap_rel = 1e-7
        settings.max_iter = 200000
        settings.compute_timings = settings.check_duality_gap = True
        settings.initial_guess = qp.InitialGuess.NO_INITIAL_GUESS
        solver.init(
            P, q,
            E if equality.size else None, b if equality.size else None,
            C if inequality.size else None, low if inequality.size else None, up if inequality.size else None,
        )
        solver.solve()
        result = solver.results
        flag = result.info.status
        if flag != qp.QPSolverOutput.PROXQP_SOLVED:
            return str(flag), None, None
        signed = result.z
    else:
        solver = lib.SparseSolver()
        settings = solver.settings
        settings.eps_abs = settings.eps_rel = 1e-10 if oracle else 1e-7
        settings.max_iter = 1000000 if oracle else 200000
        settings.compute_timings = True
        settings.verbose = False
        solver.setup(P, q, E, b, C, low, up)
        flag = solver.solve()
        if flag != lib.Status.PIQP_SOLVED:
            return str(flag), None, None
        result = solver.result
        signed = result.z_u - result.z_l
    if np.shape(result.y) != equality.shape or np.shape(signed) != inequality.shape:
        raise ArithmeticError(f"{method} returned a dual vector with the wrong dimension")
    y = np.zeros(A.shape[0])
    y[equality] = result.y
    y[inequality] = signed
    return str(flag), result.x, y


def solve_candidate(method: Method, P, q, A, lower, upper, *, oracle=False) -> Candidate:
    """Solve an already certified lift using the exact backend and registered settings."""
    if method not in _DEPENDENCIES or (oracle and method == "proxqp"):
        raise ValueError("unknown comparator or unregistered oracle backend")
    _validate(P, q, A, lower, upper)
    lib, version = _import_pinned(method)
    bounded = np.isfinite(lower) | np.isfinite(upper)
    equal = bounded & (lower == upper)
    equality = np.flatnonzero(equal)
    inequality = np.flatnonzero(bounded & ~equal)
    if method == "clarabel":
        status, x, y = _clarabel(lib, P, q, A, lower, upper, equality, inequality, oracle=oracle)
    else:
        status, x, y = _interval(lib, method, P, q, A, lower, upper, equality, inequality, oracle=oracle)
    if x is not None:
        x = np.array(x, dtype=np.float64, copy=True)
        if x.shape != q.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ArithmeticError(f"{method} returned a nonfinite or incorrectly shaped candidate")
    return Candidate(method, version, status, x, y)
