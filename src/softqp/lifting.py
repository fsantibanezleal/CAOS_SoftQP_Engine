"""Compact explicit QP lifting and original-space dual recovery."""

from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass
class LiftedProblem:
    P: object
    q: np.ndarray
    A: object
    lower: np.ndarray
    upper: np.ndarray
    dual_rows: np.ndarray
    original_n: int
    original_m: int

    @property
    def dimensions(self):
        return dict(
            variables=self.q.size, rows=self.A.shape[0], nnz=self.P.nnz + self.A.nnz
        )

    def recover(self, x, y):
        original_y = np.zeros(self.original_m)
        valid = self.dual_rows >= 0
        np.add.at(original_y, self.dual_rows[valid], y[valid])
        return x[: self.original_n], original_y


def lift_problem(problem):
    p = problem.penalty
    m = problem.lower.size
    if not (
        np.array_equal(p.lower_linear, p.upper_linear)
        and np.array_equal(p.lower_quadratic, p.upper_quadratic)
    ):
        return _asymmetric(problem)
    from .reference import Penalty as ReferencePenalty, lift, L1L2

    reference = ReferencePenalty.hard(m)
    reference.type[p.soft] = L1L2
    reference.alpha1[:] = p.lower_linear
    reference.alpha2[:] = p.lower_quadratic
    P, q, A, lower, upper = lift(
        problem.P,
        problem.q,
        sparse.csc_matrix(problem.A),
        problem.lower,
        problem.upper,
        reference,
    )
    mapping = np.full(A.shape[0], -1, dtype=np.int64)
    mapping[:m] = np.arange(m)
    extra = np.flatnonzero(
        p.soft
        & (p.lower_linear != 0)
        & (np.isfinite(problem.upper) | ~np.isfinite(problem.lower))
        & np.isfinite(problem.lower)
    )
    mapping[m : m + extra.size] = extra
    return LiftedProblem(P, q, A, lower, upper, mapping, problem.q.size, m)


def _asymmetric(problem):
    p = problem.penalty
    n = problem.q.size
    m = problem.lower.size
    A = sparse.csc_matrix(problem.A)
    rows, columns, values = [], [], []
    linear, quadratic, lower, upper, original_rows = [], [], [], [], []
    # Two nonnegative slack variables for a two-sided soft row; finite sides only.
    for i in range(m):
        if not p.soft[i]:
            out = len(lower)
            a = A.getrow(i)
            rows.extend([out] * a.nnz)
            columns.extend(a.indices)
            values.extend(a.data)
            lower.append(problem.lower[i])
            upper.append(problem.upper[i])
            original_rows.append(i)
            continue
        for sign, bound, c, d in (
            (1.0, problem.lower[i], p.lower_linear[i], p.lower_quadratic[i]),
            (-1.0, problem.upper[i], p.upper_linear[i], p.upper_quadratic[i]),
        ):
            if not np.isfinite(bound):
                continue
            j = len(linear)
            out = len(lower)
            a = A.getrow(i)
            rows.extend([out] * a.nnz + [out])
            columns.extend(list(a.indices) + [n + j])
            values.extend(list(a.data) + [sign])
            lower.append(bound if sign > 0 else -np.inf)
            upper.append(np.inf if sign > 0 else bound)
            original_rows.append(i)
            linear.append(c)
            quadratic.append(d)
    k = len(linear)
    B = sparse.coo_matrix((values, (rows, columns)), shape=(len(lower), n + k)).tocsc()
    B = sparse.vstack(
        [B, sparse.hstack([sparse.csc_matrix((k, n)), sparse.eye(k)])], format="csc"
    )
    P = sparse.block_diag(
        [problem.P, sparse.diags(quadratic) if k else sparse.csc_matrix((0, 0))],
        format="csc",
    )
    mapping = np.r_[original_rows, np.full(k, -1)].astype(np.int64)
    return LiftedProblem(
        P,
        np.r_[problem.q, linear],
        B,
        np.r_[lower, np.zeros(k)],
        np.r_[upper, np.full(k, np.inf)],
        mapping,
        n,
        m,
    )
