"""Exact recession proof for the original float64 hard QP."""

from fractions import Fraction

import numpy as np
from scipy import sparse

from .units import exact_float64_vector


def _fraction(value):
    return Fraction.from_float(float(value))


def _products(matrix, vector):
    rows = matrix.tocsr()
    for i in range(rows.shape[0]):
        start, end = rows.indptr[i:i + 2]
        yield sum(_fraction(rows.data[j]) * vector[rows.indices[j]] for j in range(start, end))


def _exact_psd(matrix):
    if (matrix != matrix.T).nnz or np.any(matrix.diagonal() < 0):
        return False
    off_diagonal = matrix - sparse.diags(matrix.diagonal())
    off_diagonal.eliminate_zeros()
    if not off_diagonal.nnz:
        return True
    count, labels = sparse.csgraph.connected_components(off_diagonal, directed=False)
    if np.max(np.bincount(labels), initial=0) > 32:
        return False
    for label in range(count):
        indices = np.flatnonzero(labels == label)
        if indices.size == 1:
            continue
        exact = [[_fraction(v) for v in row] for row in matrix[indices][:, indices].toarray()]
        for k, row in enumerate(exact):
            pivot = row[k]
            if pivot < 0 or (pivot == 0 and any(row[k + 1:])):
                return False
            if pivot:
                for i in range(k + 1, len(exact)):
                    for j in range(i, len(exact)):
                        exact[i][j] -= row[i] * row[j] / pivot
                        exact[j][i] = exact[i][j]
    return True


def certify_recession(problem, anchor, direction, *, variable_units=None):
    """Verify a normalized exact dyadic certificate. Return False if proof is unavailable."""
    from .backend import _validate

    try:
        _validate(problem.P, problem.q, problem.A, problem.lower, problem.upper)
        if hasattr(problem, "penalty"):
            problem.penalty.validate(problem.A.shape[0])
            if np.any(problem.penalty.soft):
                return False
        elif problem.groups:
            return False
        arrays = [
            exact_float64_vector(vector, problem.q.size, name)
            for vector, name in (
                (anchor, "anchor"), (direction, "direction"),
                (np.ones(problem.q.size) if variable_units is None else variable_units, "variable_units"),
            )
        ]
        x, d, units = ([_fraction(value) for value in vector] for vector in arrays)
        if min(units) <= 0 or max(abs(v / u) for v, u in zip(d, units)) != 1:
            return False
    except (ArithmeticError, ValueError, TypeError):
        return False

    if sum(_fraction(q) * v for q, v in zip(problem.q, d)) >= 0 or any(_products(problem.P, d)):
        return False
    for lo, up, value, ray in zip(
        problem.lower, problem.upper, _products(problem.A, x), _products(problem.A, d)
    ):
        if np.isfinite(lo) and (value < _fraction(lo) or ray < 0):
            return False
        if np.isfinite(up) and (value > _fraction(up) or ray > 0):
            return False
    return _exact_psd(problem.P)


def screens_recession(matrices, anchor, direction):
    """Apply the R4 numerical screens to already normalized float64 data."""
    P, q, A, lower, upper = matrices
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        activity = A @ anchor
        ray_activity = A @ direction
        curvature = P @ direction
        slope = float(q @ direction)
        matrix_norm = float(sparse.linalg.norm(P, ord=np.inf))
        if not all(np.isfinite(value).all() for value in (
            activity, ray_activity, curvature, slope, matrix_norm
        )):
            return False
        feasible = np.maximum(lower - activity, activity - upper)
        lower_error = -ray_activity[np.isfinite(lower)]
        upper_error = ray_activity[np.isfinite(upper)]
        return (
            np.max(feasible, initial=0.0) <= 1e-8
            and np.max(np.abs(curvature), initial=0.0) / max(1.0, matrix_norm) <= 1e-9
            and np.max(lower_error, initial=0.0) <= 1e-9
            and np.max(upper_error, initial=0.0) <= 1e-9
            and slope <= -1e-8
        )
