"""Scale a hard lift with declared physical reference magnitudes."""

import numpy as np
from scipy import sparse


def exact_float64_vector(value, length, name, *, allow_infinity=False):
    """Copy numerical inputs while refusing conversion that changes a supplied value."""
    raw = np.asarray(value, dtype=object)
    if raw.shape != (length,):
        raise ValueError(f"{name} requires {length} numeric entries")
    result = np.empty(length, dtype=np.float64)
    for index, item in enumerate(raw):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} requires real numeric entries")
        if isinstance(item, (int, np.integer)):
            item = int(item)
        elif not isinstance(item, (float, np.floating)):
            raise ValueError(f"{name} requires real numeric entries")
        try:
            converted = float(item)
        except OverflowError as error:
            raise ValueError(f"{name} exceeds the float64 range") from error
        if item != converted or (not allow_infinity and not np.isfinite(converted)):
            raise ValueError(f"{name} requires values exactly representable in float64")
        result[index] = converted
    return result


def _scale_nonzero(values, scales):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        answer = values * scales
    if np.any((values != 0) & (answer == 0)):
        raise ArithmeticError("unit conversion lost a nonzero coefficient")
    return answer


def _scaled_matrix(matrix, left, right, denominator):
    if not matrix.nnz:
        return matrix.copy()
    coo = matrix.tocoo()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        factors = (left[coo.row] * right[coo.col]) / denominator
    data = _scale_nonzero(coo.data, factors)
    return sparse.csc_matrix((data, (coo.row, coo.col)), shape=matrix.shape)


def scale_lift(matrices, dx, dr, sf):
    P, q, A, lo, up = matrices
    if sf == 1.0 and np.all(dx == 1.0) and np.all(dr == 1.0):
        return matrices
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        inverse_rows = 1.0 / dr
        P_hat = _scaled_matrix(P, dx, dx, sf)
        A_hat = _scaled_matrix(A, inverse_rows, dx, 1.0)
        q_hat = _scale_nonzero(q, dx / sf)
        lo_hat, up_hat = _scale_nonzero(lo, inverse_rows), _scale_nonzero(up, inverse_rows)
    if not np.array_equal(lo == up, lo_hat == up_hat):
        raise ArithmeticError("unit conversion changed interval equality classification")
    return P_hat, q_hat, A_hat, lo_hat, up_hat


def separable_scales(problem, lifted, dr):
    p = problem.penalty
    symmetric = (np.array_equal(p.lower_linear, p.upper_linear)
                 and np.array_equal(p.lower_quadratic, p.upper_quadratic))
    if symmetric:
        quadratic = p.soft & (p.lower_linear == 0)
        relaxed = p.soft & ~quadratic
        aux_units = np.r_[dr[quadratic], dr[relaxed]]
        additional_rows = dr[relaxed]
    else:
        finite_counts = np.isfinite(problem.lower).astype(int) + np.isfinite(problem.upper)
        aux_units = np.repeat(dr[p.soft], finite_counts[p.soft])
        additional_rows = aux_units
    row_units = np.r_[dr[lifted.dual_rows[lifted.dual_rows >= 0]], additional_rows]
    return aux_units, row_units
