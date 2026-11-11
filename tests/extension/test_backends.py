"""Analytic, original-coordinate checks for the official raw adapters."""

import numpy as np
import pytest
from scipy import sparse

from softqp.comparators.backend import solve_candidate


BACKENDS = [
    ("clarabel", "clarabel", "Solved"),
    ("proxqp", "proxsuite", "QPSolverOutput.PROXQP_SOLVED"),
    ("piqp", "piqp", "Status.PIQP_SOLVED"),
]


@pytest.mark.parametrize("method, package, expected_status", BACKENDS)
@pytest.mark.parametrize("cost, expected_x, expected_dual", [
    (-3.0, 1.0, 2.0),
    (3.0, 0.75, -3.75),
])
def test_signed_original_rows(method, package, expected_status, cost, expected_x, expected_dual):
    pytest.importorskip(package)
    P = sparse.eye(2, format="csc")
    q = np.array([cost, -1.0])
    A = sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    lower, upper = np.array([-np.inf, 0.75, 0.25]), np.array([np.inf, 1.0, 0.25])
    actual = solve_candidate(method, P, q, A, lower, upper)
    assert actual.native_status == expected_status
    np.testing.assert_allclose(actual.x, [expected_x, 0.25], atol=3e-6, rtol=0)
    np.testing.assert_allclose(actual.y, [0.0, expected_dual, 0.75], atol=3e-6, rtol=0)
    np.testing.assert_allclose(P @ actual.x + q + A.T @ actual.y, 0, atol=3e-6, rtol=0)


@pytest.mark.parametrize("method, package, expected_status", BACKENDS)
def test_no_finite_constraints(method, package, expected_status):
    pytest.importorskip(package)
    result = solve_candidate(
        method, sparse.eye(1, format="csc"), np.array([-2.0]),
        sparse.csc_matrix((0, 1)), np.empty(0), np.empty(0),
    )
    assert result.native_status == expected_status
    np.testing.assert_allclose(result.x, [2], atol=1e-6, rtol=0)
    assert result.y.size == 0


@pytest.mark.parametrize("method, package, _", BACKENDS)
def test_infeasible_input_does_not_produce_a_candidate(method, package, _):
    pytest.importorskip(package)
    result = solve_candidate(
        method, sparse.eye(1, format="csc"), np.array([0.0]),
        sparse.csc_matrix([[1.0], [1.0]]),
        np.array([1.0, -np.inf]), np.array([np.inf, 0.0]),
    )
    expected_failures = {
        "clarabel": {"PrimalInfeasible"},
        "proxqp": {"QPSolverOutput.PROXQP_PRIMAL_INFEASIBLE", "QPSolverOutput.PROXQP_MAX_ITER_REACHED"},
        "piqp": {"Status.PIQP_PRIMAL_INFEASIBLE", "Status.PIQP_MAX_ITER_REACHED"},
    }
    assert result.native_status in expected_failures[method]
    assert result.x is None and result.y is None


@pytest.mark.parametrize("method", ("clarabel", "proxqp", "piqp"))
def test_adversarial_asymmetry_is_rejected_before_loading_backend(method, monkeypatch):
    def reject_import(*args):
        pytest.fail("invalid input reached native import")

    monkeypatch.setattr("softqp.comparators.backend._import_pinned", reject_import)
    with pytest.raises(ValueError, match="exactly symmetric"):
        solve_candidate(
            method, sparse.csc_matrix([[1.0, 1e-14], [0.0, 1.0]]), np.zeros(2),
            sparse.csc_matrix((0, 2)), np.empty(0), np.empty(0),
        )
