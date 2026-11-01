"""Native side-specific penalties must agree with an independent lifted QP."""

import numpy as np
from scipy import sparse
import pytest
from softqp.problem import Problem, Penalty
from softqp.solvers import solve


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("target", [-4.0, -0.2, 4.0])
def test_asymmetric_native_matches_lifting(scale, target):
    p = Problem(
        sparse.eye(1, format="csc"),
        np.array([-target]),
        sparse.csc_matrix([[3.0]]),
        np.array([-1.0]),
        np.array([2.0]),
        Penalty(
            np.array([0.4]),
            np.array([1.3]),
            np.array([0.8]),
            np.array([2.1]),
            np.array([True]),
        ),
    )
    a, b = [
        solve(p, method, scaling=10 if scale else 0)
        for method in ["asymmetric", "lifted"]
    ]
    assert a["status"] == b["status"] == "solved"
    assert a["x"] == pytest.approx(b["x"], abs=3e-6)
    assert a["metrics"]["objective"] == pytest.approx(
        b["metrics"]["objective"], abs=1e-7
    )
    assert a["metrics"]["stationarity_residual"] < 1e-5
    assert a["dimensions"]["variables"] == 1
    assert a["dimensions"]["rows"] == 1


@pytest.mark.parametrize("seed", range(8))
def test_random_mixed_rows(seed):
    rng = np.random.default_rng(seed)
    n, m = 7, 13
    A = sparse.csc_matrix(rng.normal(size=(m, n)))
    B = rng.normal(size=(n, n))
    soft = np.arange(m) % 3 != 0
    p = Problem(
        sparse.csc_matrix(B.T @ B + 0.2 * np.eye(n)),
        rng.normal(size=n) * 5.0,
        A,
        np.full(m, -0.3),
        np.full(m, 0.8),
        Penalty(*(rng.uniform(0.1, 2.0, m) for _ in range(4)), soft),
    )
    a, b = [solve(p, method) for method in ["asymmetric", "lifted"]]
    assert a["status"] == b["status"] == "solved"
    assert a["x"] == pytest.approx(b["x"], abs=2e-5)
    assert a["metrics"]["objective"] == pytest.approx(
        b["metrics"]["objective"], abs=1e-6
    )
    assert a["metrics"]["hard_violation"] < 2e-6


@pytest.mark.parametrize("quadratic", [0.0, 2.0])
@pytest.mark.parametrize(
    "bounds", [(-np.inf, 1.0), (-1.0, np.inf), (0.0, 0.0), (-np.inf, np.inf)]
)
def test_linear_equal_free_and_one_sided_rows(quadratic, bounds):
    p = Problem(
        sparse.eye(1, format="csc"),
        np.array([-3.0]),
        sparse.eye(1, format="csc"),
        np.array([bounds[0]]),
        np.array([bounds[1]]),
        Penalty(
            np.array([0.4]),
            np.array([0.8]),
            np.array([quadratic]),
            np.array([quadratic * 2]),
            np.array([True]),
        ),
    )
    a, b = [solve(p, method) for method in ["asymmetric", "lifted"]]
    assert a["status"] == b["status"] == "solved"
    assert a["x"] == pytest.approx(b["x"], abs=2e-5)
    assert a["metrics"]["objective"] == pytest.approx(
        b["metrics"]["objective"], abs=1e-6
    )
