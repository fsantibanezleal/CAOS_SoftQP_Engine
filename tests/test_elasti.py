from dataclasses import replace
import numpy as np
import pytest
from scipy import sparse
from softqp.problem import Problem, Penalty
from softqp.elasti import solve_elastiqp
from softqp.solvers import solve, UnsupportedProblemError


@pytest.mark.parametrize("target", [-3.0, 3.0])
@pytest.mark.parametrize("scale", [0.2, 1.0, 4.0])
def test_elasti_side_weights_preserve_original_row_scale(target, scale):
    problem = Problem(
        sparse.eye(1, format="csc"),
        np.array([-target]),
        sparse.csc_matrix([[scale]]),
        np.array([-scale]),
        np.array([scale]),
        Penalty(
            np.array([0.3 / scale]),
            np.array([0.7 / scale]),
            np.zeros(1),
            np.zeros(1),
            np.ones(1, bool),
        ),
    )
    actual = solve_elastiqp(problem, eps_abs=1e-9, eps_rel=1e-9)
    oracle = solve(problem, "lifted", eps_abs=1e-9, eps_rel=1e-9)
    assert actual["status_code"] == 1
    assert actual["x"] == pytest.approx(oracle["x"], abs=3e-6)
    assert actual["metrics"]["objective"] == pytest.approx(
        oracle["metrics"]["objective"], abs=1e-7
    )
    assert actual["metrics"]["stationarity_residual"] < 1e-6


def test_elasti_preserves_a_hard_equality():
    problem = Problem(
        sparse.eye(2, format="csc"),
        np.array([-3.0, 2.0]),
        sparse.csc_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([-1.0, -1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        Penalty(
            np.array([0.3, 0.3, 0.0]),
            np.array([0.7, 0.7, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.array([True, True, False]),
        ),
    )
    actual = solve_elastiqp(problem)
    oracle = solve(problem, "lifted")
    assert actual["status_code"] == 1
    assert actual["x"] == pytest.approx(oracle["x"], abs=2e-6)
    assert actual["metrics"]["hard_violation"] < 1e-7
    assert actual["metrics"]["stationarity_residual"] < 1e-6


def test_elasti_refuses_quad_costs_and_hard_inequalities():
    from test_formulations import scalar

    with pytest.raises(UnsupportedProblemError, match="quadratic"):
        solve_elastiqp(scalar())
    problem = replace(
        scalar(), penalty=Penalty.uniform(1, linear=1, quadratic=0, soft=False)
    )
    with pytest.raises(UnsupportedProblemError, match="hard inequalities"):
        solve_elastiqp(problem)
