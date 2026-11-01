from dataclasses import replace
import numpy as np
import pytest
from scipy import sparse
from softqp.flex import solve_flex
from softqp.problem import Problem, Penalty
from softqp.solvers import solve, UnsupportedProblemError


@pytest.mark.parametrize("target", [-3.0, 0.0, 3.0])
@pytest.mark.parametrize("weight", [0.2, 1.0, 4.0])
def test_flex_paper_reconstruction_matches_scalar_exact_penalty(target, weight):
    problem = Problem(
        sparse.eye(1, format="csc"),
        np.array([-target]),
        sparse.eye(1, format="csc"),
        np.array([-1.0]),
        np.array([1.0]),
        Penalty.uniform(1, linear=weight, quadratic=0),
    )
    result = solve_flex(problem, eps_abs=1e-9, eps_rel=1e-9)
    oracle = solve(problem, "lifted", eps_abs=1e-9, eps_rel=1e-9)
    assert result["status_code"] == 1
    assert result["x"] == pytest.approx(oracle["x"], abs=3e-6)
    assert result["metrics"]["objective"] == pytest.approx(
        oracle["metrics"]["objective"], abs=1e-7
    )
    assert result["backend_settings"]["complementarity_residual"] < 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_flex_weighted_sides_match_independent_lifting(seed):
    rng = np.random.default_rng(seed)
    n = 6
    m = 10
    a = rng.uniform(0.2, 1.3, (2, m))
    problem = Problem(
        sparse.eye(n, format="csc"),
        rng.normal(size=n) * 3,
        sparse.csc_matrix(rng.normal(size=(m, n))),
        np.full(m, -0.3),
        np.full(m, 0.5),
        Penalty(a[0], a[1], np.zeros(m), np.zeros(m), np.ones(m, bool)),
    )
    result = solve_flex(problem)
    oracle = solve(problem, "lifted", eps_abs=1e-9, eps_rel=1e-9)
    assert result["status_code"] == 1
    assert result["metrics"]["objective"] == pytest.approx(
        oracle["metrics"]["objective"], abs=2e-6
    )
    assert result["x"] == pytest.approx(oracle["x"], abs=3e-5)


def test_flex_refuses_changed_objective_and_hard_rows():
    from test_formulations import scalar

    with pytest.raises(UnsupportedProblemError, match="quadratic"):
        solve_flex(scalar())
    p = scalar()
    p = replace(p, penalty=Penalty.uniform(1, linear=1, quadratic=0, soft=False))
    with pytest.raises(UnsupportedProblemError, match="hard"):
        solve_flex(p)


def test_flex_does_not_manufacture_a_solution_on_iteration_limit():
    from test_formulations import scalar

    p = replace(scalar(), penalty=Penalty.uniform(1, linear=0.3, quadratic=0))
    result = solve_flex(p, max_iter=1)
    assert (
        result["status_code"] != 1 and result["x"] is None and result["metrics"] is None
    )
