"""Independent scalar analytic oracles for the native binding."""
import numpy as np
from scipy import sparse
import pytest
from softqp import OSQP, PenaltyType

@pytest.mark.parametrize("target,bound,expected", [(3., 1., 5./3.), (-3., -1., -5./3.)])
def test_quadratic_penalty_matches_analytic_minimum(target, bound, expected):
    # min .5(x-target)^2 + (x-bound)^2 beyond the selected bound.
    solver = OSQP()
    lower, upper = (-np.inf, bound) if target > 0 else (bound, np.inf)
    solver.setup(sparse.eye(1, format="csc"), np.array([-target]),
                 sparse.eye(1, format="csc"), np.array([lower]), np.array([upper]),
                 verbose=False, eps_abs=1e-9, eps_rel=1e-9, polishing=False)
    solver.setup_penalty(PenaltyType.OSQP_PENALTY_L1L2, alpha1=0., alpha2=2.)
    result = solver.solve()
    assert result.info.status == "solved"
    assert result.x[0] == pytest.approx(expected, abs=1e-7)
    assert solver.slack(result.x)[0] == pytest.approx(bound-expected, abs=1e-7)

def test_hard_constraint_stays_hard():
    solver = OSQP()
    solver.setup(sparse.eye(1, format="csc"), np.array([-3.]),
                 sparse.eye(1, format="csc"), np.array([-np.inf]), np.array([1.]),
                 verbose=False, eps_abs=1e-9, eps_rel=1e-9)
    result = solver.solve()
    assert result.x[0] == pytest.approx(1., abs=1e-7)
