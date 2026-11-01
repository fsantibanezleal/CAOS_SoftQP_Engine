import numpy as np
import pytest
from softqp.problem import Penalty
from softqp.solvers import solve
from test_formulations import scalar


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("target", [-3.0, 3.0])
def test_same_native_backend_lift_matches_independent_stock(asymmetric, target):
    penalty = Penalty(
        np.array([0.3]),
        np.array([1.2 if asymmetric else 0.3]),
        np.array([2.0]),
        np.array([4.0 if asymmetric else 2.0]),
        np.array([True]),
    )
    problem = scalar(target, penalty=penalty)
    stock = solve(problem, "lifted", reference_backend="stock")
    native = solve(problem, "lifted", reference_backend="native")
    assert native["status_code"] == stock["status_code"] == 1
    assert native["x"] == pytest.approx(stock["x"], abs=2e-6)
    assert native["metrics"]["objective"] == pytest.approx(
        stock["metrics"]["objective"], abs=1e-7
    )
    assert native["metrics"]["stationarity_residual"] < 1e-6
    assert native["dimensions"] == stock["dimensions"]


def test_reference_backend_is_not_silently_ignored():
    with pytest.raises(ValueError, match="reference_backend"):
        solve(scalar(), reference_backend="unknown")
