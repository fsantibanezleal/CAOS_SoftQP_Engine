import numpy as np
from scipy import sparse
import pytest
from softqp.problem import Problem, Penalty
from softqp.solvers import solve


def scalar(target=3.0, lower=-1.0, upper=1.0, penalty=None):
    return Problem(
        sparse.eye(1, format="csc"),
        np.array([-target]),
        sparse.eye(1, format="csc"),
        np.array([lower]),
        np.array([upper]),
        penalty or Penalty.uniform(1, quadratic=2.0),
    )


@pytest.mark.parametrize("target", [-3.0, 0.0, 3.0])
def test_native_and_lifting_agree(target):
    a, b = [solve(scalar(target), method) for method in ("generalized", "lifted")]
    assert a["status"] == b["status"] == "solved"
    assert a["x"] == pytest.approx(b["x"], abs=2e-6)
    assert a["metrics"]["objective"] == pytest.approx(
        b["metrics"]["objective"], abs=1e-8
    )
    assert a["dimensions"]["variables"] == 1
    assert b["dimensions"]["variables"] == 2


def test_asymmetric_lifting_matches_analytic_solution():
    p = Penalty(
        np.array([1.0]),
        np.array([2.0]),
        np.array([2.0]),
        np.array([4.0]),
        np.array([True]),
    )
    # Upper branch: x-3 + 2 + 4(x-1)=0 => x=1. Lower cost does not enter.
    result = solve(scalar(penalty=p), "lifted")
    assert result["x"][0] == pytest.approx(1.0, abs=2e-6)
    with pytest.raises(ValueError, match="symmetric"):
        solve(scalar(penalty=p), "generalized")


@pytest.mark.parametrize(
    "mutate", [lambda p: np.array([float("nan")]), lambda p: np.array([True])]
)
def test_invalid_linear_objective_is_rejected(mutate):
    p = scalar()
    object.__setattr__(p, "q", mutate(p))
    with pytest.raises(ValueError, match="q"):
        p.validate()


def test_nonconvex_objective_is_rejected():
    p = scalar()
    object.__setattr__(p, "P", sparse.csc_matrix([[-1.0]]))
    with pytest.raises(ValueError, match="semidefinite"):
        p.validate()


def test_sparse_array_inputs_preserve_lifting_semantics():
    p = scalar()
    object.__setattr__(p, "A", sparse.csc_array(p.A))
    object.__setattr__(p, "P", sparse.csc_array(p.P))
    a, b = [solve(p, method) for method in ["asymmetric", "lifted"]]
    assert a["x"] == pytest.approx(b["x"], abs=2e-6)


@pytest.mark.parametrize("method", ["generalized", "asymmetric", "lifted"])
def test_unconstrained_convex_qp(method):
    p = Problem(
        sparse.eye(1, format="csc"),
        np.array([-3.0]),
        sparse.csc_matrix((0, 1)),
        np.array([]),
        np.array([]),
        Penalty.uniform(0),
    )
    assert solve(p, method)["x"] == pytest.approx([3.0], abs=1e-6)
