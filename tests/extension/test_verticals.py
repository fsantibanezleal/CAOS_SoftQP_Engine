"""Public comparator API controls with analytic original-coordinate optima."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from softqp.comparators.backend import Candidate
from softqp.comparators.certificates import certify_recession
from softqp.peak import PeakGroup, PeakProblem, PreparedPeakSolver, solve_peak
from softqp.problem import Penalty, Problem
from softqp.sessions import PreparedSolver
from softqp.solvers import solve

from test_backends import BACKENDS


def hard_problem(curvature, coefficient=None):
    return Problem(
        sparse.csc_matrix([[curvature]]), np.array([-1.0]),
        sparse.csc_matrix((0, 1)) if coefficient is None else sparse.csc_matrix([[coefficient]]),
        np.empty(0) if coefficient is None else np.array([-np.inf]),
        np.empty(0) if coefficient is None else np.array([1.0]),
        Penalty.uniform(int(coefficient is not None), soft=False),
    )


@pytest.mark.parametrize("method, package, _", BACKENDS)
def test_original_units_signed_duals_and_updates(method, package, _):
    pytest.importorskip(package)
    p = Problem(
        sparse.csc_matrix([[0.01]]), np.array([-0.4]), sparse.csc_matrix([[1.0]]),
        np.array([10.0]), np.array([10.0]),
        Penalty(np.array([0.1]), np.array([0.2]), np.array([0.02]), np.array([0.03]), np.array([True])),
    )
    options = dict(variable_units=[10.0], row_units=[10.0])
    cold = solve(p, method, **options)
    prepared = PreparedSolver(p, method, **options)
    for result in (cold, prepared.solve()):
        assert result["status"] == "candidate"
        np.testing.assert_allclose(result["x"], [12.5], atol=4e-5)
        np.testing.assert_allclose(result["y"], [0.275], atol=2e-5)
        assert result["metrics"]["objective"] == pytest.approx(-3.625, abs=1e-7)
        assert result["unit_map"]["rows"] == [10.0]
        assert result["workspace_reused"] is False
    prepared.update(q=[0.3], lower=[20.0], upper=[20.0])
    result = prepared.solve()
    assert result["status"] == "candidate"
    np.testing.assert_allclose(result["y"], [-0.366666667], atol=2e-5)
    np.testing.assert_allclose(result["x"], [6.666666667], atol=2e-4)
    with pytest.raises(ValueError, match="finite-side pattern"):
        prepared.update(upper=[np.inf])
    prepared.reset()
    assert prepared.update_count == 1
    assert prepared.solve()["metrics"]["objective"] == pytest.approx(result["metrics"]["objective"])


@pytest.mark.parametrize("method, package, _", BACKENDS)
def test_peak_weighted_dual_and_epigraph(method, package, _):
    pytest.importorskip(package)
    p = PeakProblem(
        sparse.eye(1, format="csc"), np.array([-19.0]), sparse.eye(1, format="csc"),
        np.array([-np.inf]), np.array([0.0]), (PeakGroup([0], upper_scale=4.0, linear=0.5, quadratic=1.0),),
    )
    for result in (solve_peak(p, method), PreparedPeakSolver(p, method).solve()):
        assert result["status"] == "candidate"
        np.testing.assert_allclose(result["x"], [1.0], atol=1e-5)
        np.testing.assert_allclose(result["y"], [18.0], atol=1e-5)
        epigraph = result["metrics"]["epigraph"]
        np.testing.assert_allclose(epigraph["peak"], [4.0], atol=1e-5)
        np.testing.assert_allclose(epigraph["lambda_sides"], [4.5], atol=1e-5)


def test_tiny_dyadic_bounded_counterexamples(monkeypatch):
    status = "QPSolverOutput.PROXQP_DUAL_INFEASIBLE"
    monkeypatch.setattr("softqp.comparators.vertical.solve_candidate",
                        lambda *args: Candidate("proxqp", "0.7.2", status, None, None))
    assert certify_recession(hard_problem(0.0), [0], [1])
    assert solve(hard_problem(0.0), "proxqp")["status"] == "unbounded"
    for problem in (hard_problem(2.0**-34), hard_problem(0.0, 2.0**-34)):
        assert not certify_recession(problem, [0], [1])
        result = solve(problem, "proxqp")
        assert result["status"] == "inconclusive"
        assert result["native_status"] == status
        assert result["certificate"] is None
    assert not certify_recession(hard_problem(0.0), [0], [0])
    assert not certify_recession(hard_problem(0.0), [0], [-1])
    assert not certify_recession(hard_problem(0.0), [0], [np.inf])


@pytest.mark.parametrize("method, package, _", BACKENDS)
def test_real_unbounded_and_tiny_curvature(method, package, _):
    pytest.importorskip(package)
    exact = solve(hard_problem(0.0), method)
    assert exact["status"] == "unbounded", exact
    assert exact["certificate"] == {"anchor": [0.0], "direction": [1.0]}
    bounded = solve(hard_problem(2.0**-34), method)
    assert bounded["status"] != "unbounded", bounded


def test_false_success_is_rejected_in_original_coordinates(monkeypatch):
    p = replace(hard_problem(1.0, 1.0), upper=np.array([101.0]))
    monkeypatch.setattr("softqp.comparators.vertical.solve_candidate",
                        lambda *args: Candidate("proxqp", "0.7.2", "QPSolverOutput.PROXQP_SOLVED",
                                                np.array([101.000002]), np.array([-100.000002])))
    result = solve(p, "proxqp")
    assert result["status"] == "accuracy-rejected"
    assert result["metrics"]["H"] == pytest.approx(2e-6)


def test_registered_reference_runs_all_three_backends():
    for _, package, _ in BACKENDS:
        pytest.importorskip(package)
    pytest.importorskip("osqp")
    from softqp.comparators.oracle import reference_objective

    result = reference_objective(hard_problem(1.0))
    assert result["status"] == "oracle-agreed", result
    assert list(result["references"]) == ["stock-osqp", "clarabel", "piqp"]
    assert result["objective"] == pytest.approx(-0.5)
    assert all(value["passed"] for value in result["references"].values())


def test_exact_recession_rejects_indefinite_hessian_and_nonfeasible_anchor():
    p = Problem(sparse.csc_matrix([[0.0, 0.0], [0.0, -2.0**-45]]),
                np.array([-1.0, 0.0]), sparse.csc_matrix((0, 2)), np.empty(0), np.empty(0),
                Penalty.uniform(0, soft=False))
    assert not certify_recession(p, [0, 0], [1, 0])
    assert not certify_recession(hard_problem(0.0, 1.0), [1.0 + 2.0**-34], [-1.0])
    assert not certify_recession(replace(hard_problem(0.0, 1.0), q=np.array([1.0])),
                                 [1.0 + 2.0**-34], [-1.0])


def test_cached_csc_format_cannot_hide_duplicate_indices(monkeypatch):
    A = sparse.csc_matrix([[1.0], [2.0]])
    assert A.has_canonical_format
    A.indices[1] = 0
    p = replace(hard_problem(1.0), A=A, lower=np.array([-np.inf, -np.inf]),
                upper=np.array([1.0, 2.0]), penalty=Penalty.uniform(2, soft=False))

    def reject_import(*args):
        pytest.fail("malformed matrix reached a backend")

    monkeypatch.setattr("softqp.comparators.backend._import_pinned", reject_import)
    with pytest.raises(ValueError, match="sorted unique"):
        solve(p, "clarabel")


def test_repeated_invalid_vector_keeps_previous_snapshot():
    p = hard_problem(1.0, 1.0)
    workspace = PreparedSolver(p, "clarabel")
    with pytest.raises(ValueError, match="prepared q"):
        workspace.update(q=["0"])
    with pytest.raises(ValueError, match="equality classification"):
        workspace = PreparedSolver(replace(p, lower=np.array([0.0])), "clarabel")
        workspace.update(lower=[1.0])
    assert workspace.update_count == 0
    assert workspace.problem.lower.tolist() == [0.0]
    assert workspace.problem.q.tolist() == [-1.0]


@pytest.mark.parametrize("method, package", (("proxqp", "proxsuite"), ("piqp", "piqp")))
def test_lossy_integer_anchor_cannot_certify_native_unbounded(method, package):
    pytest.importorskip(package)
    lower = np.array([2.0**53])
    p = Problem(sparse.csc_matrix((2, 2)), np.array([0.0, -1.0]),
                sparse.csc_matrix([[1.0, 0.0]]), lower, lower, Penalty.uniform(1, soft=False))
    assert certify_recession(p, [2**53, 0], [0, 1])
    assert not certify_recession(p, [2**53 + 1, 0.0], [0, 1])
    wrong = solve(p, method, recession_certificate=([2**53 + 1, 0.0], [0, 1]))
    assert wrong["status"] == "inconclusive", wrong
    assert wrong["certificate"] is None
    with pytest.raises(ValueError, match="exactly representable"):
        PreparedSolver(p, method).update(lower=[2**53 + 1], upper=[2**53 + 1])
    with pytest.raises(ValueError, match="exactly representable"):
        solve(p, method, variable_units=[2**53 + 1, 1.0])


def test_exact_descent_still_requires_the_registered_numerical_screen(monkeypatch):
    p = replace(hard_problem(0.0), q=np.array([-2.0**-34]))
    status = "QPSolverOutput.PROXQP_DUAL_INFEASIBLE"
    monkeypatch.setattr("softqp.comparators.vertical.solve_candidate",
                        lambda *args: Candidate("proxqp", "0.7.2", status, None, None))
    assert certify_recession(p, [0.0], [1.0])
    assert solve(p, "proxqp")["status"] == "inconclusive"
