from dataclasses import replace
import numpy as np
import pytest
from softqp.sessions import PreparedSolver
from softqp.solvers import solve
from softqp.problem import Penalty, Problem
from scipy import sparse
from test_formulations import scalar


@pytest.mark.parametrize("method", ["asymmetric", "generalized", "lifted"])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_streamed_vectors_match_cold_independent_lifting(method, asymmetric):
    p = Penalty(
        np.array([0.3]),
        np.array([1.2 if asymmetric else 0.3]),
        np.array([2.0]),
        np.array([4.0 if asymmetric else 2.0]),
        np.array([True]),
    )
    problem = scalar(penalty=p)
    if method == "generalized" and asymmetric:
        with pytest.raises(ValueError, match="symmetric"):
            PreparedSolver(problem, method)
        return
    session = PreparedSolver(problem, method, adaptive_rho_interval=25)
    for target, bound in [(-3.0, 1.0), (-2.9, 0.9), (0.0, 0.8), (2.9, 0.9), (3.0, 1.0)]:
        q = np.array([-target])
        lower = np.array([-bound])
        upper = np.array([bound])
        session.update(q=q, lower=lower, upper=upper)
        result = session.solve()
        oracle = solve(replace(problem, q=q, lower=lower, upper=upper), "lifted")
        assert result["status_code"] == oracle["status_code"] == 1
        assert result["x"] == pytest.approx(oracle["x"], abs=2e-6)
        assert result["metrics"]["objective"] == pytest.approx(
            oracle["metrics"]["objective"], abs=1e-7
        )
        assert result["metrics"]["stationarity_residual"] < 1e-6
    assert session.update_count == 5
    assert session.setup_wall_seconds > 0 and session.update_wall_seconds > 0


@pytest.mark.parametrize("method", ["asymmetric", "lifted"])
def test_snapshot_isolation_and_rejected_update_preserve_workspace(method):
    problem = scalar()
    session = PreparedSolver(problem, method)
    problem.q[0] = 100
    assert session.solve()["x"][0] > 1
    snapshot = session.problem
    snapshot.q[0] = 200
    with pytest.raises(ValueError):
        session.update(q=[True])
    with pytest.raises(ValueError, match="finite-side"):
        session.update(lower=[-np.inf])
    assert session.update_count == 0 and session.solve()["x"][0] > 1
    session.reset()
    assert session.solve()["x"][0] > 1


@pytest.mark.parametrize("backend", ["stock", "native"])
def test_one_sided_lifted_stream_recovers_dual(backend):
    problem = scalar(lower=-np.inf)
    session = PreparedSolver(problem, "lifted", reference_backend=backend)
    session.update(q=[-4.0])
    result = session.solve()
    assert result["metrics"]["stationarity_residual"] < 1e-6
    assert result["y"][0] > 0


@pytest.mark.parametrize("method", ["asymmetric", "lifted"])
def test_updated_hard_row_remains_hard(method):
    problem = Problem(
        sparse.eye(1, format="csc"),
        np.array([-3.0]),
        sparse.eye(1, format="csc"),
        np.array([-1.0]),
        np.array([1.0]),
        Penalty.uniform(1, soft=False),
    )
    session = PreparedSolver(problem, method)
    session.update(lower=[0.25], upper=[0.25])
    result = session.solve()
    assert result["x"] == pytest.approx([0.25], abs=1e-7)
    assert result["metrics"]["hard_violation"] < 1e-7


@pytest.mark.parametrize("seed", range(4))
def test_daqp_active_state_updates_match_cold_independent_lifting(seed):
    from softqp.sessions import DaqpWorkspace

    rng = np.random.default_rng(seed)
    n = 5
    m = 9
    penalty = Penalty(
        np.full(m, 0.3),
        np.full(m, 1.2),
        np.full(m, 2.0),
        np.full(m, 4.0),
        np.r_[np.ones(m - 2, bool), np.zeros(2, bool)],
    )
    problem = Problem(
        sparse.eye(n, format="csc"),
        rng.normal(size=n),
        sparse.csc_matrix(rng.normal(size=(m, n))),
        np.full(m, -1.0),
        np.full(m, 1.0),
        penalty,
    )
    workspace = DaqpWorkspace(problem)
    for step in range(5):
        q = rng.normal(size=n) * 2
        lower = problem.lower + 0.1 * np.sin(step)
        upper = problem.upper + 0.1 * np.sin(step)
        workspace.update(q=q, lower=lower, upper=upper)
        result = workspace.solve()
        oracle = solve(replace(problem, q=q, lower=lower, upper=upper), "lifted")
        assert result["status_code"] in (1, 2)
        assert result["x"] == pytest.approx(oracle["x"], abs=3e-6)
        assert result["metrics"]["objective"] == pytest.approx(
            oracle["metrics"]["objective"], abs=1e-6
        )
        assert result["metrics"]["hard_violation"] < 1e-7
        assert result["metrics"]["stationarity_residual"] < 1e-6
    assert workspace.update_count == 5


def test_daqp_workspace_rejects_invalid_and_structural_edits():
    from softqp.sessions import DaqpWorkspace

    problem = scalar(penalty=Penalty.uniform(1, soft=False))
    workspace = DaqpWorkspace(problem)
    with pytest.raises(ValueError):
        workspace.update(q=[True])
    with pytest.raises(ValueError, match="equality"):
        workspace.update(lower=[0.2], upper=[0.2])
    assert workspace.update_count == 0
    assert workspace.solve()["x"] == pytest.approx([1.0], abs=1e-7)
