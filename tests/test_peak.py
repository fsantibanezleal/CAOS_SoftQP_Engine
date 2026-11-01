"""Independent epigraph and existing row-split GSP refute the new native path."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from softqp.peak import PeakGroup, PeakProblem, PreparedPeakSolver, solve_peak
from softqp.solvers import UnsupportedProblemError


def sample(seed=0, quadratic=0):
    rng = np.random.default_rng(seed)
    n, m = 5, 9
    B = rng.normal(size=(n, n))
    A = np.vstack([np.eye(n)[:2], rng.normal(size=(m-2, n))])
    groups = tuple(PeakGroup(np.array(rows), rng.uniform(.4, 2, len(rows)),
                             rng.uniform(.4, 2, len(rows)), linear=weight, quadratic=quadratic)
                   for rows, weight in (([2, 3, 4], .7), ([5, 6, 7, 8], 1.3)))
    return PeakProblem(sparse.csc_matrix(B.T @ B + .5*np.eye(n)), rng.normal(size=n)*4,
                       sparse.csc_matrix(A), np.r_[[-2., -2.], np.full(7, -.3)],
                       np.r_[[2., 2.], np.full(7, .4)], groups)


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('quadratic', [0., .8])
def test_native_matches_independent_compact_epigraph_and_existing_gsp(seed, quadratic):
    problem = sample(seed, quadratic)
    native = solve_peak(problem)
    oracle = solve_peak(problem, 'epigraph')
    methods = [native, oracle]
    if quadratic == 0:
        methods.append(solve_peak(problem, 'row-split-gsp'))
    for result in methods:
        assert result['status'] == 'solved'
        assert result['x'] == pytest.approx(oracle['x'], abs=3e-6)
        assert result['metrics']['objective'] == pytest.approx(oracle['metrics']['objective'], abs=3e-6)
        assert result['metrics']['hard_violation'] <= 1e-6
        assert result['metrics']['normalized_stationarity'] <= 1e-6
        assert result['metrics']['normalized_fenchel_residual'] <= 1e-6
        assert result['metrics']['normalized_dual_domain_excess'] <= 1e-6
    assert native['dimensions']['variables'] == 5 and native['dimensions']['rows'] == 9
    assert oracle['dimensions']['variables'] == 7
    if quadratic == 0:
        assert methods[-1]['dimensions']['rows'] == 16


@pytest.mark.parametrize('method', ['native', 'epigraph', 'row-split-gsp'])
def test_vector_stream_matches_independent_cold_oracle_and_rejects_partial_invalid_updates(method):
    problem = sample(9)
    workspace = PreparedPeakSolver(problem, method)
    for step in range(5):
        current = replace(problem, q=problem.q+step*.2, lower=problem.lower-step*.01,
                          upper=problem.upper+step*.03)
        workspace.update(q=current.q, lower=current.lower, upper=current.upper)
        actual = workspace.solve()
        expected = solve_peak(current, 'epigraph')
        assert actual['status'] == 'solved'
        assert actual['x'] == pytest.approx(expected['x'], abs=3e-6)
        assert actual['update_count'] == step+1
    before = workspace.problem
    for kwargs in ({'q': np.full(5, np.nan)}, {'q': np.zeros(4)},
                   {'q': np.zeros(5), 'lower': before.upper+1},
                   {'lower': np.full(9, -np.inf)}, {'upper': np.full(9, 1e26)}):
        with pytest.raises((ValueError, UnsupportedProblemError)):
            workspace.update(**kwargs)
        assert np.array_equal(workspace.problem.q, before.q)
        assert np.array_equal(workspace.problem.lower, before.lower)
    workspace.reset()
    assert workspace.solve()['x'] == pytest.approx(expected['x'], abs=3e-6)
    snapshot = workspace.problem
    snapshot.q[:] = 1234
    assert np.array_equal(workspace.problem.q, before.q)


@pytest.mark.parametrize('method', ['native', 'epigraph', 'row-split-gsp'])
def test_zero_cost_group_is_free_not_hard(method):
    problem = PeakProblem(sparse.eye(2, format='csc'), np.array([-3., 2.]),
                          sparse.eye(2, format='csc'), np.zeros(2), np.zeros(2),
                          (PeakGroup([0, 1], linear=0, quadratic=0),))
    result = solve_peak(problem, method)
    assert result['status'] == 'solved'
    assert result['x'] == pytest.approx([3, -2], abs=1e-6)
    assert result['metrics']['penalty'] == 0
    assert result['metrics']['normalized_fenchel_residual'] <= 1e-7


def test_coupled_dual_budget_and_directional_scales_do_not_become_separable():
    problem = PeakProblem(sparse.eye(2, format='csc'), np.zeros(2), sparse.eye(2, format='csc'),
                          np.zeros(2), np.zeros(2), (PeakGroup([0, 1]),))
    bad = problem.evaluate(np.zeros(2), np.array([.75, .75]))
    assert bad['normalized_dual_domain_excess'] > 0
    assert bad['approximate_fenchel_residual'] is None
    with pytest.raises(ValueError):
        replace(problem, groups=(PeakGroup([0, 1]), PeakGroup([1]))).validate()
    for group in (PeakGroup([0.5]), PeakGroup([0, 0]), PeakGroup([2]), PeakGroup([]),
                  PeakGroup([0], lower_scale=0), PeakGroup([0], upper_scale=np.inf),
                  PeakGroup([0], linear=-1), PeakGroup([0], quadratic=np.nan)):
        with pytest.raises(ValueError):
            replace(problem, groups=(group,)).validate()


@pytest.mark.parametrize('settings', [{'alpha': 1.6}, {'scaling': 10},
                                    {'adaptive_rho': True}, {'polishing': True}])
def test_restricted_settings_are_rejected(settings):
    with pytest.raises(UnsupportedProblemError):
        PreparedPeakSolver(sample(), **settings)


def test_squared_peak_rejects_unsupported_published_native_comparator():
    with pytest.raises(UnsupportedProblemError, match='squared'):
        solve_peak(sample(quadratic=1), 'row-split-gsp')


@pytest.mark.parametrize('linear,quadratic,expected_status', [(2., 0., 'dual infeasible'),
                                                           (2., 1., 'solved'),
                                                           (0., 0., 'dual infeasible')])
def test_recession_uses_group_maximum_and_quadratic_recession(linear, quadratic, expected_status):
    problem = PeakProblem(sparse.csc_matrix((1, 1)), np.array([-3.]),
                          sparse.csc_matrix([[1.], [1.]]), -np.ones(2), np.ones(2),
                          (PeakGroup([0, 1], linear=linear, quadratic=quadratic),))
    result = solve_peak(problem)
    assert result['status'] == expected_status
    if quadratic:
        assert result['x'] == pytest.approx([2.], abs=2e-6)
    else:
        assert result['x'] is None and result['metrics'] is None


def test_infeasible_hard_rows_cannot_be_rescued_by_an_unrelated_soft_group():
    problem = PeakProblem(sparse.eye(1, format='csc'), np.array([0.]),
                          sparse.csc_matrix([[1.], [1.], [1.]]),
                          np.array([1., -np.inf, -1.]), np.array([np.inf, 0., 1.]),
                          (PeakGroup([2]),))
    for method in ('native', 'epigraph', 'row-split-gsp'):
        result = solve_peak(problem, method)
        assert result['status'] == 'primal infeasible'
        assert result['x'] is None and result['metrics'] is None


def test_maximal_float_spacing_does_not_accept_the_wrong_active_prefix():
    from softqp import ext_builtin
    largest = np.finfo(float).max
    rho = np.full(2, 1e-308)
    w, _ = ext_builtin.directional_peak_prox(np.full(2, largest), np.ones(2), np.ones(2), rho, 1., 0.)
    expected_ratio = 1 - 1/(2*1e-308*largest)
    assert w/largest == pytest.approx([expected_ratio]*2, rel=1e-14)


@pytest.mark.parametrize('unit', [1e-150, 1e-14, 1., 1e14, 1e150])
def test_native_kernel_unit_rescaling_counterexample(unit):
    from softqp import ext_builtin
    w, _ = ext_builtin.directional_peak_prox(np.array([2., 1.]), np.full(2, unit),
                                            np.full(2, unit), np.ones(2), 2.5/unit, 0.)
    assert w == pytest.approx([.25, .25], abs=1e-13)


def test_settings_snapshot_cache_and_rejection_are_consistent():
    workspace = PreparedPeakSolver(sample())
    solver = workspace._solver
    snapshot = solver._solver.get_settings()
    snapshot.alpha = 1.6
    assert solver._solver.get_settings().alpha == 1
    for request in ({'alpha': 1.6}, {'rho': 1.}, {'alpha': 1.6, 'rho': .5},
                    {'polishing': True}, {'scaling': 10}, {'adaptive_rho': True}):
        with pytest.raises(Exception):
            solver.update_settings(**request)
        assert solver.settings.alpha == solver._solver.get_settings().alpha == 1
        assert solver.settings.rho == solver._solver.get_settings().rho == .1
    solver.settings.alpha = 1.6  # A corrupted Python cache cannot poison a valid update.
    solver.update_settings(eps_abs=1e-9)
    assert solver.settings.alpha == 1
    assert workspace.solve()['status'] == 'solved'


def test_near_equality_rho_reclassification_is_rejected_before_native_update():
    problem = PeakProblem(sparse.eye(1, format='csc'), np.array([-2.]),
                          sparse.csc_matrix([[1.], [1.]]), np.zeros(2), np.ones(2),
                          (PeakGroup([1]),))
    workspace = PreparedPeakSolver(problem)
    with pytest.raises(ValueError, match='classification'):
        workspace.update(upper=np.array([1e-6, 1.]))
    assert workspace.problem.upper == pytest.approx([1., 1.])


@pytest.mark.parametrize('row_scale', [1e-12, 1e-8, 1e-4])
def test_small_quadratic_recession_is_not_false_unboundedness(row_scale):
    problem = PeakProblem(sparse.csc_matrix((1, 1)), np.array([-1.]),
                          sparse.csc_matrix([[row_scale]]), np.zeros(1), np.zeros(1),
                          (PeakGroup([0], 1/row_scale, 1/row_scale, linear=0, quadratic=1),))
    # The independent epigraph normalizes to min .5*x^2-x, optimum x=1.
    oracle = solve_peak(problem, 'epigraph')
    assert oracle['status'] == 'solved' and oracle['x'] == pytest.approx([1.], abs=1e-6)
    result = solve_peak(problem, row_normalization=False, max_iter=50, check_termination=1)
    assert 'infeasible' not in result['status']
    # Fixed unscaled ADMM may be slow here. An iteration limit is unresolved,
    # not permission to return a false unboundedness certificate.
    if result['status'] != 'solved':
        assert result['x'] is None and result['metrics'] is None
    normalized = solve_peak(problem, max_iter=2000)
    assert normalized['status'] == 'solved'
    assert normalized['x'] == pytest.approx([1.], abs=1e-6)
    assert normalized['metrics']['objective'] == pytest.approx(-.5, abs=1e-7)
    assert normalized['metrics']['normalized_stationarity'] <= 1e-6


def test_roundoff_dual_repair_does_not_hide_large_original_space_stationarity():
    problem = PeakProblem(sparse.csc_matrix((1, 1)), np.array([1.]),
                          sparse.csc_matrix([[1e16]]), np.array([-np.inf]), np.array([np.inf]),
                          (PeakGroup([0], linear=0, quadratic=0),))
    result = problem.evaluate([0.], [-1e-16])
    assert result['exact_dual_domain'] is False
    assert result['raw_normalized_stationarity'] == 0
    assert result['repaired_normalized_stationarity'] == 1
    assert result['normalized_stationarity'] == 1
    assert result['normalized_fenchel_residual'] == 0
    assert result['normalized_dual_repair'] < 1e-7


def test_tiny_explicit_negative_curvature_is_not_silently_regularized():
    problem = PeakProblem(sparse.csc_matrix([[-1e-12]]), np.array([0.]),
                          sparse.csc_matrix([[1.]]), np.array([-1.]), np.array([1.]),
                          (PeakGroup([0]),))
    with pytest.raises(ValueError, match='semidefinite'):
        problem.validate()


@pytest.mark.parametrize('seed', range(3))
def test_ordinary_epigraph_daqp_matches_stock_oracle_and_retains_vector_workspace(seed):
    problem = sample(seed, quadratic=.8)
    workspace = PreparedPeakSolver(problem, 'epigraph-daqp')
    for step in range(4):
        updated = replace(problem, q=problem.q + step*.3)
        workspace.update(q=updated.q)
        actual, expected = workspace.solve(), solve_peak(updated, 'epigraph')
        assert actual['status_code'] == 1
        assert actual['x'] == pytest.approx(expected['x'], abs=2e-6)
        assert actual['metrics']['objective'] == pytest.approx(expected['metrics']['objective'], abs=2e-6)
        assert actual['metrics']['normalized_stationarity'] <= 1e-6
        assert actual['metrics']['normalized_fenchel_residual'] <= 1e-6
        assert actual['update_count'] == step+1
    workspace.reset()
    assert workspace.solve()['x'] == pytest.approx(expected['x'], abs=2e-6)
    assert workspace.solve()['reset_refactorized'] is True
    assert workspace.reset_wall_seconds > 0
    with pytest.raises(UnsupportedProblemError, match='positive definite'):
        PreparedPeakSolver(sample(seed, quadratic=0), 'epigraph-daqp')


def test_numerical_failure_carries_no_finite_solution_or_certificate_and_recovers():
    problem = sample()
    workspace = PreparedPeakSolver(problem)
    assert workspace.solve()['status'] == 'solved'
    solver = workspace._solver
    solver.warm_start(x=np.full(5, np.nan))
    result = solver.solve(raise_error=False)
    assert result.info.status == 'numerical error'
    for value in (result.x, result.y, result.prim_inf_cert, result.dual_inf_cert):
        assert np.isnan(value).all()
    for field in ('obj_val', 'penalty_val', 'dual_obj_val', 'duality_gap', 'prim_res', 'dual_res'):
        assert np.isnan(getattr(result.info, field))
    workspace.reset()
    assert workspace.solve()['status'] == 'solved'


def test_penalty_row_normalization_is_covariant_in_original_physical_units_and_updates():
    problem = sample(17, quadratic=.8)
    factors = np.r_[np.ones(2), np.logspace(-12, 12, 7)]
    groups = tuple(replace(g, lower_scale=np.asarray(g.lower_scale)/factors[np.asarray(g.rows)],
                           upper_scale=np.asarray(g.upper_scale)/factors[np.asarray(g.rows)])
                   for g in problem.groups)
    transformed = replace(problem, A=sparse.csc_matrix(sparse.diags(factors) @ problem.A),
                          lower=problem.lower*factors, upper=problem.upper*factors, groups=groups)
    first, second = PreparedPeakSolver(problem), PreparedPeakSolver(transformed)
    for step in range(4):
        q = problem.q+step*.1
        first.update(q=q, lower=problem.lower-step*.01, upper=problem.upper+step*.02)
        second.update(q=q, lower=(problem.lower-step*.01)*factors,
                      upper=(problem.upper+step*.02)*factors)
        a, b = first.solve(), second.solve()
        assert a['status'] == b['status'] == 'solved'
        assert a['x'] == pytest.approx(b['x'], abs=2e-6)
        assert a['metrics']['objective'] == pytest.approx(b['metrics']['objective'], abs=2e-6)
        assert b['metrics']['normalized_stationarity'] <= 1e-6
        assert b['metrics']['normalized_fenchel_residual'] <= 1e-6
        assert b['dimensions']['rows'] == problem.A.shape[0]


def test_transformed_bound_overflow_rejects_before_mutating_a_reference_workspace():
    problem = PeakProblem(sparse.eye(1, format='csc'), np.array([0.]),
                          sparse.csc_matrix([[1e-300]]), np.array([0.]), np.array([0.]),
                          (PeakGroup([0], lower_scale=1e300, upper_scale=1e300),))
    for method in ('native', 'epigraph', 'row-split-gsp'):
        workspace = PreparedPeakSolver(problem, method)
        with pytest.raises((ValueError, FloatingPointError)):
            workspace.update(upper=np.array([1e20]))
        assert workspace.problem.upper[0] == 0
