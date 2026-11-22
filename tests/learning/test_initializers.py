"""Explicit synthetic software controls for the two R4 initialized solvers."""

from hashlib import sha256

import numpy as np
import pytest
from scipy import sparse

from softqp.learning.parameters import HEADS, SEEDS, bundle_sha256, load_weights, torch, weight_bytes
from softqp.learning.fixed_point import FixedPointProblem, L2WSHead, Normalizer, State, coordinate_names
from softqp.learning.graph import ActiveSetGNN, OneSidedLift, batch_graphs, robust_labels
from softqp.learning import solvers
from softqp.learning.solvers import (
    CheckpointUnavailable, DAQP_SETTINGS, InferenceBinding, PreparedLearningProblem, UnsupportedInput,
    ReferenceEvidence, _daq_native, _stock_setup, backend_settings, cold_labels, repair_active_set,
    solve_daqp, solve_osqp,
)
from softqp.problem import Penalty, Problem
from softqp.lifting import lift_problem


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def problem_data(q=(-2.0, -0.25), P=((2.0, 0.2), (0.2, 1.0))):
    return Problem(
        sparse.csc_matrix(np.array(P)), np.array(q),
        sparse.csc_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
        np.array([-np.inf, 0.25, -2.0]), np.array([0.4, 0.25, np.inf]),
        Penalty.uniform(3, linear=0.0, quadratic=0.0, soft=False),
    )


def _numpy_step(P, q, A, lo, up, state):
    x, y, w = state
    K = P + 1e-6 * np.eye(q.size) + 0.1 * A.T @ A
    x_next = np.linalg.solve(K, 1e-6 * x - q + A.T @ (0.1 * w - y))
    z = A @ x_next
    w = np.clip(z + y / 0.1, lo, up)
    return x_next, y + 0.1 * (z - w), w


def test_registered_features_and_numeric_refusal():
    lift = PreparedLearningProblem(problem_data())
    assert len(coordinate_names(*lift.shape)) == 23
    assert lift.features().tolist() == [
        2.0, 0.2, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -2.0, -0.25,
        0.0, 0.25, -2.0, 0.4, 0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    ]
    with pytest.raises(ValueError, match="nonrepresentable-transform"):
        FixedPointProblem([[1.0]], [0.0], [[1e308]], [-1.0], [1.0])
    with pytest.raises(ValueError, match="unsupported-input"):
        L2WSHead(1, 1, 41001)


def test_k5_unroll_gradient_and_actual_stock_c_workspace():
    physical = problem_data()
    P, q, A = physical.P.toarray(), physical.q, physical.A.toarray()
    lo, up = physical.lower, physical.upper
    fp = FixedPointProblem(P, q, A, lo, up)
    start = tensor([-2.5, 1.75, 0.75, -0.2, -0.45]).requires_grad_()
    z0 = start.detach().numpy()
    stock_matrices = physical.P, q, physical.A, lo, up
    x0, y0 = z0[:2], z0[2:]
    check = x0, y0, A @ x0
    state = fp.initial_state(start)
    for k in range(1, 7):
        state = fp.transition(state)
        check = _numpy_step(P, q, A, lo, up, check)
        for actual, independent in zip(state, check):
            np.testing.assert_allclose(actual.detach(), independent, atol=4e-15, rtol=0)
        stock = _stock_setup(stock_matrices, max_iter=k, check_termination=0)
        stock.warm_start(x=x0, y=y0)
        c_result = stock.solve(raise_error=False)
        np.testing.assert_allclose(c_result.x, check[0], atol=2e-13, rtol=0)
        np.testing.assert_allclose(c_result.y, check[1], atol=2e-13, rtol=0)
        if k == 5:
            after_five = check
    target = sum(np.mean((a - b)**2) for a, b in zip(check, after_five)) / 3
    loss = fp.residual_loss(start)
    assert float(loss.detach()) == pytest.approx(target, abs=1e-15)
    analytical, = torch.autograd.grad(loss, start)
    numeric = []
    for direction in torch.eye(start.numel(), dtype=torch.float64):
        upper = fp.residual_loss(start.detach() + 1e-5 * direction)
        lower = fp.residual_loss(start.detach() - 1e-5 * direction)
        numeric.append(float((upper - lower) / 2e-5))
    np.testing.assert_allclose(analytical, numeric, atol=1e-12, rtol=2e-5)
    assert torch.count_nonzero(analytical) == start.numel()
    assert fp.residual_loss(start, steps=0) != loss


@pytest.mark.parametrize("value,derivative", [(-2, 0), (-1, 0), (0, 1), (1, 0), (2, 0)])
def test_clamp_strict_interval_and_equality(value, derivative):
    fp = FixedPointProblem([[1.0]], [0.0], [[0.0], [0.0]], [-1.0, 0.5], [1.0, 0.5])
    y = tensor([value * 0.1, 0.05]).requires_grad_()
    next_state = fp.transition(State(tensor([0]), y, tensor([0, 0])))
    assert next_state.w.tolist() == [float(np.clip(value, -1, 1)), 0.5]
    assert torch.autograd.grad(next_state.w.sum(), y)[0].tolist() == [derivative * 10.0, 0.0]


def test_gradients_reach_actual_system_factor_and_q():
    P, q, A = (tensor(v).requires_grad_() for v in ([[1.5]], [-0.5], [[1.2]]))
    fp = FixedPointProblem(P, q, A, [-0.15], [0.15])
    loss = fp.residual_loss(tensor([0.3, -0.4]))
    assert all(bool(torch.isfinite(g).all() and (g != 0).all()) for g in torch.autograd.grad(loss, (P, q, A)))


def test_three_streams_exact_names_no_global_seed_mutation():
    initial_rng = torch.random.get_rng_state()
    graph_hashes, head_hashes = set(), set()
    for seed in SEEDS:
        graph = ActiveSetGNN(seed, 0.25)
        raw = weight_bytes(graph)
        assert raw == weight_bytes(ActiveSetGNN(seed, 0.25))
        assert len(raw) == 52098 * 8
        assert dict(graph.named_parameters())["output_layer.lin3.bias"].item() == pytest.approx(np.log(1 / 3))
        assert graph.output_layer.lin1.bias.item() == 0
        assert not any(name.endswith("lin2.bias") for name, _ in graph.named_parameters())
        graph_hashes.add(sha256(raw).hexdigest())
        model = L2WSHead(2, 3, seed)
        raw = weight_bytes(model)
        assert raw == weight_bytes(L2WSHead(2, 3, seed))
        assert len(raw) == 20229 * 8
        head_hashes.add(bundle_sha256({model.head: model}))
        for receipt in model.initialization:
            if receipt.name.endswith(".weight"):
                expected = sha256(f"softqp-learning-r2|L-1|{seed}|H-2-3|{receipt.name}".encode()).digest()
                assert receipt.initialization_seed == int.from_bytes(expected[:8], "little") % 2**63
    assert len(HEADS) == 14 and len(graph_hashes) == len(head_hashes) == 3
    assert torch.equal(torch.random.get_rng_state(), initial_rng)


def test_numeric_checkpoint_rejects_executable_nonfinite_and_bad_hash():
    model = L2WSHead(2, 3, 41001)
    raw = weight_bytes(model)
    bad = np.array([np.inf], dtype="<f8").tobytes() + raw[8:]
    for invalid, digest in ((raw, "0" * 64), (b"pickle", sha256(b"pickle").hexdigest()),
                            (bad, sha256(bad).hexdigest())):
        with pytest.raises(ValueError, match="(digest|size|nonfinite)"):
            load_weights(model, invalid, expected_sha256=digest)
        assert weight_bytes(model) == raw


def test_real_osqp_uses_original_lift_and_verified_model_normalizer():
    prepared = PreparedLearningProblem(problem_data())
    normalizer = Normalizer.from_admitted([0] * 23, [1] * 23, ["train-1"], [1])
    model = L2WSHead(2, 3, 41001)
    weights = sha256(weight_bytes(model)).hexdigest()
    binding = InferenceBinding(prepared.transform_sha256, weights, normalizer.numeric_sha256)
    reference = prepared.reference()
    assert reference["status"] == "oracle-agreed"
    result = solve_osqp(prepared, model=model, normalizer=normalizer, binding=binding, reference=reference)
    assert result["status"] == "accepted"
    assert np.any(result["initial_y"] != 0)
    assert result["checkpoint_sha256"] == weights
    cold = solve_osqp(prepared, mode="cold", reference=reference)
    assert cold["status"] == "accepted"
    np.testing.assert_allclose(result["x"], cold["x"], atol=1e-6, rtol=0)
    with pytest.raises(CheckpointUnavailable, match="mismatched learning input"):
        solve_osqp(prepared, model=model, normalizer=normalizer,
                   binding=InferenceBinding("0" * 64, weights, normalizer.numeric_sha256))
    with pytest.raises(CheckpointUnavailable, match="missing normalizer"):
        solve_osqp(prepared, model=model, normalizer=normalizer,
                   binding=InferenceBinding(prepared.transform_sha256, weights, "0" * 64))
    assert backend_settings()["osqp"]["rho_is_vec"] is False


def test_graph_equation_permutation_label_mask_and_batching():
    lift = PreparedLearningProblem(problem_data()).one_sided()
    graph = lift.graph()
    assert [side.side for side in lift.row_map] == ["upper", "equality", "lower"]
    assert graph.x.dtype == graph.edge_weight.dtype == torch.float64
    assert graph.edge_index.dtype == torch.int64
    assert graph.inequality_mask.tolist() == [False, False, True, False, True]
    model = ActiveSetGNN(41001, 0.3)
    layer = model.input_layer
    out = layer(graph.x, graph.edge_index, graph.edge_weight)
    independent = layer.lin3(graph.x).clone()
    for (j, i), weight in zip(graph.edge_index.T.tolist(), graph.edge_weight):
        independent[i] += weight * (layer.lin1(graph.x[j]) - layer.lin2(graph.x[i]))
    torch.testing.assert_close(out, independent, atol=2e-15, rtol=2e-15)
    order = [3, 1, 4, 2, 0]
    shuffled = graph.clone()
    shuffled.x = graph.x[order]
    inv = torch.argsort(torch.tensor(order))
    shuffled.edge_index = inv[graph.edge_index]
    other = OneSidedLift.from_interval(np.eye(1), np.array([0.0]), np.empty((0, 1)),
                                      np.empty(0), np.empty(0)).graph()
    assert other.x.tolist() == [[0.0] * 4]
    torch.testing.assert_close(model(shuffled), model(graph)[order], atol=2e-13, rtol=0)
    torch.testing.assert_close(model(batch_graphs([graph, other])),
                               torch.cat([model(graph), model(other)]), atol=2e-13, rtol=0)
    labels = robust_labels(lift, np.array([0.4, 0.25]), np.array([0.7, 10.0, 0.0]))
    assert labels.mask.tolist() == [False, False, True, False, True]
    assert labels.target.tolist() == [0, 0, 1, 0, 0]
    ambiguous = robust_labels(lift, np.array([0.4, 0.25]), np.array([1e-7, 10.0, 0.0]))
    assert not ambiguous.mask[2]
    bad_signed_dual = robust_labels(lift, np.array([0.4, 0.25]), np.array([0.7, 10.0, -2e-7]))
    assert not bad_signed_dual.mask[4]


def test_osqp_nonunity_physical_maps_for_warm_start_and_recovery():
    canonical = problem_data()
    dx, dr, sf = np.array([2., 3.]), np.array([4., 5., 6.]), 2.
    physical = Problem(
        sparse.csc_matrix(sf * canonical.P.toarray() / (dx[:, None] * dx)),
        sf * canonical.q / dx,
        sparse.csc_matrix(dr[:, None] * canonical.A.toarray() / dx),
        dr * canonical.lower, dr * canonical.upper, canonical.penalty,
    )
    prepared = PreparedLearningProblem(physical, variable_units=dx, row_units=dr, objective_unit=sf)
    torch.testing.assert_close(prepared.features(), PreparedLearningProblem(canonical).features(),
                               atol=2e-16, rtol=2e-16)
    normalizer = Normalizer.from_admitted([0] * 23, [1] * 23, ["train-1"], [1])
    model = L2WSHead(2, 3, 41001)
    weights = sha256(weight_bytes(model)).hexdigest()
    binding = InferenceBinding(prepared.transform_sha256, weights, normalizer.numeric_sha256)
    result = solve_osqp(prepared, model=model, normalizer=normalizer, binding=binding,
                        reference=prepared.reference())
    assert result["status"] == "accepted"
    prediction = model(prepared.features()).detach().numpy()
    np.testing.assert_allclose(result["initial_x"], dx * prediction[:2], atol=1e-15)
    np.testing.assert_allclose(result["initial_y"], sf / dr * prediction[2:], atol=1e-15)
    np.testing.assert_allclose(result["x"] / dx, [0.4, 0.25], atol=1e-6)


def test_real_daqp_labels_and_network_initialization():
    prepared = PreparedLearningProblem(problem_data())
    reference = prepared.reference()
    frame, detail = cold_labels(prepared, reference=reference)
    assert detail["status"] == "accepted"
    assert detail["sense"].tolist() == [0, 5, 0]
    assert frame.labels.target.tolist() == [0, 0, 1, 0, 0]
    model = ActiveSetGNN(41001, 0.25)
    binding = InferenceBinding(prepared.transform_sha256, sha256(weight_bytes(model)).hexdigest(), threshold=0.1)
    result = solve_daqp(prepared, model=model, binding=binding, reference=reference)
    assert result["status"] == "accepted"
    assert np.any(result["scores"] != 0)
    assert all(attempt.flags[1] == 5 for attempt in result["attempts"])
    assert result["attempts"][0].flags == tuple(np.where(
        prepared.one_sided().equality, 5, result["scores"] > binding.threshold
    ))
    with pytest.raises(ValueError, match="oracle-failed"):
        cold_labels(prepared, reference=None)
    singular = PreparedLearningProblem(problem_data(P=((1.0, 1.0), (1.0, 1.0))))
    with pytest.raises(UnsupportedInput, match="exactly positive definite"):
        singular.one_sided()
    assert backend_settings()["daqp-label"]["time_limit"] == 30


def test_candidate_cannot_supply_its_own_reference_or_mutate_trio_evidence():
    prepared = PreparedLearningProblem(problem_data())
    own = solve_daqp(prepared, mode="cold")
    assert own["status"] == "candidate"
    forged = dict(status="oracle-agreed", objective=own["metrics"]["objective"],
                  transform_sha256=prepared.transform_sha256)
    reference = prepared.reference()
    detached = reference.as_dict()
    detached["references"].pop("piqp")
    detached["objective"] = 123.0
    assert set(reference["references"]) == {"stock-osqp", "piqp", "clarabel"}
    assert reference["objective"] != 123.0
    for invalid in (forged, detached, reference.as_dict(), ReferenceEvidence()):
        for operation in (cold_labels,
                          lambda p, **r: solve_daqp(p, mode="cold", **r),
                          lambda p, **r: solve_osqp(p, mode="cold", **r)):
            with pytest.raises(ValueError, match="independent three-backend reference evidence"):
                operation(prepared, reference=invalid)
    changed = PreparedLearningProblem(problem_data(q=(-3.0, -0.25)))
    with pytest.raises(ValueError, match="independent reference transform mismatch"):
        cold_labels(changed, reference=reference)
    assert cold_labels(prepared, reference=reference)[1]["status"] == "accepted"


def test_public_learning_projections_cannot_mutate_prepared_reference_input():
    prepared = PreparedLearningProblem(problem_data())
    reference = prepared.reference()
    identity = prepared.transform_sha256
    expected = prepared.features()
    projection = prepared.fixed_point()
    projection.P.zero_()
    projection.A.zero_()
    projection.q[0] = 101
    projection.lower[-1] = -3
    projection.upper.fill_(0)
    one_sided = prepared.one_sided()
    one_sided.P.fill(0)
    one_sided.q[0] = 101
    one_sided.G.fill(0)
    one_sided.b.fill(0)
    features = prepared.features()
    features.zero_()
    assert prepared.transform_sha256 == identity
    torch.testing.assert_close(prepared.features(), expected, atol=0, rtol=0)
    assert prepared.fixed_point().lower[-1] == -2
    assert cold_labels(prepared, reference=reference)[1]["status"] == "accepted"

    q = tensor([-2.0])
    lower = np.array([-0.5])
    projection = FixedPointProblem([[1.0]], q, [[1.0]], lower, [0.5])
    q.zero_()
    lower[0] = -9
    assert projection.q[0] == -2
    assert projection.lower[0] == -0.5


def test_reference_binds_original_recovery_even_when_lifted_coefficients_match():
    soft = Problem(sparse.csc_matrix([[1.0]]), np.array([-2.0]),
                   sparse.csc_matrix([[1.0]]), np.array([-np.inf]),
                   np.array([0.5]), Penalty.uniform(1, quadratic=1.0))
    lifted = lift_problem(soft)
    hard = Problem(lifted.P, lifted.q, lifted.A, lifted.lower, lifted.upper,
                   Penalty.uniform(lifted.lower.size, linear=0.0, quadratic=0.0, soft=False))
    soft_prepared, hard_prepared = PreparedLearningProblem(soft), PreparedLearningProblem(hard)
    torch.testing.assert_close(soft_prepared.features(), hard_prepared.features(), atol=0, rtol=0)
    reference = soft_prepared.reference()
    assert reference["status"] == "oracle-agreed"
    with pytest.raises(ValueError, match="independent reference transform mismatch"):
        cold_labels(hard_prepared, reference=reference)


def test_cold_labels_gate_is_tighter_than_ordinary_candidate_gap(monkeypatch):
    problem = Problem(sparse.csc_matrix([[1.0]]), np.array([-10.0]),
                      sparse.csc_matrix([[1.0]]), np.array([20.0]), np.array([np.inf]),
                      Penalty.uniform(1, linear=0.0, quadratic=0.0, soft=False))
    prepared = PreparedLearningProblem(problem)
    reference = prepared.reference()
    assert reference["status"] == "oracle-agreed"
    x = np.array([20.0 + 5e-9])
    witness = (x, 0.0, 1, dict(setup_time=0.0, solve_time=0.0, iterations=1, lam=x - 10.0))
    monkeypatch.setattr(solvers, "_daq_native", lambda *_: witness)
    ordinary = solve_daqp(prepared, mode="cold", reference=reference)
    assert ordinary["status"] == "accepted"
    assert 1e-8 < ordinary["objective_gap"] < 1e-6
    assert ordinary["metrics"]["F"] < 1e-8 and ordinary["metrics"]["K"] < 1e-8
    with pytest.raises(ValueError, match="cold DAQP label rejected: accuracy-rejected"):
        cold_labels(prepared, reference=reference)


def test_global_retry_order_single_cold_and_other_failure():
    lift = OneSidedLift.from_interval(np.eye(2), np.zeros(2),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
        np.array([-np.inf, -np.inf, -np.inf, 0.0]), np.zeros(4))
    info = dict(setup_time=0.0, solve_time=0.0, iterations=0)
    native = lambda flag: (None, None, flag, info)
    _, trace = repair_active_set(lift, [0.9, 0.8, 0.7, 0.999], 0.5, lambda _: native(-6))
    assert [r.removed_row for r in trace] == [2, 1, 0, None]
    assert [r.flags for r in trace] == [(1, 1, 1, 5), (1, 1, 0, 5), (1, 0, 0, 5), (0, 0, 0, 5)]
    _, tied = repair_active_set(lift, [0.0, 0.8, 0.8, 0.0], 0.5, lambda _: native(-6))
    assert [r.removed_row for r in tied] == [1, 2, None]
    _, failure = repair_active_set(lift, [0.9, 0.8, 0.7, 0.99], 0.5, lambda _: native(-1))
    assert len(failure) == 1 and failure[0].removed_row is None
    _, no_prediction = repair_active_set(lift, [0, 0, 0, 1], 0.5, lambda _: native(-6))
    assert len(no_prediction) == 1 and no_prediction[0].flags == (0, 0, 0, 5)


def test_real_daqp_nc14_keeps_success_and_working_set_affects_native_iterations():
    lift = OneSidedLift.from_interval(np.eye(2), np.repeat(-1.0, 2),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.repeat(-np.inf, 3), np.zeros(3))
    native, trace = repair_active_set(lift, [0.9, 0.8, 0.7], 0.5,
                                      lambda flags: _daq_native(lift, flags, DAQP_SETTINGS))
    assert [row.flags for row in trace] == [(1, 1, 1)]
    assert trace[0].removed_row is None and trace[0].native_flag == 1
    cold = _daq_native(lift, np.zeros(3, dtype=np.intc), DAQP_SETTINGS)
    assert native[3]["iterations"] == 2 and cold[3]["iterations"] == 3
    np.testing.assert_allclose(native[0], [0.0, 0.0], atol=1e-7, rtol=0)


def test_real_daqp_rank_failure_retains_inconsistent_equalities():
    lift = OneSidedLift.from_interval(np.eye(2), np.repeat(-1.0, 2),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([0.0, 1.0, -np.inf]), np.array([0.0, 1.0, 0.0]))
    native, trace = repair_active_set(lift, [0.2, 0.1, 0.9], 0.5,
                                      lambda flags: _daq_native(lift, flags, DAQP_SETTINGS))
    assert [row.flags for row in trace] == [(5, 5, 1), (5, 5, 0)]
    assert [row.removed_row for row in trace] == [2, None]
    assert native[2] == -6 and trace[0].native_flag == -6
