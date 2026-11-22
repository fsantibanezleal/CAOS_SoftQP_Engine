"""Source, loss, calibration and numeric resume controls on small synthetic inputs."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from softqp.learning.parameters import HEADS, torch, weight_bytes
from softqp.learning.fixed_point import FixedPointProblem, L2WSHead, Normalizer
from softqp.learning.graph import (
    ActiveSetGNN, GraphFrame, Labels, Observation, OneSidedLift, calibrate, classification_loss,
    classification_metrics, fit_classes, robust_labels,
)
from softqp.learning.training import HeadTrainer, SourceUnit, balance_sources, weighted_units


SOURCE_DIGEST = sha256(b"synthetic controls").hexdigest()


def source(name, split, frames, *, family="N01", regime=0):
    return SourceUnit(name, family, regime, split, tuple(frames))


def fixed(shape, q):
    n, m = shape
    A = np.zeros((m, n))
    A[0, 0] = 1
    A[1, 1] = 1
    lo, up = np.zeros(m), np.ones(m)
    lo[1], up[1] = 0.2, 0.2
    return FixedPointProblem(np.eye(n), np.repeat(q, n), A, lo, up)


def test_source_balancing_conditioning_frames_and_train_only_fit():
    frames = [fixed((2, 3), q) for q in (1., 3., 4.)]
    a = source("01", "train", frames[:2])
    b = source("02", "train", [frames[2]], regime=1)
    c = source("03", "train", [frames[2]], family="N12")
    weights = balance_sources([a, b, c], "train")
    assert [w for _, w in weights] == [0.25, 0.25, 0.5]
    selected = weighted_units([weights[0], weights[2]], "train", condition=True)
    assert [w for _, w in selected] == [1/3, 2/3]
    norm = Normalizer.fit([weights[0], weights[2]])
    assert norm.source_ids == ("01", "03")
    assert norm.frame_weights == (1/6, 1/6, 2/3)
    expected_mean = 1/6 * 1 + 1/6 * 3 + 2/3 * 4
    expected_scale = np.sqrt(sum(w * (q - expected_mean)**2 for w, q in zip(norm.frame_weights, (1, 3, 4))))
    assert norm.mean[9].item() == pytest.approx(expected_mean)
    assert norm.scale[9].item() == pytest.approx(expected_scale)
    assert norm.scale[0].item() == 1
    with pytest.raises(ValueError, match="outside its admitted phase"):
        Normalizer.fit([(replace(a, split="test"), 1)])
    with pytest.raises(ValueError, match="duplicate"):
        Normalizer.fit([(a, 0.5), (a, 0.5)])


def graph_frame(*, masked=True):
    lift = OneSidedLift.from_interval(np.eye(1), np.array([-1.0]), np.array([[1.0], [-1.0], [0.0]]),
                                     np.array([-np.inf, -np.inf, 0.0]), np.array([0.2, 0.1, 0.0]))
    label = robust_labels(lift, np.array([0.2]), np.array([0.8, 0.0, 0.0]))
    if not masked:
        label = Labels(torch.zeros_like(label.target), torch.zeros_like(label.mask))
    return GraphFrame(lift.graph(), label)


def test_class_statistics_zero_masks_and_weighted_softplus():
    graph, empty = graph_frame(), graph_frame(masked=False)
    train = balance_sources(
        [source("s1", "train", [graph, empty]), source("s2", "train", [empty])], "train")
    stat = fit_classes(train)
    assert stat.mass0 == stat.mass1 == 0.125
    assert stat.weights == (1, 1)
    logits = torch.tensor([1e20, 3., -2., 1e20], dtype=torch.float64, requires_grad=True)
    loss = classification_loss(logits, graph.labels, stat)
    expected = (torch.nn.functional.softplus(-logits[1]) + torch.nn.functional.softplus(logits[2])) / 2
    torch.testing.assert_close(loss, expected)
    analytical, = torch.autograd.grad(loss, logits)
    assert analytical[0] == analytical[3] == 0
    zero = classification_loss(logits, empty.labels, stat)
    assert zero.item() == 0
    assert (torch.autograd.grad(zero, logits)[0] == 0).all()
    with pytest.raises(ValueError, match="positive weighted mass"):
        fit_classes([(source("s1", "train", [empty]), 1)])


def test_calibration_source_macro_ties_threshold_and_held_out_undefined():
    graph, empty = graph_frame(), graph_frame(masked=False)
    p = torch.tensor([1., 0.9, 0.1, 1.], dtype=torch.float64)
    observed = Observation(p, graph.labels, graph.graph.inequality_mask)
    empty_obs = Observation(p, empty.labels, graph.graph.inequality_mask)
    cal = [(source("s1", "calibration", [observed, empty_obs]), 0.5),
           (source("s2", "calibration", [empty_obs]), 0.5)]
    calibrated = calibrate(cal)
    assert calibrated["threshold"] == 0.8
    assert calibrated["scores"] == (0.5,)*8 + (0.0,)
    assert calibrated["zero_mask_frames"] == 2
    assert calibrated["brier"] == pytest.approx(0.01)
    assert calibrated["ece"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="all graph masks are empty"):
        calibrate([(source("s2", "calibration", [empty_obs]), 1)])
    with pytest.raises(ValueError, match="outside its admitted phase"):
        calibrate([(source("s2", "test", [observed]), 1)])
    result = classification_metrics([(source("s1", "test", [observed]), 1)], threshold=0.8)
    assert result["roc"] == ((0, 0), (0, 1), (1, 1))
    assert result["f1"] == result["coverage"] == 1
    one_class = Observation(p, Labels(graph.labels.target, graph.labels.target.bool()),
                            graph.graph.inequality_mask)
    result = classification_metrics([(source("s1", "test", [one_class]), 1)], threshold=0.8)
    assert result["roc"] is result["pr"] is None
    assert result["undefined_reason"] == "missing-class"


def assert_resume(make):
    uninterrupted = make()
    initial = uninterrupted.snapshot()
    uninterrupted.advance()
    first = uninterrupted.snapshot()
    uninterrupted.advance()
    latest = uninterrupted.snapshot()
    assert latest.best != initial.best
    assert torch.random.get_rng_state().dtype == torch.uint8
    resumed = make()
    resumed.restore(first, expected_sha256=first.sha256)
    assert resumed.snapshot() == first
    resumed.advance()
    assert resumed.snapshot() == latest
    assert len(resumed.history) == 3
    assert resumed.history[1].gradient_finite and resumed.history[1].gradient_nonzero
    assert resumed.history[1].optimizer_steps == 1
    assert resumed.history[1].permutation_sha256
    assert initial.best == initial.initial and first.best != initial.best
    with pytest.raises(ValueError, match="digest mismatch"):
        resumed.restore(replace(first, rng=b""), expected_sha256=first.sha256)
    mutated = replace(first, rng=b"")
    with pytest.raises(ValueError, match="stale best epoch, stopping or RNG"):
        resumed.restore(mutated, expected_sha256=mutated.sha256)
    assert resumed.snapshot() == latest


def test_l2ws_genuine_resume_and_immutable_best():
    train = [(source("s1", "train", [fixed((2, 3), -1.8), fixed((2, 3), -1.9)]), 1)]
    val = [(source("v1", "validation", [fixed((2, 3), -1.85)]), 1)]
    normalizer = Normalizer.fit(train)
    loss = lambda model, frame: frame.loss(model, normalizer)
    assert_resume(lambda: HeadTrainer(L2WSHead(2, 3, 41001), train, val, loss,
                                      source_sha256=SOURCE_DIGEST))


def test_graph_genuine_resume_finite_gradient_and_validation_masks():
    graph, empty = graph_frame(), graph_frame(masked=False)
    train = [(source("s1", "train", [graph, empty]), 1)]
    val = [(source("v1", "validation", [graph]), 1)]
    statistics = fit_classes(train)
    loss = lambda model, frame: frame.loss(model, statistics)
    make_model = lambda: ActiveSetGNN(41002, statistics.positive_prior)
    assert_resume(lambda: HeadTrainer(make_model(), train, val, loss, source_sha256=SOURCE_DIGEST))
    with pytest.raises(ValueError, match="all graph masks are empty"):
        HeadTrainer(make_model(), train, [(source("v1", "validation", [empty]), 1)],
                    loss, source_sha256=SOURCE_DIGEST)


@pytest.mark.parametrize("shape", HEADS)
def test_registered_head_numeric_optimizer_resume(shape):
    train = [(source("s1", "train", [fixed(shape, -1.8)]), 1)]
    val = [(source("v1", "validation", [fixed(shape, -1.85)]), 1)]
    # Shape inventory and resume across all heads use one analytic source each.
    loss = lambda model, frame: frame.residual_loss(model(frame.features()))
    def make():
        return HeadTrainer(L2WSHead(*shape, 41003), train, val, loss, source_sha256=SOURCE_DIGEST)
    trainer = make()
    trainer.advance()
    resume = trainer.snapshot()
    assert len(resume.moments) == len(resume.current) * 2
    trainer.advance()
    expected_digest = trainer.snapshot().sha256
    del trainer
    trainer = make()
    trainer.restore(resume, expected_sha256=resume.sha256)
    assert trainer.snapshot().sha256 == resume.sha256
    del resume
    trainer.advance()
    assert trainer.snapshot().sha256 == expected_digest


def test_whole_source_batches_use_fixed_weights_and_one_step():
    graph, empty = graph_frame(), graph_frame(masked=False)
    train = balance_sources([source(f"s{i}", "train", [graph, empty]) for i in range(9)], "train")
    val = [(source("v1", "validation", [graph]), 1)]
    stats = fit_classes(train)
    loss = lambda model, frame: frame.loss(model, stats)
    trainer = HeadTrainer(ActiveSetGNN(41002, stats.positive_prior), train, val, loss,
                          source_sha256=SOURCE_DIGEST)
    controls = []
    original_step = trainer.optimizer.step

    def capture_step():
        controls.append(tuple(p.grad.detach().clone() for p in trainer.parameters))
        # Keep the model fixed to test the exact reduction across the final short batch.
        return None

    trainer.optimizer.step = capture_step
    row = trainer.advance()
    trainer.optimizer.step = original_step
    actual = [left + right for left, right in zip(*controls)]
    expected = torch.autograd.grad(loss(trainer.model, graph) / 2, trainer.parameters)
    assert row.optimizer_steps == 2 and row.zero_mask_frames == 9
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=3e-16, rtol=3e-14)
