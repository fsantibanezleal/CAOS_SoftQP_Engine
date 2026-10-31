# SoftQP

SoftQP exposes a pinned native OSQP solver with generalized soft-constraint penalties. It preserves the authors' C implementation and Python binding, with a separate import namespace and reproducible local-source build.

The implementation supports the published generalized projection method and a candidate asymmetric separable elastic extension. Native asymmetric solutions are checked against independently lifted QPs; no novelty or performance improvement is claimed.

## Install from source

Python 3.10 or newer, a C/C++ compiler, and CMake are required.

```bash
python -m pip install .
python -m pytest
```

## Penalties

Call `softqp.OSQP().setup(...)` with the usual QP matrices and bounds, then `setup_penalty(...)` with per-row types, linear and quadratic weights. Hard constraints use `PenaltyType.OSQP_PENALTY_NONE`. Quadratic, L1, elastic-net, Huber and grouped norms follow the pinned upstream implementation. See [the upstream tutorial](https://github.com/A2R-Lab/OSQPSlack).

## Provenance

The native source is the A2R-Lab OSQP fork at `2500405a1bc63387721feec64f0c050fa99223dd`; the binding is `d6df7f987ba40e75bfac338ab19a97b2c92f7334`. Both are Apache-2.0. Original attribution is preserved in [NOTICE](NOTICE) and `native/`.

The mathematical reference is [Schwan, Lew and Plancher (2026)](https://arxiv.org/abs/2610.35631). No paper timing is a measurement of this distribution.

## Original-space API

`softqp.problem.Problem` declares a symmetric PSD Hessian, sparse constraint matrix, interval bounds and `Penalty` arrays. `softqp.solvers.solve(problem, method)` accepts `lifted`, `generalized` (symmetric weights) or `asymmetric` (side-specific elastic weights). It reports original-space objective components, physical violations, hard-row error, stationarity, dimensions and solver work. Failed solves carry no purported solution.

The asymmetric update supports the CPU builtin backend. Its derivation, sign conventions and limitations are in [penalty semantics](docs/penalty-semantics.md). `softqp.reference` preserves the authors' lifting helpers for Huber and grouped penalties, with their MIT notice.
