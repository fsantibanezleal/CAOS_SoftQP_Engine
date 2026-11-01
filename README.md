# SoftQP

SoftQP exposes a pinned native OSQP solver with generalized soft-constraint penalties. It preserves the authors' C implementation and Python binding, with a separate import namespace, portable wheels and a reproducible local-source build.

The implementation supports the published generalized projection method and a candidate asymmetric separable elastic extension. Native asymmetric solutions are checked against independently lifted QPs; no novelty or performance improvement is claimed.

## Install

Version [0.1.0 is published on PyPI](https://pypi.org/project/softqp/0.1.0/). Portable wheels support CPython 3.12 on Windows x64 and Linux x86_64.

```bash
python -m pip install "softqp==0.1.0"
python -m pip install "softqp[comparisons]==0.1.0"  # optional DAQP and ElastiQP adapters
```

The [GitHub release](https://github.com/fsantibanezleal/CAOS_SoftQP_Engine/releases/tag/v0.01.000) contains the same source archive and wheels, with SHA-256 checksums. Every public file was matched to the tagged workflow output, and a fresh public-index Windows install passed all 94 native tests and the installed-license checks. Linux and Windows wheels pass isolated installed tests in the release workflow.

## Install from source

A C/C++ compiler and CMake are required for source installation. The source and wheel procedure is documented in [releasing](docs/releasing.md).

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

`softqp.problem.Problem` declares a symmetric PSD Hessian, sparse constraint matrix, interval bounds and `Penalty` arrays. `softqp.solvers.solve(problem, method)` accepts `lifted`, `generalized` (symmetric weights) `asymmetric` (side-specific elastic weights), or `daqp` with the `comparisons` extra (a dense active-set reference). It reports original-space objective components, physical violations, hard-row error, stationarity, dimensions and solver work. Failed solves carry no purported solution.

The asymmetric update supports the CPU builtin backend. Its derivation, sign conventions and limitations are in [penalty semantics](docs/penalty-semantics.md). `softqp.reference` preserves the authors' lifting helpers for Huber and grouped penalties, with their MIT notice.

The DAQP adapter preserves original-scale side costs, exports backend settings, and refuses unsupported objectives. See [DAQP comparison](docs/daqp-comparison.md) for its positive-definite and quadratic-side restrictions.

Symmetric penalties use the authors' compact signed/shared-slack lifting helper. Asymmetric penalties use explicit nonnegative lower and upper slacks. Performance comparisons must include the compact symmetric baseline rather than forcing an unnecessarily large general formulation.

## Directional coupled peaks (0.2.0 development)

The source development API adds `PeakGroup`, `PeakProblem`, `PreparedPeakSolver` and `solve_peak` for disjoint directional peak penalties, native original-row updates and exact compact/row-split/DAQP comparisons. The currently published 0.1.0 wheel does not contain this API. See [directional peak semantics and restrictions](docs/directional-peaks.md). No priority or runtime advantage is established.

## Repeated vector updates

```python
from softqp.sessions import PreparedSolver
workspace = PreparedSolver(problem, method="asymmetric")
first = workspace.solve()
workspace.update(q=next_q, lower=next_lower, upper=next_upper)
next_result = workspace.solve()
```

The workspace owns a copy of the input problem. Matrices, hard/soft row policy, side coefficients and finite-side pattern remain fixed. Invalid updates are rejected before the native workspace changes. `problem` returns a copy for inspection, and `reset()` zeros warm iterates while keeping the factorization. Generalized and compact lifted workspaces are also available; lifted workspaces default to the same native backend, with `reference_backend="stock"` for independent checking.

`setup_wall_seconds` includes initial validation and assembly; `update_wall_seconds` includes validation and vector construction. The returned native setup/solve timers keep their backend meaning. A session's `elapsed_seconds` measures that solve call including original-space evaluation, whereas cold `solve()` retains its documented timing scope. A benchmark must time its full declared operation explicitly and never sum repeated initial setup fields as if every stream step created a solver. Warm-state reuse does not guarantee a runtime gain.

`DaqpWorkspace` in `softqp.sessions` uses DAQP 0.10.3's documented vector update API without resending sense flags, preserving its active state. It retains original-scale side weights, hard-row policy, finite-side pattern and hard equality classification. Random mixed-row streams match independent cold lifting. It provides the same setup/update wall timers as the native workspace, so a comparison need not handicap DAQP by rebuilding it at every frame.
