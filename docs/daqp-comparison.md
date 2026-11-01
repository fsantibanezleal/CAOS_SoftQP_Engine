# DAQP side-specific reference

Install `softqp[comparisons]` to use `solve(problem, 'daqp')`. The reference uses DAQP 0.10.3's `Model` API and its documented original-scale lower/upper weights, not global normalized defaults. The canonical linear costs map to `w_l` and `w_u`; each positive quadratic coefficient b maps to reciprocal weight rho=1/b. `w_soft=0` ensures a zero linear coefficient remains zero when DAQP falls back from an individual zero weight to its default.

The adapter accepts positive definite Hessians and strictly positive quadratic costs on finite soft sides. It does not add a regularizer or silently replace a pure L1 side. Hard equality rows use the active/immutable equality sense; soft rows retain the soft sense. Dense conversion is limited to 2048 variables and 8192 rows. An unsupported formulation raises `UnsupportedProblemError`, distinct from invalid data or a failed solve.

The common `eps_abs` sets DAQP's primal and dual tolerances; its actual backend settings are exported. DAQP has no ADMM rho adaptation or OSQP-style relative stopping tolerance, so those settings are not represented as equivalent. Comparisons must use independently measured original-space residuals and objective gaps. Native status 1 is optimal; native status 2 is soft optimal and is also a successful solve. The original status code is retained.

The objective and physical violations are recomputed outside DAQP. Tests use non-unit rows to check weight scaling and compare lower/upper violations and a hard equality against independently lifted QPs. A semidefinite test verifies that the adapter refuses the case rather than modifying its objective.

Sources: [DAQP soft constraints](https://darnstrom.github.io/daqp/start/advanced/soft) and the [official implementation](https://github.com/darnstrom/daqp), MIT.
