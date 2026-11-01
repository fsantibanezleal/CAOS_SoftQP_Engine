# FlexQP reconstruction and its comparison boundary

The adapter in `softqp.flex` implements fixed-parameter FlexQP from Oshin et al., [arXiv:2512.01565v2](https://arxiv.org/html/2512.01565v2), equations 23-26 and Algorithm 1. It is an independently authored paper-equation reconstruction, not the authors' code or their trained Deep FlexQP policy.

It converts each finite side of a pure L1 soft interval into an elastic inequality. Vector side weights are allowed by the paper's exact-penalty discussion. Original hard rows and quadratic costs are refused; they are never silently relaxed or removed. Equality relaxation and learned parameter policies are outside this adapter. A dense 2048-variable / 8192-finite-side budget bounds its use.

The implementation uses the positive definite Schur system of the paper with fixed `sigma_x=1e-6`, `sigma_s=1`, `rho=1` and `alpha=1.6`. The proximal terms regularize ADMM subproblems and do not add an objective regularizer. Convergence is declared from original stationarity and the projected dual-box condition for weighted hinges. The tolerances, iteration cap, complementarity diagnostic and factor dimension accompany every result. An iteration limit returns no fabricated solution.

Exact recovery of a feasible hard QP requires sufficiently large L1 weights relative to optimal multipliers; finite small weights can intentionally trade violations for base cost. The adapter claims no universal feasibility, infeasibility certificate or superiority. Its Python/SciPy loop is a reference implementation: timing against mature C solvers measures this implementation as well as the algorithm. Treat such timing separately from native-backend projection/lifting measurements.

Validation covers analytic lower/upper scalar optima with both low and exact-penalty weights, random weighted side costs against independent stock-OSQP lifting, refusal of changed objectives/hard rows, and honest iteration-limit outcomes.

## ElastiQP comparison

`softqp.elasti.solve_elastiqp` consumes the published ElastiQP 0.2.0 dual active-set backend (Apache-2.0; https://github.com/StanfordASL/elastiqp). It retains pure L1 soft inequalities and hard equalities. Original hard inequalities, quadratic soft costs and non-positive-definite Hessians are explicitly unsupported by this bounded adapter. No objective regularizer is inserted; the proximal parameter is set to zero for this positive-definite comparison. Tests verify physical row scaling, asymmetric side costs, independent lifted parity, stationarity and exact hard equalities. Actual numeric backend settings accompany results. Install the optional `comparisons` dependencies to use it.
