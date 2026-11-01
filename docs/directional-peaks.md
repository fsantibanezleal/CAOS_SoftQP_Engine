# Directional coupled peak penalties

This API is under development for distribution 0.2.0. Public distribution 0.1.0 does not contain it. The method is a restricted native research implementation, with no established timing advantage or historical priority claim. The original negative 0.1.0 experiment remains unchanged.

## Declared objective

`PeakProblem` minimizes `0.5*x.T*P*x + q.T*x` plus one cost per disjoint soft group. Ungrouped interval rows remain hard. For physical row residual `r=Ax-clip(Ax,l,u)`, a group's cost is

```text
M(r) = max(0, max_i upper_scale_i*r_i, max_i -lower_scale_i*r_i)
cost = linear*M(r) + 0.5*quadratic*M(r)^2
```

Scales are strictly positive and declared in physical units; both cost coefficients are nonnegative. Increasing one row's violation can change the optimal correction of another row. Overlapping/empty groups, duplicate row membership, invalid intervals and negative curvature are rejected. A group with both coefficients zero has no bound cost and is genuinely free. It does not become a hard group.

```python
import numpy as np
from scipy import sparse
from softqp import PeakGroup, PeakProblem, solve_peak

problem = PeakProblem(
    P=sparse.eye(2, format="csc"), q=np.array([-2., -1.]),
    A=sparse.eye(2, format="csc"), lower=np.zeros(2), upper=np.zeros(2),
    groups=(PeakGroup([0, 1], lower_scale=[1., 1.], upper_scale=[2., 1.],
                      linear=0.7, quadratic=0.5),),
)
candidate = solve_peak(problem, "native")
oracle = solve_peak(problem, "epigraph")
print(candidate["metrics"]["objective"], oracle["metrics"]["objective"])
```

## Exact comparisons and contribution boundary

- `native` keeps the original number of decision variables and interval rows. It changes the native C projection, original objective, conjugate and recession paths.
- `epigraph` uses independent stock OSQP with one shared nonnegative peak variable per nonzero finite-sided group, omitting inactive zero-cost/free groups. It is a compact exact QP, not a per-row slack inflation.
- `row-split-gsp` uses the existing published symmetric native NORMINF implementation. Each finite upper side becomes `cU*A*x <= cU*u`; each finite lower side becomes `-cL*A*x <= -cL*l`, both with lower bound minus infinity. A symmetric peak group then represents the same directional linear cost. This comparator supports `quadratic=0`; the pinned native NORMINF API does not implement the squared peak even though the published theorem is broader.
- `epigraph-daqp` uses ordinary hard-constraint DAQP on the compact epigraph. This direct non-proximal adapter sets `eps_prox=0` and requires a positive definite Hessian. Pure linear epigraph variables or a semidefinite base Hessian can make this adapter incompatible; no regularizer is inserted. DAQP itself also supports PSD/LP objectives through [proximal-point iterations](https://darnstrom.github.io/daqp/); the separately reviewed `epigraph-daqp-prox` path uses those iterations on the unchanged original PSD epigraph. No permanent Hessian shift is introduced. Install the `comparisons` extra for DAQP.

Directional peak expressibility, infinity penalties, sorting/thresholding, Moreau decomposition and fixed-size ADMM linear systems have prior art. The candidate being tested is preservation of the original interval-row system compared with exact row splitting and compact epigraphs. No first-ever theorem or speed claim follows from that difference.

Primary references: [Schwan, Lew and Plancher (2026)](https://arxiv.org/html/2610.35631v1), [Chancelier and De Lara](https://arxiv.org/pdf/2001.10441), [Perez et al.](https://www.cs.cornell.edu/gomes/pdf/2022_perez_jartint_projection.pdf), and [Parikh and Boyd](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf).

`epigraph-daqp-prox` supports automatic (`eps_prox=-1e-6`) or forced positive proximal parameters and explicit `eta_prox`. In pinned DAQP 0.10.3, `Model.setup(None)` fails before the native LP code. For an identically zero Hessian, default `lp_mode="one-shot"` therefore uses the documented `daqp.solve(None, ...)` path. It reports `one_shot_lp=True` and `workspace_reused=False`; its actual setup cost is included in each solve. Optional `lp_mode="model"` retains an explicit zero-Hessian proximal Model workspace; its observed unbounded LP iteration limits remain unresolved, not false optimal solutions. Both policies must be identified before a comparison is frozen.

DAQP accepts `eps_abs` as its absolute primal/dual tolerance and a positive integer `max_iter`. The peak adapter rejects an explicitly supplied `eps_rel` because DAQP has no corresponding relative stopping control; original normalized residuals are still independently reported. Nonfinite/invalid controls are rejected. DAQP's positive 4 (optimal inexact) and 5 (feasible) flags retain candidate vectors for the original-space accuracy gate, with `inexact_candidate=True`; they do not collide with OSQP's different code meanings and do not automatically qualify an observation.

For nearly singular Hessians, a small negative computed eigenvalue is not by itself proof of negative curvature. Diagonal PSD blocks are checked directly. Numerically ambiguous blocks up to 32 variables use exact dyadic-rational Schur elimination on the submitted matrix, preserving its entries; zero/negative-pivot cases distinguish PSD from actual tiny negative curvature. Larger block-diagonal matrices can use the same small-block checks. A larger coupled ambiguous block is explicitly unsupported by this certification budget, not declared indefinite or silently regularized.

## Proximal and convergence scope

Let a proper closed convex violation penalty satisfy `Psi(D*w)<=Psi(w)` for every diagonal contraction with entries in [0,1], and let `Psi(0)=min Psi=0`. For every nonempty closed box C, minimal corrective slack gives `g_C(z)=Psi(z-project_C(z))`. This condition is also necessary if the identity is required for all boxes: choose C with coordinate endpoints `-D*w` and `-w`, and evaluate at zero. Coordinatewise sign symmetry is sufficient but not necessary. Corrective slack has the opposite sign from the violation residual, so its cost is `Phi(xi)=Psi(-xi)`.

Clipping any candidate to the segment between `project_C(v)` and v reduces both this reduced penalty and a positive diagonal quadratic distance. Thus the group update is `project_C(v)+prox_Psi^H(v-project_C(v))`. For the peak family, sort `c_i*abs(r_i)` and solve the consistent active-prefix scalar equation. Each boundary uses finite local floating-point neighbor spacing; an absolute tolerance floor would violate unit invariance. Nonrepresentable intermediate arithmetic fails explicitly.

By default a fixed setup-time row normalization uses D_i=max(cL_i,cU_i) on soft rows and D_i=1 on hard rows: A_bar=D*A, bounds_bar=D*bounds, c_bar=c/D, y_original=D*y_bar. D stays immutable across updates. This selects the physical metric D*H_bar*D, and is not a claim of unchanged iterates at the old unscaled scalar rho. It improves covariance when the same physical row is expressed in different units. `row_normalization=False` preserves the explicitly unscaled research mode.

Native peaks require a fixed positive metric, one finite scalar `0 < alpha < 2` (default 1.6), direct linear solves, no adaptive rho, no automatic OSQP/Ruiz scaling and no polishing. With at least one declared group, alpha is fixed for the directional workspace lifetime; changing it requires a new workspace. An empty group list installs an ordinary hard-QP backend and retains that backend's settings-update policy. Unsupported settings, penalty mutations, directional code generation and directional sensitivities are rejected. These are actual API restrictions. The theoretical diagonal-metric identity is broader than the public settings interface; grouped rows currently inherit the native fixed group rho convention.

For the exact two-block consensus splitting, the quadratic/linear-graph block has Hessian `P+S+A.T*H*A`, positive definite for S>0. The other block is the attained positive-metric proximal map. Under a finite saddle-point assumption, whitening the fixed metric gives the standard ADMM residual/objective/dual convergence result. It supplies no rate, hard-feasibility guarantee, adaptive-policy result or finite-precision proof. See [Boyd et al., section 3.2](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) and the [subproblem-existence caveat of Chen, Sun and Toh](https://arxiv.org/abs/1507.02051).

Fixed relaxation uses the existing consensus update `h=alpha*u_next+(1-alpha)*v`, followed by `v_next=prox_G(h+d)` and `d_next=d+h-v_next`. After one complete step, `s=v+d` obeys `s_next=(1-alpha/2)*s+(alpha/2)*R_F*R_G(s)`. Whitening preserves this identity. The reflected proximal composition is nonexpansive, so the strict interior interval gives an averaged map with a fixed point under the same saddle assumption. This is established relaxed ADMM machinery, with no new priority claim; see [Boyd et al., sections 3.4.2-3.4.3](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf). Endpoints are excluded: valid zero-cost examples can stall at zero or cycle at two. Values near endpoints may require arbitrarily many iterations. The proximal penalty coefficients do not acquire an alpha factor.

## Diagnostics and failure semantics

Original-space metrics report base/group/total objective, both violation directions, hard-row error and stationarity. The group's dual gauge is `L(y)=sum(y_positive/cU + (-y)_positive/cL)`. Its conjugate is `(max(L-a,0)^2)/(2*b)` for b>0, and the indicator of `L<=a` for b=0. It is a joint budget, not independent per-row caps. The box support is also required.

`exact_dual_domain` records exact support/budget membership. Numerical diagnostics separately project forbidden-side multipliers and linear-group budget excess, expose the repaired multiplier and repair magnitude, and recompute both Fenchel and stationarity with it. The common stationarity field conservatively includes raw and repaired values. A tiny multiplier multiplied by a huge A can otherwise hide a large original-space error. An out-of-domain original multiplier is never relabeled as an exact finite conjugate.

Quadratic group recession blocks every nonzero finite-side violating direction. Hard violating directions also block an unboundedness certificate. The peak path conservatively requires exactly zero computed `P*d` for its dual-infeasibility candidate; ambiguous numerical directions remain unresolved rather than being declared unbounded by a row-unit tolerance. The initial unnormalized development run had six equivalent-objective row scalings reach their iteration limits. Those failures are retained; the later fixed normalization is tested separately for physical covariance and update/recovery correctness. Badly conditioned inputs can still reach an iteration limit. An iteration limit is not a correctness or speed success.

Finite interval endpoints of magnitude 1e25 or above are rejected before the native sentinel can silently make them infinite. Transformed nonrepresentable data, nonfinite original metrics and kernel arithmetic fail explicitly. Numerical failure uses its own status and real IEEE NaN solution/certificate/diagnostic markers. Returned successful/approximate statuses still require the declared original-space accuracy gate for an experiment.

## Updates and timings

```python
from softqp import PreparedPeakSolver

workspace = PreparedPeakSolver(problem, "native")
first = workspace.solve()
workspace.update(q=problem.q + 0.1)
next_result = workspace.solve()
workspace.reset()
```

The workspace owns copies; snapshots cannot mutate it. Vector updates preserve P/A, group membership, coefficients, finite-side pattern and native hard-row rho classification (including its near-equality tolerance). Invalid updates are rejected before mutation. Native/OSQP reset zeros warm state while retaining the existing factorization. DAQP's pinned API has no reset operation: its reset explicitly rebuilds and reports `reset_refactorized` and `reset_wall_seconds`. Normal DAQP Model vector updates retain the active workspace. The one-shot LP policy has no persistent active state and no reset refactorization; its per-solve setup and policy flags remain explicit.

`solve_peak` measures the complete cold operation: validation, assembly, setup, solve and physical recovery. Prepared setup/update wall times are separate from native backend timers; prepared solve elapsed time includes physical diagnostics. Do not sum repeated initial setup fields as if each frame creates a solver, or omit baseline construction/recovery from a registered whole-operation comparison.
