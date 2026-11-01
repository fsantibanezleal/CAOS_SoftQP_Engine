# Interval penalties and reflected projection

Let `C = [l,u]` be a nonempty closed interval, possibly unbounded. Let `phi` be a proper closed convex scalar function with `phi(0)=min(phi)=0`. For a constraint value z define `g(z)=inf{phi(xi): z+xi in C}`. Convexity and the minimum at zero imply that phi cannot decrease moving away from zero on either ray. If z belongs to C, xi=0 attains the infimum. If z>u, the largest feasible negative xi is u-z and is optimal; if z<l, the smallest feasible positive xi is l-z and is optimal. Thus `g(z)=phi(Pi_C(z)-z)`.

The two exterior branches are convex affine compositions and join the zero interval with nondecreasing one-sided slopes. Their endpoint values agree with zero and are lower semicontinuous because phi is closed. Therefore g is closed convex. The argument also covers a single endpoint, a half-line and the free interval, where the corresponding empty exterior branches disappear. The claim extends by summing independent row functions. Coupled penalties need a separate derivation.

For rho>0, minimize `g(z)+rho/2*(z-v)^2`. If v lies in C, z=v is optimal. Above C, a candidate below u is dominated by u because both the penalty and distance to v decrease moving toward u. Hence write z=u+s with s>=0; the problem is the reflected scalar proximal problem. A convex penalty minimized at zero has a proximal value between zero and its input, so the sign restriction is automatic. The below-C argument is identical with negative residual. Consequently:

`prox_{g/rho}(v)=Pi_C(v)+prox_{phi(-.)/rho}(v-Pi_C(v))`.

For side-specific elastic penalties on residual r, `a_lower*(-r)_+ + a_upper*(r)_+ + .5*b_lower*(-r)_+^2 + .5*b_upper*(r)_+^2`, the upper proximal offset is `(rho*r-a_upper)_+/(rho+b_upper)` and the lower offset is `-(rho*abs(r)-a_lower)_+/(rho+b_lower)`. Positive residual denotes an upper violation, while original signed slack is negative. The public coefficients are named by the violated bound to avoid that sign ambiguity.

The reduced problem splits as `min f(x)+g(z), Ax=z`. It is convex for symmetric positive semidefinite P and separable convex row penalties. Standard two-block ADMM convergence requires a solution/saddle point and the usual exact subproblem conditions. The native implementation retains OSQP's proximal regularization, KKT system, scaling and row types. Its extra work is per-row arithmetic and two m-length coefficient arrays; it adds no primal variables or constraint rows. This statement concerns matrix dimensions and does not assert faster convergence or runtime.

Native changes include the generalized projection, primal cost, conjugate value, scaling and recession cost. The conjugate uses the linear/quadratic coefficients selected by the dual sign, and a pure linear side has a bounded dual domain. Quadratic sides block escaping recession directions; linear sides charge their slope. Hard rows retain their interval recession cone. CPU builtin backend is the supported configuration. Updating asymmetric penalty types/weights and code generation are not public supported operations in this initial API; use a fresh solver with the desired coefficients.

Tests compare scaled and unscaled native solutions against independent explicit nonnegative lower/upper slacks solved by the stock OSQP distribution. Analytic scalar oracles check sign conventions and hard rows. These checks cannot establish novelty or performance improvement.

Related work: [OSQP](https://osqp.org/docs/solver/), [generalized slack projection](https://arxiv.org/abs/2610.35631), and [DAQP side-specific penalties](https://darnstrom.github.io/daqp/start/advanced/soft). Side-specific costs are already known; the candidate extension and its implementation remain subject to comparative evaluation.
