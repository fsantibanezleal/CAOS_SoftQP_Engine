"""Paper-equation reconstruction of fixed-parameter FlexQP, not author code.

Oshin et al., arXiv:2512.01565v2, equations 23-26 and Algorithm 1.
The compatible adapter retains pure L1 soft inequalities only; it never softens
an original hard row or replaces a quadratic operating cost. No learned policy.
"""

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .solvers import UnsupportedProblemError


class FlexWorkspace:
    def __init__(
        self,
        problem,
        *,
        rho=1.0,
        sigma_x=1e-6,
        sigma_s=1.0,
        alpha=1.6,
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=100000,
    ):
        import copy

        start = time.perf_counter()
        self.problem = copy.deepcopy(problem)
        problem = self.problem
        problem.validate()
        for name, value in dict(
            rho=rho, sigma_x=sigma_x, sigma_s=sigma_s, eps_abs=eps_abs, eps_rel=eps_rel
        ).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 2:
            raise ValueError("alpha must be between zero and two")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        p = problem.penalty
        finite_l = np.isfinite(problem.lower)
        finite_u = np.isfinite(problem.upper)
        if np.any(~p.soft & (finite_l | finite_u)):
            raise UnsupportedProblemError(
                "FlexQP reconstruction relaxes its inequalities; original hard rows are not softened"
            )
        if np.any(p.soft & finite_l & (p.lower_quadratic != 0)) or np.any(
            p.soft & finite_u & (p.upper_quadratic != 0)
        ):
            raise UnsupportedProblemError(
                "FlexQP reconstruction requires pure L1 costs; quadratic costs are not removed"
            )
        rows = np.r_[np.flatnonzero(finite_u), np.flatnonzero(finite_l)]
        signs = np.r_[np.ones(finite_u.sum()), -np.ones(finite_l.sum())]
        mu = np.r_[p.upper_linear[finite_u], p.lower_linear[finite_l]]
        if np.any(mu <= 0):
            raise UnsupportedProblemError(
                "FlexQP reconstruction requires positive finite-side L1 costs"
            )
        if problem.q.size > 2048 or rows.size > 8192:
            raise UnsupportedProblemError(
                "FlexQP reconstruction exceeds its dense reference budget"
            )
        self.G = problem.A[rows].toarray() * signs[:, None]
        self.h = np.r_[problem.upper[finite_u], -problem.lower[finite_l]]
        self.mu = mu
        self.rows = rows
        self.signs = signs
        self.rho = rho
        self.sigma_x = sigma_x
        self.sigma_s = sigma_s
        self.alpha = alpha
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter
        self.coefficient = 1 / (1 / sigma_s + 1 / rho)
        K = (
            problem.P.toarray()
            + sigma_x * np.eye(problem.q.size)
            + self.coefficient * (self.G.T @ self.G)
        )
        self.factor = cho_factor(K, lower=True, check_finite=True)
        self.reset()
        self.setup_seconds = time.perf_counter() - start

    def reset(self):
        self.x = np.zeros(self.problem.q.size)
        self.s = np.zeros(self.h.size)
        self.z = np.zeros(self.h.size)
        self.w = np.zeros(self.h.size)
        self.y = np.zeros(self.h.size)

    def solve(self):
        start = time.perf_counter()
        p = self.problem
        x, s, z, w, y = (
            value.copy() for value in (self.x, self.s, self.z, self.w, self.y)
        )
        status = "maximum iterations reached"
        code = 7
        stationarity = float("inf")
        complementarity = float("inf")
        for iteration in range(1, self.max_iter + 1):
            rhs = (
                self.sigma_x * x
                - p.q
                + self.coefficient
                * self.G.T
                @ (self.h - s + w / self.sigma_s + z - y / self.rho)
            )
            xt = cho_solve(self.factor, rhs, check_finite=False)
            nu = self.coefficient * (
                self.G @ xt + s - w / self.sigma_s - z + y / self.rho - self.h
            )
            st = s - w / self.sigma_s - nu / self.sigma_s
            zt = z - y / self.rho + nu / self.rho
            x = self.alpha * xt + (1 - self.alpha) * x
            sr = self.alpha * st + (1 - self.alpha) * s
            zr = self.alpha * zt + (1 - self.alpha) * z
            s = np.maximum(sr + w / self.sigma_s, 0)
            arg = zr + y / self.rho
            z = np.maximum(arg - self.mu / self.rho, 0) - np.maximum(
                -arg - self.mu / self.rho, 0
            )
            w = w + self.sigma_s * (sr - s)
            y = y + self.rho * (zr - z)
            if not np.isfinite(x).all():
                status = "nonfinite iterate"
                code = 10
                break
            px = p.P @ x
            gy = self.G.T @ y
            gx = self.G @ x
            stationarity = float(np.max(np.abs(px + p.q + gy), initial=0))
            complementarity = float(
                np.max(
                    np.abs(y - np.clip(y + self.rho * (gx - self.h), 0, self.mu)),
                    initial=0,
                )
                / self.rho
            )
            dual_tol = self.eps_abs + self.eps_rel * max(
                np.max(np.abs(px), initial=0),
                np.max(np.abs(p.q), initial=0),
                np.max(np.abs(gy), initial=0),
            )
            primal_tol = self.eps_abs + self.eps_rel * max(
                np.max(np.abs(gx), initial=0), np.max(np.abs(self.h), initial=0)
            )
            if stationarity <= dual_tol and complementarity <= primal_tol:
                status = "solved"
                code = 1
                break
        elapsed = time.perf_counter() - start
        self.x, self.s, self.z, self.w, self.y = x, s, z, w, y
        original_y = np.zeros(p.lower.size)
        np.add.at(original_y, self.rows, self.signs * y)
        metrics = p.evaluate(x) if code == 1 else None
        if metrics is not None:
            metrics["stationarity_residual"] = stationarity
        k = self.h.size
        return dict(
            method="flex",
            status=status,
            status_code=code,
            x=x.tolist() if code == 1 else None,
            y=original_y.tolist() if code == 1 else None,
            metrics=metrics,
            dimensions=dict(
                variables=p.q.size + 2 * k,
                rows=2 * k,
                nnz=int(p.P.nnz + 2 * np.count_nonzero(self.G) + 2 * k),
            ),
            iterations=iteration,
            setup_seconds=self.setup_seconds,
            solve_seconds=elapsed,
            elapsed_seconds=self.setup_seconds + elapsed,
            backend_settings=dict(
                rho=self.rho,
                sigma_x=self.sigma_x,
                sigma_s=self.sigma_s,
                alpha=self.alpha,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter,
                factor_variables=p.q.size,
                complementarity_residual=complementarity,
                paper_reconstruction=1,
            ),
        )


def solve_flex(problem, **settings):
    return FlexWorkspace(problem, **settings).solve()
