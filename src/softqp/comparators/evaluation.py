"""Independent original-coordinate evaluation for extension candidates."""

import numpy as np


def _inf(values):
    return float(np.max(np.abs(values), initial=0.0))


def _conjugate(price, quadratic, magnitude):
    excess = np.maximum(magnitude - price, 0.0)
    out = np.zeros_like(excess)
    mask = quadratic > 0
    out[mask] = (excess[mask] / quadratic[mask]) * (0.5 * excess[mask])
    return out


def evaluate(problem, x, y, dx, dr, objective_unit):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        px, z = problem.P @ x, problem.A @ x
        if not np.isfinite(px).all() or not np.isfinite(z).all():
            raise ArithmeticError("nonrepresentable matrix product")
        lo = np.maximum(problem.lower - z, 0.0)
        up = np.maximum(z - problem.upper, 0.0)
        positive = np.where(np.isfinite(problem.upper), np.maximum(y, 0.0), 0.0)
        negative = np.where(np.isfinite(problem.lower), np.maximum(-y, 0.0), 0.0)
        phi = np.zeros_like(y)
        conjugate = np.zeros_like(y)
        if hasattr(problem, "penalty"):
            p = problem.penalty
            hard = ~p.soft
            bounds = p.soft & (p.upper_quadratic == 0)
            positive[bounds] = np.minimum(positive[bounds], p.upper_linear[bounds])
            bounds = p.soft & (p.lower_quadratic == 0)
            negative[bounds] = np.minimum(negative[bounds], p.lower_linear[bounds])
            s = p.soft
            phi[s] = (p.lower_linear[s] + 0.5 * p.lower_quadratic[s] * lo[s]) * lo[s]
            phi[s] += (p.upper_linear[s] + 0.5 * p.upper_quadratic[s] * up[s]) * up[s]
            conjugate[s] = _conjugate(p.upper_linear[s], p.upper_quadratic[s], positive[s])
            conjugate[s] += _conjugate(p.lower_linear[s], p.lower_quadratic[s], negative[s])
        else:
            labels, a, b, cl, cu = problem.layout()
            hard = labels < 0
            peaks = []
            for g, group in enumerate(problem.groups):
                rows = labels == g
                polar = np.sum(positive[rows] / cu[rows] + negative[rows] / cl[rows])
                if group.quadratic == 0 and polar > group.linear:
                    positive[rows] *= group.linear / polar
                    negative[rows] *= group.linear / polar
                    polar = group.linear
                peak = max(np.max(cl[rows] * lo[rows]), np.max(cu[rows] * up[rows]))
                peaks.append(float(peak))
        ybar = positive - negative
        support = np.zeros_like(y)
        support[ybar > 0] = problem.upper[ybar > 0] * ybar[ybar > 0]
        support[ybar < 0] = problem.lower[ybar < 0] * ybar[ybar < 0]
        inner = ybar * z
        if not hasattr(problem, "penalty"):
            for g, group in enumerate(problem.groups):
                rows = np.flatnonzero(labels == g)
                first = rows[0]
                peak = peaks[g]
                phi[first] = (a[first] + 0.5 * b[first] * peak) * peak
                polar = np.sum(positive[rows] / cu[rows] + negative[rows] / cl[rows])
                conjugate[first] = _conjugate(a[[first]], b[[first]], np.array([polar]))[0]
                for values in (support, inner):
                    total = np.sum(values[rows])
                    values[rows] = 0
                    values[first] = total
        parts = np.stack((phi, support, conjugate, inner))
        residual = np.abs(phi + support + conjugate - inner)
        f = _inf(residual / np.maximum(objective_unit, np.max(np.abs(parts), axis=0)))
        totals = np.sum(parts, axis=1)
        f_all = abs(totals[0] + totals[1] + totals[2] - totals[3]) / max(objective_unit, _inf(totals))
        gradients, normalized = [], []
        for dual in (y, ybar):
            aty = problem.A.T @ dual
            gradient = px + problem.q + aty
            gradients.append(gradient.tolist())
            normalized.append(_inf(dx * gradient / objective_unit) / max(
                1.0, _inf(dx * px / objective_unit), _inf(dx * problem.q / objective_unit),
                _inf(dx * aty / objective_unit),
            ))
        hard_error = np.maximum(lo[hard], up[hard])
        base = float((0.5 * x) @ px + x @ problem.q)
        objective = base + totals[0]
        metrics = dict(
            base_objective=base, penalty_objective=float(totals[0]), objective=float(objective),
            row_values=z.tolist(), lower_violation=lo.tolist(), upper_violation=up.tolist(),
            slack=(lo - up).tolist(), hard_violation=_inf(hard_error),
            stationarity_residual=_inf(gradients[0]), raw_stationarity=gradients[0],
            repaired_stationarity=gradients[1], repaired_y=ybar.tolist(),
            H=_inf(hard_error / dr[hard]), K=max(normalized), F=f, F_all=float(f_all),
            D=_inf(dr * (y - ybar) / objective_unit) / max(1.0, _inf(dr * y / objective_unit)),
        )
        if not hasattr(problem, "penalty"):
            metrics["peak_values"] = peaks
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ArithmeticError("nonrepresentable original-space evaluation")
        return metrics
