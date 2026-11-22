"""Public method-extension solve paths and reusable input snapshots."""

import copy
from dataclasses import replace
import time

import numpy as np

from .backend import _DEPENDENCIES, _validate, solve_candidate
from .certificates import certify_recession, screens_recession
from .evaluation import evaluate
from .units import exact_float64_vector, scale_lift, separable_scales
from ..lifting import lift_problem


METHODS = tuple(_DEPENDENCIES)
_SUCCESS = {"clarabel": "Solved", "proxqp": "QPSolverOutput.PROXQP_SOLVED",
            "piqp": "Status.PIQP_SOLVED"}
_UNBOUNDED = {
    "DualInfeasible", "AlmostDualInfeasible",
    "QPSolverOutput.PROXQP_DUAL_INFEASIBLE", "Status.PIQP_DUAL_INFEASIBLE",
}
_LIMITS = {"H": 1e-6, "K": 1e-5, "F": 1e-5, "F_all": 1e-5, "D": 1e-7}


def _units(value, length, name):
    converted = np.ones(length) if value is None else exact_float64_vector(value, length, name)
    if np.any(converted <= 0):
        raise ValueError(f"{name} requires positive unit magnitudes")
    return converted


class ComparatorSession:
    """Prepare a fixed input layout. Each solve starts a fresh official backend."""

    def __init__(self, problem, method, *, variable_units=None, row_units=None,
                 objective_unit=1.0, peak_units=None, recession_certificate=None):
        start = time.perf_counter()
        _validate(problem.P, problem.q, problem.A, problem.lower, problem.upper)
        self.problem = copy.deepcopy(problem)
        self.method = method
        self.peak = not hasattr(problem, "penalty")
        self.dx = _units(variable_units, self.problem.q.size, "variable_units")
        self.dr = _units(row_units, self.problem.lower.size, "row_units")
        self.dt = _units(peak_units, len(self.problem.groups) if self.peak else 0, "peak_units")
        self.sf = _units([objective_unit], 1, "objective_unit")[0]
        if recession_certificate is not None and (
            not isinstance(recession_certificate, (tuple, list)) or len(recession_certificate) != 2
        ):
            raise ValueError("recession_certificate requires an anchor and a direction")
        self.certificate = copy.deepcopy(recession_certificate)
        self.data = self._prepare(self.problem)
        self.setup_wall_seconds = time.perf_counter() - start
        self.update_wall_seconds = 0.0
        self.update_count = 0

    def _prepare(self, problem):
        _validate(problem.P, problem.q, problem.A, problem.lower, problem.upper)
        if self.peak:
            from ..peak import _assemble, _vectors
            problem.validate()
            assembly = _assemble(problem, "epigraph")
            q, lo, up = _vectors(problem, assembly)
            matrices = assembly["P"], q, assembly["A"], lo, up
            hard = assembly["kind"] == 0
            side = np.abs(assembly["kind"]) == 1
            labels = problem.layout()[0]
            side_groups = labels[assembly["rows"][side]]
            auxiliary_units = self.dt[np.unique(side_groups)]
            constraint_units = np.r_[self.dr[assembly["rows"][hard]],
                                     self.dt[side_groups], auxiliary_units]

            def recover(x, y):
                return x[:problem.q.size], assembly["recover"] @ y
        else:
            # Use the same exact-symmetry and PSD guard for both problem families.
            from ..peak import _certify_psd
            _certify_psd(problem.P)
            problem.penalty.validate(problem.A.shape[0])
            lift = lift_problem(problem)
            matrices = lift.P, lift.q, lift.A, lift.lower, lift.upper
            recover, assembly = lift.recover, None
            auxiliary_units, constraint_units = separable_scales(problem, lift, self.dr)
        variable_units = np.r_[self.dx, auxiliary_units]
        normalized = scale_lift(matrices, variable_units, constraint_units, self.sf)
        _validate(*normalized)
        return matrices, recover, assembly, normalized, variable_units, constraint_units

    def update(self, *, q=None, lower=None, upper=None):
        start = time.perf_counter()
        p = self.problem
        replacements = {}
        for name, value in (("q", q), ("lower", lower), ("upper", upper)):
            if value is None:
                continue
            replacements[name] = exact_float64_vector(
                value, getattr(p, name).size, f"prepared {name}", allow_infinity=name != "q"
            )
        candidate = replace(p, **replacements)
        for old, new in ((p.lower, candidate.lower), (p.upper, candidate.upper)):
            if not np.array_equal(np.isfinite(old), np.isfinite(new)):
                raise ValueError("prepared workspace requires an unchanged finite-side pattern")
        hard = (p.layout()[0] < 0) if self.peak else ~p.penalty.soft
        if not np.array_equal((p.lower == p.upper)[hard], (candidate.lower == candidate.upper)[hard]):
            raise ValueError("hard equality classification must stay fixed")
        data = self._prepare(candidate)
        self.problem, self.data = candidate, data
        self.update_count += 1
        self.update_wall_seconds = time.perf_counter() - start

    def reset(self):
        """No backend warm state is retained between comparator solves."""

    def _certificate(self, candidate):
        if self.certificate is not None:
            anchor, direction = self.certificate
        else:
            anchor, direction = np.zeros_like(self.problem.q), candidate.direction
            if direction is not None:
                direction = direction[:anchor.size] * self.dx
                scale = np.max(np.abs(direction / self.dx), initial=0.0)
                if scale > 0 and np.isfinite(scale):
                    direction = direction / scale
        if direction is None and self.problem.q.size == 1:
            direction = -np.sign(self.problem.q) * self.dx
        if direction is None or not certify_recession(self.problem, anchor, direction, variable_units=self.dx):
            return None
        anchor = np.asarray(anchor, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                passed_screen = screens_recession(self.data[3], anchor / self.dx, direction / self.dx)
        except ArithmeticError:
            passed_screen = False
        if passed_screen:
            return dict(anchor=anchor.tolist(), direction=direction.tolist())
        return None

    def _epigraph(self, x_lift, y_lift, assembly):
        p, z = self.problem, self.data[0][2] @ x_lift
        n = p.q.size
        side = np.abs(assembly["kind"]) == 1
        slack = assembly["kind"] == 2
        lambdas, mu, t = y_lift[side], -y_lift[slack], x_lift[n:]
        labels = p.layout()[0]
        side_labels = labels[assembly["rows"][side]]
        gaps = self.data[0][4][side] - z[side]
        nonnegative = np.maximum(-lambdas, 0.0) * self.dt[side_labels] / self.sf
        live = np.unique(side_labels)
        dual_signs = np.r_[nonnegative, np.maximum(-mu, 0.0) * self.dt[live] / self.sf]
        residuals = []
        for index, group_id in enumerate(live):
            group = p.groups[group_id]
            prices = lambdas[side_labels == group_id]
            terms = group.linear, group.quadratic * t[index], np.sum(prices), mu[index]
            denominator = max(self.sf / self.dt[group_id], abs(terms[0]), abs(terms[1]),
                              np.sum(np.abs(prices)), abs(terms[3]))
            residuals.append(abs(terms[0] + terms[1] - terms[2] - terms[3]) / denominator)
        lhs = np.r_[lambdas * z[side], np.zeros_like(t)]
        rhs = np.r_[lambdas * self.data[0][4][side], mu * t]
        complements = np.abs(lhs - rhs) / np.maximum(self.sf, np.maximum(np.abs(lhs), np.abs(rhs)))
        feasibility = np.r_[np.maximum(-gaps, 0) / self.dt[side_labels],
                            np.maximum(-t, 0) / self.dt[live]]
        accepted = (
            all(np.isfinite(v).all() for v in (z, complements, residuals, dual_signs, feasibility))
            and np.max(dual_signs, initial=0.0) <= 1e-7
            and np.max(complements, initial=0.0) <= 1e-5
            and max(residuals, default=0.0) <= 1e-5
            and np.max(feasibility, initial=0.0) <= 1e-6
        )
        return dict(accepted=bool(accepted), peak=t.tolist(), group_ids=live.tolist(),
                    side_rows=assembly["rows"][side].tolist(), side_signs=assembly["kind"][side].tolist(),
                    lambda_sides=lambdas.tolist(), mu=mu.tolist(), stationarity=residuals,
                    complementarity=complements.tolist(), dual_sign_error=dual_signs.tolist(),
                    feasibility=feasibility.tolist())

    def recover_metrics(self, candidate):
        _, recover, assembly, _, dx_lift, dr_lift = self.data
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x_lift = candidate.x * dx_lift
            y_lift = candidate.y * (self.sf / dr_lift)
            x, y = recover(x_lift, y_lift)
            if not np.isfinite(x_lift).all() or not np.isfinite(y_lift).all():
                raise ArithmeticError("nonrepresentable recovered lift")
            metrics = evaluate(self.problem, x, y, self.dx, self.dr, self.sf)
            if self.peak:
                metrics["epigraph"] = self._epigraph(x_lift, y_lift, assembly)
        return x, y, metrics

    def solve(self):
        start = time.perf_counter()
        matrices, recover, assembly, normalized, dx_lift, dr_lift = self.data
        P, q, A, lower, upper = matrices
        package, version = _DEPENDENCIES[self.method]
        result = dict(method=self.method, status="inconclusive", status_code=0,
                      native_status=None, backend_package=package, backend_version=version,
                      x=None, y=None, metrics=None, certificate=None,
                      dimensions=dict(variables=q.size, rows=A.shape[0], nnz=P.nnz + A.nnz),
                      unit_map=dict(variables=self.dx.tolist(), rows=self.dr.tolist(),
                                    peaks=self.dt.tolist(), objective=float(self.sf),
                                    lifted_variables=dx_lift.tolist(), lifted_rows=dr_lift.tolist()),
                      iterations=None, setup_seconds=None, solve_seconds=None,
                      setup_wall_seconds=self.setup_wall_seconds, workspace_reused=False,
                      update_count=self.update_count, update_wall_seconds=self.update_wall_seconds)
        try:
            candidate = solve_candidate(self.method, *normalized)
            result.update(native_status=candidate.native_status, iterations=candidate.iterations,
                          setup_seconds=candidate.setup_seconds, solve_seconds=candidate.solve_seconds)
            if candidate.native_status in _UNBOUNDED:
                proof = self._certificate(candidate)
                if proof is not None:
                    result.update(status="unbounded", status_code=-1, certificate=proof)
                else:
                    result["reason"] = "exact-recession-proof-unavailable"
            elif candidate.native_status == _SUCCESS[self.method] and candidate.x is not None:
                x, y, metrics = self.recover_metrics(candidate)
                accepted = all(metrics[k] <= limit for k, limit in _LIMITS.items())
                if self.peak:
                    accepted = accepted and metrics["epigraph"]["accepted"]
                result.update(status="candidate" if accepted else "accuracy-rejected", status_code=int(accepted),
                              x=x.tolist(), y=y.tolist(), metrics=metrics)
            else:
                result["reason"] = "native-status-without-certified-result"
        except ImportError as error:
            result.update(status="backend-unavailable", reason=str(error))
        except (RuntimeError, ValueError) as error:
            result.update(status="native-failure", reason=str(error))
        except ArithmeticError as error:
            result.update(status="accuracy-rejected", reason=str(error))
        result["elapsed_seconds"] = time.perf_counter() - start
        return result
