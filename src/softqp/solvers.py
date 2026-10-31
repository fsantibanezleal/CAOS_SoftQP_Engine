"""Full native generalized projection and independent explicit slack lifting."""
import time
import numpy as np
from scipy import sparse
from .problem import Problem

def solve(problem: Problem, method='generalized', **settings):
    problem.validate()
    start = time.perf_counter()
    defaults = dict(verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=100000, polishing=False)
    defaults.update(settings)
    p = problem.penalty
    n = problem.q.size
    m = problem.lower.size
    if method in ('generalized', 'asymmetric'):
        if method == 'generalized' and not (np.array_equal(p.lower_linear, p.upper_linear) and
                np.array_equal(p.lower_quadratic, p.upper_quadratic)):
            raise ValueError('generalized method requires symmetric penalties; use lifted for side-specific costs')
        from .interface import OSQP, PenaltyType
        solver = OSQP()
        solver.setup(sparse.triu(problem.P, format='csc'), problem.q, problem.A,
                     problem.lower, problem.upper, **defaults)
        types = np.where(p.soft, int(PenaltyType.OSQP_PENALTY_L1L2), int(PenaltyType.OSQP_PENALTY_NONE)).astype(np.int32)
        if method == 'asymmetric':
            solver._solver.setup_asymmetric_penalty(types, *[np.ascontiguousarray(a,dtype=np.float64) for a in
                (p.lower_linear,p.lower_quadratic,p.upper_linear,p.upper_quadratic)])
        else:
            solver.setup_penalty(types, alpha1=p.lower_linear, alpha2=p.lower_quadratic)
        result = solver.solve()
        x, y = result.x, result.y
        dimensions = dict(variables=n, rows=m, nnz=problem.P.nnz+problem.A.nnz)
    elif method == 'lifted':
        import osqp
        A = sparse.csc_matrix(problem.A)
        rows, columns, values = [], [], []
        linear, quadratic, lower, upper, original_rows = [], [], [], [], []
        # Two nonnegative slack variables for a two-sided soft row; finite sides only.
        for i in range(m):
            if not p.soft[i]:
                out = len(lower)
                a = A.getrow(i)
                rows.extend([out]*a.nnz); columns.extend(a.indices); values.extend(a.data)
                lower.append(problem.lower[i]); upper.append(problem.upper[i]); original_rows.append(i)
                continue
            for sign, bound, c, d in ((1., problem.lower[i], p.lower_linear[i], p.lower_quadratic[i]),
                                       (-1., problem.upper[i], p.upper_linear[i], p.upper_quadratic[i])):
                if not np.isfinite(bound):
                    continue
                j = len(linear); out = len(lower); a = A.getrow(i)
                rows.extend([out]*a.nnz+[out]); columns.extend(list(a.indices)+[n+j]); values.extend(list(a.data)+[sign])
                lower.append(bound if sign > 0 else -np.inf)
                upper.append(np.inf if sign > 0 else bound)
                original_rows.append(i); linear.append(c); quadratic.append(d)
        k = len(linear)
        B = sparse.coo_matrix((values,(rows,columns)), shape=(len(lower), n+k)).tocsc()
        main_rows = len(lower)
        B = sparse.vstack([B, sparse.hstack([sparse.csc_matrix((k,n)), sparse.eye(k)])],format='csc')
        P = sparse.block_diag([problem.P, sparse.diags(quadratic) if k else sparse.csc_matrix((0,0))],format='csc')
        solver = osqp.OSQP()
        solver.setup(sparse.triu(P,format='csc'), np.r_[problem.q, linear], B,
                     np.r_[lower,np.zeros(k)], np.r_[upper,np.full(k,np.inf)], **defaults)
        result = solver.solve()
        x = result.x[:n]
        y = np.zeros(m)
        np.add.at(y, original_rows, result.y[:main_rows])
        dimensions = dict(variables=n+k, rows=B.shape[0], nnz=P.nnz+B.nnz)
    else:
        raise ValueError(f'unknown method: {method}')
    elapsed = time.perf_counter()-start
    metrics = problem.evaluate(x) if result.info.status_val in (1,2) else None
    if metrics is not None:
        metrics['stationarity_residual'] = float(np.max(np.abs(problem.P@x + problem.q + problem.A.T@y),initial=0.))
    return dict(method=method, status=result.info.status, status_code=int(result.info.status_val),
                x=x.tolist() if metrics is not None else None, y=y.tolist() if metrics is not None else None, metrics=metrics, dimensions=dimensions,
                iterations=int(result.info.iter), setup_seconds=float(result.info.setup_time),
                solve_seconds=float(result.info.solve_time), elapsed_seconds=elapsed)
