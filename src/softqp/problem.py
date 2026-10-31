"""Original-space contracts and objective recomputation, independent of a solver."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

@dataclass(frozen=True)
class Penalty:
    lower_linear: np.ndarray
    upper_linear: np.ndarray
    lower_quadratic: np.ndarray
    upper_quadratic: np.ndarray
    soft: np.ndarray

    @classmethod
    def uniform(cls, m, linear=0., quadratic=1., soft=True):
        return cls(*(np.full(m, v) for v in (linear, linear, quadratic, quadratic)),
                   np.full(m, soft, dtype=bool))

    def validate(self, m):
        for name in ('lower_linear', 'upper_linear', 'lower_quadratic', 'upper_quadratic'):
            a = np.asarray(getattr(self, name))
            if a.shape != (m,) or a.dtype.kind not in 'fi' or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f'{name}: expected {m} finite nonnegative coefficients')
        if np.asarray(self.soft).shape != (m,) or np.asarray(self.soft).dtype.kind != 'b':
            raise ValueError('soft: expected a boolean mask with one value per row')

@dataclass(frozen=True)
class Problem:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    penalty: Penalty

    def validate(self):
        n = self.P.shape[0]
        m = self.A.shape[0]
        if n == 0 or self.P.shape != (n, n) or self.A.shape[1] != n:
            raise ValueError('matrix dimensions do not define a nonempty QP')
        for name, a, shape in [('q', self.q, (n,)), ('lower', self.lower, (m,)), ('upper', self.upper, (m,))]:
            if np.asarray(a).shape != shape or np.asarray(a).dtype.kind not in 'fi' or np.isnan(a).any():
                raise ValueError(f'{name}: invalid shape, type or NaN')
        if not np.isfinite(self.q).all() or not np.isfinite(self.P.data).all() or not np.isfinite(self.A.data).all():
            raise ValueError('QP coefficients must be finite')
        if (self.lower > self.upper).any() or np.isposinf(self.lower).any() or np.isneginf(self.upper).any():
            raise ValueError('invalid interval bounds')
        delta = self.P - self.P.T
        if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
            raise ValueError('P must be symmetric, not just its upper triangle')
        smallest = np.linalg.eigvalsh(self.P.toarray())[0] if n <= 128 else eigsh(self.P, k=1, which='SA', return_eigenvectors=False)[0]
        if smallest < -1e-9 * max(1., sparse.linalg.norm(self.P)):
            raise ValueError('P must be positive semidefinite')
        self.penalty.validate(m)

    def evaluate(self, x):
        x = np.asarray(x)
        if x.shape != self.q.shape or not np.isfinite(x).all():
            raise ValueError('solution must be finite with one value per variable')
        z = self.A @ x
        lower = np.maximum(self.lower-z, 0.)
        upper = np.maximum(z-self.upper, 0.)
        p = self.penalty
        soft = p.soft
        base = float(.5*x @ (self.P @ x) + self.q @ x)
        cost = float(np.sum((p.lower_linear*lower + p.upper_linear*upper +
                            .5*p.lower_quadratic*lower**2 + .5*p.upper_quadratic*upper**2)[soft]))
        return dict(base_objective=base, penalty_objective=cost, objective=base+cost,
                    lower_violation=lower.tolist(), upper_violation=upper.tolist(),
                    slack=(lower-upper).tolist(),
                    hard_violation=float(np.max((lower+upper)[~soft], initial=0.)))
