# Changelog

## 0.01.000, unreleased

- Build the pinned OSQPSlack native core in a distinct `softqp` namespace.
- Validate original-space QPs and side-specific penalties.
- Solve independent explicit lifting using the stock OSQP distribution.
- Add native asymmetric projection, objective, conjugate, scaling and recession paths.
- Compare native and lifted solutions using analytic and generated tests.
- Add the optional DAQP 0.10.3 side-specific reference with independent objective and equality checks.
