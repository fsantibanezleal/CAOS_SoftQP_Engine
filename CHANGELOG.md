# Changelog

## 0.01.000, 2026-11-01

- Build the pinned OSQPSlack native core in a distinct `softqp` namespace.
- Validate original-space QPs and side-specific penalties.
- Solve independent explicit lifting using the stock OSQP distribution.
- Add native asymmetric projection, objective, conjugate, scaling and recession paths.
- Compare native and lifted solutions using analytic and generated tests.
- Add the optional DAQP 0.10.3 side-specific reference with independent objective and equality checks.
- Add fixed-shape native and DAQP workspaces for repeated vector updates.
- Add fixed-parameter FlexQP reconstruction and the published ElastiQP 0.2.0 adapter on compatible pure L1 cases.
- Publish CPython 3.12 Windows x64 and Linux x86_64 wheels plus a source archive on PyPI as `softqp==0.1.0`.
- Verify public distribution hashes and a fresh public-index install, including all 94 native tests and dependency notices.
