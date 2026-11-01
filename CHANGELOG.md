# 0.02.000 / distribution 0.2.0 (unreleased)

- Add original-row directional coupled peak updates, compact stock-OSQP epigraph, existing symmetric row-split GSP and ordinary hard-epigraph DAQP comparisons.
- Include faithful proximal-point DAQP PSD/LP comparison, explicit one-shot/persistent LP policies, original-objective checks and backend-specific inexact candidates.
- Certify small ambiguous PSD blocks exactly without a Hessian shift and validate all exposed comparator controls.
- Add original-space group/dual diagnostics, fixed-shape warm updates and conservative hard/quadratic recession handling.
- Support fixed finite relaxation `0<alpha<2` (default 1.6), with immutable metric/alpha and no automatic scaling/polishing; reject unsupported code generation and derivatives.
- Correct adversarially found unit/maximal-float thresholds, settings/cache/allocation boundaries and invalid output markers.
- Generalize validated source-archive and exact-tag release checks. Original 0.1.0 artifacts and negative experiment stay immutable; no new speed or priority claim.

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
