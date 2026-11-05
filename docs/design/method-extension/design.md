# Prospective engine boundaries and implementation order

This unit extends the separately versioned `softqp` engine from source `722380f60ad6d9dc8fe273d0939242fb5876a1cc`. The current release remains 0.2.0 until an independently accepted implementation is ready for normal versioning and promotion. The product orchestrates data, blocking, registration, training jobs, evaluation, replay, manuscripts and deployment; the engine owns transformations, numerical algorithms, learned initializers and original-space recovery. It must not obtain historical native internals through a new mutable backdoor.

The complete reviewed mathematical, numerical, learning, timing and recovery specifications are exported unmodified under `specification/`. Their raw SHA-256 identities are recorded in `reviewed-design.json`. They are prospective requirements, not evidence of binary installation, completed training, performance improvement or a new original scientific contribution. The original private evidence bundle additionally preserves rejected revisions, exact expected keysets and primary-source receipts. This public engine export does not pretend to be that complete registration bundle.

## Dependency order

1. Declare typed engine/product boundaries and EARS requirements; implement both-side admission adversaries. Independently review the contracts before algorithm implementations consume them.
2. Admit a new isolated CPU Python 3.12 runtime using the reviewed exact candidate versions and resolved dependency/license locks. A matching wheel filename alone is insufficient: binary imports, minimal official calls, NumPy compatibility and representative memory probes must pass.
3. Implement the reversible transformations and Clarabel, ProxQP and PIQP verticals against independent original-space checks and exact small controls.
4. Implement the differentiated OSQP map and graph-guided DAQP initialization, with gradient, permutation, seed, calibration, serialization and real downstream-solver controls.
5. Independently review implemented source and real production paths. Freeze exact source/lock identities and a prospective registration before generating new scientific data, labels, weights or the held-out campaign.
6. Train and evaluate the exact declared populations with resumable jobs. A failed, incompatible, rejected or timed-out cell remains present. Only then export the new replay/models, update the technical report from the authoritative LaTeX base, review the rendered artifacts, and perform normal release/publication/deployment gates.

## Engine boundaries

`learning-original` admits original QP coefficients, units, semantics and source identity. `learning-transform` retains the reversible primal/dual and epigraph/cone map. `learning-features` supplies the declared ordered coordinate vector or typed graph and a training-only normalizer identity. `learning-checkpoint` contains numeric weights and declarative architecture/provenance; optimizer resume state is a separate trusted local artifact. `learning-calibration` binds the graph decision threshold to its exact checkpoint and calibration split. `learning-method-result` retains native status, recovered original results, diagnostics, all timing phases and explicit failures.

The remaining dataset, split, labels, evaluation and export-index boundaries belong to the product. All contracts remain versioned, strict and source-bound. New boundaries are separate from the immutable historical result schemas. The implementation must supply real examples and producer/consumer tests; this design does not claim those gates passed.

## Numerical distinctions

Transforms may share epigraph primitives; Clarabel's conic homogeneous embedding, ProxQP's proximal augmented-Lagrangian algorithm and PIQP's proximal interior-point algorithm remain genuine distinct algorithms. The learned OSQP map trains a parameter-to-iterate MLP through solver steps, then uses stock OSQP. Graph-guided DAQP learns active-set information from a different representation and target, then calls the real DAQP solver. It is never counted as a second name for the first network.

Exact certificate checks are separate from numerical screens. In particular, a residual below 1e-9 cannot prove a nullspace or a recession sign. The reviewed dyadic controls with coefficient 2^-34 are bounded problems whose numerical screens can pass. Public unboundedness requires the complete exact rational certificate; absent proof is inconclusive/refused.

CPU, single-thread execution is canonical. Current direct candidate pins are Clarabel 0.11.1, ProxSuite 0.7.2, PIQP 0.6.4, DAQP 0.10.3, stock OSQP 1.1.3, PyTorch 2.15.0+cpu and PyG 2.8.1. NumPy/SciPy and transitive versions remain subject to the isolated binary admission. An admission failure requires a separately reviewed compatible profile before registration; it does not permit an unrecorded downgrade of existing experiment or notebook environments.
