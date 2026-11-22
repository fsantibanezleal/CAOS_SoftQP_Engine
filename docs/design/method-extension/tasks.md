# Implementation and acceptance map

Every row is pending. These are prospective units with explicit dependencies; no previous implementation is retroactively declared to have followed this plan.

| Unit | Responsibilities and requirements | Depends on | Acceptance |
|---|---|---|---|
| E1 | Typed engine boundary schemas/readers, source identity, external path rules; ME-001, ME-010, ME-011, ME-014 | Accepted design | Real valid producer/consumer examples and mutation refusals; independent contract review |
| P1 | Product dataset/split/label/evaluation/index contracts; complete key inventory; ME-012, ME-016 | Accepted design | Both-side semantic and provenance adversaries; no historical bytes changed |
| E2 | Isolated dependency, binary, license and memory admission | E1, P1 | Exact resolved locks, official native calls, actual resource observations |
| E3 | Exact equivalent transforms, primal/dual recovery and independent checks; ME-002, ME-003, ME-005, ME-006 | E1, E2 | Small exact controls, directional/group compatibility, certificate counterexamples |
| E4 | Official Clarabel, ProxQP and PIQP solver verticals; ME-004, ME-012, ME-013 | E3 | Real solves, equivalent original objectives, failure/resource controls |
| E5 | Differentiated fixed-point method and public stock OSQP warm start; ME-007, ME-008 | E3 | Independent unroll parity, gradients and source-bound inference |
| E6 | LEConv graph method, DAQP initialization and calibrated numeric exports; ME-009, ME-010, ME-011 | E3 | Architecture/seed/permutation/mask/repair and actual downstream solve controls |
| I1 | Independent implementation and source admission; ME-015 | E4, E5, E6, P1 | Frozen reviewed source/lock/registration identities |
| P2 | New generation, labels, training/calibration, complete held-out bake and export; ME-014, ME-016 | I1 | All expected keys, actual checkpoints and held-out metrics, explicit negative outcomes |
| R1 | Current report, method documentation/notebooks/media, full UI and promotion/publication/live deployment | P2 | Applicable content, independent scientific/visual and exact-source release gates |

Engine unit tests cannot substitute for independent registration, trained checkpoints, product qualification or live deployment proof.
