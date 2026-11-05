# Prospective method-extension CPU profile

Status: candidate binary admission; independent review pending. This new profile is separate from historical SDKs and experiments. It targets CPython 3.12 on Windows AMD64 with CPU float64, one numerical worker/thread and an 8 GiB admission RSS ceiling. At least 20 GiB external free disk is required.

Consumed: None. Produced: None. Locks and admission receipts are dependency/build evidence, not scientific artifacts or an additional learning contract. No generator, labels, optimizer updates, training, benchmark or learned method is implemented here.

Exact direct candidates are Clarabel 0.11.1, ProxSuite 0.7.2, PIQP 0.6.4, DAQP 0.10.3, stock OSQP 1.1.3, PyTorch 2.15.0+cpu, PyG 2.8.1, NumPy 2.4.6 and SciPy 1.18.1, with unchanged softqp 0.2.0 and ElastiQP 0.2.0 for coexistence. A failed candidate requires independent review of a compatible proposal before replacement; no silent downgrade is permitted.

Use a new external runtime/cache/temp root. Reproduce the complete hash lock once frozen, then run `python environments/method-extension-cpu/verify.py --check` from the repository root. The verifier calls the actual native solvers on two analytic equality/lower/upper controls, a true L1 ElastiQP control, CPU linalg/autograd and actual LEConv forward/backward. It also checks the declared 52,098-parameter graph shape and a transient representative coordinate-head kernel. These are untrained binary controls, never learned/scientific results.

The caller enforces the owned process-tree deadline/RSS/disk limits. Cold imports have a 180-second bound; numerical toy calls have a 10-second bound. BLAS/OpenMP/MKL/Torch are set to one thread. Clarabel's binary fixture explicitly uses max_threads=1 for that resource rule; its original defaults remain exported separately. Full scientific settings/parity, all method verticals, training and the 15,456-input/42,816-status reader throughput remain later gates.
