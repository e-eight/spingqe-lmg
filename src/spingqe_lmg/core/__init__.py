"""Shared machinery used by multiple evaluator backends.

Caching (``caching.py``), matrix-free statevector ops (``statevec_ops.py``),
Numba CPU/CUDA kernels (``numba_kernels.py``, ``numba_cuda_kernels.py``), and
parity-even sector utilities (``parity.py``). Backend classes themselves live
in ``evaluators/``.
"""
