"""numba.cuda kernels for the fused GPU evaluator primitives.

Drop-in replacement for `_cuda_evaluators.py`'s `apply_gate`/`eval_energy`,
which compile a hand-rolled PyTorch C++/CUDA extension
(`evaluators/csrc/evaluators.cu`) via `torch.utils.cpp_extension.load`. Both
`IncrementalGPUEvaluator` and `ParityEvenGPUEvaluator` (statevec_gpu.py) call
these two functions with identical signatures regardless of backend.

No `_load_ops()`-style lazy singleton is needed here: `@cuda.jit` decoration
builds a Dispatcher object, not a driver context, so compilation naturally
defers to the first real kernel launch on a real GPU. See
tests/test_numba_cuda_evaluators.py::test_importing_numba_cuda_evaluators_does_not_require_a_gpu
for the regression test that backs this claim.
"""

from __future__ import annotations

import numba
import torch
from numba import cuda

THREADS_PER_BLOCK = 256


@cuda.jit(device=True)
def _popcount_cuda(x):
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


@cuda.jit
def _apply_gate_kernel(
    psi_in, psi_out, xor_mask, parity, has_parity, c_phase_re, c_phase_im, cos_h, sin_h
):
    """Per-thread gate application on the statevector.

    psi_out[k] = cos_h*psi_in[k] - i*sin_h*c_phase*sign[k]*psi_in[k^xor_mask]
    """
    k = cuda.grid(1)
    n = psi_in.shape[0]
    if k >= n:
        return
    src = k ^ xor_mask
    ppsi = psi_in[src]
    if has_parity:
        s = parity[k]
        ppsi = complex(ppsi.real * s, ppsi.imag * s)
    c_phase = complex(c_phase_re, c_phase_im)
    ppsi = ppsi * c_phase
    psi_k = psi_in[k]
    psi_out[k] = complex(
        cos_h * psi_k.real + sin_h * ppsi.imag,
        cos_h * psi_k.imag - sin_h * ppsi.real,
    )


@cuda.jit
def _eval_energy_kernel(psi, h_diag, xor_masks, sign_masks, c_re, c_im, block_sums):
    """Per-thread accumulate + block reduction for the fused energy evaluation.

    Computes parity on-the-fly via ``_popcount_cuda(k & sign_masks[j]) & 1``
    (maps to single PTX ``popc`` instruction), matching the CPU numba kernel
    in ``numba_kernels.py``.  Eliminates the precomputed ``K_off x 2^N``
    float32 parity tensor (88 GB at N=26).
    """
    tid = cuda.threadIdx.x
    k = cuda.grid(1)
    n = psi.shape[0]
    k_off = xor_masks.shape[0]

    sdata = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=numba.float64)

    val = 0.0
    if k < n:
        psi_k = psi[k]
        re = psi_k.real
        im = psi_k.imag
        val = h_diag[k] * (re * re + im * im)
        for j in range(k_off):
            src = k ^ xor_masks[j]
            psi_src = psi[src]
            sm = sign_masks[j]
            if sm == 0:
                s = 1.0
            else:
                s = -1.0 if (_popcount_cuda(k & sm) & 1) else 1.0
            sre = psi_src.real * s
            sim = psi_src.imag * s
            dot_re = re * sre + im * sim
            dot_im = re * sim - im * sre
            val += dot_re * c_re[j] - dot_im * c_im[j]

    sdata[tid] = val
    cuda.syncthreads()

    stride = THREADS_PER_BLOCK // 2
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        block_sums[cuda.blockIdx.x] = sdata[0]


def apply_gate(psi_in, xor_mask, parity, c_phase_re, c_phase_im, cos_h, sin_h):
    n = psi_in.shape[0]
    psi_out = torch.empty_like(psi_in)
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    has_parity = parity.numel() > 0
    _apply_gate_kernel[blocks, THREADS_PER_BLOCK](
        cuda.as_cuda_array(psi_in),
        cuda.as_cuda_array(psi_out),
        int(xor_mask),
        cuda.as_cuda_array(parity),
        has_parity,
        float(c_phase_re),
        float(c_phase_im),
        float(cos_h),
        float(sin_h),
    )
    return psi_out


def eval_energy(psi, identity_coeff, h_diag, xor_masks, sign_masks, c_re, c_im):
    n = psi.shape[0]
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    block_sums = torch.zeros(blocks, dtype=torch.float64, device=psi.device)
    _eval_energy_kernel[blocks, THREADS_PER_BLOCK](
        cuda.as_cuda_array(psi),
        cuda.as_cuda_array(h_diag),
        cuda.as_cuda_array(xor_masks),
        cuda.as_cuda_array(sign_masks),
        cuda.as_cuda_array(c_re),
        cuda.as_cuda_array(c_im),
        cuda.as_cuda_array(block_sums),
    )
    return block_sums.sum() + identity_coeff
